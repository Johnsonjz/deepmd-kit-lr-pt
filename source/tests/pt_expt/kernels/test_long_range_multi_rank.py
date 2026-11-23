# SPDX-License-Identifier: LGPL-3.0-or-later
"""The share of a domain of the frame virial of `latent_charge_sum`.

The operator sums one frame across the domains of an MPI communicator: each
rank passes the charged nodes it owns, the call gathers them and evaluates the
whole frame, and every output is the part of the caller's nodes.  The per-node
outputs (energy, charge gradient, force, atomic virial) can therefore be
checked against a single-domain evaluation of the whole frame on each rank
alone, while the per-frame virial is a share: only the sum over the ranks is
the frame virial ``W``.  A backend that returned ``W`` itself on every rank
would be counted once per rank by the caller's frame sum, and the long-range
pressure would drift with the rank count.

Run under MPI, on a GPU, with at least two ranks::

    mpirun -np 2 python -m pytest \
        source/tests/pt_expt/kernels/test_long_range_multi_rank.py

The module skips without the operator, without CUDA, or without an OpenMPI
communicator of two or more ranks.
"""

from __future__ import (
    annotations,
)

import array
import atexit
import ctypes

import pytest
import torch

from deepmd.pt_expt.kernels.long_range import (
    latent_charge_sum,
    op_available,
)
from deepmd.pt_expt.long_range.kernel import (
    SOGKernel,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not op_available(),
    reason="the long-range operator is CUDA only",
)

COULOMB = 14.3996454784255
#: The tag of the exchange of the virial shares, and the number of its entries.
TAG = 7
ENTRIES = 9


def _mpi():
    """The MPI library, the world communicator handle and ``MPI_DOUBLE``.

    The operator reads an ``MPI_Comm`` variable at the address it is given, so
    the test hands it a variable that holds the handle of the world
    communicator.  OpenMPI exports that handle as the address of its global
    ``ompi_mpi_comm_world``; MPICH has no such global, so both handles come
    from ``MPI_Comm_f2c`` / ``MPI_Type_f2c`` of the Fortran handles mpi4py
    knows.  ``(None, None, None)`` without an MPI to read.
    """
    for name in ("libmpi.so.12", "libmpi.so.40", "libmpi.so"):
        try:
            mpi = ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        world, mpi_double = _handles(mpi)
        if world is None:
            continue
        initialized = ctypes.c_int()
        mpi.MPI_Initialized(ctypes.byref(initialized))
        if not initialized.value:
            mpi.MPI_Init(None, None)
            # A rank that initializes MPI has to finalize it, or mpirun
            # reports the whole job as an abnormal termination.
            atexit.register(mpi.MPI_Finalize)
        return mpi, world, mpi_double
    return None, None, None


def _handles(mpi):
    """The world communicator and the double datatype, through any MPI."""
    try:
        from mpi4py import MPI
    except ImportError:
        MPI = None
    if MPI is not None and hasattr(mpi, "MPI_Comm_f2c"):
        mpi.MPI_Comm_f2c.argtypes = [ctypes.c_int]
        mpi.MPI_Comm_f2c.restype = ctypes.c_void_p
        mpi.MPI_Type_f2c.argtypes = [ctypes.c_int]
        mpi.MPI_Type_f2c.restype = ctypes.c_void_p
        return (
            ctypes.c_void_p(mpi.MPI_Comm_f2c(int(MPI.COMM_WORLD.py2f()))),
            ctypes.c_void_p(mpi.MPI_Type_f2c(int(MPI.DOUBLE.py2f()))),
        )
    try:
        return (
            ctypes.c_void_p(
                ctypes.addressof(ctypes.c_char.in_dll(mpi, "ompi_mpi_comm_world"))
            ),
            ctypes.c_void_p(
                ctypes.addressof(ctypes.c_char.in_dll(mpi, "ompi_mpi_double"))
            ),
        )
    except (ValueError, AttributeError):
        return None, None


def _system(atoms: int = 192, length: float = 12.4447):
    generator = torch.Generator(device="cpu").manual_seed(0)
    coord = (
        torch.rand(atoms, 3, generator=generator, dtype=torch.float64) * length
    ).cuda()
    charge = torch.randn(atoms, 1, generator=generator, dtype=torch.float32).cuda()
    mask = torch.ones(atoms, dtype=torch.bool, device="cuda")
    n_node = torch.tensor([atoms], dtype=torch.long, device="cuda")
    total = torch.zeros(1, dtype=torch.float64, device="cuda")
    box = torch.tensor(
        [[length, 0.0, 0.0, 0.0, length, 0.0, 0.0, 0.0, length]],
        dtype=torch.float64,
    )
    return coord, charge, mask, n_node, total, box


@pytest.mark.parametrize("mesh_backend", ["bspline", "midtown"])
@pytest.mark.parametrize("slab", [False, True])
def test_rank_shares_add_up_to_the_frame_virial(
    mesh_backend: str, slab: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shares of the ranks sum to the frame virial of one domain.

    ``slab`` exercises the z-slab distributed mesh (two NCCL transposes); at
    two ranks the slab and the replicated solve must return the same per-node
    parts and the same frame virial.
    """
    monkeypatch.delenv("SOG_MIDTOWN_SLAB", raising=False)
    monkeypatch.delenv("SOG_MIDTOWN_FULLBRICK", raising=False)
    if mesh_backend == "midtown":
        monkeypatch.setenv("SOG_MIDTOWN_FORM_A", "0")
    mpi, world, mpi_double = _mpi()
    if mpi is None:
        pytest.skip("the test needs an MPI world communicator (mpi4py or OpenMPI)")
    rank = ctypes.c_int()
    size = ctypes.c_int()
    mpi.MPI_Comm_rank(world, ctypes.byref(rank))
    mpi.MPI_Comm_size(world, ctypes.byref(size))
    rank, size = rank.value, size.value
    if size < 2:
        pytest.skip("the test needs at least two MPI ranks (mpirun -np 2)")
    if slab and torch.cuda.device_count() < size:
        pytest.skip("the slab solve needs one GPU per rank")
    torch.cuda.set_device(rank % torch.cuda.device_count())

    coord, charge, mask, n_node, total, box = _system()
    kernel = SOGKernel(
        amplitude=[0.7, -0.2, 0.1], width=[0.8, 1.7, 3.2], precision="float32"
    )
    common = {
        "kernel": kernel,
        "channel": 0,
        "prefactor": COULOMB,
        "box": box,
        "order": 6,
        "spacing": 0.4,
        "mesh_backend": mesh_backend,
    }
    reference = latent_charge_sum(coord, charge, mask, n_node, total, **common)
    if slab:
        monkeypatch.setenv("SOG_MIDTOWN_SLAB", "1")

    # The nodes of this domain.  The slab solve needs contiguous z slabs
    # (the LAMMPS processor grid is 1 x 1 x nprocs); the replicated solve
    # accepts the same split, so use it for both.
    atoms = coord.shape[0]
    if slab:
        # Assign atoms to the z slab that contains them, exactly as a
        # 1 x 1 x nprocs LAMMPS processor grid would.  Splitting by atom
        # count would let a rank own atoms on both sides of its slab.
        length = float(box[0, 0])
        slab_of = (coord[:, 2] / length * size).floor().clamp(0, size - 1)
        owned = torch.nonzero(slab_of == rank, as_tuple=False).flatten()
    else:
        owned = torch.arange(rank, atoms, size, dtype=torch.long, device="cuda")
    handle = ctypes.c_void_p(world.value)
    communicator = torch.tensor(
        [ctypes.addressof(handle)], dtype=torch.int64, device="cpu"
    )
    share = latent_charge_sum(
        coord[owned].contiguous(),
        charge[owned].contiguous(),
        mask[owned].contiguous(),
        torch.tensor([owned.numel()], dtype=torch.long, device="cuda"),
        total,
        communicator=communicator,
        **common,
    )
    # The per-node outputs are the parts of the caller's nodes: they must equal
    # the single-domain ones there, whatever the number of ranks.
    for name, got, ref in (
        ("energy", share[0], reference[0][owned]),
        ("charge_gradient", share[1], reference[1][owned]),
        ("force", share[2], reference[2][owned]),
        ("atom_virial", share[3], reference[3][owned]),
    ):
        got = got.detach().to(torch.float64)
        ref = ref.detach().to(torch.float64)
        scale = max(float(ref.abs().max()), 1e-12)
        error = float((got - ref).abs().max()) / scale
        assert error < 5e-3, f"{name} of rank {rank}: relative error {error:.3e}"

    # The frame virial is the sum of the shares: rank 0 receives those of the
    # other ranks and adds its own.  A rank that returned the frame virial
    # itself would make this sum ``size`` times the frame virial.
    mine = array.array("d", share[4][0].detach().cpu().tolist())
    if rank == 0:
        share_sum = array.array("d", mine)
        status = ctypes.create_string_buffer(256)
        for source in range(1, size):
            other = array.array("d", [0.0] * ENTRIES)
            mpi.MPI_Recv(
                ctypes.cast(other.buffer_info()[0], ctypes.c_void_p),
                ENTRIES,
                mpi_double,
                source,
                TAG,
                world,
                status,
            )
            for k, value in enumerate(other):
                share_sum[k] += value
        frame = reference[4][0].detach().to(torch.float64).cpu()
        scale = max(float(frame.abs().max()), 1.0)
        for k in range(ENTRIES):
            assert abs(share_sum[k] - float(frame[k])) / scale < 5e-3, (
                f"{mesh_backend}: the shares of {size} ranks sum to "
                f"{share_sum[k]:.6f} for component {k} of the frame virial "
                f"{float(frame[k]):.6f}"
            )
    else:
        mpi.MPI_Send(
            ctypes.cast(mine.buffer_info()[0], ctypes.c_void_p),
            ENTRIES,
            mpi_double,
            0,
            TAG,
            world,
        )


@pytest.mark.parametrize("mesh_backend", ["bspline", "midtown"])
def test_an_empty_rank_joins_the_slab_collective(
    mesh_backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rank with no atoms still enters the z-slab NCCL collectives.

    This is the shape a LAMMPS ``1 x 1 x nprocs`` grid produces when one
    subdomain is empty.  The call must return empty per-node outputs on that
    rank and finite shares on the owner, not abort inside a kernel launch or
    hang in an all-to-all.
    """
    mpi, world, mpi_double = _mpi()
    if mpi is None:
        pytest.skip("the test needs an MPI world communicator (mpi4py or OpenMPI)")
    rank = ctypes.c_int()
    size = ctypes.c_int()
    mpi.MPI_Comm_rank(world, ctypes.byref(rank))
    mpi.MPI_Comm_size(world, ctypes.byref(size))
    rank, size = rank.value, size.value
    if size < 2:
        pytest.skip("the test needs at least two MPI ranks (mpirun -np 2)")
    if torch.cuda.device_count() < size:
        pytest.skip("the slab solve needs one GPU per rank")
    torch.cuda.set_device(rank % torch.cuda.device_count())

    coord, charge, mask, n_node, total, box = _system()
    owned = (
        torch.arange(coord.shape[0], dtype=torch.long, device="cuda")
        if rank == 0
        else torch.empty(0, dtype=torch.long, device="cuda")
    )
    kernel = SOGKernel(
        amplitude=[0.7, -0.2, 0.1], width=[0.8, 1.7, 3.2], precision="float32"
    )
    handle = ctypes.c_void_p(world.value)
    communicator = torch.tensor(
        [ctypes.addressof(handle)], dtype=torch.int64, device="cpu"
    )
    monkeypatch.delenv("SOG_MIDTOWN_SLAB", raising=False)
    monkeypatch.setenv("SOG_MIDTOWN_SLAB", "1")
    if mesh_backend == "midtown":
        monkeypatch.setenv("SOG_MIDTOWN_FORM_A", "0")
    out = latent_charge_sum(
        coord[owned].contiguous(),
        charge[owned].contiguous(),
        mask[owned].contiguous(),
        torch.tensor([owned.numel()], dtype=torch.long, device="cuda"),
        total,
        communicator=communicator,
        kernel=kernel,
        channel=0,
        prefactor=COULOMB,
        box=box,
        order=6,
        spacing=0.4,
        mesh_backend=mesh_backend,
    )
    torch.cuda.synchronize()
    n = owned.numel()
    assert out[0].shape == (n, 1)
    assert out[1].shape == (n, 1)
    assert out[2].shape == (n, 3)
    assert out[3].shape == (n, 9)
    assert out[4].shape == (1, 9)
    assert torch.isfinite(out[4]).all()
