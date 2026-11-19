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


def _world():
    """The OpenMPI library and an ``MPI_Comm`` variable holding MPI_COMM_WORLD.

    The operator reads an ``MPI_Comm`` variable at the address it is given, so
    the test hands it the address of a variable that holds the handle of the
    world communicator.  The handle of OpenMPI is the address of its global
    ``ompi_mpi_comm_world``.  ``(None, None)`` without an OpenMPI to read.
    """
    for name in ("libmpi.so.40", "libmpi.so"):
        try:
            mpi = ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        try:
            world = ctypes.c_void_p(
                ctypes.addressof(ctypes.c_char.in_dll(mpi, "ompi_mpi_comm_world"))
            )
        except (ValueError, AttributeError):
            continue
        initialized = ctypes.c_int()
        mpi.MPI_Initialized(ctypes.byref(initialized))
        if not initialized.value:
            mpi.MPI_Init(None, None)
            # A rank that initializes MPI has to finalize it, or mpirun
            # reports the whole job as an abnormal termination.
            atexit.register(mpi.MPI_Finalize)
        return mpi, world
    return None, None


def _address(mpi, name: str) -> ctypes.c_void_p:
    """The handle of the MPI object ``ompi_<name>``, such as a datatype."""
    return ctypes.c_void_p(ctypes.addressof(ctypes.c_char.in_dll(mpi, name)))


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
def test_rank_shares_add_up_to_the_frame_virial(mesh_backend: str) -> None:
    """The shares of the ranks sum to the frame virial of one domain."""
    mpi, world = _world()
    if mpi is None:
        pytest.skip("the test needs an OpenMPI world communicator")
    rank = ctypes.c_int()
    size = ctypes.c_int()
    mpi.MPI_Comm_rank(world, ctypes.byref(rank))
    mpi.MPI_Comm_size(world, ctypes.byref(size))
    rank, size = rank.value, size.value
    if size < 2:
        pytest.skip("the test needs at least two MPI ranks (mpirun -np 2)")
    torch.cuda.set_device(rank % torch.cuda.device_count())
    mpi_double = _address(mpi, "ompi_mpi_double")

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

    # The nodes of this domain, a strided split so that the gathered frame
    # interleaves the ranks as a domain decomposition does.
    atoms = coord.shape[0]
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
