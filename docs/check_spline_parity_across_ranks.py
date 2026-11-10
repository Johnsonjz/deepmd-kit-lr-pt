# Standalone verification tool, not package code.
# ruff: noqa
"""Spline-mode parity across ranks: run under mpirun -np {1,2,4}.

Based on slab_parity.py.  Parameterised by MESH_BACKEND ("midtown" or
"quads") and ORDER (4/6) so the z-slab distributed solve can be checked for
every spline mode against the replicated one.  SOG_MIDTOWN_SLAB=1 selects the
slab path (else replicated).

usage:  mpirun -np 2 python quad_parity.py <tag>
env:    BOX_LX/LY/LZ, NATOMS, SPACING, ORDER, MESH_BACKEND, SOG_MIDTOWN_SLAB
"""

import ctypes
import os
import sys

LX = float(os.environ.get("BOX_LX", 18.0))
LY = float(os.environ.get("BOX_LY", 18.0))
LZ = float(os.environ.get("BOX_LZ", 18.0))
N = int(os.environ.get("NATOMS", 400))
OUT = os.environ.get("OUT", "/tmp/lr_md")
os.makedirs(OUT, exist_ok=True)
SP = float(os.environ.get("SPACING", 0.5))
ORDER = int(os.environ.get("ORDER", 6))
BACKEND = os.environ.get("MESH_BACKEND", "midtown")
COUL = 14.3996454784255

_HOLDER = None
_LIB = None
TAG_OF_RUN = os.environ.get("TAG", sys.argv[1] if len(sys.argv) > 1 else "?")
RANK = 0
NP = 1


def comm_address():
    global _HOLDER, _LIB, RANK, NP
    lib = ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL)
    lib.MPI_Init(None, None)
    world = ctypes.c_char.in_dll(lib, "ompi_mpi_comm_world")
    handle = ctypes.c_void_p(ctypes.addressof(world))
    rank = ctypes.c_int()
    size = ctypes.c_int()
    lib.MPI_Comm_rank(handle, ctypes.byref(rank))
    lib.MPI_Comm_size(handle, ctypes.byref(size))
    _HOLDER, _LIB, RANK, NP = handle, lib, rank.value, size.value
    if not os.environ.get("SLAB_SAME_GPU"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(RANK)
    return ctypes.addressof(handle)


ADDR = comm_address()

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    from deepmd.pt_expt.kernels.long_range import latent_charge_sum, op_available
    from deepmd.pt_expt.long_range.kernel import SOGKernel

    op_available()
    comm = torch.tensor([ADDR], dtype=torch.int64)

    g = torch.Generator().manual_seed(11)
    coord_all = torch.rand(N, 3, generator=g, dtype=torch.float64) * torch.tensor(
        [LX, LY, LZ], dtype=torch.float64
    )
    ch_all = torch.randn(N, 1, generator=g, dtype=torch.float32)

    # The slab solver assumes rank == z-slab: each rank must own the atoms of
    # the continuous subdomain [r*LZ/np, (r+1)*LZ/np), exactly as LAMMPS's
    # `processors 1 1 np` decomposition gives.  Splitting by atom count instead
    # (argsort) lets the boundary atom overhang into the neighbour's band, which
    # on a fine grid (small cells) trips the solver's z-subdomain guard.
    if os.environ.get("NOZBAND"):
        idx = np.arange(N)
    elif os.environ.get("IDXSPLIT"):
        # Naive contiguous index split: rank r owns atoms [r*N/np, (r+1)*N/np)
        # of the ORIGINAL (random) order, so a rank's atoms span the whole box.
        # This is the decomposition SOG_MIDTOWN_FULLBRICK handles; without the
        # full-brick switch the z-subdomain guard must reject it.
        lo = RANK * N // NP
        hi = (RANK + 1) * N // NP
        idx = np.arange(lo, hi)
    elif os.environ.get("COUNTSPLIT"):
        zkey = np.argsort(coord_all[:, 2].numpy(), kind="stable")
        lo = RANK * N // NP
        hi = (RANK + 1) * N // NP
        idx = zkey[lo:hi]
    else:
        z = coord_all[:, 2].numpy()
        band = np.minimum((z / LZ * NP).astype(int), NP - 1)
        idx = np.nonzero(band == RANK)[0]
    coord = coord_all[idx].cuda()
    ch = ch_all[idx].cuda()
    n = idx.size
    np.save(f"{OUT}/par_{TAG_OF_RUN}_i{RANK}.npy", idx)
    mask = torch.ones(n, dtype=torch.bool, device="cuda")
    nn = torch.tensor([n], dtype=torch.long, device="cuda")
    tot = torch.zeros(1, dtype=torch.float64, device="cuda")
    box = torch.tensor([[LX, 0, 0, 0, LY, 0, 0, 0, LZ]], dtype=torch.float64)
    k = SOGKernel(
        amplitude=[0.7, -0.2, 0.1], width=[0.8, 1.7, 3.2], precision="float32"
    )

    e, cg, f, av, vi = latent_charge_sum(
        coord,
        ch,
        mask,
        nn,
        tot,
        kernel=k,
        channel=0,
        prefactor=COUL,
        box=box,
        order=ORDER,
        spacing=SP,
        mesh_backend=BACKEND,
        communicator=comm,
    )
    torch.cuda.synchronize()
    tag = TAG_OF_RUN
    np.save(f"{OUT}/par_{tag}_f{RANK}.npy", f.double().reshape(-1).cpu().numpy())
    np.save(f"{OUT}/par_{tag}_e{RANK}.npy", e.double().reshape(-1).cpu().numpy())
    np.save(f"{OUT}/par_{tag}_v{RANK}.npy", vi.double().reshape(-1).cpu().numpy())
    _LIB.MPI_Finalize()


if __name__ == "__main__":
    main()
