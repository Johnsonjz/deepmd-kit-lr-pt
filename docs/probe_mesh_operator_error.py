# Standalone verification tool, not package code.
# ruff: noqa
"""Print the real operator error behind an `Expected a proper Tensor` crash.

A `.pt2` runs its custom operators through the AOTI proxy executor, which
catches an operator exception, prints one `[E...] shim_common.cpp` line to
stderr, and leaves that operator's output tensors undefined.  The LAMMPS side
then dies in `flatten_canonical_atom_virial` with

    Expected a proper Tensor but got None (or an undefined Tensor in C++)

which says nothing about the failure.  This probe runs the same artifact
through the Python `pt_expt` path, where the shim's message is visible and the
call returns `nan` instead of aborting.

Usage (from the repo root, `module load gcc/13`):

    python docs/probe_mesh_operator_error.py lr_mid4ik
    SOG_MIDTOWN_SLAB=1 python docs/probe_mesh_operator_error.py lr_mid4ik

The default artifact directory is the frozen equal-accuracy set used by
docs/MIDTOWN_VS_BSPLINE_BENCH.md.
"""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
REPO = "/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt"
sys.path.insert(0, REPO)

FROZEN = "/data/home/public/jiangzhen/dp/sog_work/mdsuite/frozen/midtown_vs_bspline"


def frame(n: int = 64, length: float = 12.0):
    """A small neutral two-type periodic frame on a cubic grid."""
    k = int(round(n ** (1.0 / 3.0)))
    grid = np.linspace(0.0, length, k, endpoint=False)
    coord = np.array([[x, y, z] for x in grid for y in grid for z in grid])
    atype = np.zeros(len(coord), dtype=int)
    atype[1::2] = 1
    box = np.array([length, 0, 0, 0, length, 0, 0, 0, length])
    return coord, atype, box


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "lr_mid4ik"
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    path = name if os.path.isabs(name) else f"{FROZEN}/{name}.pt2"

    from deepmd.infer.deep_eval import DeepEval

    coord, atype, box = frame()
    print(f"artifact : {path}", flush=True)
    print(
        f"env      : SOG_MIDTOWN_SLAB={os.environ.get('SOG_MIDTOWN_SLAB', '<unset>')}",
        flush=True,
    )
    dp = DeepEval(path, device=device)
    try:
        out = dp.eval(
            coord.reshape(1, -1),
            box.reshape(1, -1),
            np.array([atype]),
            do_atomic_virial=True,
        )
    except Exception as exc:  # noqa: BLE001 - the point is to show any failure
        print(f"RAISED {type(exc).__name__}: {str(exc).splitlines()[0][:400]}")
        return 1

    energy = float(np.ravel(out[0])[0])
    status = (
        "NaN  (the operator threw and the shim swallowed it)"
        if not np.isfinite(energy)
        else "finite"
    )
    print(f"energy   : {energy:.9f}  -> {status}")
    print(f"force max: {np.abs(out[1]).max():.6e}")
    return 0 if np.isfinite(energy) else 2


if __name__ == "__main__":
    raise SystemExit(main())
