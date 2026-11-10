# Standalone verification tool, not package code.
# ruff: noqa
"""Compare two tagged parity runs: per-rank virial (not summed) + per-atom E/F.

Per-atom E and F are put back in global atom order using the recorded index
arrays (`_i{r}.npy`), so the comparison is valid for any per-rank split -- the
continuous z-band split (rank r owns a z-slice, not a contiguous index slice)
scatters each rank's atoms through the global index, and a bare concatenation
would misalign them.
"""

import glob
import os
import sys

import numpy as np

OUT = os.environ.get("OUT", "/tmp/lr_md")


def _rank_key(path):
    # ..._e<R>.npy -> R, so ranks sort numerically (e10 after e2).
    return int(path.rsplit("_", 1)[1].split(".")[0][1:])


def load(tag):
    e_files = sorted(glob.glob(f"{OUT}/par_{tag}_e*.npy"), key=_rank_key)
    f_files = sorted(glob.glob(f"{OUT}/par_{tag}_f*.npy"), key=_rank_key)
    i_files = sorted(glob.glob(f"{OUT}/par_{tag}_i*.npy"), key=_rank_key)
    I = np.concatenate([np.load(x) for x in i_files])
    perm = np.argsort(I)
    E = np.concatenate([np.load(x) for x in e_files])[perm]
    F = np.concatenate([np.load(x) for x in f_files]).reshape(-1, 3)[perm]
    V = np.load(sorted(glob.glob(f"{OUT}/par_{tag}_v0.npy"))[0])  # rank 0
    return E, F, V


a, b = sys.argv[1], sys.argv[2]
Ea, Fa, Va = load(a)
Eb, Fb, Vb = load(b)
print(
    f"{a:10s} E={Ea.sum():.15e} vir={Va.sum():.12e} rmsF={np.sqrt((Fa**2).sum()):.15e}"
)
print(
    f"{b:10s} E={Eb.sum():.15e} vir={Vb.sum():.12e} rmsF={np.sqrt((Fb**2).sum()):.15e}"
)
print(
    f"  dE={Eb.sum() - Ea.sum():+.6e}  dVir={Vb.sum() - Va.sum():+.6e}  "
    f"maxdF={np.abs(Fb - Fa).max():.3e}  maxdEatom={np.abs(Eb - Ea).max():.3e}"
)
