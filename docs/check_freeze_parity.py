#!/usr/bin/env python3
"""Freeze-time deployment parity gate.

Compares a frozen `.pt2` artifact (AOTInductor program) against its training
checkpoint evaluated eagerly, on one real frame. This catches the class of
defect where the *compiled* program is not numerically equivalent to the eager
model -- e.g. AOTInductor miscompiling an autograd graph under dynamic shapes --
before the artifact ships.

A faithful artifact agrees with the eager checkpoint to ~1e-5 eV/atom and
~1e-4 eV/A. A fused-op artifact (DPA4C canonical, DPA4-B-lineage 1F+1B) is
bitwise. Anything above the tolerance refuses.

Usage:
    freeze_parity_gate.py <model.ckpt-N.pt> <model.pt2> [--system DIR] [--frame 0]
Exit code 0 on pass, 1 on fail.
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def load_frame(system: str, frame: int):
    setd = None
    for root, _dirs, files in os.walk(system):
        if "coord.npy" in files:
            setd = root
            break
    if setd is None:
        raise SystemExit(f"no set.000 under {system}")
    coord = np.load(os.path.join(setd, "coord.npy"))[frame].reshape(1, -1, 3)
    box = np.load(os.path.join(setd, "box.npy"))[frame].reshape(1, 3, 3)
    force = np.load(os.path.join(setd, "force.npy"))[frame].reshape(-1, 3)
    types = np.loadtxt(os.path.join(system, "type.raw")).astype(int)
    return coord, box, force, types


def eval_dp(path: str, coord, box, types):
    from deepmd.infer import DeepPot

    dp = DeepPot(path)
    e, f, v = dp.eval(coord, box, types)
    return float(np.ravel(e)[0]), np.asarray(f).reshape(-1, 3), np.asarray(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("artifact")
    ap.add_argument(
        "--system",
        default="/data/home/public/jiangzhen/dp/dp_example/water-scan0/data/split/test/data1",
    )
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--tol-e", type=float, default=1e-3, help="eV/atom")
    ap.add_argument("--tol-f", type=float, default=1e-3, help="eV/A")
    args = ap.parse_args()

    coord, box, _dft, types = load_frame(args.system, args.frame)
    e_ck, f_ck, _ = eval_dp(args.ckpt, coord, box, types)
    e_pt, f_pt, _ = eval_dp(args.artifact, coord, box, types)

    natoms = types.shape[0]
    de = abs(e_pt - e_ck) / natoms
    df = float(np.sqrt(((f_pt - f_ck) ** 2).sum(1).mean()))
    print(f"checkpoint   E = {e_ck:.6f} eV")
    print(f"artifact     E = {e_pt:.6f} eV")
    print(f"|dE|         {de:.3e} eV/atom   (tol {args.tol_e:g})")
    print(f"rms|dF|      {df:.3e} eV/A      (tol {args.tol_f:g})")
    ok = de <= args.tol_e and df <= args.tol_f
    print("PARITY_PASS" if ok else "PARITY_FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
