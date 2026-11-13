#!/usr/bin/env python3
"""Tier-2 equivalence: compare two equiv_dump.py directories.

Reports, per model, max|dF| and rms|dF| (with the force scale) so the number can
be read as an absolute and as a relative change, plus max|dq| for the charges.

Acceptance, with the shipped float32 mesh engine:
  * a change confined to one path leaves the other paths *bitwise* equal;
  * any change that touches a weight table, an accumulator or the spread moves
    the force rms by about 1e-6 relative or less (the float32 floor);
  * the total energy is unchanged to the last printed digit.
A difference above ~1e-4 relative is a real algorithmic change, not rounding.
"""
import glob
import math
import os
import sys

if len(sys.argv) != 3:
    raise SystemExit("usage: equiv_compare.py <dirA> <dirB>")
A, B = sys.argv[1], sys.argv[2]


def read_dump(path):
    lines = open(path).readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("ITEM: ATOMS"))
    cols = lines[start].split()[2:]
    ix = {c: i for i, c in enumerate(cols)}
    rows = {}
    for line in lines[start + 1:]:
        f = line.split()
        if len(f) < len(cols):
            continue
        rows[int(f[ix["id"]])] = {
            "q": float(f[ix["q"]]),
            "f": (float(f[ix["fx"]]), float(f[ix["fy"]]), float(f[ix["fz"]])),
        }
    return rows


worst = 0.0
for a in sorted(glob.glob(f"{A}/*.dump")):
    tag = os.path.basename(a)[:-5]
    b = f"{B}/{tag}.dump"
    if not os.path.exists(b):
        print(f"{tag:28s} missing counterpart")
        continue
    da, db = read_dump(a), read_dump(b)
    ids = sorted(da)
    if ids != sorted(db):
        print(f"{tag:28s} atom id mismatch")
        continue
    df = [math.dist(da[i]["f"], db[i]["f"]) for i in ids]
    dq = [abs(da[i]["q"] - db[i]["q"]) for i in ids]
    rms = math.sqrt(sum(x * x for x in df) / len(df))
    fmag = math.sqrt(sum(sum(c * c for c in da[i]["f"]) for i in ids) / len(ids))
    rel = rms / fmag if fmag else float("nan")
    worst = max(worst, rel)
    print(f"{tag:28s} max|dF|={max(df):.3e} rms|dF|={rms:.3e} "
          f"(rms|F|={fmag:.3e}, rel={rel:.2e}) max|dq|={max(dq):.2e}")
print(f"\nworst relative force change: {worst:.2e}")
