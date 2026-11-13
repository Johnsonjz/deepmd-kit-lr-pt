#!/usr/bin/env python3
"""Emit the LaTeX rows of both performance tables from the benchmark CSVs.

Every number in Sec. 6.4 of the paper comes from these CSVs, so the tables can
be regenerated (and audited) without re-running anything:

  $BENCH/matrix_midtown/backend_9_98304.csv     eight backend configurations
  $BENCH/matrix_midtown/backend_9_331776.csv    same at 331,776 atoms
  $BENCH/matrix_maxsize/maxsize_6model_final.csv  six models, np = 1/2/4

usage: paper_tables.py [backend|maxsize|all]
"""
import csv
import os
import sys
from collections import defaultdict

BENCH = "/data/home/public/jiangzhen/dp/sog_work/bench"
MID = f"{BENCH}/matrix_midtown"
MAXS = os.environ.get("MAXSIZE_CSV",
                        f"{BENCH}/matrix_maxsize/maxsize_6model_final.csv")

LAB = {"dpa4c": "DPA4C", "dpa4c-sog": "DPA4C-SOG", "dpa4": "DPA4-2L",
       "dpa4-1L": "DPA4-1L", "dpa4-sog": "SOG-2L", "dpa4-sog-1L": "SOG-1L"}
ORDER = ["dpa4c", "dpa4c-sog", "dpa4", "dpa4-1L", "dpa4-sog", "dpa4-sog-1L"]
# (backend, order, mode) -> the row label of the paper's table
BACKEND_ROWS = [
    ("bspline", "4", "ad", "B-spline  & 4 & AD"),
    ("bspline", "6", "ad", "B-spline  & 6 & AD"),
    ("midtown", "4", "ad", "CubeS$_2$ & 4 & AD"),
    ("midtown", "6", "ad", "CubeS$_2$ & 6 & AD"),
    ("midtown", "4", "ik", "CubeS$_2$ & 4 & ik"),
    ("midtown", "6", "ik", "CubeS$_2$ & 6 & ik"),
    ("quads", "4", "ik", "QuadS     & 4 & ik"),
    ("quads", "6", "ik", "QuadS     & 6 & ik"),
]


def backend_table():
    r98 = {(r["backend"], r["order"], r["mode"]): r
           for r in csv.DictReader(open(f"{MID}/backend_9_98304.csv"))}
    r331 = {(r["backend"], r["order"], r["mode"]): r
            for r in csv.DictReader(open(f"{MID}/backend_9_331776.csv"))}
    sr98 = next(r for (b, _, _), r in r98.items() if b == "-")
    print(f"% SR baseline: {float(sr98['step_ms']):.2f} ms, "
          f"{sr98['peak_mib']} MiB (98,304 atoms)")
    for be, order, mode, label in BACKEND_ROWS:
        k = (be, order, mode)
        a, b = r98[k], r331[k]
        grid = a["grid"].replace("x", "^3$").replace("216^3$", "216^3")
        g = a["grid"].split("x")[0]
        print(f"{label} & ${g}^3$ & {float(a['step_ms']):.2f} & "
              f"{float(a['lr_ms']):.2f} & {a['lr_mib']} & {float(b['lr_ms']):.2f} \\\\")


def maxsize_table():
    d = defaultdict(list)
    for r in csv.DictReader(open(MAXS)):
        try:
            t = float(r["loop_s_per_step"])
        except (TypeError, ValueError):
            t = None
        d[(r["model"], int(r["np"]))].append(
            (int(r["n"]), t, float(r["gpu_peak_mib"] or 0)))
    for m in ORDER:
        for np_ in (1, 2, 4):
            rows = sorted(d.get((m, np_), []), key=lambda x: x[0])
            ok = [x for x in rows if x[1]]
            if not ok:
                continue
            n, t, mem = ok[-1]
            fail = next((f"{r[0]:,}" for r in rows if r[0] > n and not r[1]), "---")
            print(f"{LAB[m]:10s} & {np_} & {n:,} & {mem:.0f} & "
                  f"{0.0432 / t:.2f} & {fail} \\\\")


what = sys.argv[1] if len(sys.argv) > 1 else "all"
if what in ("backend", "all"):
    backend_table()
if what in ("maxsize", "all"):
    maxsize_table()
