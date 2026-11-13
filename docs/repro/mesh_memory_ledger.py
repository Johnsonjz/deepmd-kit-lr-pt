#!/usr/bin/env python3
"""Static device-buffer ledger: B-spline vs midtown (current vs target).

The peak-memory tables in ``docs/MIDTOWN_VS_BSPLINE_BENCH.md`` and
``sog_work/bench/matrix_midtown/backend_9_*.csv`` come from an ``nvidia-smi``
poll of the whole GPU.  They mix the mesh buffers with the one-time green build,
the cuFFT work area, the AOTI graph pool and any other process on the card, so
they cannot say which engine's *buffers* are larger.  This script computes the
buffers from the layouts in the source, so the two can be compared on their own
terms.

Layouts (all sizes in bytes, N nodes, G = nx*ny*nz grid points, order p):

  B-spline ``mesh_sum.cu``
      fixed-point mesh          8 G
      float mesh (also potential)  4 G
      half spectrum (float2)    4 G
      3 axis stencils           36 p N   (int offset + float value + float slope)
      node charge               4 N
      owner potential           4 N
      outputs                  60 N   (energy 8 + gradient 4 + force 12 + virial 36)
      ---------------------------------------------------------------
      total                    16 G + (36 p + 68) N

  midtown current ``midtown_sum.cu`` + ``sog_gpu.cu``
      green table (full nx stride)  8 G
      real density / potential       4 G
      fixed-point mesh + spectrum    8 G
      FracNode                       24 N
      charges q                      8 N
      d_vpot (double)                8 N
      d_force (double)               24 N
      outputs                        60 N
      ---------------------------------------------------------------
      total                    20 G + 124 N

  midtown target (this proposal)
      green table (compact half spectrum)  4 G
      real density / potential             4 G
      fixed-point mesh + spectrum          8 G
      FracNode                             24 N
      charges q                            8 N
      d_vpot (float)                        4 N
      d_force (float)                      12 N
      outputs                              60 N
      ---------------------------------------------------------------
      total                    16 G + 108 N

Neither column includes the cuFFT work area (``cufftGetSize``), the FFT plan
twiddles, the CUDA context or the model.  Those are reported separately by
``SOG_MEM_BREAKDOWN=1`` and are common to both engines at the same grid.

Usage:
    python docs/repro/mesh_memory_ledger.py --n 98304 --grid-mid 96 --grid-bs 108
    python docs/repro/mesh_memory_ledger.py --n 331776 --grid-mid 144 --grid-bs 160
"""

from __future__ import annotations

import argparse

MIB = 1024.0 * 1024.0


def ledger(n: int, g_mid: int, g_bs: int, order: int = 6) -> dict[str, dict[str, float]]:
    outputs = 60 * n
    bs = {
        "fixed mesh (8 B/pt)": 8 * g_bs,
        "float mesh + potential (4 B/pt)": 4 * g_bs,
        "half spectrum (4 B/pt)": 4 * g_bs,
        f"axis stencils (36*{order} B/atom)": 36 * order * n,
        "node charge + owner potential (8 B/atom)": 8 * n,
        "outputs (60 B/atom)": outputs,
    }
    mid_now = {
        "green table, full nx (8 B/pt)": 8 * g_mid,
        "density + potential (4 B/pt)": 4 * g_mid,
        "fixed mesh + spectrum (8 B/pt)": 8 * g_mid,
        "FracNode (24 B/atom)": 24 * n,
        "q (8 B/atom)": 8 * n,
        "d_vpot double (8 B/atom)": 8 * n,
        "d_force double (24 B/atom)": 24 * n,
        "outputs (60 B/atom)": outputs,
    }
    mid_new = {
        "green table, compact (4 B/pt)": 4 * g_mid,
        "density + potential (4 B/pt)": 4 * g_mid,
        "fixed mesh + spectrum (8 B/pt)": 8 * g_mid,
        "FracNode (24 B/atom)": 24 * n,
        "q (8 B/atom)": 8 * n,
        "d_vpot float (4 B/atom)": 4 * n,
        "d_force float (12 B/atom)": 12 * n,
        "outputs (60 B/atom)": outputs,
    }
    return {"bspline": bs, "midtown-current": mid_now, "midtown-target": mid_new}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=98304)
    parser.add_argument("--grid-mid", type=int, default=96)
    parser.add_argument("--grid-bs", type=int, default=108)
    parser.add_argument("--order", type=int, default=6)
    args = parser.parse_args()

    n = args.n
    g_mid = args.grid_mid**3
    g_bs = args.grid_bs**3
    rows = ledger(n, g_mid, g_bs, args.order)
    totals = {k: sum(v.values()) / MIB for k, v in rows.items()}

    print(
        f"N = {n}, order = {args.order}, grid midtown {args.grid_mid}^3 "
        f"({g_mid/1e6:.2f} M), B-spline {args.grid_bs}^3 ({g_bs/1e6:.2f} M)"
    )
    for name in ("bspline", "midtown-current", "midtown-target"):
        print(f"\n{name}:")
        for key, value in rows[name].items():
            print(f"  {key:52s} {value/MIB:9.1f} MiB")
        print(f"  {'total (no cuFFT work area)':52s} {totals[name]:9.1f} MiB")
    print(
        f"\nmidtown / B-spline total: current "
        f"{totals['midtown-current']/totals['bspline']:.2f}, target "
        f"{totals['midtown-target']/totals['bspline']:.2f}"
    )
    print(
        f"\nPer grid point: B-spline {16:.0f} B, midtown target {16:.0f} B; "
        f"per atom: B-spline {36*args.order+68:.0f} B, midtown target {108:.0f} B."
    )


if __name__ == "__main__":
    main()
