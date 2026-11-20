# Paper Tables 5 and 6 — 2026-11-20

Rerun on the current `c560-test` / `pr/long-range-complete` tree with the
**dpa4_sog_fix r_c=5 Å** campaign artifacts, following
`sog_work/docs/PERF_BENCH_WORKFLOW.md`:

- step time = second `Loop time` of `run 3` + `run 8`;
- ns/day = 0.0432 / s_per_step;
- peak memory = `nvidia-smi` sample minus the pre-run baseline;
- small 9,216-atom cells use `USE_SLAB=0` (replicated mesh); the
  1,119,744-atom cells use the z-slab.

## Table 5

9,216 atoms per rank on 1/2/4 GPUs; 2/4 GPU replicate the cell along z
(18,432 / 36,864 atoms), exactly the Figure 10 convention.  Additional
1,119,744-atom rows for the four DPA4C-family models (one common cell).

- Raw 9,216: `table5_9216_20261120.csv`
- Raw 1.12M: `table5_1119744_20261120.csv`
- Artifacts: `$E2E/frozen_v2` (eight campaign models, r_c=5 Å).

All eight models run at 9,216 atoms per rank.  The two-layer message-passing
models use 23.8 GiB per GPU at 2/4 ranks; the one-layer models 16.1 GiB.

## Table 6

DPA4C-SOG, equal 1e-4 energy target, 9,216 and 1,119,744 atoms, one GPU.
B-spline / CubeS2 / QuadS, order 4 and 6.

- Raw 9,216: `table6_9216_20261120.csv`
- Raw 1.12M: `table6_1119744_20261120.csv`
- Artifacts: `frozen_v2/dpa4c.pt2` (SR baseline), `frozen_v2/dpa4c-sog-bs4.pt2`,
  `frozen_v2/dpa4c-sog-bs6.pt2`, `frozen_v2/dpa4c-sog-mid4ad.pt2`, and
  `mid6fix.pt2` / `qs4fix.pt2` / `qs6fix.pt2` frozen from
  `runs/dpa4_sog_fix/water/dpa4c-sog/model.ckpt-200000.pt` with the current
  tree.

At 9,216 atoms all variants are within 0.04 ms of total step time; CubeS2-4 has
the smallest total (2.402 ms).  At 1.12M QuadS-6 is fastest (17.05 ms LR,
127.57 ms total), CubeS2-6 is 17.55 ms, B-spline-6 19.35 ms, B-spline-4
33.57 ms.  B-spline-6 has the smallest LR memory (3,556 MiB).
