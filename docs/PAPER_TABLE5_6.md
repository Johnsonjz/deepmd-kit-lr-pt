# Paper Tables 5 and 6 — 2026-11-20

Benchmarks rerun on the current `c560-test` / `pr/long-range-complete`
tree (P0 slab, 6th-order production path, fixed frame-virial share).

## Table 5

- 9,216 atoms per rank; 1/2/4 GPUs.  Multi-GPU runs replicate the cell along
  z (18,432 / 36,864 atoms).
- Additional 1,119,744-atom rows for DPA4C, DPA4C-SOG, DPA4C-Plus,
  DPA4C-SOG-Plus.
- Artifacts re-frozen with the current tree:
  `t5_dpa4.pt2`, `t5_dpa4-1L.pt2`, `t5_dpa4-sog.pt2`, `t5_dpa4-sog-1L.pt2`,
  `t5_dpa4c.pt2`, `t5_dpa4c-sog.pt2`, `t5_dpa4c-plus.pt2`,
  `t5_dpa4c-plus-sog.pt2`.
- Raw: `table5_20261120.csv`.

DPA4-2L and DPA4-SOG-2L complete on one GPU at 9,216 atoms but not when the
cell is replicated (24 GiB card limit); the table marks those cells with a
dash.  This differs from the old 20,736-atom table.

## Table 6

- DPA4C-SOG, equal 1e-4 energy target, 9,216 and 1,119,744 atoms.
- B-spline / CubeS2 / QuadS, order 4 and 6, AD force path.
- Artifacts: `bs4.pt2`, `bs6.pt2`, `mid4.pt2`, `mid6.pt2`, `qs4.pt2`,
  `qs6.pt2`; short-range baseline `sr.pt2`.
- Raw: `table6_9216_20261120.csv`, `table6_1119744_20261120.csv`.

At 9,216 atoms all six variants are within 0.06 ms; at 1.12M CubeS2-6 is
fastest (17.87 ms LR, 107.70 ms total), B-spline-6 is 19.20 ms, and B-spline-4
is 27.18 ms.  QuadS-6 is 18.67 ms.
