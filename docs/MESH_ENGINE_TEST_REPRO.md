# Mesh engine: test, verification and reproduction

This document pins down how the two particle-mesh engines of the lr-pt tree —
the cardinal B-spline (`source/op/pt/long_range/mesh_sum.cu`) and the
midtown CubeS$_2$/QuadS family (`source/op/pt/long_range/midtown_sum.cu` +
`midtown/sog_gpu.cu`) — are built, frozen, tested and benchmarked, together with
every artifact and binary that the current numbers were produced from.

The scripts it refers to live in `docs/repro/`; the benchmark drivers live in
`/data/home/public/jiangzhen/dp/sog_work/bench/` and are referenced by absolute
path (their module-level constants are reproduced here so the recipes survive a
move).

The optimization plan that follows from these numbers — the per-atom/per-node
cost split, the static buffer ledger, and the B-spline-shaped target
architecture — is in [`MIDTOWN_PERF_REDESIGN.md`](MIDTOWN_PERF_REDESIGN.md).

---

## 1. Environment and build

### 1.1 Environment

```bash
source /etc/profile.d/modules.sh
module load gcc/13 cmake/3.26.3 openmpi/4.1.4
source /data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt/docs/repro/env.sh
```

| component | path / version |
|---|---|
| Python / torch | `/data/home/jiangzhen/.conda/envs/lr212/bin/python`, torch 2.12.1+cu126 |
| CUDA | `/usr/local/cuda-12.4`, `CMAKE_CUDA_ARCHITECTURES=89` (RTX 4090) |
| MPI | `/data/app/openmpi-4.1.4/install/openmpi/4.1.4/bin/mpiCC` |
| LAMMPS binary | `/data/home/public/jiangzhen/dp/lmp_kokkos_install/bin/lmp` (Kokkos) |
| Kokkos plugin | `/tmp/kk_lrpt_plugin/dpplugin.so` → `/tmp/kk_lrpt_build/libdeepmd_lmp.so` |
| op install dir | `/data/home/public/jiangzhen/dp/lr212_install_lrpt/deepmd/lib` |

`LAMMPS_PLUGIN_PATH` must be a directory that holds **only** the `dpplugin.so`
symlink; pointing it at a CMake build directory loads zero plugins.

### 1.2 Building the operator

The mesh engines are compiled into `libdeepmd_op_pt.so` (a 319 MB torch
extension).  Only that target needs rebuilding for a change under
`source/op/pt/long_range/`; a one-file change takes 13 s.

```bash
BUILD=/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt/build-lmp-mpi-212
make -C $BUILD deepmd_op_pt -j 20
```

The configure line (unchanged since the tree was set up) is in
[`BUILD_AND_TEST.md`](BUILD_AND_TEST.md) §1.1.

### 1.3 Install — both directories, always

```bash
B=/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt/build-lmp-mpi-212
INSTALL=/data/home/public/jiangzhen/dp/lr212_install_lrpt/deepmd/lib
REPO=/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt
cp -a $B/op/pt/libdeepmd_op_pt.so $INSTALL/
cp -a $B/op/pt/libdeepmd_op_pt.so $REPO/deepmd/lib/     # pytest + docs/check_*.py
```

The first copy is what LAMMPS loads (through `DP_BACKEND_PLUGIN_PATH`); the
second is what the Python unit tests and the `docs/check_*.py` scripts load.
Copying only the first leaves the tests silently green against the old code.

### 1.4 The Kokkos plugin

`/tmp/kk_lrpt_plugin/dpplugin.so` was built once (2026-11-12) against
`LAMMPS_SOURCE_ROOT=.../lammps_src/lammps-stable_22Jul2025_update2` with
`nvcc_wrapper` (see `BUILD_AND_TEST.md` §1.1 for the shim).  It calls the op
through the *stable* `deepmd::long_range::*` symbols and the `latent_charge_sum`
schema, so **a refactor inside the engines needs no plugin rebuild**.  A schema
change does (the op declares the argument list in `ops.cc`).

### 1.5 Compiled binaries of the current numbers

| file | size (B) | mtime |
|---|---|---|
| `build-lmp-mpi-212/op/pt/libdeepmd_op_pt.so` | 318,890,928 | 2026-11-13 01:11 |
| `lr212_install_lrpt/deepmd/lib/libdeepmd_op_pt.so` | 318,890,928 | 2026-11-13 01:11 |
| `deepmd-kit-LR-lr-pt/deepmd/lib/libdeepmd_op_pt.so` | 318,890,928 | 2026-11-13 01:11 |
| `/tmp/kk_lrpt_build/libdeepmd_lmp.so` (plugin) | 14,601,920 | 2026-11-12 17:21 |
| `lmp_kokkos_install/bin/lmp` | 1,553,784 | 2026-10-26 23:44 |

The three op copies are byte-identical; that is the invariant §1.3 protects.

---

## 2. Frozen artifacts

All artifacts are **compressed canonical** (`--compress`), which is mandatory
both for multi-rank and for a long-range module on the Kokkos device plugin.
Directory: `/data/home/public/jiangzhen/dp/sog_work/mdsuite/frozen/dpa4_campaign`.

### 2.1 Inventory

| file | backend | order | mode | spacing (Å) | size (MB) | role |
|---|---|---|---|---|---|---|
| `dpa4c.pt2` | — | — | — | — | 13.6 | short-range baseline (local) |
| `dpa4.pt2` | — | — | — | — | 31.0 | short-range, MP 2-layer |
| `dpa4-1L.pt2` | — | — | — | — | 23.6 | short-range, MP 1-layer |
| `dpa4c-sog.pt2` | CubeS₂ | 4 | ik | 0.805005 | 14.8 | **deployed SOG model** |
| `dpa4-sog.pt2` | CubeS₂ | 4 | ik | 3.853686 | 31.7 | SOG, MP 2-layer |
| `dpa4-sog-1L.pt2` | CubeS₂ | 4 | ik | 2.564212 | 24.2 | SOG, MP 1-layer |
| `dpa4c-sog-bs4.pt2` | B-spline | 4 | ad | 0.468187 | 14.8 | backend matrix |
| `dpa4c-sog-bs6.pt2` | B-spline | 6 | ad | 0.984941 | 14.8 | backend matrix |
| `dpa4c-sog-mid4ad.pt2` | CubeS₂ | 4 | ad | 0.805005 | 14.8 | backend matrix |
| `dpa4c-sog-mid6ad.pt2` | CubeS₂ | 6 | ad | 1.080138 | 14.8 | backend matrix |
| `dpa4c-sog-mid6ik.pt2` | CubeS₂ | 6 | ik | 1.080138 | 14.8 | backend matrix |
| `dpa4c-sog-quads4ik.pt2` | QuadS | 4 | ik | 0.828858 | 14.8 | backend matrix |
| `dpa4c-sog-quads6ik.pt2` | QuadS | 6 | ik | 1.234289 | 14.8 | backend matrix |
| `dpa4c-sog-mid6ad-s102.pt2` | CubeS₂ | 6 | ad | 0.984941 | 14.8 | iso-grid |
| `dpa4c-sog-mid6ik-s102.pt2` | CubeS₂ | 6 | ik | 0.984941 | 14.8 | iso-grid |
| `dpa4c-sog-qs6ad-s102.pt2` | QuadS | 6 | ad | 0.984941 | 14.8 | iso-grid |
| `dpa4c-sog-qs6ik-s102.pt2` | QuadS | 6 | ik | 0.984941 | 14.8 | iso-grid |

The backend/order/mode are compiled into the graph, so they are not in the
metadata; the **spacing is**, and it is the fingerprint that identifies an
artifact.  The five `-s102` artifacts share the deployed B-spline-6 spacing on
purpose: at 98,304 atoms all four order-6 variants then get the same $108^3$
grid, so a comparison measures the implementation rather than the
equal-accuracy grid.

### 2.2 Re-freezing

```bash
bash docs/repro/freeze_variants.sh [outdir]
```

Three traps, all of which cost a debugging session:

* **`module load gcc/13` is required.**  The AOTI wrapper is compiled by the g++
  found at freeze time; an artifact built with gcc/8 segfaults later inside
  `std::filesystem` when it runs against gcc/13's libstdc++.
* **`PYTHONHASHSEED=0` + a fresh `TORCHINDUCTOR_CACHE_DIR`.**  Otherwise
  Inductor fails with `'int_array_N' was not declared` on some checkpoints.
* **The SOG campaign checkpoints live in `old_20261112_m12/`.**  For
  `dpa4c-sog` and `dpa4-sog` the top-level `model.ckpt-200000.pt` is *not* the
  checkpoint the artifacts were frozen from; the archived one is (identical
  kernel widths: 12 Gaussians, σ_min = 2.0 Å).

---

## 3. Test tiers

Each tier answers a different question; run them in this order after a change.

### 3.1 Tier 1 — unit tests (seconds)

```bash
cd /data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt
$PY -m pytest source/tests/common/dpmodel/test_grid_rule.py -q      # 20
CUDA_VISIBLE_DEVICES=3 $PY -m pytest source/tests/pt_expt/kernels/test_mesh_backend.py -q  # 20
```

Covers the grid rule (energy/force/ik constants) and the per-backend operator
against the Python reference.  Both must be green; neither touches the GPU mesh
paths in depth.

### 3.2 Tier 2 — single-point equivalence against the previous build

Two builds of `libdeepmd_op_pt.so` cannot coexist in one process, so the
reference is a directory of dumps.

```bash
# before the change
$PY docs/repro/equiv_dump.py /tmp/ref_before 1536
# ... rebuild, install, then
$PY docs/repro/equiv_dump.py /tmp/ref_after 1536
$PY docs/repro/equiv_compare.py /tmp/ref_before /tmp/ref_after
# and the distributed path (one rank is one z-band, so no second GPU is needed)
$PY docs/repro/equiv_dump.py /tmp/slab_after 1536 --slab
$PY docs/repro/equiv_compare.py /tmp/ref_after /tmp/slab_after
```

`equiv_dump.py` runs one thermal step (fixed velocity seed) and dumps
`id type q x y z fx fy fz`, plus the total energy from the log.

**Acceptance**

| situation | expected |
|---|---|
| a path the change does not touch | `max|dF| = 0` exactly |
| a weight table / accumulator / spread touched | `rel rms|dF| ≲ 1e-6`, `max|dF| ~ 1e-5` |
| `pe` | unchanged to the last printed digit |
| anything above `1e-4` relative | a real algorithmic change, not rounding — investigate |

Measured on the current engine: untouched paths 0; touched paths
4.8–8.0 × 10⁻⁷ relative; `pe` identical.

### 3.3 Tier 3 — multi-rank bitwise parity

```bash
$PY docs/repro/mp_parity.py 12288
```

Runs five configurations × {replicated, z-slab} × np = 1/2/4 and prints the
energy per atom and `max|dF|` against np = 1 of the same mode.  The 2⁴⁰
fixed-point charge spread makes the accumulation independent of the atomic
order, so **every difference must be zero**; this is the check that catches a
halo plane, a reduction or an ownership mask.

Needs `/tmp/lmp_rank_dev.sh` (maps `OMPI_COMM_WORLD_LOCAL_RANK` to a GPU) and
np GPUs.

### 3.4 Tier 4 — performance

```bash
# same-grid, all order-6 backends at 108^3 (the implementation comparison)
$PY docs/repro/perf_iso_grid.py
# with the per-phase split of both engines (needs MEAS >= 100)
TIMING=1 MEAS=110 $PY docs/repro/perf_iso_grid.py
# replicated vs z-slab, np = 1/2/4
$PY docs/repro/perf_dist.py 98304
```

Both engines report phases under `SOG_GPU_TIMING` with the same names
(`stencil`/`spread`/`fft_fwd`/`kspace`/`fft_inv`/`gather`), so their tables can
be read side by side.  Memory is sampled every 20 ms and reported as both the
steady state (median of the last 0.8 s) and the peak: a 0.1 s sampler misses, or
catches by luck, the ~100 ms one-time allocation spike of the first call, and
that made earlier memory tables unreliable.

### 3.5 The paper's benchmark drivers

| driver | what it produces |
|---|---|
| `sog_work/bench/bench_backend_6model.py` | `matrix_midtown/backend_9_{98304,331776}.csv`: eight backend configurations at equal accuracy, `LR = SOG − SR` |
| `sog_work/bench/bench_maxsize_6model.py` | `matrix_maxsize/maxsize_6model_final.csv`: six models × np = 1/2/4 up to 2,044,416 atoms |

```bash
cd /data/home/public/jiangzhen/dp/sog_work/bench
GPU_BASE=0 N=98304  TAG=backend_9_98304  WARM=5 MEAS=20 $PY bench_backend_6model.py
GPU_BASE=0 N=331776 TAG=backend_9_331776 WARM=5 MEAS=20 $PY bench_backend_6model.py
GPU_BASE=0 WARM=3 MEAS=8 TAG=maxsize_6model_final $PY bench_maxsize_6model.py
```

Both take the timing from the **second** `Loop time` of `run WARM` + `run MEAS`
(a single `run N` folds one-time plan creation and graph capture into the
average).  `ns/day = 0.0432 / s_per_step` at 0.5 fs.  The ladders are
`[192, 1536, 12288, 41472, 98304, 255552, 526848, 1119744, 2044416]`, with
np = 2 starting at 12,288 and np = 4 at 41,472; a model whose ceiling is below
its start point needs an extra run at the lower end (this is how the
message-passing 4-GPU points were obtained).

The paper's tables are regenerated from those CSVs:

```bash
$PY docs/repro/paper_tables.py            # both tables
$PY docs/repro/paper_tables.py backend    # backend table only
```

and the figures with `papers/sog-lib-paper/tools/plot_backend_6model.py` and
`tools/plot_6model_maxsize.py` (`MAXSIZE_CSV=...` points the latter at the final
sweep).

---

## 4. End-to-end reproduction

```bash
# 1. environment and build
source /etc/profile.d/modules.sh && module load gcc/13 cmake/3.26.3
make -C $BUILD deepmd_op_pt -j 20
cp -a $BUILD/op/pt/libdeepmd_op_pt.so $INSTALL/
cp -a $BUILD/op/pt/libdeepmd_op_pt.so $REPO/deepmd/lib/

# 2. artifacts (already frozen; this re-freezes them)
bash docs/repro/freeze_variants.sh

# 3. correctness
$PY -m pytest source/tests/common/dpmodel/test_grid_rule.py -q
CUDA_VISIBLE_DEVICES=3 $PY -m pytest source/tests/pt_expt/kernels/test_mesh_backend.py -q
$PY docs/repro/equiv_dump.py /tmp/ref 1536 && $PY docs/repro/equiv_dump.py /tmp/ref_slab 1536 --slab
$PY docs/repro/equiv_compare.py /tmp/ref /tmp/ref_slab      # ~5e-7 relative
$PY docs/repro/mp_parity.py 12288                            # all zeros

# 4. performance
$PY docs/repro/perf_iso_grid.py
$PY docs/repro/perf_dist.py 98304
cd $BENCH && GPU_BASE=0 N=98304 WARM=5 MEAS=20 TAG=backend_9_98304 $PY bench_backend_6model.py
$PY $REPO/docs/repro/paper_tables.py
```

---

## 5. Traps worth remembering

* **Host vs device pointers in the slab entry points.**  `sog_gpu_*_brick_dev`
  take host `x`/`boxlo` and upload `x` themselves; passing `boxlo` straight to a
  device kernel is an illegal access that surfaces only as "the graph returned
  undefined outputs".
* **A graph error hides the real message.**  A failed `aoti_torch_empty_strided`
  inside the exported graph is reported by AOTI as "the graph returned
  undefined output(s)"; the cause is on stderr as an `[E...]` line.
* **The fp64 pipe.**  On these cards fp64 is 1/32–1/64 of fp32.  A per-atom
  prologue with three `(x - lo)/L` divisions cost 85 fp64 instructions per atom
  and was the whole cost of the QuadS gather (0.54 ms of a 0.72 ms mesh).  Check
  the SASS (`cuobjdump -sass -fun <mangled>`) for `DFMA/DMUL/DADD` before
  optimising anything else.
* **Divergent class branches.**  A CubeS₂ weight is a five-class expression;
  with the node table sorted by grid offset a warp round mixes classes and the
  branch serialises up to five ways.  Sorting the table by class halved the
  gather (0.359 → 0.186 ms at 108³).
* **`ncu` from CUDA 11.8 is unusable here** — Kokkos aborts with "likely
  mismatch of architecture".  Use event timing (`SOG_GPU_TIMING`) and SASS
  instruction counts instead.
* **`SOG_MIDTOWN_SLAB=1` selects the distributed solve even at np = 1**, which
  is how the brick kernels are verified on a single card.
* **midtown 的 CUDA graph 必须真的进图。** 两个曾同时存在的问题：CUDA 系数
  未 pin 的 `.to(kCPU)` 让 AOTI capture 直接失败、整模型回退 eager；即使
  capture "成功"，mesh kernel 若走 legacy default stream 也不会进图，
  replay 只跑几 µs 的空图。检查：直接 capture `latent_charge_sum`，
  replay 应 ≈ eager 的 mesh 时间（98k 约 0.4 ms）；改坐标再 replay，
  `max|ΔF(graph−eager)|` 必须为 0。修复见
  [`MIDTOWN_PERF_REDESIGN.md`](MIDTOWN_PERF_REDESIGN.md) §0.1。
