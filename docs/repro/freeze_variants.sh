#!/bin/bash
# Freeze every artifact of the mesh-engine study, from the campaign checkpoints.
#
# All artifacts are COMPRESSED CANONICAL (`--compress`).  That form is mandatory
# both for multi-rank and for a long-range module on the Kokkos device plugin
# (pair_style deepmd/kk), and it is what removes the multi-gigabyte fixed
# allocation of the uncompressed graph (one RTX 4090 then holds 2M atoms).
#
# The mesh backend / order / force mode / spacing are COMPUTATIONAL settings:
# freeze_lr_variants.py patches model_params.long_range in a copy of the
# checkpoint and re-freezes, leaving the trained weights untouched.
#
# Reproducibility notes:
#  * module load gcc/13 is REQUIRED: the AOTI wrapper is compiled with the g++
#    found at freeze time, and an artifact built with gcc/8 crashes later when
#    it is run against gcc/13's libstdc++ (std::filesystem).
#  * PYTHONHASHSEED=0 + a fresh TORCHINDUCTOR_CACHE_DIR work around an Inductor
#    codegen bug that otherwise fails with "'int_array_N' was not declared".
#  * The SOG campaign checkpoints of dpa4c-sog and dpa4-sog live in the archived
#    subdirectory old_20261112_m12; the model.ckpt-200000.pt at their top level
#    is not the one the artifacts were frozen from.
#
# usage: bash docs/repro/freeze_variants.sh [outdir]
set -u
source /etc/profile.d/modules.sh 2>/dev/null; module load gcc/13 2>/dev/null
export PYTHONPATH=/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt
export CUDA_HOME=/usr/local/cuda-12.4
export PYTHONHASHSEED=0
PY=/data/home/jiangzhen/.conda/envs/lr212/bin/python
CK=/data/home/public/jiangzhen/dp/runs/dpa4_campaign/water
OUT=${1:-/data/home/public/jiangzhen/dp/sog_work/mdsuite/frozen/dpa4_campaign}
V=/data/home/public/jiangzhen/dp/sog_work/tools/freeze_lr_variants.py
mkdir -p "$OUT"
rm -rf /tmp/ti_freeze; export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_freeze

# ── the six models of the campaign, as they are benchmarked ---------------
# SOG models: CubeS2, fourth order, Fourier force mode, auto grid at 1e-4.
sog() {  # name ckpt
  "$PY" "$V" --ckpt "$1" --out "$OUT/$2.pt2" \
    --backend midtown --order 4 --force-mode ik --accuracy 1e-4
}
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 "$PY" -m deepmd --pt-expt freeze \
  -c "$CK/dpa4c/model.ckpt-200000.pt"     -o "$OUT/dpa4c.pt2"     --compress
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 "$PY" -m deepmd --pt-expt freeze \
  -c "$CK/dpa4/model.ckpt-200000.pt"      -o "$OUT/dpa4.pt2"      --compress
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 "$PY" -m deepmd --pt-expt freeze \
  -c "$CK/dpa4-1L/model.ckpt-200000.pt"   -o "$OUT/dpa4-1L.pt2"   --compress
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 sog "$CK/dpa4c-sog/old_20261112_m12/model.ckpt-200000.pt" dpa4c-sog
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 sog "$CK/dpa4-sog/old_20261112_m12/model.ckpt-200000.pt"  dpa4-sog
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 sog "$CK/dpa4-sog-1L/model.ckpt-200000.pt"                dpa4-sog-1L

# ── the backend matrix on one kernel (dpa4c-sog) -------------------------
# Deployed grids: the spacing the equal-accuracy rule picks at 1e-4.
D=$CK/dpa4c-sog/old_20261112_m12/model.ckpt-200000.pt
be() {  # name backend order mode [spacing]
  local extra=()
  [ $# -ge 5 ] && extra=(--spacing "$5")
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 "$PY" "$V" --ckpt "$D" \
    --out "$OUT/$1.pt2" --backend "$2" --order "$3" --force-mode "$4" "${extra[@]}" \
    --accuracy 1e-4
}
be dpa4c-sog-bs4       bspline 4 ad
be dpa4c-sog-bs6       bspline 6 ad
be dpa4c-sog-mid4ad    midtown 4 ad
be dpa4c-sog-mid6ad    midtown 6 ad
be dpa4c-sog-mid6ik    midtown 6 ik
be dpa4c-sog-quads4ik  quads   4 ik
be dpa4c-sog-quads6ik  quads   6 ik

# ── the iso-grid set: every order-6 backend at ONE spacing ---------------
# 0.98494 A is the deployed B-spline-6 spacing; with it all four variants get
# the same 108^3 grid at 98,304 atoms, so a comparison measures the
# implementation rather than the equal-accuracy grid.
SP=0.9849412510342754
be dpa4c-sog-mid6ad-s102 midtown 6 ad "$SP"
be dpa4c-sog-mid6ik-s102 midtown 6 ik "$SP"
be dpa4c-sog-qs6ad-s102  quads   6 ad "$SP"
be dpa4c-sog-qs6ik-s102  quads   6 ik "$SP"

echo "=== frozen into $OUT ==="
ls -la "$OUT"/*.pt2
