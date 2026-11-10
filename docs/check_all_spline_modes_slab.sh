#!/bin/bash
# Spline-mode parity across ranks, one configuration per invocation.
#
#   bash docs/check_all_spline_modes_slab.sh
#
# Runs `check_spline_parity_across_ranks.py` (see docs/SLAB_IK_FAILURE_DIAGNOSIS.md
# §6.1) twice per spline/order: once on one rank with the replicated solve, once
# on two ranks with SOG_MIDTOWN_SLAB=1, then compares the two with
# `compare_spline_parity.py`.  Needs the lr212 build of the operator, i.e.
# `lr212_install_lrpt/deepmd/lib`.
set -e

CONDA=/data/home/jiangzhen/.conda/envs/lr212
ROOT=/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt
INST=/data/home/public/jiangzhen/dp/lr212_install_lrpt/deepmd/lib
MPI=/data/app/openmpi-4.1.4/install/openmpi/4.1.4/bin/mpirun
OUT=${OUT:-/tmp/lr_md}
mkdir -p "$OUT"

export LD_LIBRARY_PATH=$INST:$ROOT/build-lmp-mpi-212/lib:/usr/local/cuda-12.4/lib64:$CONDA/lib/python3.11/site-packages/torch/lib:$CONDA/lib/python3.11/site-packages/nvidia/nccl/lib:/data/app/openmpi-4.1.4/install/openmpi/4.1.4/lib:$CONDA/lib
export OMPI_MCA_btl=self,vader OMPI_MCA_pml=ob1
export PYTHONPATH=$ROOT

run() {
  # $1 = np, $2 = tag; the rest come from the caller's environment
  "$MPI" --oversubscribe -np "$1" --allow-run-as-root \
    "$CONDA/bin/python" "$ROOT/docs/check_spline_parity_across_ranks.py" "$2"
}

for backend in bspline midtown quads; do
  for order in 4 6; do
    rep="rep_${backend}${order}"
    slab="slab_${backend}${order}"
    MESH_BACKEND=$backend ORDER=$order NATOMS=400 SPACING=0.5 \
      run 1 "$rep" >"$OUT/v_$rep.log" 2>&1
    MESH_BACKEND=$backend ORDER=$order NATOMS=400 SPACING=0.5 SOG_MIDTOWN_SLAB=1 \
      run 2 "$slab" >"$OUT/v_$slab.log" 2>&1
    printf '%-12s ' "$backend-$order"
    "$CONDA/bin/python" "$ROOT/docs/compare_spline_parity.py" "$rep" "$slab" \
      | tail -1
  done
done
