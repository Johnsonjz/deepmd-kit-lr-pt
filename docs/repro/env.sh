# Shared environment of the mesh-engine verification harness.
#
# Everything below assumes the A-chain (lr-pt) tree and the lr212 conda env.
# Source this file from a bash shell:   source docs/repro/env.sh
# and the python scripts of this directory read the same values from their own
# module-level constants (kept in sync by hand; they are few).

export REPO_LRPT=/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt
export BUILD=$REPO_LRPT/build-lmp-mpi-212
export INSTALL=/data/home/public/jiangzhen/dp/lr212_install_lrpt/deepmd/lib
export FROZEN=/data/home/public/jiangzhen/dp/sog_work/mdsuite/frozen/dpa4_campaign
export BENCH=/data/home/public/jiangzhen/dp/sog_work/bench
export CKPT_RUN=/data/home/public/jiangzhen/dp/runs/dpa4_campaign/water
export LMP=/data/home/public/jiangzhen/dp/lmp_kokkos_install/bin/lmp
export KK_PLUGIN=/tmp/kk_lrpt_plugin
export OMPI=/data/app/openmpi-4.1.4/install/openmpi/4.1.4
export CONDA=/data/home/jiangzhen/.conda/envs/lr212
export TORCH=$CONDA/lib/python3.11/site-packages/torch
export NVD=$CONDA/lib/python3.11/site-packages/nvidia
export PY=$CONDA/bin/python
export CUDA_HOME=/usr/local/cuda-12.4

# The op is loaded through DP_BACKEND_PLUGIN_PATH / LD_LIBRARY_PATH; the Kokkos
# plugin through LAMMPS_PLUGIN_PATH (a directory holding only dpplugin.so).
export DP_BACKEND_PLUGIN_PATH=$INSTALL
export LAMMPS_PLUGIN_PATH=$KK_PLUGIN
export LD_LIBRARY_PATH="$OMPI/lib:$CONDA/lib:$TORCH/lib:$NVD/cuda_runtime/lib:$NVD/cuda_cupti/lib:$NVD/nvjitlink/lib:$NVD/cudnn/lib:$NVD/cublas/lib:$NVD/cusparse/lib:$NVD/cusparselt/lib:$NVD/curand/lib:$NVD/cufft/lib:$NVD/cufile/lib:$NVD/nccl/lib:$NVD/nvshmem/lib:/data/home/public/jiangzhen/dp/lmp_kokkos_install/lib:$INSTALL"
export PATH="/usr/local/cuda-12.4/bin:$OMPI/bin:$PATH"
export OMP_NUM_THREADS=1
