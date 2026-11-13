#!/bin/bash
# run.sh -- launch the Kokkos LAMMPS build with the DeePMD plugin.
#
#   ./run.sh in.deepmd_kk_sog.lammps            # 1 rank, 1 GPU
#   ./run.sh -np 4 in.deepmd_kk_sog.lammps      # 4 ranks, one GPU each
#
# The four variables below are the only host-specific settings.
set -euo pipefail

# 1. Kokkos-enabled LAMMPS (PKG_KOKKOS built with CUDA).
LMP=${LMP:-lmp}
# 2. DeePMD install prefix (holds deepmd/lib/libdeepmd_cc.so and the
#    AOTInductor runtime the frozen .pt2 needs).
DP_INSTALL=${DP_INSTALL:?set DP_INSTALL to the DeePMD install prefix}
# 3. Directory holding the plugin .so (deepmd_lmp.so, or a *plugin.so symlink).
PLUGIN_DIR=${PLUGIN_DIR:-/tmp/lmpkk_plugin}
# 4. Conda env that ships the libstdc++/CUDA runtimes the frozen artifact was
#    built against (torch's bundled CUDA must win over the system one).
CONDA=${CONDA:-${CONDA_PREFIX:-}}
if [ -z "$CONDA" ]; then
    echo "run.sh: activate the conda env the artifact was built with, or set CONDA" >&2
    exit 1
fi

TORCH_NV=$(ls -d "$CONDA"/lib/python*/site-packages/nvidia 2>/dev/null | head -n 1)
if [ -z "$TORCH_NV" ]; then
    echo "run.sh: no nvidia/ runtime tree under $CONDA/lib/python*" >&2
    exit 1
fi
export LD_LIBRARY_PATH=${CONDA}/lib:${TORCH_NV}/cuda_runtime/lib:${TORCH_NV}/cuda_cupti/lib:${TORCH_NV}/nvjitlink/lib:${LMP%/bin/lmp}/lib:${DP_INSTALL}/deepmd/lib
export DP_BACKEND_PLUGIN_PATH=${DP_INSTALL}/deepmd/lib
export LAMMPS_PLUGIN_PATH=${PLUGIN_DIR}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# MPI launcher for the multi-rank branch (module load openmpi, or give a full path).
MPIRUN=${MPIRUN:-mpirun}

NP=1
if [ "${1:-}" = "-np" ]; then NP=$2; shift 2; fi

if [ "$NP" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    exec "${LMP}" -k on g 1 "$@"
fi

# One GPU per rank: bind by LOCAL rank, not by global rank.
cat > /tmp/lmp_rank_kk.sh <<'EOS'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
exec "$LMP" -k on g 1 "$@"
EOS
chmod +x /tmp/lmp_rank_kk.sh
export LMP
exec ${MPIRUN} --oversubscribe -np "$NP" /tmp/lmp_rank_kk.sh "$@"
