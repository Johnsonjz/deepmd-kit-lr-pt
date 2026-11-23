#!/bin/bash
# Long-range multi-rank / multi-GPU regressions.
#
# Runs the Python multi-rank kernels test (the midtown/`latent_charge_sum`
# frame-virial split under an MPI communicator) and, when a LAMMPS fixture is
# provided, the host and Kokkos pair-style legs of the water-scan0 harness on
# the same ranks.  Every rank is bound to its own device, so the same script
# covers one-GPU and multi-GPU runners:
#
#   * the launcher's local rank (``OMPI_COMM_WORLD_LOCAL_RANK`` under
#     OpenMPI, ``MPI_LOCALRANKID`` under MPICH, ``SLURM_LOCALID`` under
#     Slurm) picks the device inside a rank;
#   * a single-GPU runner maps every rank to device 0, which is the
#     multi-rank case the upstream CI job already runs;
#   * a multi-GPU runner spreads the ranks over the devices it sees.
#
# Environment:
#   DP_LONG_RANGE_RANKS    rank counts to run, default "2 4"
#   DP_LONG_RANGE_TIMEOUT  per-run timeout in seconds, default 1800
#   WATER_SCAN0_*          optional LAMMPS fixture, see
#                          examples/lammps/water_scan0_lr/README.md; when unset
#                          only the Python legs run
#   LMP / LMP_KK           LAMMPS executables of the host and Kokkos builds
#
# Usage (inside the CUDA job, after the build and the fixture generators):
#   source/install/test_long_range_mpi.sh
set -euo pipefail

SCRIPT_PATH=$(dirname "$(realpath -s "$0")")
REPO_ROOT=$(realpath "${SCRIPT_PATH}/../..")
PYTHON=${PYTHON:-python}
RANKS=${DP_LONG_RANGE_RANKS:-"2 4"}
TIMEOUT=${DP_LONG_RANGE_TIMEOUT:-1800}
MPIEXEC=${MPIEXEC:-$(command -v mpiexec || command -v mpirun)}

if [[ -z "${MPIEXEC}" ]]; then
	echo "No MPI launcher found; set MPIEXEC or put mpiexec on PATH." >&2
	exit 1
fi

# Device count of this process, so the per-rank binding never names a device
# the launcher cannot see.
GPU_COUNT=$("${PYTHON}" -c 'import torch; print(max(torch.cuda.device_count(), 1))')
echo "Launching ${RANKS} ranks over ${GPU_COUNT} visible device(s) with ${MPIEXEC}"

TEST_ARGS=(
	-q -p no:cacheprovider
	"${REPO_ROOT}/source/tests/pt_expt/kernels/test_long_range_multi_rank.py"
)

run_ranks() {
	local ranks=$1
	if ! timeout "${TIMEOUT}" "${MPIEXEC}" -n "${ranks}" \
		bash -c "r=\${OMPI_COMM_WORLD_LOCAL_RANK:-\${MPI_LOCALRANKID:-\${SLURM_LOCALID:-0}}}; export CUDA_VISIBLE_DEVICES=\$((r % ${GPU_COUNT})); exec \"\$@\"" \
		bash "${PYTHON}" -m pytest "${TEST_ARGS[@]}"; then
		echo "multi-rank Python leg failed at ${ranks} ranks" >&2
		return 1
	fi
}

has_lammps_fixture() {
	[[ -n "${WATER_SCAN0_LMP_DATA:-}" && -n "${WATER_SCAN0_MODEL:-}" &&
		-n "${WATER_SCAN0_PLUGIN:-}" && -n "${LMP:-}" ]]
}

run_lammps_legs() {
	if ! has_lammps_fixture; then
		echo "No water-scan0 fixture (WATER_SCAN0_*); skipping the LAMMPS legs."
		return 0
	fi
	local host_script="${REPO_ROOT}/examples/lammps/water_scan0_lr/run_validation.sh"
	# The harness reads WATER_SCAN0_RANKS to limit the rank counts it runs, and
	# runs the Kokkos legs only when the Kokkos fixture is present.
	local run_kk=0
	if [[ -n "${LMP_KK:-}" && -n "${WATER_SCAN0_PLUGIN_KK:-}" ]]; then
		run_kk=1
	fi
	echo "Running the LAMMPS host and Kokkos legs (Kokkos=${run_kk}) via ${host_script}"
	PATH="$(dirname "${MPIEXEC}"):${PATH}" \
	WATER_SCAN0_RANKS="${RANKS}" \
	WATER_SCAN0_RUN_KK="${run_kk}" \
		"${host_script}"
}

status=0
for ranks in ${RANKS}; do
	echo "=== multi-rank Python, ${ranks} ranks ==="
	run_ranks "${ranks}" || status=1
done

run_lammps_legs || status=1

if [[ "${status}" != 0 ]]; then
	echo "Long-range MPI regressions failed." >&2
fi
exit "${status}"
