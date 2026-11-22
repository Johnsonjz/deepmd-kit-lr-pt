#!/usr/bin/env bash
# Validate a water-scan0 long-range .pt2 with host and Kokkos pair styles.
#
# Required environment:
#   WATER_SCAN0_LMP_DATA  path to water_charge.data
#   WATER_SCAN0_MODEL     path to the canonical .pt2
#   LMP                   host LAMMPS executable
#   LMP_KK                Kokkos LAMMPS executable
#   WATER_SCAN0_PLUGIN    host libdeepmd_lmp.so
#   WATER_SCAN0_PLUGIN_KK Kokkos libdeepmd_lmp.so
#   DP_BACKEND_PLUGIN_PATH / LD_LIBRARY_PATH as for the build
# Optional:
#   WATER_SCAN0_RANKS     rank counts to run, default "1 2 4"
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
: "${WATER_SCAN0_LMP_DATA:?}" "${WATER_SCAN0_MODEL:?}"
: "${LMP:?}" "${WATER_SCAN0_PLUGIN:?}"
run_kk=${WATER_SCAN0_RUN_KK:-0}
if [[ "$run_kk" == 1 ]]; then
	: "${LMP_KK:?}" "${WATER_SCAN0_PLUGIN_KK:?}"
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
failed=0
run() {
	local log=$1
	shift
	if ! "$@" >"$log" 2>&1; then
		echo "validation run failed; see $log" >&2
		failed=1
	fi
}
run_env() {
	local log=$1
	shift
	if ! env "$@" >"$log" 2>&1; then
		echo "validation run failed; see $log" >&2
		failed=1
	fi
}
for np in ${WATER_SCAN0_RANKS:-1 2 4}; do
	if [[ "$run_kk" == 1 ]]; then
		kk_ld="${WATER_SCAN0_BACKEND_PLUGIN_KK:-}:${LD_LIBRARY_PATH:-}"
		# One GPU per rank: the Kokkos binary reads the device LAMMPS was
		# launched on, so without the remap every rank would share GPU 0 and
		# the run fails once np exceeds 1.
		run_env "$tmp/kk$np.log" \
			"DP_BACKEND_PLUGIN_PATH=${WATER_SCAN0_BACKEND_PLUGIN_KK:-$DP_BACKEND_PLUGIN_PATH}" \
			"LD_LIBRARY_PATH=$kk_ld" \
			mpirun -np "$np" bash -c \
			'export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK:-0}; exec "$@"' \
			bash "$LMP_KK" -k on g 1 -sf kk \
			-var DATA "$WATER_SCAN0_LMP_DATA" -var MODEL "$WATER_SCAN0_MODEL" \
			-var PLUGIN "$WATER_SCAN0_PLUGIN_KK" -log none -in "$here/in.water.lammps"
	fi
	run "$tmp/host$np.log" mpirun -np "$np" "$LMP" \
		-var DATA "$WATER_SCAN0_LMP_DATA" -var MODEL "$WATER_SCAN0_MODEL" \
		-var PLUGIN "$WATER_SCAN0_PLUGIN" -log none -in "$here/in.water.lammps"
done
[[ "$failed" == 0 ]] || exit 1
python3 - "$tmp" <<'PY'
import pathlib, re, sys
root=pathlib.Path(sys.argv[1]); rows={}
for p in sorted(root.glob('*.log')):
    m=re.search(r'^\s*0\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)', p.read_text(), re.M)
    if m: rows[p.stem]=(float(m.group(1)),float(m.group(2)))
print(rows)
e=[v[0] for v in rows.values()]; p=[v[1] for v in rows.values()]
assert e and max(e)-min(e) < 1e-4, (max(e)-min(e), rows)
assert p and max(p)-min(p) < 0.01, (max(p)-min(p), rows)
PY
