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
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
: "${WATER_SCAN0_LMP_DATA:?}" "${WATER_SCAN0_MODEL:?}"
: "${LMP:?}" "${LMP_KK:?}" "${WATER_SCAN0_PLUGIN:?}" "${WATER_SCAN0_PLUGIN_KK:?}"

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
for np in 1 2 4; do
  mpirun -np "$np" "$LMP" -var DATA "$WATER_SCAN0_LMP_DATA" \
    -var MODEL "$WATER_SCAN0_MODEL" -var PLUGIN "$WATER_SCAN0_PLUGIN" \
    -log none -in "$here/in.water.lammps" > "$tmp/host$np.log" 2>&1 || true
  mpirun -np "$np" "$LMP_KK" -k on g 1 -sf kk \
    -var DATA "$WATER_SCAN0_LMP_DATA" -var MODEL "$WATER_SCAN0_MODEL" \
    -var PLUGIN "$WATER_SCAN0_PLUGIN_KK" \
    -log none -in "$here/in.water.lammps" > "$tmp/kk$np.log" 2>&1 || true
done
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
