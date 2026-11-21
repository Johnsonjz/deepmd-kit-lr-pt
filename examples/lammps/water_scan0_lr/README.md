# water-scan0 long-range validation

This fixture validates a two-block DPA4-SOG canonical `.pt2` on the
water-scan0 192-atom cell with the host and Kokkos pair styles.

No dataset or model is committed. Provide:

```bash
export WATER_SCAN0_LMP_DATA=/path/to/water-scan0/lmp/sog/water_charge.data
export WATER_SCAN0_MODEL=/path/to/water_scan0_lr.pt2
export WATER_SCAN0_PLUGIN=/path/to/host/libdeepmd_lmp.so
export WATER_SCAN0_PLUGIN_KK=/path/to/kokkos/libdeepmd_lmp.so
export LMP=/path/to/lmp
export LMP_KK=/path/to/kokkos/lmp
export DP_BACKEND_PLUGIN_PATH=/path/to/api_cc
./run_validation.sh
```

The harness runs `np=1/2/4`, host and Kokkos, and requires every
single-frame energy and pressure to agree. `in.water.lammps` can be used
manually by passing `-var DATA/-var MODEL/-var PLUGIN`; it is a `run 0`
input, not an MD protocol.

For the empty-rank check, use `processors 2 2 1` with a data file that
leaves one subdomain without atoms; the canonical with-comm graph and the
Kokkos ghost fold keep every rank in the same collectives.
