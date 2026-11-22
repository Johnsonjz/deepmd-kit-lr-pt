# water-scan0 long-range validation

This fixture validates a two-block DPA4-SOG canonical `.pt2` on the
water-scan0 192-atom cell. The host pair style runs by default; the Kokkos pair style also runs the
192-atom two-block case, enabled with `WATER_SCAN0_RUN_KK=1` (the harness runs
the host leg first; invoke the Kokkos leg alone if your MPI/GPU stack requires
it).
The 1.12M-atom two-block Kokkos case still hits the AOTI sort limit.

No dataset or model is committed. Provide:

```bash
export WATER_SCAN0_LMP_DATA=/path/to/water-scan0/lmp/sog/water_charge.data
export WATER_SCAN0_MODEL=/path/to/water_scan0_lr.pt2
export WATER_SCAN0_PLUGIN=/path/to/host/libdeepmd_lmp.so
export WATER_SCAN0_RUN_KK=0            # set 1 to add the Kokkos 2L leg manually
export WATER_SCAN0_PLUGIN_KK=/path/to/kokkos/libdeepmd_lmp.so
export WATER_SCAN0_BACKEND_PLUGIN_KK=/path/to/kokkos/backend
export LMP=/path/to/lmp
export LMP_KK=/path/to/kokkos/lmp
export DP_BACKEND_PLUGIN_PATH=/path/to/api_cc
./run_validation.sh
```

The harness runs `np=1/2/4` on the host; with `WATER_SCAN0_RUN_KK=1` it also
runs the Kokkos leg on the 192-atom 2L fixture. Every single-frame energy and
pressure must agree. `in.water.lammps` can be used
manually by passing `-var DATA/-var MODEL/-var PLUGIN`; it is a `run 0`
input, not an MD protocol.

For the empty-rank check, use `processors 2 2 1` with a data file that
leaves one subdomain without atoms; the canonical with-comm graph keeps every
rank in the same collectives. The 192-atom 2L Kokkos leg is supported; the
1.12M-atom 2L case is not.

For the z-slab distributed mesh, add `processors 1 1 $np` before `read_data`
and export the model with a CubeS2/QuadS or B-spline order 4/6 mesh:
`SOG_MIDTOWN_SLAB=1` enables the slab solve, and `SOG_SLAB_IK=1` adds the
Fourier-differentiated force. Every rank still joins the collective when its
subdomain has no atoms.
