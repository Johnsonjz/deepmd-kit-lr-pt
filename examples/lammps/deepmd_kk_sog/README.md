# `pair deepmd/kk` + `kspace sog/kk`: a long-range SOG model in LAMMPS

This example runs a frozen long-range (latent-charge / SOG) model through the
all-device chain:

```
pair   deepmd/kk      short-range energy and forces; writes the latent charge
                      to atom->q on the device (no host round trip)
kspace sog/kk         SOG particle-mesh solve on the same device arrays
                      (spread -> cuFFT -> Green function -> inverse -> gather)
fix    sog/response   the dq/dr term: the long-range force also acts through
                      the coordinates the charge depends on
```

`run.sh` is a thin launcher for a Kokkos-enabled LAMMPS.

## Prerequisites

- LAMMPS with `PKG_KOKKOS` and CUDA, run as `lmp -k on g 1`.
- DeePMD-kit built with the LAMMPS plugin (`deepmd_lmp.so`) and the
  `pt_expt` backend; `DP_INSTALL` is the install prefix that contains
  `deepmd/lib/libdeepmd_cc.so` and the AOTInductor runtime.
- A frozen `.pt2` that carries the latent-charge response extra
  (`model/extra/latent_charge_response.pt2`; the `*_fused.pt2` artifact is the
  usual one) and a data file whose atom types match the model's `type_map`.
- The trained SOG kernel, in the C++ convention
  `K(k^2) = sum_m amp_m exp(-bandwidth_m k^2 / 2)` with `bandwidth = sigma^2`.
  Export `amp`/`bandwidth` from the checkpoint; do not retype them.

## Run

```bash
conda activate <env the artifact was built with>
export DP_INSTALL=/path/to/deepmd/install
export LMP=/path/to/lmp                      # defaults to `lmp` on PATH
export PLUGIN_DIR=/path/to/plugin_dir        # holds dpplugin.so
./run.sh in.deepmd_kk_sog.lammps
```

Edit the `USER` block of `in.deepmd_kk_sog.lammps` first: `MODEL`, `DATA`,
`TYPES`, `KERNEL`, `NSTEPS`.

For a quick check, set `NSTEPS` to `0`; the reported `pe` should match the
host chain (`pair deepmd` + `kspace sog`) on the same artifact. The reference
case in the development tree (192-atom water, one GPU) gives
`pe = -29971.346 eV` and is bit-identical at np = 1 and np = 2.

## Multi-rank

Do **not** run this input on more than one rank: the device response path in
`fix sog/response` aborts at np > 1, and the replicated k-space virial is
scaled by np. Use the host chain (`pair deepmd` + `kspace sog`) for MPI runs.

## Notes

- `latent_charge_to_q yes` and `full_lr no` pin the pair+kspace chain. With an
  artifact that also carries the in-graph long-range graph, `full_lr` would
  otherwise be enabled and the charge write-back skipped.
- `use_gpu yes` on `kspace_style sog/kk` selects the cuFFT device path;
  without it the solver silently falls back to the host path (correct, about
  2x slower per atom).
- `fix sog/response` needs the response extra in the artifact; it also arms the
  per-atom potential that the SOG solver reports.
