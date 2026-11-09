# SeZM `.pt2`: the AOTInductor package is not a faithful lowering

2026-11-09. Supersedes the "graph route differs from dense" hypothesis in
`master-merge-test/water_dpa4_sog/md/SEZM_DEPLOY_PARITY_BUG.md`.

## The defect

A DPA4(SeZM)+SOG `.pt2` produced by AOTInductor is **not** the program its own
`torch.export` graph describes: eager, the `make_fx` trace, the exported
program and the exported program moved to CUDA all agree, and only the compiled
package diverges.

Measured on `dp_example/water-scan0/data/split/test/data1` frame 0, checkpoint
`master-merge-test/water_dpa4_sog/model.ckpt-30000.pt`, compact canonical
artifact:

| evaluation | E (eV) | dE (eV/atom) | rms dF (eV/Å) |
|---|---|---|---|
| eager `DeepPot(ckpt)` | −29925.2297 | — | — |
| `make_fx` trace, run eagerly | −29925.2297 | 1.4e-7 | 2.7e-6 |
| `torch.export` program, run eagerly (CPU) | −29925.2297 | 1.3e-7 | 2.8e-6 |
| `torch.export` program moved to CUDA, run eagerly | −29925.2297 | 7.2e-8 | 1.3e-6 |
| **AOTInductor package** | **−29931.1557** | **3.1e-2** | **0.254** |

So the trace, the export and the CUDA kernels of the custom operators are all
correct, and the error is introduced by code generation (or by an AOTInductor
operator shim), not by the model, the graph ABI, the mesh or the long-range
wiring.

## Blast radius and the guard

Everything above is about the **frozen artifact**, not about the training or
eager evaluation of the model, and not about the descriptor on its own. A
short-range-only DPA4 freeze carries the same Triton-baked graph, so the first
defect is not specific to the long-range module — the long-range work merely
gave a frame on which a 3e-2 eV/atom discrepancy is unmissable.

`master-merge-test/water_dpa4_sog/md/freeze_parity_gate.py` compares every
`.pt2` against its checkpoint on one frame and refuses on `|dE| > 1e-3 eV/atom`
or `rms|dF| > 1e-3 eV/Å`; `freeze.sh` runs it. It caught both defects.

## Two independent defects

### 1. The frozen graph does not reproduce the model — but not for the reason first supposed

The freeze pins `DP_TRITON_INFER=2` / `DP_CUDA_INFER=1` for a DPA4 target
(`_dpa4_kernel_level_defaults`), so the SeZM descriptor's accelerator paths are
traced **into** the graph as opaque custom operators
(`sezm_triton.so2_rotate_mix`, `flash_atten_aggregate`, `so2_mixing_stack`,
`edge_force_assembly`, …). Saving the freeze's exported program and running it
eagerly — no AOTInductor involved — gives −29931.1557 eV against the model's
−29925.2297.

**But the operators themselves survive the trace.** Reproducing the whole
freeze pipeline by hand — `use_mesh_evaluation`, the SO2/embedding pinning,
`run_autotune`, `_prepare_dpa4_triton_value_path_weights`,
`forward_lower_canonical_graph_exportable`, `torch.export.export`,
`_strip_shape_assertions`, `move_to_device_pass` — at the same kernel levels
gives a program that reproduces the model to 1.2e-6 eV/Å, with 16
`sezm_triton` nodes in its graph. A make_fx trace and a `torch.export` of the
plain graph lower likewise agree to 1e-6 at `DP_TRITON_INFER=2`.

What separates the faithful reconstruction from the freeze is the **trace
sample**. The freeze derives it from
`count_synthetic_graph_edges(model, nframes=11, nloc=7)` → 353 edge slots;
tracing the same pipeline at that size raises

```
GuardOnDataDependentSymNode: Could not guard on data-dependent expression
zuf0 (unhinted). ... consider guard_or_false / guard_or_true /
statically_known_true.
```

That is a Python-level `int(tensor.shape[k])` on an **unbacked symbolic**
extent inside the accelerator path (candidates:
`kernels/triton/sezm/radial_mix.py:647,684` `rank = int(compact.shape[-1])`,
`so2_value_path.py:2487,2531` `c_wide = int(x.shape[2])`,
`so2_rotation.py:1166`). Under
`prefer_deferred_runtime_asserts_over_guards=True` such a guard is resolved
from the sample and, when `_strip_shape_assertions` then replaces the deferred
check with `True`, nothing catches the mismatch at run time.

The mechanism is therefore: **a shape guard in the accelerator path is
specialised on the trace sample, and the deployment runs on other shapes.**
This is the lead for an upstream report; the operator-lifting hypothesis above
is *not* confirmed and the earlier wording of this document overstated it.

Tracing on the reference kernels avoids the path entirely and is what the
freeze now does.


### 2. `aot_inductor.use_runtime_constant_folding` corrupts a correct graph

At `DP_TRITON_INFER=0` the graph is correct (5.9e-8 eagerly), but the package
the canonical freeze produced from it was not (6.35e-3 eV/atom). The one option
the canonical path added was
`aot_inductor.use_runtime_constant_folding = True`; with it off the package
passes:

```
DP_TRITON_INFER=0 + use_runtime_constant_folding off
  checkpoint E = -29925.229676   artifact E = -29925.229672
  |dE| 1.74e-08 eV/atom   rms|dF| 2.18e-06 eV/A   PARITY_PASS
```

The option is now not enabled for the SeZM family; those artifacts carry no
charge-state constants for it to fold in the first place.


## Where the defect is, precisely


Further isolation, all on the compact canonical artifact:

| step | dE (eV/atom) | rms dF (eV/Å) |
|---|---|---|
| the **freeze's own** exported program (`/tmp/freeze_ep.pt2`), run eagerly | 5.9e-8 | 1.4e-6 |
| that same exported program, recompiled by a clean harness with `build_inductor_compile_options(inference=True)` + `assert_indirect_indexing=False` + `realize_opcount_threshold=0` + `_opaque_slice_backward` | 5.2e-8 | 2.2e-6 |
| the freeze's package itself | 6.3e-3 … 3.1e-2 | 0.23 … 0.25 |

So the trace, the export **and** the exported program are all correct, and
AOTInductor compiles that same program correctly under a plain option set. The
difference is in the **option set the canonical freeze adds**:

```python
if metadata.get("lower_input_kind") in ("dpa1_canonical", "dpa4c_canonical"):
    lower_configs["aot_inductor.use_runtime_constant_folding"] = True
```

plus `_canonical_symbolic_shapes` (`torch.fx.experimental._config.patch(
backed_size_oblivious=True)`) around the compile. Compiling the saved freezes
program with `use_runtime_constant_folding=True` returns **NaN** from the
reloaded archive, so that pass is at minimum fragile for this family.

## Ruled out (measured, not argued)

| candidate | measurement |
|---|---|
| long range k-sum vs frozen mesh (the freeze rewrites `k_grid_mode` to `pme` and pins `pme_spacing = 2.90 Å`) | k-sum vs mesh: dE 5.3e-8 eV/atom, dF 1.3e-6 |
| the freeze's kernel pinning (`SO2Linear._force_block_diag_matmul`, `GeometricInitialEmbedding._force_fused_scatter`) | dE 2.2e-8, dF 7.9e-6 |
| the CPU reference vs the CUDA kernel of `deepmd::latent_charge_sum` | exported program run on CUDA: 1.1e-6 |
| `_strip_shape_assertions` applied to the exported program | 9.0e-8 / 2.3e-6 |
| `patch_inductor_force_int64_indexing()` | 9.9e-8 / 2.2e-6 |
| `_prepare_dpa4_triton_value_path_weights` (Triton SO2 value path packing) | eager: 1.1e-6 |
| `backed_size_oblivious=True` during the trace | 1.3e-7 / 2.7e-6 |
| `use_mesh_evaluation` | see above |
| TF32 (`allow_tf32` / `set_float32_matmul_precision`) | only 2.2e-4 eV/atom, 1.2e-2 eV/Å |
| `DP_TRITON_INFER` / `DP_CUDA_INFER` eager sweep {0,0}…{2,1} | all −29925.2297 |
| `_strip_shape_assertions` disabled in the freeze | unchanged |
| `DP_CUDA_INFER` / `DP_TRITON_INFER` sweep (eager) | bit-identical |
| edge set, `destination_order`, `assembleGraph`, mesh spacing, charge neutralization, LR injection order | all measured not to move the gap |
| DPA4C canonical (`dpa4c_250k_cli_canon.pt2`) regression | still PASS 1.4e-7 / 3.7e-6 |


## Why DPA4C is bitwise exact and SeZM is not

`forward_common_lower_graph` short-circuits to `_fused_energy_force_graph(...)`
when a descriptor implements `fused_energy_force_graph`
(`deepmd/pt_expt/model/make_model.py`). `DescrptDPA4C` implements it, so the
DPA4C artifact contains opaque custom operators and AOTInductor compiles no
autograd math; `DescrptDPA4`/SeZM does not, so its whole energy+force graph is
handed to AOTInductor. The "DPA4C control" is therefore not a control for
SeZM.

## What the compiled program contains

From the export (3 635 FX nodes): 150 `aten.slice_backward`, 254
`aten.slice.Tensor`, 67 `bmm`, 48 `mm`, 13 `index_add`, plus the custom
operators. Two observations:

* **`aten.slice_backward`** is decomposed by Inductor into `new_zeros` +
  `aten.slice_scatter` with the forward slice's `end`, which is
  `sys.maxsize` for an open-ended slice. `ir.SliceView.normalize_start_end`
  then needs `sizevars.evaluate_min(INT_MAX, s)` and raises
  `evaluate_min(...) with unbacked symints`, because the export defers every
  shape guard to a runtime assert and no hint survives. Keeping the operator
  opaque (dropping it from `torch._inductor.decomposition.decompositions`)
  makes the package compile; it is emitted through the proxy executor.
* That fix is a **workaround for compilation only**; the artifact compiled that
  way still shows the 3.1e-2 / 0.254 gap, so the wrong code is elsewhere.

## The gate (added)

`master-merge-test/water_dpa4_sog/md/freeze_parity_gate.py` compares a `.pt2`
against its checkpoint on one frame and refuses on `|dE| > 1e-3 eV/atom` or
`rms|dF| > 1e-3 eV/Å`. Wired into `freeze.sh`.

## Do not re-investigate

Edge-set parity, `destination_order`, `assembleGraph`, mesh spacing, charge
neutralization, LR injection order, kernel inference levels, device pinning
during the trace, the frozen mesh rewrite. All measured not to move this gap.
