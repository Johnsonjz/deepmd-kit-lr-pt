# SeZM canonical deployment — status (2026-11-09, updated)

## Delivered: the compact canonical `--compress` artifact is faithful

```
./freeze.sh model.ckpt-30000.pt model_30k_canon_fixed.pt2 0 --compress
  checkpoint E = -29925.229690   artifact E = -29925.229672
  |dE| 9.6e-08 eV/atom   rms|dF| 2.18e-06 eV/A   PARITY_PASS
```

LAMMPS, `run 0` on `sog_work/bench/data_192.lmp`, single rank:

| | PE (eV) | max\|dF\| (eV/Å) | rms\|dF\| |
|---|---|---|---|
| checkpoint (`DeepPot`) | −29946.757634 | — | — |
| `model_30k_canon_f32.pt2`, 1 rank | −29946.758 | 6.6e-6 | 3.1e-6 |

Three freeze-time defects had to be fixed first (all in
`deepmd/pt_expt/utils/serialization.py` unless noted, all DPA4/SeZM-only):

1. **`_sezm_canonical_reference_kernels()`** — the DPA4 family's accelerator
   levels were baked into the exported graph; a graph carrying the
   `sezm_triton::*` operators does not reproduce the model once it leaves the
   Python module (−29931.156 eV against −29925.230 even when run eagerly with no
   AOTInductor). The compact canonical lower of the family is now traced on the
   reference kernels.
2. **no `aot_inductor.use_runtime_constant_folding`** for the family — that
   pass corrupted an otherwise-correct program (6.3e-3 eV/atom against 1.7e-8).
3. **`graph_edge_dtype` reports `float32` for the canonical lower kinds**
   (`deepmd/pt_expt/model/graph_lower.py`) — the compact ABI is traced with
   float32 edges and the C++ ingestion requires them, but the SeZM atomic model
   reported its native float64, so LAMMPS refused the artifact with "compact
   canonical graph artifacts require float32 edge vectors".

## Open: multi-rank (np ≥ 2) — a defect of the canonical lower itself

| run | PE (eV) | vs single rank |
|---|---|---|
| full model (SeZM+SOG), 1 rank | −29946.758 | — |
| full model (SeZM+SOG), 2 ranks (`processors 1 1 2`) | −29931.239 | **+15.52 eV**, rms\|dF\| 0.96 eV/Å |
| **short-range-only SeZM**, 1 rank | −29393.266 | — |
| **short-range-only SeZM**, 2 ranks | −29389.077 | **+4.19 eV**, rms\|dF\| 2.68 eV/Å |
| DPA4C canonical control, 1 rank | −29946.558 | — |
| DPA4C canonical control, 2 ranks | −29946.558 | identical, max\|dF\| 1e-5 |

The short-range-only artifact is frozen from `model_30k_sr.pt` (no
`long_range` key), and it passes the parity gate at
2.7e-7 eV/atom / 6.9e-6 eV/Å. It still breaks at np = 2, so **the long-range
wiring is exonerated**: the defect is in the SeZM compact canonical lower
itself, on a ghosted node axis.

Ruled out as well: the adapter's ghost arithmetic. Relabelling 4 or 16 nodes of
a single-domain frame as ghosts (`n_local < n_node`) keeps the compact lower
and the autograd topology lower within 3e-8 eV/atom and 1.3e-6 eV/Å of each
other; the C++ does pass the MPI communicator
(`pair_deepmd_kokkos.cpp`: `world_ptr = long_range && multi_rank ? &world :
nullptr`) and does require the source CSR (`canonical_requires_source_csr`).

### Root cause: the compact canonical lower has no cross-rank exchange

`sezm_canonical_energy_force` calls `forward_common_lower_graph` with
`comm_dict=None` — the canonical adapter reconstructs the graph in Python from
the dual CSR and never opens a `border_op` exchange. A descriptor whose message
passing extends across a rank boundary therefore keeps the ghost features of
its **first** block for every later one.

That is exactly the observed signature: the energy stays close
(1.4e-4 relative) because it is a per-node sum, while the forces are wrong
everywhere (rms 2.68 eV/Å against |F|rms 1.52), uniformly rather than
concentrated on the boundary atoms, and identically for an x- and a
z-decomposition (−29389.174 and −29389.077 eV).

DPA4C is unaffected because it is **not** a message-passing descriptor
(`has_message_passing: False` in its metadata, against `True` for SeZM).

`_collect_metadata` now records `canonical_single_rank_only` for such an
artifact and logs a warning. Running the multi-rank case needs the canonical
lower to thread `comm_dict` through `canonical_graph_edge_inputs` and the
descriptor, i.e. a with-comm canonical artifact — which
`_needs_with_comm_artifact` currently answers `False` for, because DPA4's dense
and canonical lowers are documented as raising on `comm_dict`.



See `SEZM_AOTI_MISCOMPILE.md` for the full isolation and the ruled-out list.

## The model-level long-range connection

The compact canonical ABI carries no reciprocal-space topology. A message-passing
SeZM model therefore cannot evaluate its long-range module on the topology
(autograd) path, whose mesh sizing detaches to NumPy. It now evaluates it through
the **existing fused `deepmd::latent_charge_sum` operator**, whose mesh follows
the cells inside the compiled program. No fused SeZM descriptor kernel and no
full-LR 1F+1B artifact is required.

The mechanism is the DPA4C canonical construction, applied to an autograd
descriptor:

| step | where |
|---|---|
| descriptor `gg` (invariant node features) | `DescrptDPA4.call_graph`, unchanged |
| long-range energy / charge response / fixed-charge force / virial | `LatentChargeLongRange.energy_force_from_coord` → `deepmd::latent_charge_sum` |
| charge response reaches the positions | a **zero-valued linear surrogate** `g_lr · (gg − detach(gg))` added to the per-node energy, so the caller's single `edge_vec` backward carries it |
| force / virial at fixed charges | written out of band into `long_range_op["out"]` and added after that backward |

$$
E \leftarrow E_{\mathrm{sr}} + E_{\mathrm{lr}} + \underbrace{g_{\mathrm{lr}}\cdot\left(g-\operatorname{detach}g\right)}_{=\,0\ \text{in value}},\qquad
\mathbf F = -\frac{dE_{\mathrm{sr}}}{d\mathbf r} - g_{\mathrm{lr}}\frac{dg}{d\mathbf r} + \mathbf F_{\mathrm{lr}}\big|_q
$$

### Files

| file | change |
|---|---|
| `deepmd/dpmodel/atomic_model/dp_atomic_model.py` | `forward_atomic_graph` gains `long_range_op`; `_long_range_from_op` evaluates the module through the operator and adds the surrogate |
| `deepmd/dpmodel/atomic_model/base_atomic_model.py` | `forward_common_atomic_graph` forwards `long_range_op` |
| `deepmd/pt_expt/model/make_model.py` | `forward_common_lower_graph` gains `long_range_op`, collects the operator's out-of-band outputs and calls `_add_fused_long_range` after the backward |
| `deepmd/pt_expt/kernels/dpa4/canonical.py` | `canonical_long_range_op_eligible`; `sezm_canonical_energy_force` takes `coord` / `communicator` / `box` and builds the operator context |
| `deepmd/pt_expt/model/ener_model.py` | the SeZM+LR branch of `forward_lower_canonical_graph` no longer raises |
| `deepmd/pt_expt/utils/serialization.py` | `_canonical_requires_source_csr` returns `True` for SeZM (its force assembly walks both CSRs) |

### Numerical equivalence (eager, `md/lr_op_equivalence.py`)

Frame 0 of `water-scan0/data/split/test/data1`, checkpoint `model.ckpt-30000.pt`:

| quantity | graph lower (topology AD) vs fused op | compact canonical lower vs graph lower |
|---|---|---|
| energy | 2.3e-10 | 4.0e-10 |
| atom_energy | 4.6e-9 | 5.2e-9 |
| force | 3.3e-7 | 3.5e-7 |
| virial | 2.0e-7 | 6.1e-7 |
| atom_virial | 1.5e-6 | 1.5e-6 |

Both routes agree with the autograd topology path to ~1e-6 relative. The model
layer is correct; what remains is compilation.

## The remaining blocker: AOTInductor and `aten.slice_backward`

`--compress` freezes to `lower_kind=dpa4c_canonical`, traces with `make_fx`,
exports with `torch.export`, and then **AOTInductor compilation fails**:

```
torch._inductor.exc.InductorError: LoweringException:
  TypeError: evaluate_min(9223372036854775807, s65) with unbacked symints
  target: aten.slice_scatter.default
```

### Root cause (found)

The exported graph has **no** `aten.slice_scatter`; it has 150
`aten.slice_backward` nodes (the force backward slices on the node axis) and 327
open-ended `aten.slice` nodes (`end = sys.maxsize`). Inductor decomposes

```python
# torch/_decomp/decompositions.py::slice_backward
grad_input = grad_output.new_zeros(input_sizes)
return torch.slice_scatter(grad_input, grad_output, dim, start, end, step)
```

so each open-ended backward slice becomes `new_zeros` + `slice_scatter` with
`end = 9223372036854775807`. `ir.SliceView.normalize_start_end` must clamp that
`end` against the symbolic node-axis size, and `sizevars.evaluate_min(INT_MAX,
s65)` cannot decide `min(INT_MAX, s65)` because `s65` is an **unbacked** symbol
(the export defers every shape guard to a runtime assert, so no hint survives).

The graph lower does not hit this because the C++ interface assembles the edge
tensors there instead of producing a graph with open-ended node-axis slices —
and even where a `slice_scatter` survives, it falls back to the proxy executor
(`"aten.slice_scatter.default is missing a c-shim implementation"`).

### Fix under test

Keep `aten.slice_backward` as an **extern aten kernel** instead of letting
Inductor decompose it, i.e. drop it from
`torch._inductor.decomposition.decompositions` before
`aoti_compile_and_package`. The op is then emitted through the proxy executor,
which clamps `end` with the concrete runtime size.

## Do not re-investigate

* The long-range operator's mesh sizing (`box` is read inside C++; nothing
  detaches to NumPy during the trace).
* The charge-response chain rule (verified to 1e-6 against the topology path).
* `canonical_requires_source_csr` (SeZM needs the source CSR, and now says so).
* `k_grid_mode='ellipsoid'`: `use_mesh_evaluation()` rewrites it to the mesh at
  freeze, which the operator serves.
