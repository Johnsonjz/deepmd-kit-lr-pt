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

## Multi-rank (np ≥ 2): delivered

The compact canonical lower now carries the descriptor's cross-rank border
exchange, so a frame decomposed over ranks reproduces the single-domain run.

| run (`data_192.lmp`, `run 0`) | PE (eV) | press (atm) | max\|dF\| vs np = 1 |
|---|---|---|---|
| 1 rank | −29946.758 | −3151.5968 | — |
| 2 ranks (`processors 1 1 2`) | −29946.758 | −3151.5949 | 1.0e-5 (rms 2.3e-6) |
| 4 ranks (`processors 1 1 4`) | −29946.758 | −3151.5929 | 1.0e-5 (rms 2.5e-6) |

`|F|rms` of the frame is 1.404 eV/Å, so the force agreement is at the
single-precision floor — the same 1e-5 the DPA4C control shows.

Throughput (30 NVT steps, `timestep 0.0005`, one GPU per rank, 12 288 atoms):

| stage | np = 1 | np = 2 | np = 4 |
|---|---|---|---|
| canonical artifacts first working | 1.97 steps/s | 2.64 | 4.23 |
| + accelerator kernels (default) | 1.97 | 2.69 | 4.40 |
| + pinned border staging | **2.17** | **3.41** | **4.93** |

The border exchange was the one multi-rank cost DPA4 pays and DPA4C does not.
It is gone: at two ranks the step costs 8.80 s against the 8.75 s of an
artifact that performs no exchange at all. It was never the bytes — a
row-only staging that moved 5–10× fewer bytes bought nothing; the pageable
host buffer was forcing a slow, synchronizing round trip four times per step.

Where the remaining time goes: an eager step at 1536 atoms takes 0.080 s and
the compiled canonical artifact 0.063 s, while **DPA4C needs 0.078 s at
12288 atoms**. The SeZM descriptor costs several times more per atom and that
gap does not move with the kernel level — it is the message-passing stack
itself, not a missing fast kernel. Closing it needs a fused SeZM operator
(the B-lineage route), which is a project of its own.

### Memory at extreme sizes (331 776 atoms, 149.3 Å, 24 GB card)

The mesh is **replicated** on every rank, so it is a per-rank cost that rank
count does not divide. Measured peak per rank, with the same model frozen once
with the long-range module and once without:

| DPA4C (`lr_bs6` vs `lr_sr`) | np = 1 | np = 2 | np = 4 |
|---|---|---|---|
| with the module | 4452 MB | 3110 MB | 2852 MB |
| without | 3174 MB | 2128 MB | 2182 MB |
| **long-range delta** | **1278 MB** | **982 MB** | **670 MB** |

The replicated mesh is 216³ points = 80 MB of complex64 by itself; the rest is
the operator's work arrays and cuFFT workspace. A z-slab distributed mesh would
divide it, saving up to (np−1)/np of that delta — of order 0.5–1 GB per rank at
np = 4, about 20 % of that run's peak. **It is a memory lever, not a speed one**
(time-wise it is 2.5 %), and only for a fine mesh.

For SeZM+SOG the mesh is 40³ = 0.5 MB (spacing 3.98 Å), so distributed and
replicated differ by 0.4 MB: pointless.

What does limit SeZM is elsewhere. Canonical peak per rank at 12 288 atoms:

| np = 1 | np = 2 | np = 4 |
|---|---|---|
| 20.3 GB | 15.8 GB | 10.3 GB |

A run at 20 736 atoms fails with `aoti_torch_empty_strided` running out of
memory on a 24 GB card, so this artifact tops out between 12 288 and 20 736
atoms. Most of that is a per-rank floor of about 7 GB (context, runtime,
AOTI/allocator pools) that ranks do not divide; the split part is roughly
13 GB at np = 1.

## Known limitation

A rank that owns no atom still takes the empty-subdomain early return and never
joins the artefact's collectives, so a decomposition that empties a rank would
hang rather than answer. The DPA4C route pads two phantom local atoms for this;
the compact canonical ABI has no place for them yet, and this run does not hit
the case.

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
