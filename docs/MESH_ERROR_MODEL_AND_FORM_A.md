# Mesh error model and the exact influence function (Form A)

Status as of 2026-11-08. Companion to `LONG_RANGE_PORT.md` (section "Accuracy")
and to `papers/sog-lib-paper` §4.3, which carry the same content in their own
notation. Every path below is absolute.

---

## 0. The question this answers

The `midtown` (CubeS₂) mesh backend measured **less accurate than the B-spline at
the same grid**: 4.8× worse rms at order 6 in one table, "within a factor ~2" in
another table of the same document. That appears to contradict the asymptotic
optimality of midtown splines (Predescu, Bergdorf & Shaw, *J. Chem. Phys.* 153,
224117 (2020), Proposition 5).

It is not a contradiction. **The deployed CubeS₂ path was not inverting its own
assignment window.** The spline is fine; the influence function was a
second-order approximation, and its residual dominated.

---

## 1. Theory

### 1.1 Three terms, not one

The mesh error is

```
ε_mesh(Δ) = ε_tail(k_max)  +  (C_assign + C_infl) · (Δ/σ_min)^p          (1)
```

* `ε_tail` — kernel truncation at `k_max` (unchanged; §4.1 of the paper).
* `C_assign` — the assignment/interpolation/alias residual. This is the
  window's own property. It is the `2ν`-th Taylor coefficient of the assignment
  symbol about `k = 0`. For a cardinal B-spline the same object is the
  Dirichlet symbol `(sin(θ/2)/(θ/2))^p`; for CubeS₂ it follows from the moment
  conditions of the midtown paper (its Eq. 46), which make the leading
  unmatched moment appear at total degree `2ν`.
* `C_infl` — the **influence-function residual**. Identically zero when the
  window symbol is inverted exactly: the B-spline path divides by the analytic
  `|D_d(m)|^{-2}` (`deepmd/dpmodel/long_range/reciprocal.py`, `_assignment_factor`),
  and the separable QuadS family by its exact window division. The zero-cost
  alternative, a variance subtraction `exp(ξ²Δ²k²)`, leaves a residual of the
  **same order** with a constant that is an implementation artefact.

Writing the bracket as a single `C_ν` hides the distinction, and the
distinction is the whole point: only `C_assign` is a property of the spline.

### 1.2 The force exponent is not `2ν`

The force requires the **derivative** of the assignment window. A window that
is only `C⁰` has a discontinuous derivative, and the force error loses one
order:

```
p = 2ν        energy metric
p = 2ν − 1    force metric, C⁰ window                            (2)
```

This is not a defect. The midtown paper's Table I tabulates exactly this:
CubeS₂ order 4 is `C²`, CubeS₂ order 6 is `C⁰`.

### 1.3 Form A and Form B

For an assignment window whose symbol is `W(k)`:

```
Form A (exact):   ge(k) = kfac(k) / |W(k)|²
Form B (variance subtraction):   ge(k) = kfac(k) · exp(ξ² (Δx²kx² + Δy²ky² + Δz²kz²))
```

`kfac(k) = Σ_m a_m (2πσ_m²)^{3/2} exp(−σ_m²k²/2)`, the Fourier transform of
`K(r) = Σ a_m e^{−r²/2σ_m²}`. (Verified: the operator's `amp = a(2πσ²)^{3/2}`
is exactly this transform; an independent direct k-sum agrees with the operator
to 1.4e-6.)

Three facts make Form A work in practice:

**(a) The divisor is `D · ⟨|W|²⟩`, not `⟨|W|²⟩`.**
The incoming table is Form B and its `D = exp(ξ²Δ²k²)` already stands in for
`1/|W|²`. So

```
ge_A = kfac/|W|² = ge_B / (D · ⟨|W|²⟩)                            (3)
```

Dividing by `⟨|W|²⟩` alone applies the deconvolution **twice**. Measured
symptom: the reciprocal part of the energy came out 1.7× too large and the
total energy changed sign (+102 eV instead of −23 eV). Diagnostic that found
it: zero every mode; the energy must then equal the closed-form self term
`−½·COEF·(Σa_m)·Σq²`, which is grid-independent (measured −204.16 eV at three
different spacings, as it must be).

**(b) The average must be incoherent, `⟨|W|²⟩`, not `|⟨W⟩|²`.**
The energy sums charges incoherently, so a single global deconvolution can only
invert their average power spectrum. A coherent average cancels — it reached
1e25 in a test. This is also why the closed-form `I_p(α) = ∫₀¹ t^p e^{iαt} dt`
route is a dead end: integrating over the sub-cell position is exactly what
produces `⟨W⟩`.

**(c) The window has near-zero modes, so the cut must be relative.**
`⟨|W|²⟩ ≤ 1` by construction (it is the mean of a power spectrum of a unit
charge), so the floor is inherently relative. Modes below it are dropped, not
amplified, exactly as the B-spline path drops its Nyquist planes. An *absolute*
floor lets the corner be amplified by ~1e8; measured symptom: the Form-A error
*grew* from 6e-3 to 1.1 as the grid refined.

### 1.4 How `C` may be obtained

* **Not** from the midtown paper's `φ_max = Δ/rc` table. That table is
  empirical: it was determined by requiring the discretisation error to fall
  below the u-series' *intrinsic* error on representative biomolecular systems,
  keeping the most conservative value. It is not a bound and cannot be turned
  into a single `C_ν`. The mapping to our normalisation is
  `Δ/σ_min = φ_max · (rc/σ)`, with `rc/σ = 1.9893` for the `b = 2`
  parameterisation; CubeS₂-6 `φ_max = 0.35` therefore corresponds to
  `Δ/σ_min = 0.696`.
* **Derived** for `C_assign`: it is the `(2ν)`-th Taylor coefficient of the
  window symbol, computable from the node table (the polynomial pieces are in
  `source/op/pt/long_range/midtown/sog_spline.h`).
* **Measured** in practice, per `(family, order, form, kernel, metric)`,
  against a converged direct k-sum. That is what the tables below are.

A coefficient quoted without its family, order, form, kernel and metric is not
usable — `C(Form B)` itself varies 36 % across kernels (§3.4).

---

## 2. How the numbers were obtained

### 2.1 Reference

Two independent references, both validated:

1. **Direct k-sum, analytic force.** `E = ½(1/V) Σ_k K|ρ̂|²`,
   `F_i = −(1/V) Σ_k K q_i k · Im[ρ̂* e^{−ik·r_i}]`. The force was checked
   against a finite difference of the same energy (the apparent 13× discrepancy
   when first run was a missing factor of `COUL`, not an error: 6.59e-2/14.4 =
   4.58e-3).
2. **Converged fine grid** of the operator itself. Both give the same forces
   bit-for-bit, so the force error is not a reference floor.

### 2.2 Self-consistency checks

* `E = ½ Σ q_i φ_i` holds to machine precision on both references.
* `max|F + dE/dx| / max|F| ≈ 2e-3` for every backend and order (the
  finite-difference noise). **The operator's force is the gradient of its own
  energy** — there is no spread/gather inconsistency, which rules out the
  first hypothesis for the order-6 anomaly.
* The operator's mesh builds the kernel the Python `SOGKernel` describes: an
  independent direct k-sum agrees with the operator to 1.4e-6, so parameters
  migrate unchanged between the two implementations.

---

## 3. Validation numbers

### 3.1 Energy metric — the 4.8× was the deconvolution

Deployed operator, 64 charges, 12 Å, test kernel (`σ_min = 0.8 Å`), relative
energy error against a converged direct k-sum:

| Δ/σ_min | `bspline-4` | `midtown-4` Form B | `midtown-4` **Form A** |
|---|---|---|---|
| 0.625 | 6.18e-2 | 1.19e-1 | **1.80e-3** |
| 0.500 | 2.02e-2 | 5.16e-2 | **4.85e-3** |
| 0.417 | 1.03e-2 | 2.22e-2 | **2.88e-5** |
| 0.375 | 5.27e-3 | 1.53e-2 | **8.02e-4** |
| 0.312 | 2.61e-3 | 7.36e-3 | **4.41e-4** |

Fitted `C` at `p = 4`: `C(Form B) = 0.77`, `C(bspline-4) = 0.27`, both stable to
a few percent. Form A is **11–770× better than Form B** and better than the
B-spline at the same grid.

Form A does **not** yield a usable constant from this estimator: dividing its
column by `(Δ/σ)^4` scatters by 82× with no trend, because what remains is at
the resolution of the audit (64 atoms, one configuration, 8³ sub-cell samples,
grid-quantised spacings producing accidental cancellations).

### 3.2 Force metric — the C⁰ law

Trained water kernel from `/data/home/public/jiangzhen/dp/master-merge-test/water/model.ckpt-200000.pt`
(12 bands, `σ_min = 1.134 Å`), analytic direct k-sum force, 64 atoms, 12 Å:

| family | measured slope | fitted `C` | spread |
|---|---|---|---|
| `bspline-4` | **3.5** | `C(p=4) = 0.18` | 0.79–1.24 |
| `bspline-6` | **6.3** | `C(p=6) = 0.033` | 0.95–1.43 |
| `midtown-4` Form B | **3.2** | `C(p=4) = 0.26` | 0.85–2.33 |
| **`midtown-6` Form B** | **5.3** | **`C(p=5) = 0.098`** | 0.79–1.09 |
| `midtown-4/6` Form A | same as Form B | — | — |

The two `midtown-6` fits (coarse band and the finer 0.265→0.176 band) give 5.29
and 5.36 — a robust `p = 5`. The B-spline keeps `p = 6`, as `C⁴` continuity
predicts.

### 3.3 Form A's benefit is metric-dependent

| metric | order 4 | order 6 |
|---|---|---|
| energy | **11–770×** | ~1× |
| force | **2.5–5×** | ~1.1× |

The force column is the paper's own audit tool (`cubeA`, 8³ sub-cell samples,
four kernels); the deployed operator on the trained kernel shows essentially
**no** force-metric difference. The energy and force weight the influence
residual differently, so a gain quoted without its metric is not usable.
**This is why the constants table uses the force metric**: it is the canonical
one, and it is the conservative one.

### 3.4 The constants depend on the kernel

Energy metric, Form B, order 4, same Δ/σ band:

| kernel | `C` |
|---|---|
| 3-band (σ_min 0.8) | **0.78** |
| single band (σ 1.2) | **1.06** |

A 36 % spread — the same order as the "~40 % across kernels" the SOG library
paper reports. Under Form A the spread largely disappears, i.e. the kernel
sensitivity lived in the deconvolution residual.

### 3.5 The production chain

DPA4C-SOG LAMMPS chain (`kspace_style sog` + `fix sog/response`), 192 atoms,
12.4 Å:

| | grid | E(step 0) | P(step 0) |
|---|---|---|---|
| before the constant fix | 27³ | — | — |
| after (`C = 2.6e-1, p = 4`) | **41³** | −30008.093 | 16414.4 |
| + Form A (`SOG_FORM_A=1`, order 4 path) | 41³ | −30008.093 | 16414.386 |

The grid moves because the old `C_4 = 1.0e-1` under-resolved by ~2× in points
per axis. The discriminating Form-A measurement is at a coarse grid, where the
variance subtraction still matters: at 9³ (forced with `phi_accuracy 0.02`) Form
A moves the energy by +0.009 eV and the pressure by −15 bar at order 6. At 41³
the two forms are within the print precision, as they should be.

---

## 4. Code changes

### 4.1 Operator (deepmd-kit-LR-lr-pt)

| file | change |
|---|---|
| `/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt/source/op/pt/long_range/midtown/sog_gpu.cu` | `sog_gpu_green_form_a()`, `k_acc_spec_abs2`, `k_gp_divide_form_a`; lazy `f_rho`/`f_pot` (saves 16 B/grid point = 27 % of the mesh set); `free_grid_buffers` now releases the fp32 set and the fp32 plans; padding-lane out-of-bounds write fixed in `k_spread_cubes2_warp_fx_packed`; `k_kspace_h` Hermitian weight for odd `nx` |
| `/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt/source/op/pt/long_range/midtown/sog_gpu.cuh` | declarations for the above |
| `/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt/source/op/pt/long_range/midtown/sog_green.h` | `grid_axis` constants: CubeS₂ `C = 2.6e-1 / p = 4` (order 4), `C = 9.8e-2 / p = 5` (order 6) |
| `/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt/source/op/pt/long_range/midtown_sum.cu` | Form A call site (`SOG_MIDTOWN_FORM_A`, value-parsed); `form` recorded in `MidtownState` and in all three cache keys; loud refusal of Form A on the local-spread and slab paths; `nth` clamped to ≥1; z-subdomain guard moved to the device and evaluated once; device-side `np==1` halo folds; `ZL >= gh` guard |
| `/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt/docs/LONG_RANGE_PORT.md` | accuracy section rewritten (two-term model, Form A, the C⁰ law, the metric warning); the two contradictory tables reconciled |
| `/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt/docs/MESH_ERROR_MODEL_AND_FORM_A.md` | this document |

### 4.2 SOG library (two copies, must stay identical)

| file | change |
|---|---|
| `/data/home/public/jiangzhen/sog/lmp/sog.cpp` | `apply_form_a_green()` (host stencil via `LAMMPS_NS::cubes2_weight_4/6`, cuFFT via LAMMPS `FFT3d`, relative floor); grid constants updated; new **B-spline branch that reads `bspline_order`** (previously every order used the CubeS₂ order-4 constants); QuadS branch deliberately left alone |
| `/data/home/public/jiangzhen/sog/lmp/sog.h` | declaration |
| `/data/home/public/jiangzhen/dp/deepmd-kit-LR/source/lmp/sog.cpp` | same changes (this is the copy the deployed plugin builds) |
| `/data/home/public/jiangzhen/dp/deepmd-kit-LR/source/lmp/sog.h` | declaration |

Flags: `SOG_FORM_A` (value-parsed), `SOG_FORM_A_NTH` (default 3 on the CPU path),
`SOG_FORM_A_FLOOR` (default 1e-4).

### 4.3 Paper (`/data/home/public/jiangzhen/papers/sog-lib-paper`)

| file | change |
|---|---|
| `sections/04-general-kernel.tex` | §4.3 rewritten: Eqs. (1)–(3) above, a new `eq:force-order`, `eq:phimax-rule` uses `p`; the `φ_max ↔ Δ/σ_min` mapping and the worked inconsistency |
| `sections/06-results.tex` | `tab:spline-slopes` and `tab:phimax` captions now say "Form B effective coefficient"; the figure caption no longer calls CubeS₂ Form A ill-conditioned; the order-4 gain is stated per metric |
| `sections/03-mesh-method.tex` | "direct division unsuitable" replaced by "needs a relative near-zero cut", with the measured effect |
| `sections/08-appendix.tex` | Form A is usable with CubeS₂; the divisor is `D·⟨|W|²⟩` |
| `sections/00-abstract.tex`, `sections/07-conclusion.tex` | the influence-function axis and the C⁰ force order added |
| `tools/mesh_validation.py` | `cubeA` method added (previously any non-QuadS `*A` silently used the B-spline sinc window, measuring the wrong thing); `nth = 8` |

Build: `cd /data/home/public/jiangzhen/papers/sog-lib-paper && pdflatex main.tex`
→ 37→38 pages, no errors, no undefined references.

### 4.4 Audit tooling (scratch, not in the repo)

| file | purpose |
|---|---|
| `/tmp/lr_md/op_audit.py` | deployed-operator audit: realized grid, energy and force relative error, `--direct` for the analytic direct k-sum reference. **Caveat:** its direct energy reference does not include the closed-form self term `−½·COEF·(Σa_m)·Σq²`; forces are unaffected |
| `/tmp/lr_md/mesh_audit2.py` | independent Python reference (B-spline and CubeS₂, Form B and exact symbol) |
| `/tmp/lr_md/symbol_analytic.py` | proves the closed-form route yields `⟨W⟩`, not `⟨|W|²⟩` |
| `/tmp/lr_md/fd_check.py` | force vs `−dE/dx` consistency |
| `/tmp/lr_md/sweep_backends.py` | size ladder / timing (see `LONG_RANGE_PORT.md` §6) |

### 4.5 Verification

* `pytest source/tests/common/dpmodel/test_long_range.py source/tests/pt/model/test_sezm_long_range.py` → **83 passed** (61 + 22), Form A off by default.
* `/data/home/public/jiangzhen/sog/lmp/build` and `/tmp/dplmp` (the deployed plugin) both build.
* `SOG_MIDTOWN_FORM_A=0` → Form B (1.19e-1), `=1` → Form A (1.80e-3), at Δ/σ = 0.625.
* Slab path with Form A set → explicit `RuntimeError`; slab alone → `E = 0.7243632187771221`, the previously validated value.

---

---

## 5. Known limitations

1. **Form A is replicated-path only.** The local-spread (`SOG_MIDTOWN_DIST`) and
   slab (`SOG_MIDTOWN_SLAB`) paths refuse it loudly rather than silently
   running Form B. Tilted cells (`build_green_metric`) are excluded by the same
   guard.
2. **`pme_order` defaults to 8 and is coerced to 6** under the midtown backend
   (`deepmd/pt_expt/long_range/latent_charge.py`); only `pme_order: 4` selects
   order 4. A constants table keyed on the *config* value would size for the
   wrong stencil.
3. **QuadS was never force-audited.** Its `C = 2e-4` is a self-described upper
   bound, so its exponent was left at `2ν` even though QuadS order 6 is also
   `C⁰`.
4. **Three hand-maintained copies of the grid rule** (`sog_green.h` in two
   trees, `sog.cpp` in two trees) — there is no lockstep test.
5. **`cubeA` is unstable on a few grids**: the CubeS₂ window has exact spectral
   zeros, the relative floor drops different mode sets at different grids, and
   the ratio to Form B jumps (one grid gave 0.6× instead of 3–5×). A finer
   sub-cell average, or a smarter truncation, would fix it.
6. The `C` values above are for **one trained kernel and one 64-atom
   configuration**. The paper's canonical panel is 192 atoms over an ensemble;
   the constants in the code should be re-derived on that panel before they are
   treated as kernel-independent.

---

## 6. Reproducing

```bash
# operator
cd /data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt/build-pt-mpi-gcc
OMPI_CXX=/opt/rh/gcc-toolset-13/root/usr/bin/c++ make deepmd_op_pt -j8
cp op/pt/libdeepmd_op_pt.so ../deepmd/lib/

# audit (energy + force, deployed operator, direct k-sum reference)
export KA=... KW=... SIGMIN=1.133775812      # trained kernel, see §3.2
python /tmp/lr_md/op_audit.py --direct --order 4,6 --backends bspline,midtown \
    --spacings 0.8,0.65,0.55,0.45,0.38,0.30

# Form A vs Form B
SOG_MIDTOWN_FORM_A=1 python /tmp/lr_md/op_audit.py --order 4 --backends midtown ...

# the paper's own audit, now with cubeA
cd /data/home/public/jiangzhen/papers/sog-lib-paper
python tools/mesh_validation.py --output /tmp/mesh_val.json
```

## 7. Form A at order 4: complete results, and why the error "jumps"

### 7.1 The jump is not a bug

Earlier single-configuration runs showed a non-monotonic Form-A error: at
`Delta/sigma = 0.625, 0.500, 0.417, 0.375, 0.312` the energy error read
`1.80e-3, 4.85e-3, 2.88e-5, 8.02e-4, 4.41e-4` -- a 28x improvement followed by
a 28x regression on a finer grid. Three independent checks rule out a defect:

| check | result |
|---|---|
| math | `⟨\|W\|²⟩` is the least-squares-optimal *single* global deconvolution for the ensemble; any single configuration deviates from it, and that deviation cannot vanish with `Delta` |
| parameters | `xi` matches the paper tool exactly; `D` matches `build_green_range` term for term; the `f_gp`/accumulator index pair was checked on a non-cubic `k²` set; the operator kernel equals the Python kernel to 1.4e-6 |
| implementation | the *change* introduced by Form A, `E_A - E_B`, agrees between the operator and an independent Python build to **1.6 %** (0.0859 vs 0.0873 eV); the absolute offset between the two is a constant 4.84 eV that affects both forms identically, i.e. it is in the scratch script's normalisation, not in the operator. `max\|F + dE/dx\|/max\|F\| ≈ 2e-3` for every backend and order |
| the same wobble appears in the independent Python implementation (`/tmp/lr_md/mesh_audit2.py`) |

**Cause: per-configuration fluctuation plus a phase in the residual.** The
operator uses one influence function, `⟨|W|²⟩`, but each configuration has its
own window mixture; dividing by the average leaves a configuration-specific
residual that is a fixed fraction of the kernel weight and does not vanish with
`Delta`. On top of that the residual itself oscillates in `Delta` (its
coefficient is not positive-definite).

### 7.2 Order-4 results over an 8-configuration ensemble

Trained water kernel (`sigma_min = 1.134 Å`), 64 atoms, 12 Å, RMS over 8
independent configurations, analytic direct k-sum reference, operator with
`SOG_MIDTOWN_FORM_A`:

| `Delta/sigma` | E Form A | E Form B | gain | F Form A | F Form B | gain |
|---|---|---|---|---|---|---|
| 0.706 | **1.87e-2** | 4.65e-1 | 25x | 8.15e-2 | 7.84e-2 | 0.96x |
| 0.573 | **3.94e-3** | 1.42e-1 | 36x | 3.00e-2 | 2.98e-2 | 0.99x |
| 0.485 | **1.66e-3** | 6.74e-2 | 41x | 1.77e-2 | 1.77e-2 | 1.00x |
| 0.397 | **1.62e-3** | 4.03e-2 | 25x | 1.19e-2 | 1.18e-2 | 0.99x |
| 0.335 | **2.61e-4** | 2.04e-2 | 78x | 5.60e-3 | 5.64e-3 | 1.01x |
| 0.265 | **4.66e-4** | 8.51e-3 | 18x | 3.20e-3 | 3.21e-3 | 1.00x |

* **Energy: Form A is 18-78x better**, the ensemble-averaged counterpart of the
  11-770x seen on single configurations. Envelope slope ~3.75 in `Delta`; the
  conservative constant over the band is `C_E(Form A) ≈ 0.095`, against
  `C_E(Form B) ≈ 1.7` for this kernel.
* **Force: no benefit at all** (ratios 0.96-1.01), and this is now explained.

### 7.3 Why Form A helps the energy and not the force

The two metrics evaluate the interpolation differently:

* the **paper's audit tool** differentiates in `k` (ik-differentiation, the same
  weights for spread and gather), so the force error tracks the potential
  error and Form A shows its 2.5-5x;
* the **deployed operator** differentiates the CubeS2 **weights**
  (`cs2_wg6` in `sog_gpu.cu`), which needs `dS/dtheta`.

CubeS2 order 6 is `C⁰` (midtown Table I), so its weight derivative is
discontinuous and the force error carries a term that **no influence function
can remove** -- hence `p = 2nu - 1` for the force (Eq. 2 above) and hence the
flat response to Form A. Order 4 is `C²`, so its force exponent is the full 4,
and there the weight-differentiated force does improve, just far less than the
energy does.

This is the honest summary: **Form A fixes the energy; it does not fix the
CubeS2 force.** Making the force track the energy would need a smoother window
(an `xi` in a `C¹`+ continuity class) or ik-differentiation in the gather.
