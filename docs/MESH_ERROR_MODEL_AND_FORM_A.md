# Mesh error model, the exact influence function (Form A), and the grid rule

Reference test model for every number below:
`/data/home/public/jiangzhen/dp/master-merge-test/water/model.ckpt-150000.pt`
(trained `dpa4c-sog`, 12 SOG bands, `sigma_min = 2.0 Å`).
One earlier data set, from `model.ckpt-200000.pt` (`sigma_min = 1.134 Å`) of the
same run, is quoted only where the kernel dependence is the point.

---

## 0. Summary

* The `midtown` (CubeS₂) mesh measured less accurate than the B-spline at the
  same grid **on the force metric**. The cause was **not** the spline: the
  deployed path inverted its assignment window only approximately (a variance
  subtraction), and that residual dominated the *energy*. The force gap is the
  window's own smoothness, not the deconvolution.
* The mesh error has **three** terms, not one, and the **force** exponent is one
  lower than the energy's: `p_force = 2ν − 1`. Measured on the reference kernel:
  **4.64 ≈ 5 at order 6** and **2.82 ≈ 3 at order 4** — *not* `2ν`.
* Replacing the approximation by the exact measured window symbol ("Form A")
  lowers the order-4 **energy** error by **10–64×** on the same grid (a wash at
  order 6), and leaves the **force** unchanged to within ~4% — the force is
  Form-A-invariant because it is set by the window *derivative*, which no
  influence function can alter. Form A is therefore the **default at both CubeS₂
  orders** (was: order 4 only).
* On the force metric the B-spline is still strictly better than CubeS₂ at equal
  grid — **~6.8× at order 6** (4.7–9.8×) and **~1.8× at order 4** (1.5–2.1×),
  and the gap widens toward fine grids. On the *energy* metric Form A closes it
  (order 6 a wash, order 4 ~12× **better** than the B-spline).
* The grid rule now follows the measured law, per backend and spline order, and
  is live in the operator's Python layer. It never places a grid coarser than
  the legacy `0.75 π / k_max`. Because the force is Form-A-invariant, the
  force-based constants did not move when Form A became the order-6 default.

---

## 1. The problem

`docs/LONG_RANGE_PORT.md` reported the CubeS₂ mesh as 4.8× worse than the
B-spline at order 6 in one table and "within a factor ~2" in another, which
appeared to contradict the asymptotic optimality of midtown splines
(Predescu, Bergdorf & Shaw, *J. Chem. Phys.* **153**, 224117 (2020),
Proposition 5).

It does not. Proposition 5 is about the **assignment kernel**. The deployed
CubeS₂ path used a Green function that only approximated the inverse of that
kernel, and the approximation's residual was the dominant error.

---

## 2. Theory

### 2.1 Three terms

```
ε_mesh(Δ) = ε_tail(k_max) + (C_assign + C_infl) (Δ/σ_min)^p          (1)
```

* **`ε_tail`** — kernel truncation at `k_max`. Unchanged.
* **`C_assign`** — the assignment/interpolation/alias residual, and the
  window's own property: the `2ν`-th Taylor coefficient of the assignment
  symbol about `k = 0`. Moment matching cancels the lower terms, so the first
  unmatched one is at `2ν`. For a cardinal B-spline the same object is the
  Dirichlet symbol `(sin(θ/2)/(θ/2))^p`.
* **`C_infl`** — the influence-function residual. **Identically zero when the
  window symbol is inverted exactly**: the B-spline path divides by the
  analytic `|D_d(m)|^{-2}` (`deepmd/dpmodel/long_range/reciprocal.py`,
  `_assignment_factor`), and the separable QuadS family by its exact window
  division. The zero-cost alternative, a variance subtraction
  `exp(ξ²Δ²k²)`, leaves a residual of the **same order** with a constant that
  is an implementation artefact.

Collapsing the bracket into one `C_ν` hides exactly the thing that matters:
only `C_assign` is a property of the spline.

### 2.2 The force exponent is not always `2ν`

The force needs the **derivative** of the assignment window. A window that is
only `C⁰` has a discontinuous derivative and loses one order:

```
p = 2ν        energy metric
p = 2ν − 1    force metric, C⁰ window                              (2)
```

The midtown paper's Table I lists CubeS₂ order 4 as `C²` and order 6 as `C⁰`.
On the reference kernel the fitted force exponent is **4.64 at order 6** and
**2.82 at order 4** (§4.6) — both ≈ `2ν − 1`, so the weight-differentiated force
loses roughly one order at *both* CubeS₂ orders, not only the `C⁰` case. The
B-spline loses less: **5.46 at order 6** (quintic, `C⁴`) and **3.06 at order 4**
(cubic, `C²`), i.e. about half an order. The B-spline's smoother window is the
whole reason it beats CubeS₂ on force at equal grid (§4.6): it keeps
`p_force ≈ order` while CubeS₂ drops to `2ν − 1`. Eq. (2) is the clean
theoretical case; the shipped rule (§5.1) uses the measured exponent per
`(backend, order)` instead of extrapolating from it.

### 2.3 Form A and Form B

```
Form A (exact):              ge(k) = kfac(k) / |W(k)|²
Form B (variance subtrac.):  ge(k) = kfac(k) · exp(ξ²(Δx²kx² + Δy²ky² + Δz²kz²))
kfac(k) = Σ_m a_m (2πσ_m²)^{3/2} exp(−σ_m²k²/2)
```

`kfac` is the Fourier transform of `K(r) = Σ a_m e^{−r²/2σ_m²}`; the operator
builds exactly this (`amp = a(2πσ²)^{3/2}`), and an independent direct k-sum
agrees with the operator to 1.4e-6, so parameters migrate unchanged between the
two implementations.

Three facts make Form A work:

**(a) The divisor is `D · ⟨|W|²⟩`, not `⟨|W|²⟩`.** The incoming table is Form B
and its `D = exp(ξ²Δ²k²)` already stands in for `1/|W|²`:

```
ge_A = kfac/|W|² = ge_B / (D · ⟨|W|²⟩)                              (3)
```

Dividing by `⟨|W|²⟩` alone applies the deconvolution **twice**. Measured
symptom: the reciprocal energy came out 1.7× too large and the total energy
changed sign. Diagnostic: zero every mode; the energy must then equal the
closed-form self term `−½·COEF·(Σa_m)·Σq²`, which is grid-independent (it
reproduces to all printed digits at every spacing).

**(b) The average must be incoherent, `⟨|W|²⟩`, not `|⟨W⟩|²`.** The energy sums
charges incoherently, so one global deconvolution can only invert their average
power spectrum. A coherent average cancels — it reached 1e25 in a test. This
also rules out the closed-form `I_p(α) = ∫₀¹ t^p e^{iαt} dt` route: integrating
over the sub-cell position is precisely what yields `⟨W⟩`.

**(c) The near-zero cut must be relative.** `⟨|W|²⟩ ≤ 1` by construction, so the
floor is inherently relative. Modes below it are dropped rather than amplified,
as the B-spline path drops its Nyquist planes. An absolute floor lets the corner
be amplified by ~1e8; measured symptom: the Form-A error *grew* from 6e-3 to 1.1
as the grid refined.

### 2.4 How `C` may be obtained — and why it is kernel-specific

Not from the midtown paper's `φ_max = Δ/rc` table: that is empirical, defined by
requiring the discretisation error to fall below the u-series' *intrinsic*
error on representative biomolecular systems. It is not a bound and cannot be
turned into a constant. The mapping to our normalisation is
`Δ/σ_min = φ_max · (rc/σ)` (`rc/σ = 1.9893` for `b = 2`), so CubeS₂-6
`φ_max = 0.35` is `Δ/σ_min = 0.696`.

`C_assign` is derivable in principle (the `(2ν)`-th Taylor coefficient of the
symbol), but in practice both `C_assign` and `C_infl` are **measured** per
`(family, order, form, kernel)`, against a converged direct k-sum. §4.4 shows
why the kernel has to be in the key: two checkpoints of the same training
differ by ~3.7× in the force constant.

---

## 3. Verification method

### 3.1 Reference

`E = ½(1/V) Σ_k K|ρ̂|²`, `F_i = −(1/V) Σ_k K q_i k · Im[ρ̂* e^{−i k·r_i}]`,
with `K` the same kernel the operator builds. Cross-checked three ways:

* the force against a finite difference of the same energy (an apparent 13×
  discrepancy was a missing `COUL`, not an error: 6.59e-2/14.4 = 4.58e-3);
* a converged fine grid of the operator gives the same forces bit-for-bit, so
  the force error is not a reference floor;
* `E = ½ Σ q_i φ_i` holds to machine precision.

### 3.2 Self-consistency of the operator

`max|F + dE/dx| / max|F| ≈ 2e-3` for every backend and order — the
finite-difference noise. The operator's force **is** the gradient of its own
energy, which rules out a spread/gather inconsistency as the cause of the
order-6 behaviour.

### 3.3 Force estimator: AD/weights versus ik

Both production backends differentiate the **assignment weights**:

* the B-spline path computes only the energy in
  `deepmd/dpmodel/long_range/reciprocal.py::mesh_energy`; the coordinate force
  is the autograd derivative through the B-spline weights and the interpolation
  target;
* the midtown gather differentiates the CubeS₂ weights analytically
  (`cs2_wg6` in `source/op/pt/long_range/midtown/sog_gpu.cu`).

So the backend comparison is on equal footing. The paper's own audit tool
(`papers/sog-lib-paper/tools/mesh_validation.py`) instead uses
**ik-differentiation** (`ifftn(i k · G · R)`), a third estimator; that is why it
reports a larger Form-A force gain. ik-mode numbers are correct as ik-mode
numbers and do not describe the production force. The `p_force` of Eq. (2) is a
weight/AD-mode statement: in ik mode the force error tracks the potential error
and the exponent would be `2ν`.

---

## 4. Results — reference model `model.ckpt-150000.pt`

8 independent configurations, 64 atoms in a 12 Å cubic cell, RMS errors,
`σ_min = 2.0 Å`. `Δ/σ` = 0.650, 0.550, 0.450, 0.375, 0.300, 0.250.

### 4.1 Order 4

| Δ/σ | E Form A | E Form B | energy gain | F Form A | F Form B | force gain |
|---|---|---|---|---|---|---|
| 0.650 | **4.07e-3** | 4.04e-2 | 10× | 2.47e-2 | 2.48e-2 | 1.01× |
| 0.550 | **2.72e-4** | 1.74e-2 | 64× | 9.75e-3 | 1.06e-2 | 1.09× |
| 0.450 | **3.46e-4** | 7.07e-3 | 20× | 5.49e-3 | 5.59e-3 | 1.02× |
| 0.375 | **1.72e-4** | 5.47e-3 | 32× | 4.59e-3 | 4.77e-3 | 1.04× |
| 0.300 | **7.64e-5** | 2.26e-3 | 30× | 2.22e-3 | 2.29e-3 | 1.03× |
| 0.250 | **5.67e-5** | 1.08e-3 | 19× | 1.35e-3 | 1.38e-3 | 1.02× |

Fitted slopes 3.8 (A) / 3.6 (B) in energy, 2.8 / 2.8 in force.

### 4.2 Order 6

| Δ/σ | E Form A | E Form B | energy gain | F Form A | F Form B | force gain |
|---|---|---|---|---|---|---|
| 0.650 | 1.15e-3 | 1.20e-3 | 1.05× | 5.98e-3 | 6.02e-3 | 1.01× |
| 0.550 | 2.10e-4 | 2.29e-4 | 1.09× | 1.95e-3 | 1.96e-3 | 1.00× |
| 0.450 | 1.17e-4 | 1.21e-4 | 1.03× | 6.19e-4 | 6.22e-4 | 1.00× |
| 0.375 | 1.91e-5 | 1.77e-5 | 0.93× | 4.26e-4 | 4.28e-4 | 1.01× |
| 0.300 | 9.30e-6 | 9.21e-6 | 0.99× | 1.41e-4 | 1.41e-4 | 1.00× |
| 0.250 | 2.28e-6 | 1.81e-6 | 0.79× | 5.69e-5 | 5.70e-5 | 1.00× |

Fitted slopes 6.2 (A) / 6.5 (B) in energy, **4.6 / 4.7 in force** — the `C⁰`
prediction of Eq. (2).

### 4.3 What Form A buys, and what it does not

| | order 4 | order 6 |
|---|---|---|
| energy | **10–64×** | 0.79–1.09× (a wash) |
| force | **1.00–1.05×** | 1.00–1.01× |

Form A fixes the **energy**, not the force. That is the `C⁰` story again: the
force is limited by the discontinuity of the window derivative, which no
influence function can remove. At order 6 `C_infl` is only ~5e-3, so exact
inversion changes nothing this audit can resolve. The fitted force constants are
Form-A-invariant to within ~4% (`6.46e-2` vs `6.76e-2` at order 4, `3.52e-2` vs
`3.55e-2` at order 6), so the code defaults Form A **on at both orders**: order 6
pays a one-time `nth³`-transform cost for a strictly non-negative energy change
and no force regression, and a single default is simpler than two.

### 4.4 The constant is kernel-specific

Force constants (the force is Form-A-invariant, §4.3, so the form label is
immaterial), for the two checkpoints of one training run:

Constants evaluated **at the exponents the code ships** (`p = 3` at order 4,
`p = 5` at order 6), so they are directly comparable with §5.1:

| kernel | `σ_min` | `C_force` order 4 (`p=3`) | `C_force` order 6 (`p=5`) | measured slope, order 6 |
|---|---|---|---|---|
| `ckpt-200000` | 1.134 Å | 0.166 | 0.124 | 5.2 |
| `ckpt-150000` | 2.00 Å | 0.076 | 0.049 | 4.6 |

A factor ~2.2 at order 4 and ~2.5 at order 6, at fixed
`(family, order, form, metric)`. (The `ckpt-150000` row is the definitive
re-measurement of §4.6, which pins the shipped exponent and fits `C`; evaluated
at the free exponent instead it gives `6.46e-2` at `p = 2.82` and `3.52e-2` at
`p = 4.64` — the constant and the exponent are not separable, which is why both
are tabulated together.) The **shape** of the rule — order dependence, and
`p_force = 2ν − 1` for the `C⁰` window — survives; the **constant does not
transfer**. Any table of these constants must carry the kernel, and a caller
with a different kernel should re-derive it.

The constants shipped in the code (§5.1) are the **larger** of the two, i.e. the
conservative choice: they under-resolve a wide kernel by ~30 % in spacing rather
than over-resolve a narrow one.

### 4.5 What the older 4.8× figure was

On the earlier `ckpt-200000` kernel the energy error at order 4 was 4.65e-1
(Form B) against 1.87e-2 (Form A) at `Δ/σ = 0.706` — 25× — and against a
B-spline at the same grid the ordering inverted. The published 4.8× and the
"within a factor ~2" of the same document were both this residual, measured
with different functionals on different grids.

### 4.6 The B-spline comparison (force baseline, Form A default)

The definitive re-measurement of §4.1–4.3 (same 8 configurations, same kernel,
same grids) with the B-spline backend added, `SOG_MIDTOWN_FORM_A` default on at
both orders. Fitted exponents and constants, all backends:

| backend | order | `p_E` | `C_E` | `p_F` | `C_F` |
|---|---|---|---|---|---|
| midtown Form A | 4 | 3.77 | 7.64e-3 | 2.82 | 6.46e-2 |
| midtown Form A | 6 | 6.21 | 1.27e-2 | **4.64** | 3.52e-2 |
| midtown Form B | 4 | 3.61 | 1.64e-1 | 2.83 | 6.76e-2 |
| midtown Form B | 6 | 6.47 | 1.56e-2 | **4.65** | 3.55e-2 |
| bspline | 4 | 3.87 | 9.38e-2 | 3.06 | 4.52e-2 |
| bspline | 6 | 6.01 | 1.72e-2 | **5.46** | 1.13e-2 |

Two readings:

* **The force exponent is `2ν − 1 = 5`, not 6.** The order-6 midtown force fits
  `p_F = 4.64` (Form A) / `4.65` (Form B) — one full order below the energy's
  `6.21`/`6.47`. Order 4 fits `2.82`/`2.83` against energy `3.77`/`3.61`. This is
  the weight-differentiated force of a non-smooth window (§2.2), and it is why
  the shipped order-6 exponent is `5`, not `6`.
* **Form A leaves the force untouched.** The force fits agree to ~4% across forms
  (`C_F` 6.46e-2 vs 6.76e-2, 3.52e-2 vs 3.55e-2), while the energy moves by ~21×
  at order 4. The force-based grid rule (§5.1) is therefore unchanged by the
  Form-A default flip.

The same-mesh force comparison (midtown Form A against the B-spline):

| Δ/σ | order 4 midtown | order 4 bspline | ratio | order 6 midtown | order 6 bspline | ratio |
|---|---|---|---|---|---|---|
| 0.650 | 2.470e-2 | 1.342e-2 | 1.84× | 5.978e-3 | 1.279e-3 | 4.67× |
| 0.550 | 9.745e-3 | 6.398e-3 | 1.52× | 1.950e-3 | 3.535e-4 | 5.52× |
| 0.450 | 5.485e-3 | 3.720e-3 | 1.47× | 6.193e-4 | 1.313e-4 | 4.72× |
| 0.375 | 4.591e-3 | 2.383e-3 | 1.93× | 4.260e-4 | 5.856e-5 | 7.27× |
| 0.300 | 2.222e-3 | 1.134e-3 | 1.96× | 1.409e-4 | 1.583e-5 | 8.90× |
| 0.250 | 1.349e-3 | 6.465e-4 | 2.09× | 5.686e-5 | 5.826e-6 | 9.76× |

Geometric mean **1.80×** (order 4) and **6.81×** (order 6), growing toward fine
grids (2.09× / 9.76×) as the asymptotic regime is entered. The B-spline wins on
force purely because it is smoother: it keeps `p_F ≈ order` (5.46 / 3.06) while
CubeS₂ drops to `2ν − 1` (4.64 / 2.82). Form A has already removed `C_infl`, so
the gap is the window's non-smoothness and cannot be closed by any influence
function.

The *energy* comparison is the mirror image: on the same grids the B-spline
order-6 energy is a wash against midtown Form A (fitted `C_E` 1.72e-2 vs
1.27e-2, both `p ≈ 6`), and at order 4 the B-spline is ~12× **worse** (fitted
`C_E` 9.38e-2 vs 7.64e-3) — exact deconvolution makes CubeS₂ energy exceed the
B-spline's at order 4. The force gap is the price of the CubeS₂ window's
non-smoothness; it does not indicate a deconvolution defect.

---

## 5. Code

### 5.1 Grid rule (new, live)

`deepmd/dpmodel/long_range/grid_rule.py` — the measured law of Eq. (1) and (2):

```
FORCE_RULE[(backend, order)] = (C, p, provenance)
  bspline 4 : 0.117, p = 3.5 (measured)     bspline 7 : 0.020, p = 7 (extrap.)
  bspline 5 : 0.062, p = 4.5 (interpolated) bspline 8 : 0.012, p = 8 (extrap.)
  bspline 6 : 0.033, p = 6.0 (measured)
  midtown 4 : 0.166, p = 3 (measured)       midtown 6 : 0.124, p = 5 (measured)
spacing = min(σ_min·(ε/C)^{1/p},  0.75 π/k_max)
```

`C` is the larger of the two audited kernels and `p` is the measured slope
rounded **down**, so the rule can only over-resolve. `p` is taken from the
measurement rather than from `2ν` because both CubeS₂ orders come out about one
order low in the force (2.82 at order 4, 4.64 at order 6), while the B-spline
keeps `p ≈ order`. The constants are force-based and Form-A-invariant (§4.6), so
the Form-A default flip did not move them.

Wired into `deepmd/dpmodel/long_range/latent_charge.py`:

* `mesh_spacing()` solves the law instead of returning `0.75 π / k_max`;
* `effective_order()` — the midtown mesh has a CubeS₂ stencil of order 4 or 6
  only, so any other configured `pme_order` (including the 8 default) means 6.
  Both the spacing and the operator call go through it, which closes the trap
  where the grid was sized for a stencil the mesh does not evaluate;
* `pme_accuracy` (new field, default `None` → `1e-4`) sets the target; added to
  the schema in `deepmd/utils/argcheck.py`;
* the **B-spline order is now meaningful in both directions**: `C(order)` and
  `p(order)` make a low order refine the grid — on the reference kernel the
  B-spline at order 4 goes from 0.44 Å to 0.24 Å (513³), midtown order 4 to
  0.16 Å (766³) — and a high order coarsen it: B-spline order 8 takes 1.065 Å
  (115³) at the same `1e-4` target;
* the legacy `0.75 π/k_max` is **not** retained as a ceiling. The law is
  measured at the target spacings themselves, so the window's aliasing above
  `k_max` is already in `C`; adding the Nyquist cap back would neutralise
  exactly the high-order gain it was meant to protect. The consequence is a
  deliberate break: the default B-spline (order 8) grid is ~1.4× coarser than
  before, so results change with the model. `pme_spacing` pins the old grid
  explicitly, and `C(8)` is an extrapolation (orders 4 and 6 were measured).

### 5.2 Form A (new, default on at both orders)

| file | change |
|---|---|
| `source/op/pt/long_range/midtown/sog_gpu.cu` | `sog_gpu_green_form_a()`, `k_acc_spec_abs2`, `k_gp_divide_form_a`; the divisor is `D·⟨|W|²⟩`; the θ average is incoherent and the floor relative |
| `source/op/pt/long_range/midtown/sog_gpu.cuh` | declarations |
| `source/op/pt/long_range/midtown_sum.cu` | call site; `SOG_MIDTOWN_FORM_A` value-parsed, **unset = on at both orders** (`=0` disables); `form` recorded in `MidtownState` and in every cache key; loud refusal on the local-spread and slab paths; `nth` clamped ≥ 1 |

Cost: `nth³` transforms once per box (default `nth = 8`; ~26 ms at 98³,
cached by `box_tol`). Override with `SOG_MIDTOWN_FORM_A_NTH` /
`SOG_MIDTOWN_FORM_A_FLOOR`.

### 5.3 SOG library (`sog.cpp`, two copies kept identical)

`/data/home/public/jiangzhen/sog/lmp/sog.cpp` and
`/data/home/public/jiangzhen/dp/deepmd-kit-LR/source/lmp/sog.cpp`:
`apply_form_a_green()` behind `SOG_FORM_A`, **default on** (value-parsed, `=0`
disables); the grid constants updated to `C = 0.166 / p = 3` and `C = 0.124 / p = 5`;
a new **B-spline branch that reads `bspline_order`** (previously every order used
the CubeS₂ order-4 constants); QuadS left alone because it was never force-audited.

### 5.4 Paper

`papers/sog-lib-paper`: §4.3 rewritten to Eqs. (1)–(3) plus `eq:force-order`;
`tab:spline-slopes` and `tab:phimax` captions relabelled as Form-B effective
coefficients; `03-mesh-method` and `08-appendix` no longer call Form A
unsuitable for CubeS₂; abstract and conclusion carry the influence-function
axis; `tools/mesh_validation.py` gained a `cubeA` method (any non-QuadS `*A`
previously fell through to the B-spline sinc window).

---

## 6. Open

1. **Form A is replicated-path only.** Local spread and slab refuse it loudly.
   Tilted cells (`build_green_metric`) are excluded by the same guard.
2. **Constants are kernel-specific (§4.4)** and the shipped values are the
   conservative ones; a kernel much wider than the audited pair would be
   under-resolved by the accuracy target, and there is no Nyquist ceiling left
   to catch it — only the `pme_spacing` override.
3. `C(7)`, `C(8)` are extrapolated; only orders 4 and 6 were measured.
4. **QuadS was never force-audited**; its `C = 2e-4` is a self-described upper
   bound, so its exponent stays `2ν` even though QuadS order 6 is also `C⁰`.
5. Three hand-maintained copies of the grid rule (`sog_green.h` in two trees,
   `sog.cpp` in two trees), with no lockstep test.
6. `cubeA` in the paper tool is unstable on a few grids: the CubeS₂ window has
   exact spectral zeros and the relative floor drops different mode sets at
   different grids.
7. **CubeS₂ force stays ~2–7× behind the B-spline at equal grid (§4.6).** This
   is the window's smoothness, not the deconvolution; closing it needs a smoother
   CubeS₂ (a `C¹⁺` ξ class) or ik-differentiation, not a better influence
   function.

---

## 7. Reproducing

```bash
# operator
cd /data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt/build-pt-mpi-gcc
OMPI_CXX=/opt/rh/gcc-toolset-13/root/usr/bin/c++ make deepmd_op_pt -j8
cp op/pt/libdeepmd_op_pt.so ../deepmd/lib/

# grid rule, at the reference kernel
PYTHONPATH=.. python -c "from deepmd.dpmodel.long_range import grid_rule as g; \
  print([ (b,o,g.rule(b,o),g.spacing(b,o,2.0,5.35)) for b,o in \
  [('bspline',4),('bspline',8),('midtown',4),('midtown',6)] ])"

# ensemble audit (all three backends, force baseline — the numbers of §4.6)
export KA=... KW=...     # kernel of model.ckpt-150000.pt, see §4
python /tmp/lr_md/forma_v2.py --configs 8 --orders 4,6 \
    --spacings 1.3,1.1,0.9,0.75,0.6,0.5 --backends midtown_A,midtown_B,bspline
python /tmp/lr_md/forma_crosscheck.py
```
