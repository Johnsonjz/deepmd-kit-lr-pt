# Upstream `long_range` → pt backend: interface map and deviations

Internal note (plan W0 task 4). Baseline: `deepmodeling/deepmd-kit` master with
#6116 (`a14c52b97`). Branch `feat/long-range-pt`, created from upstream master
`0edeaef56` with no rebase of the legacy `feat/sezm-sog` line.

## 1. Module map

| Upstream (pt_expt) | pt backend | Notes |
|---|---|---|
| `deepmd/dpmodel/long_range/kernel.py` | same file, extended | `kspace_virial` added on `BaseInteractionKernel`, `LESKernel`, `SOGKernel` |
| `deepmd/dpmodel/long_range/reciprocal.py` | same file, extended | `kspace_energy` gains `want_potential` / `want_force` / `want_box`; `mesh_energy` gains `want_potential` |
| `deepmd/dpmodel/long_range/latent_charge.py` | same file, extended | `ellipsoid` / `metric-ellipsoid` modes, `bias_atom_q`, `sog_default_ladder`, `want_derivatives` |
| `deepmd/pt_expt/long_range/*` | `deepmd/pt/long_range/*` (new) | `BaseLongRange = make_base_long_range(torch.Tensor)`, kernels wrapped as `nn.Module`, `LatentChargeLongRange` subclasses the dpmodel class |
| `deepmd/dpmodel/fitting/latent_charge.py` | `deepmd/pt/model/task/latent_charge.py` (new) | pt class subclasses `InvarFitting`; registered as `GeneralFitting("latent_charge")` |
| `forward_common_lower_graph` (pt_expt) | `SeZMModel.core_compute` hooks | no pt graph lower; see §3 |
| `source/op/pt/long_range/*` | unchanged (upstream) | C++ side untouched; the `midtown` backend is not implemented (plan W4.2) |

## 2. Registration and capability

- `BaseDescriptor` (`deepmd/dpmodel/descriptor/make_base_descriptor.py`) gains
  `supports_long_range() -> bool`, default `False`. `DescrptDPA4`,
  `DescrptDPA4C` and pt `DescrptSeZM` return `True`.
- `check_long_range_support` reads the flag first and keeps the legacy
  `isinstance(DescrptDPA4C)` fallback for descriptors written before the flag.
- `argcheck.long_range_args(("pt", "pt_expt"))`; the `sezm` / `dpa4` model args
  attach it. `k_grid_mode ∈ {ragged, fixed, pme, ellipsoid, metric-ellipsoid}`.
- Legacy `sezm_sog` / `dpa4_sog` / `sog_sezm` / `dpa4c_sog` model types raise a
  `ValueError` pointing at `sog_work/tools/convert_sog_to_long_range.py`.

## 3. pt host

`SeZMModel.core_compute` runs `_apply_long_range_correction` *before*
`edge_energy_deriv`:

1. topology build:
   - `fixed` / `ellipsoid` / `metric-ellipsoid`: torch-only builder
     (`_build_long_range_topologies_pt`) so `use_compile` can trace it;
   - `pme` / `ragged`: `build_topologies` in eager;
2. `long_range.call_graph(descriptor, atype, node_mask, n_node, topologies,
   want_potential=True, want_derivatives=True)`;
3. `fit_ret["energy"] += e_lr_atom`, so the charge response rides the single
   edge backward;
4. `_apply_long_range_derivatives` adds the closed-form direct force and cell
   virial (k-sum modes) or falls back to autograd over the topology
   `coord` / `strain` leaves (mesh and pair modes).

`comm_dict` (multi-rank) raises `NotImplementedError`: deployment uses the
pt_expt `.pt2` path.

## 4. k grids

| mode | builder | cell | compile |
|---|---|---|---|
| `ragged` | `build_k_grid` | any | no (eager) |
| `fixed` | `build_fixed_k_grid` | any | yes |
| `pme` | `build_mesh_grid` | any | no (eager, logged warning) |
| `ellipsoid` | `build_ellipsoid_k_grid` | orthogonal only | yes |
| `metric-ellipsoid` | `build_metric_ellipsoid_k_grid` | any (triclinic) | yes |

`fixed` / `ellipsoid` / `metric-ellipsoid` are sized by
`LatentChargeLongRange.compute_input_stats` (largest sampled cell) or by an
explicit `k_grid_bounds`. A frame that outgrows the static grid logs a warning
instead of silently truncating.

## 5. Deviations from the plan

| Plan item | Actual |
|---|---|
| W1.0: `core_compute` already provides `box`, and `_apply_long_range_*` are existing insertion points | Baseline has no `box` in `core_compute` and neither hook exists; both were added in this branch |
| W1.4: legacy types are load-time aliases with `log.warning` | They are hard-rejected with a migration hint; the legacy classes do not exist in this tree, so an alias could not load them |
| W1.4: `dp --pt freeze` emits a pt_expt `.pt2` | `freeze_pt2.py` raises `NotImplementedError` for a long-range checkpoint; users go through `dp convert-backend` + `dp --pt-expt freeze` |
| W3.3: `RSP_USE_POTENTIAL=1` φ-VJP response path behind a switch | Descoped by decision 2026-11-07. The energy injection + edge backward is the same VJP and additionally carries the charge-constraint Jacobian; the unused `want_potential=True` call in the pt host was removed. The module still supports `want_potential` for callers that need φ (it is covered by tests) |
| W4.2: `mesh_backend="midtown"` | Single-rank orthogonal SOG path implemented (`long_range/midtown_sum.cu` + the copied `midtown/` device kernels). Triclinic cells, `midtown_dist` (per-rank slab) and the device-resident call are still open |
| D12: `sigma = nlayers * rcut / 1.989...` | Implemented as `sog_default_ladder` with `nlayers = n_blocks + 1` (the geometric embedding reaches `rcut` before the first block) |

## 6. Mesh backends

`latent_charge_sum(..., mesh_backend="bspline" | "midtown")`; the default keeps
the upstream SPME path and every existing caller and frozen graph unchanged.

`midtown` (`source/op/pt/long_range/midtown_sum.cu` + `midtown/`) evaluates a
periodic batch with the CubeS2 splines and cuFFT of the SOG library. The first
version is narrow and raises on everything it does not serve:

| dimension | supported |
|---|---|
| kernel | SOG only (the op's `a_k 2^{-c_k r^2}` maps to the normalized `amp (2πσ²)^{-3/2}` and `bw = σ²`) |
| channels | 1 |
| cell | orthogonal and triclinic (triclinic runs in fractional coordinates with the metric `k^2 = (2*pi)^2 m^T W m`, `W = A^{-1} A^{-T}`; the force rotates by `A^{-T}` and the volume is `|det A|`) |
| ranks | 1 (`midtown_dist` is not implemented) |
| spline order | 6 (88-node stencil, the default) or 4 (32-node stencil, selected with `pme_order: 4`) |

Measured at 192 atoms, 12 Å, SOG 3 bands: the backends agree to 3.6e-4
(force) at 0.8 Å spacing, 2.1e-5 at 0.5 Å, 4.1e-6 at 0.35 Å — they converge to
the same answer.

Cost per call (RTX 4090, 50 back-to-back calls, the steady state of a mesh step):

| backend | 1536 atoms, 0.5 Å | 1536, 0.25 Å | 12288 atoms, 0.5 Å | 12288, 0.25 Å |
|---|---|---|---|---|
| `bspline` | 0.100 ms | 0.117 ms | 0.191 ms | 0.474 ms |
| `midtown` (order 6) | 0.137 ms (1.4x) | 0.159 ms (1.4x) | 0.229 ms (1.2x) | 0.666 ms (1.4x) |

The gap was 4.0x at 12288 / 0.5 Å at the start of this round and 2.1x after the
first pass.  Order 4 still runs the unpacked spread, so it is now slower than
order 6 at 12288 (0.26 ms) and is not the recommended setting either way -- see
the accuracy note below.

### What the remaining 1.2x is, and where the wins came from

Entry-to-entry and per-stage CPU clocks inside the operator, `nsys` on the
kernels, and standalone micro-benchmarks (each reproducing the kernel's cost and
checked bit-for-bit against it) drove four changes; three of them landed.

**The host is not the bottleneck.** The operator's own CPU time is ~113 µs per
call (`pack` 17, `hostparse` 14, `solve` 53, `finish` 28), against ~150 µs for
the B-spline path.  What reads as host time is the launch queue -- the per-call
wall time tracks the grid size -- and the fixes were to remove every wait the
operator used to take: coefficients arrive on the host (no upload and readback
per call), the total charge is read in place on the device, the frame boundaries
of a single-frame batch are the batch size, and the virial comes from a
nine-thread kernel instead of `index_select` + `to(float)` + `copy_`.  Thirteen
ATen kernels per call became four.

**Spread: 113 -> 26 µs.** The atomics were never the problem.  The per-atom
weight table is, because the degree-5 CubeS2 weights are evaluated in fp64 --
their literals are `double`, so `(1.0/12.0)*t*t*t*t*t` promotes the whole
polynomial -- which is ~120 fp64 operations per atom on a card whose fp64 pipe
is 1/64 of fp32.  `k_spread_cubes2_warp_fx_packed` puts four atoms in a warp:
lanes 0..3 compute the positions, 24 lanes (six per atom) fill the four tables in
one body, one polynomial per lane, and the 88-node stencil is then scattered for
all four atoms, with a wrap-free interior path for atoms two cells from the
boundary.  Bit-identical to the old kernel.

**k-space: 70 -> 14 µs.** Also fp64-bound, for a different reason: three
correctly rounded divisions per mode (`kx = 2*pi*mx/lx`, with `lx` a runtime
value, so nvcc cannot reassociate) plus seven fp64 accumulations cost more than
all the memory traffic.  The reciprocal lengths come from the host, one thread
owns three adjacent kh columns of one y row and strides z, so kx, ky and the six
k products leave the inner loop and the 64-bit index multiply becomes pointer
increments; the seven sums accumulate in fp32 per mode and widen once per
thread.  The potential spectrum -- the only output the inverse FFT sees -- is
bit-identical; the reduced scalars move by ~5e-8 relative.

**Gather: 101 -> 86 µs.** The same three divisions, removed the same way.  The
line-vectorised gather that motivated this round was built, measured and
rejected: 0.91x, because with a physically meaningful potential mesh the 88
scattered reads are almost free (they only looked expensive against an all-zero
mesh, where the fp64 accumulation of zero operands changes the instruction
mix).  What actually costs in the gather is the fp64 reduction (13.6 µs, no
precision-neutral replacement) and those three divisions (15.4 µs).

### The separable Green function does not reach the real space

Section 3 of the SOG-library paper splits Form-B into rank-one terms,
`G_B(k) = sum_m a_m prod_d exp(-(sigma_m^2/2 - sigma_{s,d}^2) k_d^2)`.  The
B-spline's `build_green_range` in `midtown/sog_green.h` already does exactly
this: it evaluates `exp(-bandwidth_m k_d^2 / 2)` once per axis per band and
combines the three by multiply.  The paper is explicit that this separability
belongs to the *multiplier*, not to the CubeS2 window -- "the spreading window
remains non-separable" -- so it cannot make the spread or the gather cheaper,
which is where the remaining gap is.

### Slab (z-decomposed) solve: state

`midtown_sum_slab` (`SOG_MIDTOWN_SLAB=1`, default off) decomposes z across ranks and
runs a two-transpose 3-D FFT: brick spread -> NCCL reverse halo -> (x,y) FFT and
pack -> NCCL all-to-all -> z FFT, green, inverse -> NCCL accumulator reduce ->
inverse all-to-all -> inverse (x,y) FFT -> NCCL forward halo -> brick gather.
Memory is `1/np` of the replicated mesh and the density never crosses a rank.
The device-side API (`sog_gpu_setup_slab`, `sog_gpu_spread_brick_dev`,
`sog_gpu_slab_xyfft_pack_brick_dev`, `sog_gpu_slab_unpack_green_izpack_dev`,
`sog_gpu_slab_unpack_ixyfft_to_brick_dev`, `sog_gpu_nccl_*`) was already ported from
the SOG library and is proven; only the driver is new.  The reference driver is
`/data/home/public/jiangzhen/sog/op/sog_kspace/sog_kspace_op.cc`.

Two environment traps, both hit and fixed here:

- `ncclCommInitRank` segfaults on this machine because NCCL picks the real NIC for
  bootstrap.  Single-node runs need `NCCL_SOCKET_IFNAME=lo` and `NCCL_IB_DISABLE=1`;
  `ensure_nccl` sets both unless the caller already did.
- `at::from_blob(host_ptr, {n}, coord.options().dtype(kDouble))` builds a CUDA tensor
  around a host pointer (`pointer resides on host memory`).  Use plain `at::kDouble`
  and `.to(device)`.

**Status: correct at np = 1, 2, 4.**  It used to be wrong by a factor ~1400 in the energy
(E -743.7 against -0.517, virial -1858 against -45.73, max|dF| 15.8) and the guess was a
missing normalisation.  It was not: the driver built its green tables with
`sog_green::build_green` (full `nx*ny*nz`, x-stride `nx`) and sliced them with an x-stride
of `nx`, while `k_green_pencil` reads them as the **compact half-spectrum**,
`gi = kh + Nh*(iyl + YL*iz)` with `Nh = nx/2+1`.  Every mode therefore read the green value
of a different `(kh, iy, iz)` cell.  `gn = nx*YL*nz` is about twice the compact size, so the
kernel only ever read inside the (wrongly laid out) buffer -- no OOB, no NaN, no error: the
"plausible but wrong, no fingerprint" failure the reference's own comment warns about.  The
reference uses `build_green_slab_half` with `gn = (nx/2+1)*YL*nz`; the port now does the same.

That the energy and the virial were off by *different* factors (1438x and 40.6x) is the
fingerprint of a per-mode corruption rather than a scalar: the frame energy is
`0.5*Σ q_i φ_i` from the gathered potential, the virial is the mode sum `red[1..6]` from the
same spectrum.  The energy is also a near-total cancellation -- with `kernel_zero = Σ a_m`
the self term is `0.5*prefactor*kernel_zero*Σq²` (1903 eV for the 400-atom test) and the
answer is +0.72 eV, so a 3.8e-4 relative error in the reciprocal sum is amplified by ~2600
into the frame energy.  Compare energies at the atom level, or on a system that does not
cancel, or you will chase precision forever.

Three more defects were found and fixed while validating:

- The green/state cache is one process-wide `MidtownState` shared by the replicated, local
  and slab paths.  Nothing recorded *which* device layout built it, so the second path in a
  process reused the first one's tables -- and the reverse order left the slab scratch
  allocated below what the replicated path indexes.  The cache now keys on a mode flag plus
  the slab's `(ylo, YL)`.
- `free_grid_buffers` released only the fp64 set, so every rebuild (a grid change under NPT,
  and every slab<->replicated switch) leaked the whole fp32 mesh set -- 48 B per grid point
  plus three cuFFT plans, 162 MB per switch at 150^3.  It now frees it and `sog_gpu_destroy`
  routes through it.
- The reference's `TORCH_CHECK(ZL >= gh_hi)` and its per-rank z-subdomain check were
  missing.  Without the second one a rank that owns atoms anywhere in the box spreads them
  onto its own slab, the out-of-range nodes are silently dropped by the spread's
  `if(gx>=0&&gy>=0&&gz>=0)` guard, and the solve returns a plausible wrong number -- this is
  what the first np=2 run did.  Both checks are back, with the reference's message.

One more, found only because the memory change above moved the allocation layout and it then
faulted: in `k_spread_cubes2_warp_fx_packed` the padding lanes of a partly filled pack
(`i >= nlocal`) were marked **interior** (`sh_intr = 1`) while sitting at `(0,0,0)`.  The
interior path indexes `base + sh_off[k]` with no wrap, and at `(0,0,0)` that is `rho[-2]` and
`rho[-2 - nx]`, i.e. *before* the mesh.  The value added is zero (`qs == 0`), so no result was
ever wrong, but the address was out of bounds and had been landing in whatever allocation
happened to precede the mesh.  Padding lanes now take the wrapping path; compute-sanitizer
reports 0 errors on the replicated <-> slab alternation.

Validation (`/tmp/lr_md/slab_parity.py`, atoms assigned to the continuous z-subdomain that
`processors 1 1 np` gives):

| system | np | dE vs np=1 | max dF | dE vs replicated np=1 | dVir rel | max dF |
|---|---|---|---|---|---|---|
| 18³ Å cubic, order 6 | 2 / 4 | 6e-15 / 1e-14 | **0** | 2.3e-4 (3.2e-4 rel) | 4.9e-6 | 6.3e-6 |
| 18x22x14 Å, order 6 | 2 / 4 | 3.6e-15 / 0 | **0** | 3.1e-4 | 1.4e-4 abs | 9.5e-6 |
| 18³ Å cubic, order 4 | 2 | 0 | **0** | 2.1e-4 | 2.9e-4 abs | 2.6e-5 |
| nz=40, np=3 (uneven 13/13/14) | 3 | 4.4e-15 | **0** | 3.0e-5 | 9.8e-5 abs | 6.0e-6 |

The slab path is bit-reproducible across runs and exactly equal across ranks (forces are
*identical*, not merely close).  The residual against the replicated path is that path's
fp32 mesh and half-spectrum -- 1e-7 relative on forces.

Cost, 12288 atoms at 48.54 Å spacing 0.5 (one rank per GPU, RTX 4090).  The three
backends on the *same* harness, `mesh_backend=bspline` / `midtown` / `midtown`+slab;
midtown sizes to 100³, the B-spline to 98³ (its factor set admits 7, midtown's does not):

| backend | np=1 | np=2 | np=4 | device MiB per rank (np=1/2/4) |
|---|---|---|---|---|
| `bspline` | **0.190 ms** | 0.545 | 0.606 | **30** / 30 / 30 |
| `midtown` replicated | 0.227 | **0.444** | **0.513** | 68 / 68 / 68 |
| `midtown` slab | 1.181 | 1.568 | 1.224 | 62 / 190 / 178 |

Three things follow, and only one of them is good news for the slab.

**The slab's memory advantage does not survive NCCL.**  Its mesh buffers really do shrink
(27.3 / 13.6 / 6.8 MB against 99.2 MB), but a 25³ test grid isolates the rest: np=1 slab is
6 MiB, np=2 is 160 MiB.  `ncclCommInitRank` itself accounts for 62 MiB (measured with
`SOG_DIST_DEBUG=1`), the first collective adds ~84 more, and none of `NCCL_BUFFSIZE`,
`NCCL_MAX_NCHANNELS`, `NCCL_P2P_DISABLE` or `NCCL_NET` moves it more than 6 MiB.  So the
slab costs ~146 MiB flat per rank and only pays once the per-rank replicated mesh exceeds
that -- roughly a 120³ grid at np=4.  Below that it uses *more* memory than the backend it
replaces.

**The time split is structural, not a code defect.**  Stage timings (`SOG_SLAB_TIME=1`,
steady state, np=4): spread+reverse halo 0.345 ms, the four NCCL collectives 0.345 ms
(forward all-to-all 0.121, inverse 0.123, forward halo 0.088, accumulator reduce 0.013),
gather 0.108, z FFT + green 0.061, the two (x,y) FFT stages 0.080.  One collective costs
~0.12 ms on this box, and a three-dimensional transpose needs them.  The replicated path
needs none, which is why it cannot be beaten by a decomposition at these sizes.

**Two wins did come out of the measurement, both in the deployed path.**  The replicated
midtown solve reads only the single-precision meshes, so the fp64 set
(`d_rho/d_pot/d_ge/gf/gv`, 56 B per grid point) was pure dead weight: `sog_gpu_setup_f32`
skips it and takes the replicated path from **128 to 68 MiB** (196³: 400 MB saved per rank).
And the z-subdomain guard was copying three arrays to the host and looping over every atom
per call; it is now a device-side min/max, checked once per `(np, nz)`.

The np=1 halo folds and the forward-halo copies stay on the device
(`sog_gpu_brick_add_planes` / `_copy_planes`); routing them through host
`brick_get`/`brick_add` cost ~24 synchronising bus round trips per call and was 1.9 ms of the
original 2.2 ms.

NCCL is now needed only for np>1: at np=1 both all-to-alls are the identity permutation
(`sog_gpu_slab_copy_packed`) and the accumulator is already complete, so the backend runs
without `libnccl` -- and does not depend on a rank sending to itself, a path with no
reference counterpart.

Remaining known limitation: the driver has the full-brick geometry (`ZL == nz`) but no
full-brick branch, so a decomposition that is not 1x1xnp must be refused (it is now, by the
z-subdomain check).  The reference's full-brick path reduce-scatters the density and
all-gathers the potential; that is not ported.

### Multi-rank: what is distributed, and what that buys

Two paths exist across `np > 1`:

- **replicated** (the default): every rank gathers the whole frame and solves the
  whole mesh, then returns its owner slice.  Mesh buffers are 40 B per grid
  point per rank (`f_rho` 8, `f_gp` 8, `f_rho_r` 4, `f_rho_fx` 8, `f_pot_r` 4,
  `f_spec` 8), i.e. 40 MB per rank at 100^3 and 300 MB at 196^3.
- **local spread** (`SOG_MIDTOWN_DIST=1`, `midtown_sum_local`): each rank spreads
  only its own atoms, the densities are summed, every rank transforms the whole
  mesh, and each gathers its own atoms.

The local path used to be wrong by 5.6 eV.  The cause was three frame-wide
quantities computed from one rank's slice: the uniform charge constraint
(`csums`), the ten frame sums of `k_midtown_finish1`, and the charged-node
count.  All three are now reduced across ranks.  `red[0..6]` from the k-space
pass must NOT be reduced: it is a sum over modes of the *already summed*
density, so summing it again doubles the virial (measured: exactly 2x at np=2).
After the fix, np=2 agrees with the replicated path to max|dF| 6e-6 and
max|dE| 2e-6 (2.9e-7 relative), from 1.36 and 0.68.

It is still not the default, for a measured reason:

| np=2, 12288 atoms, 0.5 A | ms/call |
|---|---|
| replicated | 0.93 (np=1: 0.19) |
| local spread, before | 28.5 |
| + float32 device-side density reduction | 21.8 |
| + green-table cache (it was rebuilt every call) | 4.69 |

The remaining 4.7 ms is dominated by one full-grid reduction (3.3 ms measured).
`MPIX_Query_cuda_support()` returns 0 for this OpenMPI build, so every exchange
stages through pinned host memory: the density costs device-to-host, MPI, and
host-to-device of 4 MB each, to save about 0.5 ms of atom work.  Distribution
of the atom work therefore **cannot pay at these sizes on this MPI**.

What it can still buy is memory, and only with a slab decomposition: each rank
holds `np^-1` of the mesh (plus two pencil buffers) instead of all of it, and
the density never crosses ranks at all.  The transposes would have to use NCCL
(the SOG library's own multi-GPU path does exactly that, and it is device-side,
unlike this MPI).  That port is the next step.

### Accuracy: Form B threw away the midtown spline's advantage

Two error mechanisms sit behind the same exponent. The assignment window itself
contributes an interpolation/aliasing error of order `2*nu` in the spacing; the
*influence function* (the deconvolution of the window) contributes a second
`O(Delta^{2*nu})` term that is identically zero when the window symbol is
inverted exactly, as the B-spline path does:

    deepmd/dpmodel/long_range/reciprocal.py:_assignment_factor builds
    |D_d(m)|^-2 from the EXACT analytic Fourier transform of the B-spline.

The midtown backend used "Form B", `ge = kfac * exp(xi^2 Delta^2 k^2)`, an
analytic variance subtraction that only approximates `1/|W|^2`.  Its residual
dominated the error budget: measured on the deployed operator (64 charges, 12 A,
the shipped SOG kernel, relative energy error against a converged direct k-sum):

| Delta/sigma_min | bspline-4 | midtown-4 Form B | midtown-4 **Form A** |
|---|---|---|---|
| 0.625 | 6.18e-2 | 1.19e-1 | **1.80e-3** |
| 0.500 | 2.02e-2 | 5.16e-2 | **4.85e-3** |
| 0.417 | 1.03e-2 | 2.22e-2 | **2.88e-5** |
| 0.375 | 5.27e-3 | 1.53e-2 | **8.02e-4** |
| 0.312 | 2.61e-3 | 7.36e-3 | **4.41e-4** |

Fitting `C (Delta/sigma_min)^(2 nu)` over the range: `C(Form B) = 0.77` and
`C(bspline-4) = 0.27`, both stable to a few percent across the sweep.  **Form A
does not yield a usable constant from this audit**: dividing its column by
`(Delta/sigma)^4` gives 1.2e-2, 7.8e-2, 9.5e-4, 4.1e-2, 4.7e-2 at
Delta/sigma = 0.625, 0.500, 0.417, 0.375, 0.312 -- an 82x scatter with no trend,
because what is left is at the resolution of this estimator (a 64-atom neutral
cell, one configuration, 8^3 sub-cell samples, and grid-quantised spacings that
produce accidental cancellations at some grids).

**And the size of the Form-A gain depends on the metric.** These are *energy*
errors: at order 4 Form A lowers them by 11-770x. In the canonical
**force-relative** metric the honest numbers are much smaller and kernel
dependent -- the paper's own audit tool (`tools/mesh_validation.py`, `cubeA`
added, 8^3 sub-cell samples, four kernels) gives **2.5-5x at order 4 and ~1.1x
at order 6**, while the deployed operator on the trained water kernel shows
essentially no force-metric difference at either order. The energy and force
weights the influence residual differently, so a gain quoted without its metric
is not usable.
To put `C_assign` in a table it has to be measured in the canonical
force-relative metric on an ensemble of configurations, which this audit is not.

The earlier "4.8x worse than the B-spline" was this residual, and the earlier
`h^5.5` "slope" was a clean `h^4` term added to a shallower `O(Delta^4)`
residual, not a loss of order.

At order 6 the residual is much smaller (`C_infl` falls from ~5.6e-2 to
~5.1e-3), so Form A and Form B are within 1.0-1.8x of each other and the
benchmark cannot separate them at these grids.

`C` is kernel-dependent for Form B (0.78 for the 3-band test kernel vs 1.06 for
a single band of sigma = 1.2 A, a 36% spread -- the same spread the SOG library
paper reports), and nearly kernel-independent for Form A.  Any table of these
constants must therefore carry the kernel and the metric, not just a family.

#### The force exponent is not 2*nu: a C0 window loses one order

Measured against a converged direct k-sum (analytic force, trained water kernel,
sigma_min = 1.134 A, 64 atoms, 12 A), the FORCE error converges at

| family | measured slope | fitted `C` |
|---|---|---|
| bspline-4 | 3.5 | `C(p=4) = 0.18` (spread 0.79-1.24) |
| bspline-6 | 6.3 | `C(p=6) = 0.033` (0.95-1.43) |
| midtown-4 | 3.2 | `C(p=4) = 0.26` (0.85-2.33) |
| **midtown-6** | **5.3** | **`C(p=5) = 0.098` (0.79-1.09)** |

CubeS2 order 6 loses one order in the force, and the midtown paper says why:
its Table I lists CubeS2 order 4 as **C2** and order 6 as **C0**. The force needs
the *derivative* of the assignment window, so a merely continuous window cannot
deliver `2*nu`. The rule for the force is therefore

    p_force = 2*nu        if the window is at least C1
    p_force = 2*nu - 1    for a C0 window (CubeS2 order 6)

This is a property of the spline, not a defect: the energy still converges at
~2*nu (measured 6.0 for order 6), and force/energy remain consistent to the
finite-difference noise (2e-3 relative in `max|F + dE/dx| / max|F|` for every
backend and order).

The grid rule had `p = 6` with `C = 2.4e-2` for CubeS2-6, i.e. it under-resolved
order 6 by 1.6x in the spacing (0.40 vs 0.25 at the 1e-4 target).  Corrected in
`sog_green.h` (`grid_axis`) and in both `sog.cpp` copies: `C = 9.8e-2, p = 5`
for order 6 and `C = 2.6e-1, p = 4` for order 4, all force-relative against a
direct k-sum.  The B-spline branch now reads `bspline_order` (it previously sized
every order with the CubeS2 order-4 constants): `p = bspline_order`,
`C = 0.18 / 0.077 / 0.033` for orders 4 / 5 (log-interpolated) / 6.  QuadS is
left alone -- it was never force-audited, and moving its exponent on an
unmeasured constant would re-grid every QuadS run.

Effect on the production chain (DPA4C-SOG, 192 atoms): the 12.4 A box goes from
27^3 to 41^3 at the default target.

#### Form A

`SOG_MIDTOWN_FORM_A=1` (default off) replaces the variance subtraction by the
measured window symbol.  The symbol is theta-dependent for a non-separable
window, and what one global deconvolution can undo is the *incoherent* average
`<|W|^2>` over sub-cell positions (a coherent average cancels -- it is
meaningless here).  The operator builds it on the device: spread one unit
charge at `nth^3` sub-cell positions with the same `cubes2_weight_*` the host
code uses, transform with the existing R2C plan, accumulate `|W|^2`, then divide
the packed green table.  Two details are load-bearing:

- The divisor is `D * <|W|^2>`, not `<|W|^2>`.  The incoming table is Form B and
  already carries `D = exp(xi^2 Delta^2 k^2)`; dividing by `<|W|^2>` alone
  applies the deconvolution twice (symptom: the reciprocal energy came out 1.7x
  too large and the total energy changed sign).
- The near-zero floor is *relative* (`<|W|^2> <= 1` by construction).  Modes
  below it are dropped rather than amplified, exactly as the B-spline path drops
  its Nyquist planes.

`nth` defaults to 8 (512 transforms, ~26 ms at 98^3, once per box, cached by
`box_tol`).  It is the resolution limit of the measurement: `nth = 2/4/8` gave
1.8e-2 / 5.5e-3 / 1.8e-3 at `Delta/sigma = 0.625`.

`sog/lmp/sog.cpp` and `dp/deepmd-kit-LR/source/lmp/sog.cpp` carry the same
`apply_form_a_green()` behind `SOG_FORM_A` (CPU, `nth` defaults to 3).  Measured
on the DPA4C-SOG production chain at a 9^3 grid: Form A moves the energy by
+0.009 eV and the pressure by -15 bar at order 6, the size the operator audit
predicts for that grid.

Not covered yet: tilted cells (`build_green_metric`) and the slab path still run
Form B, and Form A is off by default.

One navigation-level finding does generalise: `_even_fft_size` in
`deepmd/dpmodel/utils/neighbor_graph/reciprocal_graph.py` admits prime factor 7,
so L=48.54 at 0.5 Å sizes to 98 = 2*7^2, and the radix-7 cuFFT plan costs 12.5%
of the whole B-spline call (98^3 0.178 ms vs 100^3 0.156 ms at 12288 atoms).
Restricting the factor set to {2,3,5} -- which is what the midtown path already
does -- would speed the *B-spline* path up by that much.  Not changed here,
because it moves every production mesh.

The path since the first working version (8.4 ms at 1536 atoms): device-resident
charges and outputs, deferred reductions, one host transfer per call,
warp-cooperative spread and gather, CubeS2 weights from a per-atom 1-D table,
float32 meshes with a single-precision half-spectrum R2C/C2R transform, a
packed (energy, virial) influence table, fixed-point integer atomics for the
spread, and the per-atom assembly and charge preparation folded into CUDA
kernels.  The remaining gap to the B-spline mesh is on the device, in the
real-space pair (spread and gather), not in the host plumbing: the CubeS2
stencil writes 88 non-separable nodes per atom with 64-bit atomics, where the
B-spline spread is separable and writes 216 nodes without contention.

The replicated solve now runs the mesh, the spectrum and the green tables in
**float32** (`k_kspace_f`, `cufftComplex`, a CUFFT_C2C single-precision plan)
with every reduction and the charge/energy arithmetic still in float64, and
evaluates the CubeS2 weight algebra in float.  On these cards -- all RTX 4090,
whose FP64 pipe is 1/64 of FP32 -- that was the dominant cost: the two FFTs
went from 0.34 ms to 0.05 ms at a 100³ grid, and the gather from 0.59 ms to
0.10 ms at 12288 atoms.

`midtown` started at 8.4 ms (0.5 Å).  The wins so far: device-resident charges
and outputs (no host round trip), deferred reductions (the 20 sums stay on the
device), one host transfer per call, warp-cooperative spread and gather (one
warp per atom instead of one thread), no dead gradient meshes, and no
unconditional debug-event synchronization.  The per-atom assembly likewise runs in two kernels (`k_midtown_finish1/2`)
instead of a dozen tensor ops, so no scalar crosses back to the host between
the solve and the outputs.

The hardware makes the next step unambiguous: every GPU here is an RTX 4090,
whose FP64 throughput is 1/64 of FP32.  The midtown pipeline is double
throughout -- the CubeS2 weight evaluation, the cuFFT transforms and the green
tables -- while `mesh_sum.cu` keeps the mesh, spectrum and influence in float32
and accumulates the reductions in float64.  Porting that split (plus the
half-spectrum R2C/C2R layout, which halves both the transform and the mesh
footprint) is the change that closes the gap; the physics stays float64 where
it matters (charges, energies, reductions).

The grid follows the operator contract — `n_d = ceil(|a_d| / spacing)` for
both backends — so one call means one grid whichever mesh evaluates it. The
midtown accuracy rule (1e-4 force-relative) only sizes the grid when a caller
passes no spacing.

Accuracy against a converged direct k-sum (192 atoms, 12 Å, 3 SOG bands,
`max|phi - phi_exact|` relative):

| backend | grid | error |
|---|---|---|
| `bspline` 0.5 Å | 48³ | 1.4e-3 |
| `bspline` 0.25 Å | 96³ | 8.1e-6 |
| `bspline` 0.15 Å | 160³ | 8.4e-7 |
| `midtown` 0.5 Å | 48³ | ~2e-3 (49³ estimate) |
| `midtown` 0.15 Å | 160³ | 1.4e-6 |

Read this table together with the section above: it is a *potential* metric at a
single configuration, and the two backends are within a factor ~2 there, whereas
the *energy* metric at the same grid gave 2.9x for Form B and better than the
B-spline for Form A. Different functionals weight the deconvolution residual
differently -- quote the metric with the number. CubeS₂ carries 88 support nodes
against the B-spline's 216, which is the advantage the midtown paper reports
(spreading cost, not grid points).

## 6.1 Selecting the backend

`model.long_range.mesh_backend` (`"bspline"` by default, or `"midtown"`) is
carried through the pt/pt_expt module, the operator schema (`int
mesh_backend`, 0/1) and the compressed canonical operator
(`dpa4c_canonical_compress_long_range_energy_force`), so a frozen `.pt2`
runs the midtown mesh in LAMMPS.  The midtown backend evaluates the CubeS2 stencil of
order 6, or order 4 when `pme_order: 4` is set (every other value, including the
default 8, maps to 6 -- see `deepmd/pt_expt/long_range/latent_charge.py`), and
one channel.

> A string argument to a custom operator does not survive the AOTInductor
> freeze, and the default backend is left implicit so a frozen graph keeps the
> argument count it has always had.

## 6.2 Distributed solve (experimental)

`SOG_MIDTOWN_DIST=1` selects a distributed solve: every rank spreads only its
own atoms into the full mesh, the meshes are summed over the ranks
(`sum_over_domains`), and each rank then transforms and gathers its own atoms.
The mesh is still replicated, so this distributes the particle work, not the
memory.  It is now numerically validated against the replicated solve (np = 2
agrees to max|dF| 6e-6, max|dE| 2e-6, from errors of 1.36 and 5.6 eV before the
cross-rank reductions were fixed), but it is off by default for a measured
reason: see the multi-rank table above -- 4.69 ms against 0.93 ms at np = 2,
because this OpenMPI is not CUDA-aware and every exchange stages through the
host.

The z-slab transform of `sog_kspace_op.cc`, which also cuts the per-rank mesh
to 1/np, IS now ported as `SOG_MIDTOWN_SLAB=1`; see "Slab (z-decomposed)
solve: state" for its validation and why it buys memory rather than time.

## 7. Multi-rank (`np = 1/2/4`)

LAMMPS `pair_style deepmd` with the compressed DPA4C canonical `.pt2`,
192 atoms, `run 0` (Intel MPI, `I_MPI_FABRICS=shm`):

| np | E / eV | dE / eV | P / bar | dP / bar | max abs dF |
|---|---|---|---|---|---|
| 1 | −29930.504892431563 | — | −11238.400197 | — | — |
| 2 | −29930.504892571436 | 1.4e-7 | −11238.399404 | 7.9e-4 | 1.0e-5 |
| 4 | −29930.504898145984 | 5.7e-6 | −11238.402040 | 1.8e-3 | 1.0e-5 |

The same scan with `mesh_backend="midtown"`: E = −29930.497081180554 /
−29930.497082134651 / −29930.497087312364 eV at np = 1/2/4 (dE 9.5e-7 and
6.1e-6, i.e. 3e-11 and 2e-10 relative), P = −11259.184210 / −11259.179850 /
−11259.183276 bar, and `max|dF| = 1.0e-5` (the dump print quantum) against
np=1.  The midtown backend therefore reproduces itself across ranks in the
deployed compressed-canonical path, where each rank solves the complete frame
and returns its owner slice.

Backend-to-backend agreement in the same path: at the default mesh spacing
the forces differ by 2.4e-3 rms (7.5e-4 relative), and at a fixed
`pme_spacing = 0.2` they converge to **dE = 4.3e-6 eV, dP = 2.3e-5 bar,
max|dF| = 2.0e-5 (rms 6.1e-6)** -- the default-spacing gap is resolution, not
bias.

Energy agrees to 4.7e-12 (np=2) / 1.9e-10 (np=4) relative; forces to the print
precision of the dump. The residual is the float32 accumulation order of the
descriptor under domain decomposition, not the long-range sum (whose own sums
are float64).

## 8. Tests

- `source/tests/pt/model/test_sezm_long_range.py` — pt host: build/serialize,
  periodic and non-periodic finite-difference forces, virial, φ = ∂E/∂q,
  ellipsoid/metric-ellipsoid parity, compile, data-stat sizing, warnings.
- `source/tests/common/dpmodel/test_long_range.py` — kernels, grids, φ and the
  packed-slot reduction.
- `source/tests/pt_expt/model/test_long_range.py` — upstream suite; 6 `.pt2`
  cases need a `gcc/13` libstdc++ (`GLIBCXX_3.4.30`).
