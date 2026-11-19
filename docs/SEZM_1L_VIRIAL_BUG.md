# SeZM 1L canonical virial bug

## Symptom

`dpa4-sog-1L` canonical with-comm artifact (`c560-1l.pt2`) reproduces the
single-rank energy and forces but not the pressure:

| rank | energy (eV) | press (atm) |
|---|---|---|
| 1 | -29946.665 | -3709.59 |
| 2 | -29946.665 | -4171.79 |
| 4 | -29946.665 | -5096.20 |

Forces agree with single rank at `max|dF| = 1.0e-5 eV/A`.
The 2L model (`c560-canon.pt2`) has stable pressure (~-3784.73) at 1/2/4.

The difference is linear in rank count:

```
NP2 - NP1 = -462.2 atm
NP4 - NP1 = -1386.6 atm ~ 3 x -462.2
```

## Root cause

The operator contract (`source/op/pt/long_range/ops.cc`) makes the frame
virial a **per-rank share**:

> The outputs of a rank are those of its charged nodes, and W_f is their share
> of the virial of the frame.

`mesh_sum.cu` (the `bspline` backend) honours it: a rank returns

```
share = sum_owned F(x)r + n_owned * (W - sum_all F(x)r) / N,
```

so the ranks add up to `W = -dE/d(strain)`. `direct_sum.cu` sums the per-node
virials of its owners.

`midtown_sum.cu` (the `midtown` backend) did not. After evaluating the
replicated mesh it wrote the frame virial of the whole frame into every rank's
output (`k_midtown_virial`), so each rank contributed `W`, and the caller's
sum over the ranks (LAMMPS sums the rank virials in the thermo) counted the
long-range virial `nranks` times:

```
drift = (nranks - 1) * W_lr   ->   (nranks - 1) * P_lr
```

which is exactly the observed constant step of -462.2 atm per added rank.

The earlier reading of this defect in this file attributed the drift to the
`rest` of `W` alone, added once per rank. The whole `W` (the `sum F(x)r` part
as well as the rest) was replicated; `atom_virial`, on the other hand, was
already a correct per-node share that sums to `W` over the ranks.

Why the 2L model looked fine: its artifact never ran this backend.
`model/extra/model_def_script.json` of `c560-canon.pt2` has
`long_range.mesh_backend = "bspline"` (`pme_order: null`), while `c560-1l.pt2`
has `"midtown"`, `pme_order: 6`. The bug is backend-specific, not
`n_blocks`-specific.

## What was ruled out

- SR-only 1L artifact (`c560-1l-sr.pt2`) has stable pressure at 1/2/4
  (-34282.29): short-range descriptor, border exchange and short virial are
  correct.
- `pair deepmd` and `pair deepmd/kk` show the same error: not a Kokkos issue
  (both drive the same `latent_charge_sum` operator).
- Removing the with-comm sibling is invalid: energy/pressure become wrong
  (-30089.2 / 261138.6).
- Keeping only rank 0's virial does not fix it (NP2 -3735.0, NP4 -2833.3):
  rank 0 is not the sole owner of the global virial, and the share of a rank
  is not the frame virial.

## Formula (SOG paper, Sec. 2.2, Eq. 6)

```
Xi_lr = sum_j F_lr,j (x) r_j + Xi_rec
Xi_rec = -(dE_lr/dh) h^T   at fixed positions and charges
```

`F_lr,j` is the **total** long-range force, including the charge-response
force. The response term `sum_j F_resp,j (x) r_j` is already inside the
particle part; no separate response correction is added. In the implementation
the response force reaches force and virial through the autograd term
`g_lr . (g - detach(g))` the atomic model adds to the energy, so the operator
only adds the fixed-charge moment `sum F_fixed(x)r` (a per-node quantity) and
`Xi_rec` (a per-frame quantity, one share per rank).

## Fix

`source/op/pt/long_range/midtown_sum.cu`: a new kernel
`k_midtown_virial_share` writes, for a frame,

```
sum_owned F(x)r + n_owned * (W - sum_all F(x)r) / N,
```

from the per-node moments `moments`, the frame sums `sums` and the charged-node
count the solve already produces, and the six k-space sums `red`. It is used
by all three midtown paths:

| path | change |
|---|---|
| replicated (`midtown_sum`, the default) | `W` itself when the call owns every node (single domain, unchanged bit for bit), else the owner share through `owner_begin`/`owner_count` |
| local spread (`SOG_MIDTOWN_DIST=1`, `midtown_sum_local`) | local moments plus `n_local * (W - sum_all)/N`, `sums` and the count already all-reduced |
| slab (`SOG_MIDTOWN_SLAB=1`, `midtown_sum_slab`) | same, from the slab's NCCL-reduced k-space sums |

The single-domain result is unchanged, so the midtown/bspline agreement and
the strain-derivative test of the `bspline` backend keep passing.

## Verification

### Operator level, two MPI ranks

New test `source/tests/pt_expt/kernels/test_long_range_multi_rank.py`, run as

```bash
mpirun -np 2 python -m pytest \
  source/tests/pt_expt/kernels/test_long_range_multi_rank.py
```

Each rank owns a strided slice of one 192-atom periodic frame and passes the
address of `MPI_COMM_WORLD`; the test checks that the per-node outputs of a
rank equal the single-domain evaluation on its nodes, and that the sum of the
`virial` shares over the ranks equals the single-domain frame virial.

With the unfixed `midtown` kernel the test fails exactly as the symptom says:

```
AssertionError: midtown: the shares of 2 ranks sum to 27.728397 for component
0 of the frame virial 13.864199          # 2 x W at np=2
```

while `bspline` passes; with the fix both backends pass:

```
backend=midtown ranks=2 max|dE|=3.6e-15 max|dF|=0.0e0 max|dVirial|=1.9e-05
```

(`max|dVirial|` was `1.717e+02`, the magnitude of `W` itself, before.)

### LAMMPS, 192 water, `run 0`

Artifact re-frozen from the checkpoint with this tree and the midtown backend
(`head-1l.pt2`, `--backend midtown --order 6`), identical for the two library
builds below:

| library | pair | np=1 | np=2 | np=4 |
|---|---|---|---|---|
| HEAD without the fix | `deepmd/kk` | -3709.650 | -4171.906 | -5096.420 |
| HEAD with the fix | `deepmd/kk` | -3709.651 | -3709.649 | -3709.648 |
| HEAD with the fix | `deepmd` | -3709.645 | -3709.648 | -3709.644 |

The energy stays -29946.665 eV in every run, and the fixed pressure agrees
with the uncompressed graph reference (-3708.08) and with the old single-rank
value (-3709.59) to about 1.5 atm, the accuracy of the mesh.

### Existing suites

`source/tests/pt_expt/kernels/test_mesh_backend.py` (4 passed) and
`test_long_range_sum.py` (50 passed) on a GPU: the single-domain outputs are
unchanged and the two mesh backends still agree on energy, gradient, force,
atomic virial and frame virial.

## Reproducing

- `deepmd/lib/libdeepmd_op_pt.so` is the module that registers
  `latent_charge_sum`; only it has to be rebuilt after a kernel change:

  ```bash
  make -C $BUILD deepmd_op_pt -j 8
  cp $BUILD/op/pt/libdeepmd_op_pt.so $REPO/deepmd/lib/
  ```

  With a `TORCH_CUDA_ARCH_LIST`-derived arch list that includes `compute_50`,
  `sog_gpu.cu` fails on its `atomicAdd(double*, double)`; set
  `TORCH_CUDA_ARCH_LIST=8.9` (or drop the `sm_50` `-gencode`) for the build.
- The `c560-*.pt2` artifacts were frozen against a library whose operator
  schema still carries the trailing `int force_mode=0` argument; this tree's
  schema (`ops.cc`) does not. The frozen AOTInductor program calls the
  operator through the proxy executor with the argument list of its freeze
  time, so those artifacts fail at run time with

  ```
  Error: aoti_torch_get_data_ptr(handle_.get(), &result) API call failed
  ```

  against a library rebuilt from this tree. Re-freezing from the checkpoint
  (`freeze_lr_variants.py --ckpt .../dpa4-sog-1L/model.ckpt-200000.pt
  --out head-1l.pt2 --backend midtown --order 6`) produces an artifact that
  matches this tree; the numbers above are from that artifact.

## Current state

- The fix is in `source/op/pt/long_range/midtown_sum.cu` on `c560-test`
  (3f7db81a) plus the new multi-rank test.
- 1L energy/forces unchanged, pressure now rank-independent at 1/2/4.
- 2L and `pair deepmd/kk` are unaffected (`bspline` path untouched).
