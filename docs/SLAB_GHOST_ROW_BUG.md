# Slab solve: rows without charge in the midtown kernels

## Symptom

The distributed (`SOG_MIDTOWN_SLAB=1`) solve of a DPA4C-SOG model disagreed
with the replicated solve of the same configuration, and the disagreement grew
with the number of ranks.  LAMMPS `run 0` on 1119744 water atoms, 224.0046 Å
cube, `pair deepmd/kk`, `processors 1 1 np` (midtown CubeS2 order 6):

| model | np | replicated | slab before | slab after |
|---|---|---|---|---|
| `dpa4c-sog` | 1 | -1.7464870e8 / 216.5382 atm | -1.7464870e8 / 216.53826 | -1.7464870e8 / 216.53826 |
| `dpa4c-sog` | 2 | -1.7464870e8 / 216.5382 | -1.7465057e8 / 215.85788 | -1.7464870e8 / 216.53826 |
| `dpa4c-sog` | 4 | -1.7464870e8 / 216.5382 | -1.7465215e8 / 177.65261 | -1.7464870e8 / 216.53826 |
| `dpa4c-plus-sog` | 2 | -1.7464828e8 / 325.6171 | -1.7465106e8 / 222.21694 | -1.7464828e8 / 325.61713 |
| `dpa4c-plus-sog` | 4 | -1.7464828e8 / 325.6171 | -1.7465351e8 / 127.45332 | (same as np=2) |

The slab solve at `np = 1` was already exact; only a decomposed frame drifted.

## Root cause

A decomposed frame hands `latent_charge_sum` the whole node axis of the rank:
its owned nodes, whose `node_mask` is true and whose charge is the network's,
and its ghost rows, whose `node_mask` is false and whose charge is zero.  The
midtown kernels assumed every row was charged, so each ghost row leaked into a
frame-wide quantity:

1. `k_midtown_charges1` added `1.0` to the charged-node count of every row
   (`local[0] = 1.0`), not only of the charged ones.  The uniform shift of the
   constraint is `(sum q - Q) / N`, so dividing by the local + ghost count made
   the shift too small by the ghost fraction and left the frame with a net
   charge: at np = 2 the per-rank charge sums were -129.88 and -50.16 e (a net
   -180.04 e) instead of the +/-39.84 e that cancel.  A net charge changes the
   potential at every node, which is the energy drift.  The ghost fraction
   grows with np (a rank's z-slab is thinner), which is why the drift did.
2. `k_midtown_finish1` summed `F_i (x) r_i` over every row into the frame moment
   sum `sums[1..9]`.  A ghost row carries the real force at its position but
   owns no charge, so the "rest of W" that is shared over the charged nodes,
   `(W - sum) / N`, was wrong.
3. `k_midtown_finish2` wrote that share, `(W - sum) / N`, to every row, ghost or
   not.  The replicated path returns the owner slice and never sees it, but the
   slab path forms the frame virial as the sum of the per-node shares over the
   whole node axis, so it counted the share `N_all / N_charged` times.  That is
   the pressure drift, and it grows with np for the same reason as (1).

The `bspline` backend (`mesh_sum.cu`) already masks all three reductions, and
`direct_sum.cu` sums only the per-node virials of its owners, which is why the
same frames were correct there.

## Fix

`source/op/pt/long_range/midtown_sum.cu`, three one-line changes plus comments:

```
k_midtown_charges1 : local[0] = m                       (count charged rows)
k_midtown_finish1  : local[1 + 3a + b] = m * F_a r_b     (moment of charged rows)
k_midtown_finish2  : virial9 = m * (F (x) r + (W - sum) / N)  (share of charged rows)
```

A call whose rows all carry charge (every single-domain call, and the gathered
frame of the replicated multi-rank path) has `m = 1` on every row, so its
outputs are unchanged bit for bit.  The fix also covers the local-spread path
(`SOG_MIDTOWN_DIST=1`), which passes the caller's node axis with ghosts to the
same kernels.

## Verification

### Operator level

`source/tests/pt_expt/kernels/test_long_range_sum.py`:
`test_rows_without_charge_change_nothing` evaluates a periodic frame twice, the
second time with 37 extra rows of zero charge and `node_mask == False`, and
requires every one of the five outputs on the charged rows to agree to 1e-6
relative -- including the frame virial, which the slab path forms over every
row.  It is parametrized over `bspline`, `midtown` and the slab solve.

Before the fix it fails for `midtown` and `slab` and passes for `bspline`.
A 20000-atom, 224 Å frame (spacing 0.9333, np = 2) that reproduced the bench:

| quantity | before | after |
|---|---|---|
| energy with ghosts vs without | -1718.382267 vs -1718.509873 eV (7.2e-5) | identical |
| frame virial with ghosts vs without | 1.77 eV of 5.28e3 | identical |
| net charge left by the constraint | -180.04 e | 0 |

### LAMMPS

The table above: the slab solve equals the replicated solve at np = 1, 2 and 4
for both `dpa4c-sog` and `dpa4c-plus-sog`, energy and pressure.

### Existing suites

`test_mesh_backend.py` 4 passed, `test_long_range_sum.py` 53 passed,
`test_long_range_multi_rank.py` 2 passed per rank under `mpirun -np 2`.

## Note on the local-spread path

`SOG_MIDTOWN_DIST=1` now reproduces the energy of the replicated solve
(-1.746487e8 eV at np = 2); its pressure is 216.47286 atm against 216.5382, a
3e-4 relative difference that its float32 density all-reduce may account for.
That path is opt-in and was not the subject of this report.
