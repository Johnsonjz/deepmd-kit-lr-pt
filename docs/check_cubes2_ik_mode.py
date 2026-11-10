# Standalone analysis tool, not package code.
# ruff: noqa
"""Validate the CubeS2 ik force mode against the AD mode on the shipped operator.

The two force modes share one mesh pipeline (spread, influence, transforms);
only the force law differs:

  AD : F_i = -q_i * d/dr [ sum_k w_k(r) phi_k ]      (weight differentiation)
  ik : u_a = IFFT(i k_a phi_hat);  F_i = -q_i sum_k w_k u_a[k]

AD is the exact gradient of the mesh energy and loses one order in Delta; ik is
one order more accurate but is not the gradient of E_mesh.  The measurement
below pins, on the real CUDA operator, whether the new ik branch delivers that
extra order and whether the AD branch still reproduces its pre-change numbers.

Checks, in order:
  1. AD and ik both evaluated through `latent_charge_sum(mesh_backend="midtown")`.
  2. Force error against a converged direct k-sum, over a spacing sweep.
  3. Slopes of the two error curves (per order 4 and 6).
  4. `force_mode="ik"` still rejected for the B-spline backend.
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, "/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt")

from deepmd.pt_expt.kernels.long_range import latent_charge_sum, op_available
from deepmd.pt_expt.long_range.kernel import SOGKernel

COUL = 14.3996454784255
KA = [0.1444232464, 0.02830792777, 0.01773190871, 0.001224140287] + [6.601994755e-06] * 8
KW = [2.000000004, 5.073223173, 5.096173536, 5.740239968, 15.38825675, 25.06860504,
      40.85598023, 66.58572018, 108.519222, 176.8610674, 288.2423646, 469.7679483]
SIG_MIN = min(KW)

NATOMS, L, NCONF = 64, 12.0, 32
# 2-3-5 smooth grid sizes, Delta = L/n from 1.5 A down to 0.4 A
GRIDS = [8, 10, 12, 15, 20, 24, 30]
DT = torch.float64
DEV = "cuda"


def direct_k_sum(coord, q, L, kcut=14):
    """Converged reciprocal-space reference (the complete lattice sum)."""
    c = coord.detach().double()
    qq = q.detach().double().reshape(-1)
    m = torch.arange(-kcut, kcut + 1, dtype=torch.float64)
    MZ, MY, MX = torch.meshgrid(m, m, m, indexing="ij")
    k = torch.stack([MX.reshape(-1), MY.reshape(-1), MZ.reshape(-1)], 1) * (2 * math.pi / L)
    k2 = (k * k).sum(1)
    keep = k2 > 0
    k, k2 = k[keep], k2[keep]
    K = torch.zeros_like(k2)
    for a, s in zip(KA, KW):
        K = K + a * (2 * math.pi * s * s) ** 1.5 * torch.exp(-0.5 * s * s * k2)
    ph = c @ k.T
    rho = (qq[:, None] * torch.exp(-1j * ph)).sum(0)
    E = 0.5 / L ** 3 * (K * rho.abs() ** 2).sum() * COUL
    E = E - 0.5 * COUL * float(np.sum(KA)) * float((qq * qq).sum())
    imag = (rho.conj()[None, :] * torch.exp(-1j * ph)).imag
    F = -(COUL / L ** 3) * (qq[:, None, None] * imag[:, :, None]
                            * K[None, :, None] * k[None, :, :]).sum(1)
    return float(E.item()), F.numpy().reshape(-1, 3)


def next235(n):
    """The extent the midtown backend rounds a request up to (2-3-5 only)."""
    n = max(n, 4)
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def op_force(coord, q, kernel, box, mask, nn, tot, comm, order, spacing, mode):
    e, _cg, f, _av, _vi = latent_charge_sum(
        coord.contiguous(), q.reshape(-1, 1).float().contiguous(), mask, nn, tot,
        kernel=kernel, channel=-1, prefactor=COUL, box=box, order=order,
        spacing=spacing, mesh_backend="midtown", communicator=comm,
        force_mode=mode)
    return float(e.sum()), f.double()


def fit_slope(x, y):
    """Least-squares slope of log(y) against log(x)."""
    x = np.log(np.asarray(x, float))
    y = np.log(np.asarray(y, float))
    a = np.polyfit(x, y, 1)
    return float(a[0])


def main():
    if not op_available():
        print("operator not available")
        return
    torch.manual_seed(0)
    coords, charges, refs, frms = [], [], [], []
    for s in range(NCONF):
        g = torch.Generator().manual_seed(100 + s)
        c = torch.rand(NATOMS, 3, generator=g, dtype=torch.float64) * L
        q = torch.randn(NATOMS, generator=g, dtype=torch.float32).double()
        q = q - q.mean()
        E, F = direct_k_sum(c, q, L)
        coords.append(c.to(DEV))
        charges.append(q.to(DEV))
        refs.append(torch.tensor(F, dtype=DT, device=DEV))
        frms.append(float(torch.sqrt((refs[-1] ** 2).mean())))

    kernel = SOGKernel(amplitude=KA, width=KW, precision="float32")
    box = torch.tensor([[L, 0, 0, 0, L, 0, 0, 0, L]], dtype=torch.float64)
    comm = torch.zeros(1, dtype=torch.int64)
    mask = torch.ones(NATOMS, dtype=torch.bool, device=DEV)
    nn = torch.tensor([NATOMS], dtype=torch.long, device=DEV)
    tot = torch.zeros(1, dtype=torch.float64, device=DEV)

    print("=== CubeS2 mesh backend: force error vs direct k-sum, AD vs ik ===")
    print(f"{'ord':>4} {'n':>4} {'D/sig':>7} {'AD':>11} {'ik':>11} {'ik/AD':>7} "
          f"{'dE_rel':>10}")
    res = {}
    for order in (4, 6):
        res[order] = {"dl": [], "ad": [], "ik": [], "de": []}
        for target in GRIDS:
            n = next235(target)
            spacing = L / target * (1 + 1e-12)
            e_ad, e_ik, de = [], [], []
            for s in range(NCONF):
                c = coords[s]
                q = charges[s]
                Ea, Fa = op_force(c, q, kernel, box, mask, nn, tot, comm,
                                  order, spacing, "ad")
                Ei, Fi = op_force(c, q, kernel, box, mask, nn, tot, comm,
                                  order, spacing, "ik")
                e_ad.append(float(torch.sqrt(((Fa - refs[s]) ** 2).mean())) / frms[s])
                e_ik.append(float(torch.sqrt(((Fi - refs[s]) ** 2).mean())) / frms[s])
                de.append(abs(Ei - Ea) / abs(Ea))
            dl = (L / n) / SIG_MIN
            ad = float(np.sqrt(np.mean(np.square(e_ad))))
            ik = float(np.sqrt(np.mean(np.square(e_ik))))
            derr = float(np.mean(de))
            res[order]["dl"].append(dl)
            res[order]["ad"].append(ad)
            res[order]["ik"].append(ik)
            res[order]["de"].append(derr)
            print(f"{order:>4} {n:>4} {dl:7.3f} {ad:11.3e} {ik:11.3e} "
                  f"{ik / ad:7.2f} {derr:10.2e}", flush=True)

    print("\n=== fitted slopes (last four grids) ===")
    for order in (4, 6):
        r = res[order]
        p_ad = fit_slope(r["dl"][-4:], r["ad"][-4:])
        p_ik = fit_slope(r["dl"][-4:], r["ik"][-4:])
        print(f"  order {order}:  AD p = {p_ad:5.2f}   ik p = {p_ik:5.2f}   "
              f"(Strang-Fix floor: AD {2 * order - 1}, ik {2 * order})")

    print("\n=== guard rails ===")
    try:
        latent_charge_sum(coords[0], charges[0].reshape(-1, 1).float(), mask, nn,
                          tot, kernel=kernel, channel=-1, prefactor=COUL, box=box,
                          order=6, spacing=1.0, mesh_backend="bspline",
                          communicator=comm, force_mode="ik")
        print("  FAIL: bspline accepted force_mode='ik'")
    except ValueError as exc:
        print(f"  ok: bspline rejects ik -> {str(exc)[:60]}")
    try:
        latent_charge_sum(coords[0], charges[0].reshape(-1, 1).float(), mask, nn,
                          tot, kernel=kernel, channel=-1, prefactor=COUL, box=box,
                          order=6, spacing=1.0, mesh_backend="midtown",
                          communicator=comm, force_mode="bogus")
        print("  FAIL: unknown force mode accepted")
    except ValueError as exc:
        print(f"  ok: unknown mode rejected -> {str(exc)[:60]}")


if __name__ == "__main__":
    main()
