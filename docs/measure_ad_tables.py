"""AD-mode accuracy of midtown (CubeS2) vs B-spline, energy AND force.

Companion to docs/AD_MODE_ACCURACY.md sections 2.1-2.2.  Usage:
    python docs/measure_ad_tables.py [CKPT] [OUT.json]
defaults to the sigma_min = 2.00 A kernel; the sigma_min = 1.13 A one is
.../run_200k_noramp_archive/model.ckpt-200000.pt.

Both backends are measured through the shipped operator, so the numbers include
the real influence function (CubeS2 Form A by default), the real assignment
factor and the real stencil -- no reimplementation.  Both are weight
differentiated (AD): `interpolate_owners` slopes for the B-spline, `cs2_wg6` for
CubeS2, and gather-only weight differentiation is the exact gradient of the mesh
energy, so this is a comparison of two discretizations of the same functional.

  reference : direct k-sum of the same SOG kernel, kcut = 14 at L = 12 A
              (k_max = 7.3 1/A, exp(-sigma^2 k^2/2) ~ 1e-46: converged)
  ensemble  : 32 independent 64-atom configurations, neutral charges
  metric    : per configuration |dE|/|E| and rms|dF|/rms|F|, RMS over the set
  abscissa  : the spacing the mesh ACTUALLY has, Delta = L/n, on a ladder of
              sizes both backends accept verbatim (nothing rounds)
"""

# Standalone analysis tool, not package code: module-level imports, print and
# unannotated helpers are deliberate.
# ruff: noqa
import math
import os

import numpy as np
import torch

import sys
COUL = 14.3996454784255
KEY = "model.Default.atomic_model.long_range.kernel"
CKPT = (sys.argv[1] if len(sys.argv) > 1 else
        "/data/home/public/jiangzhen/dp/master-merge-test/water/"
        "run_150k_widthfloor_archive/model.ckpt-150000.pt")
_sd = torch.load(CKPT, map_location="cpu")["model"]
KA = _sd[KEY + ".amplitude"].detach().cpu().double().numpy().tolist()
KW = np.exp(_sd[KEY + ".log_width"].detach().cpu().double().numpy()).tolist()
OUT = sys.argv[2] if len(sys.argv) > 2 else ""

NATOMS, L, NCONF = 64, 12.0, 32
SMIN = min(KW)
GRIDS = [6, 8, 10, 12, 16, 18, 20, 24, 30, 40, 48, 60, 75, 100]
FLOOR = 3e-5
# band over which the law is fitted, per order, in Delta/sigma_min
BAND = {4: (0.15, 0.75), 6: (0.25, 0.75)}


def direct_k_sum(coord, q, L, kcut=14):
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


def fit(dl, err):
    x = np.log(np.asarray(dl)); y = np.log(np.asarray(err))
    A = np.vstack([np.ones_like(x), x]).T
    (lnc, p), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ np.array([lnc, p])
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return p, math.exp(lnc), r2


def main():
    from deepmd.pt_expt.kernels.long_range import latent_charge_sum, op_available
    from deepmd.pt_expt.long_range.kernel import SOGKernel
    op_available()
    k = SOGKernel(amplitude=KA, width=KW, precision="float32")
    box = torch.tensor([[L, 0, 0, 0, L, 0, 0, 0, L]], dtype=torch.float64)
    comm = torch.zeros(1, dtype=torch.int64)

    coords, charges, refs, frms = [], [], [], []
    for s in range(NCONF):
        g = torch.Generator().manual_seed(100 + s)
        c = torch.rand(NATOMS, 3, generator=g, dtype=torch.float64) * L
        q = torch.randn(NATOMS, generator=g, dtype=torch.float32).double()
        q = q - q.mean()
        E, F = direct_k_sum(c, q, L)
        coords.append(c)
        charges.append(q)
        refs.append(E)
        frms.append(float(np.sqrt((F ** 2).mean())))
        refs[-1] = (E, F)

    def ev(coord, q, order, spacing, backend):
        n = coord.shape[0]
        mask = torch.ones(n, dtype=torch.bool, device="cuda")
        nn = torch.tensor([n], dtype=torch.long, device="cuda")
        tot = torch.zeros(1, dtype=torch.float64, device="cuda")
        os.environ.pop("SOG_MIDTOWN_FORM_A", None)   # shipped default (Form A)
        e, _cg, f, _av, _vi = latent_charge_sum(
            coord.cuda(), q.reshape(-1, 1).float().cuda(), mask, nn, tot,
            kernel=k, channel=0, prefactor=COUL, box=box, order=order,
            spacing=spacing, mesh_backend=backend, communicator=comm)
        torch.cuda.synchronize()
        return float(e.sum().item()), f.double().cpu().numpy().reshape(-1, 3)

    print(f"# kernel {os.path.basename(os.path.dirname(CKPT))}  "
          f"sigma_min={SMIN:.4f} A")
    print(f"{'backend':>10} {'ord':>3} {'n':>4} {'D/sig':>7} {'dE/E':>11} {'dF/F':>11}")
    res = {}
    for backend in ("bspline", "midtown"):
        for order in (4, 6):
            key = (backend, order)
            res[key] = {"dl": [], "e": [], "f": []}
            for ng in GRIDS:
                sp = L / ng * (1.0 + 1e-12)
                de, df = [], []
                for s in range(NCONF):
                    E, F = ev(coords[s], charges[s], order, sp, backend)
                    de.append(abs(E - refs[s][0]) / abs(refs[s][0]))
                    df.append(float(np.sqrt(((F - refs[s][1]) ** 2).mean())) / frms[s])
                dl = (L / ng) / SMIN
                e = float(np.sqrt(np.mean(np.square(de))))
                f = float(np.sqrt(np.mean(np.square(df))))
                res[key]["dl"].append(dl); res[key]["e"].append(e); res[key]["f"].append(f)
                print(f"{backend:>10} {order:>3} {ng:4d} {dl:7.3f} {e:11.3e} {f:11.3e}",
                      flush=True)

    print("\n=== log-log fits, err = C (Delta/sigma_min)^p ===")
    print(f"{'backend':>10} {'ord':>3} {'band':>11} {'p_E':>6} {'C_E':>10} {'R2_E':>6} "
          f"{'p_F':>6} {'C_F':>10} {'R2_F':>6}")
    out = {}
    for (backend, order), r in res.items():
        lo, hi = BAND[order]
        sel = [i for i, d in enumerate(r["dl"]) if lo - 1e-9 <= d <= hi + 1e-9]
        pE, cE, r2E = fit([r["dl"][i] for i in sel], [r["e"][i] for i in sel])
        pF, cF, r2F = fit([r["dl"][i] for i in sel], [r["f"][i] for i in sel])
        out[(backend, order)] = (pE, cE, pF, cF)
        print(f"{backend:>10} {order:>3} {lo:.2f}-{hi:.2f} {pE:6.2f} {cE:10.3e} "
              f"{r2E:6.3f} {pF:6.2f} {cF:10.3e} {r2F:6.3f}")

    globals()["res"] = res
    print("\n=== midtown / bspline at matched Delta/sigma_min ===")
    for order in (4, 6):
        for i, ng in enumerate(GRIDS):
            dl = (L / ng) / SMIN
            if dl < BAND[order][0] - 1e-9:
                continue
            be, bt = res[("bspline", order)], res[("midtown", order)]
            print(f"  order {order}  D/sig={dl:5.3f}  n={ng:3d}  "
                  f"E {bt['e'][i]/be['e'][i]:6.2f}x   F {bt['f'][i]/be['f'][i]:6.2f}x")


if __name__ == "__main__":
    import json
    main()
    if OUT:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in res.items()}, open(OUT, "w"))
        print("saved", OUT)
