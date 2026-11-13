"""ik-mode accuracy of midtown (CubeS2) vs B-spline, energy AND force.

Companion to docs/IK_MODE_ACCURACY.md.  Usage:
    python docs/measure_ik_tables.py [CKPT] [OUT.json]
defaults to the sigma_min = 2.00 A kernel; the sigma_min = 1.13 A one is
.../run_200k_noramp_archive/model.ckpt-200000.pt.

Three rows per order are measured through the SHIPPED operator, so the numbers
include the real influence function (CubeS2 Form A by default), the real
assignment factor and the real stencil:

    (bspline, ad)  the reference long-range backend
    (midtown, ad)  CubeS2 with weight differentiation (the default force mode)
    (midtown, ik)  CubeS2 with Fourier differentiation (force_mode="ik")

AD is the exact gradient of the mesh energy; ik differentiates the potential in
Fourier space, which removes the extra order the C0 window costs AD.  Energy is
mode independent (same phi_hat), so the energy rows of (midtown, ad) and
(midtown, ik) must agree to float32.

  reference : direct k-sum of the same SOG kernel, kcut = 14 at L = 12 A
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
import sys

import numpy as np
import torch

# Import the tree that owns the operator, whatever the invocation directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COUL = 14.3996454784255
KEY = "model.Default.atomic_model.long_range.kernel"

# Optional geometric u-series kernel (RBSOG_SISC 2023 Table 1, C1 construction).
#   SOG_GEOM="b,sigma_min,omega,M"  ->  sigma_m = sigma_min b^m,
#   a_m = (pi/2)^-1/2 ln(b)/sigma_m, a_0 *= omega  (k-space b_m = 4 pi ln b sigma_m^2).
GEOM = os.environ.get("SOG_GEOM", "")
# Energy-error normalisation: "k" divides by the k-space reference energy Ek
# (the mesh error lives entirely in the k-space part; the self-energy is exact),
# "e" (default) divides by the total reference energy E.
ENORM = os.environ.get("SOG_ENORM", "e")
if GEOM:
    _b, _smin, _omega, _M = [float(x) for x in GEOM.split(",")]
    KW = [_smin * _b**m for m in range(int(_M))]
    KA = [(math.pi / 2) ** -0.5 * math.log(_b) / s for s in KW]
    KA[0] *= _omega
    CKPT = f"geom_b{_b:g}_s{_smin:g}_w{_omega:g}_M{int(_M)}"
else:
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
# band over which the law is fitted, per order, in Delta/sigma_min
BAND = {4: (0.15, 0.75), 6: (0.25, 0.75)}
# (backend, force mode) rows, every order
ROWS = [
    ("bspline", "ad"),
    ("midtown", "ad"),
    ("midtown", "ik"),
    ("quads", "ad"),
    ("quads", "ik"),
]


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
    Ek = float(E.item())
    E = E - 0.5 * COUL * float(np.sum(KA)) * float((qq * qq).sum())
    imag = (rho.conj()[None, :] * torch.exp(-1j * ph)).imag
    F = -(COUL / L ** 3) * (qq[:, None, None] * imag[:, :, None]
                            * K[None, :, None] * k[None, :, :]).sum(1)
    return float(E.item()), F.numpy().reshape(-1, 3), Ek


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

    coords, charges, refs, frms, eknorms = [], [], [], [], []
    for s in range(NCONF):
        g = torch.Generator().manual_seed(100 + s)
        c = torch.rand(NATOMS, 3, generator=g, dtype=torch.float64) * L
        q = torch.randn(NATOMS, generator=g, dtype=torch.float32).double()
        q = q - q.mean()
        E, F, Ek = direct_k_sum(c, q, L)
        coords.append(c)
        charges.append(q)
        refs.append((E, F))
        eknorms.append(abs(Ek))
        frms.append(float(np.sqrt((F ** 2).mean())))

    def ev(coord, q, order, spacing, backend, mode):
        n = coord.shape[0]
        mask = torch.ones(n, dtype=torch.bool, device="cuda")
        nn = torch.tensor([n], dtype=torch.long, device="cuda")
        tot = torch.zeros(1, dtype=torch.float64, device="cuda")
        os.environ.pop("SOG_MIDTOWN_FORM_A", None)   # shipped default (Form A)
        e, _cg, f, _av, _vi = latent_charge_sum(
            coord.cuda(), q.reshape(-1, 1).float().cuda(), mask, nn, tot,
            kernel=k, channel=0, prefactor=COUL, box=box, order=order,
            spacing=spacing, mesh_backend=backend, communicator=comm,
            force_mode=mode)
        torch.cuda.synchronize()
        return float(e.sum().item()), f.double().cpu().numpy().reshape(-1, 3)

    print(f"# kernel {CKPT if GEOM else os.path.basename(os.path.dirname(CKPT))}  "
          f"sigma_min={SMIN:.4f} A  M={len(KA)}")
    print(f"{'backend':>9} {'mode':>3} {'ord':>3} {'n':>4} {'D/sig':>7} "
          f"{'dE/E':>11} {'dF/F':>11}")
    res = {}
    for backend, mode in ROWS:
        for order in (4, 6):
            key = (backend, mode, order)
            res[key] = {"dl": [], "e": [], "f": []}
            for ng in GRIDS:
                sp = L / ng * (1.0 + 1e-12)
                de, df = [], []
                for s in range(NCONF):
                    E, F = ev(coords[s], charges[s], order, sp, backend, mode)
                    de.append(abs(E - refs[s][0])
                              / (eknorms[s] if ENORM == "k" else abs(refs[s][0])))
                    df.append(float(np.sqrt(((F - refs[s][1]) ** 2).mean())) / frms[s])
                dl = (L / ng) / SMIN
                e = float(np.sqrt(np.mean(np.square(de))))
                f = float(np.sqrt(np.mean(np.square(df))))
                res[key]["dl"].append(dl); res[key]["e"].append(e); res[key]["f"].append(f)
                print(f"{backend:>9} {mode:>3} {order:>3} {ng:4d} {dl:7.3f} "
                      f"{e:11.3e} {f:11.3e}", flush=True)

    print("\n=== log-log fits, err = C (Delta/sigma_min)^p ===")
    print(f"{'backend':>9} {'mode':>3} {'ord':>3} {'band':>9} {'p_E':>6} {'C_E':>10} "
          f"{'R2_E':>6} {'p_F':>6} {'C_F':>10} {'R2_F':>6}")
    out = {}
    for (backend, mode, order), r in res.items():
        lo, hi = BAND[order]
        sel = [i for i, d in enumerate(r["dl"]) if lo - 1e-9 <= d <= hi + 1e-9]
        pE, cE, r2E = fit([r["dl"][i] for i in sel], [r["e"][i] for i in sel])
        pF, cF, r2F = fit([r["dl"][i] for i in sel], [r["f"][i] for i in sel])
        out[f"{backend}|{mode}|{order}"] = (pE, cE, pF, cF)
        print(f"{backend:>9} {mode:>3} {order:>3} {lo:.2f}-{hi:.2f} {pE:6.2f} "
              f"{cE:10.3e} {r2E:6.3f} {pF:6.2f} {cF:10.3e} {r2F:6.3f}")

    print("\n=== midtown/bspline at matched Delta/sigma_min (F ratio) ===")
    for order in (4, 6):
        for i, ng in enumerate(GRIDS):
            dl = (L / ng) / SMIN
            if dl < BAND[order][0] - 1e-9:
                continue
            be = res[("bspline", "ad", order)]
            ba = res[("midtown", "ad", order)]
            bi = res[("midtown", "ik", order)]
            print(f"  order {order}  D/sig={dl:5.3f}  n={ng:3d}  "
                  f"E ik/bs {bi['e'][i]/be['e'][i]:5.2f}x  "
                  f"F ad/bs {ba['f'][i]/be['f'][i]:5.2f}x  "
                  f"F ik/bs {bi['f'][i]/be['f'][i]:5.2f}x")

    if OUT:
        import json
        raw = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res.items()}
        json.dump({"fits": out, "raw": raw}, open(OUT, "w"))
        print("saved", OUT)


if __name__ == "__main__":
    main()
