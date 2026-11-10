# Standalone analysis tool, not package code: module-level imports, print and
# unannotated helpers are deliberate.
# ruff: noqa
"""ik vs AD force, measured in one pipeline.

Both differentiation schemes are implemented on the SAME mesh pipeline (same
spread, same deconvolution, same transforms), so the difference between the two
force laws isolates the differentiation scheme and nothing else.

  AD : F_i = -q_i * d/dr [ sum_k w_k(r) phi_k ]        (weight differentiation)
  ik : u_a = IFFT(i k_a phi_hat);  F_i = -q_i * sum_k w_k u_a[k]

The AD branch is validated against the shipped CUDA operator before its numbers
are used: if the prototype's AD force does not reproduce `latent_charge_sum`
(bspline backend) to float32 precision, the prototype is wrong and its ik
branch cannot be trusted either.

Abscissa is the real mesh spacing Delta = L/n, and the ladder uses mesh sizes
both backends accept verbatim (even and 2-3-5 smooth), so no two points share a
mesh and no nominal-spacing quantisation enters the fit.
"""
import math
import os
import sys

import numpy as np
import torch

COUL = 14.3996454784255
KA = [0.1444232464, 0.02830792777, 0.01773190871, 0.001224140287] + [6.601994755e-06] * 8
KW = [2.000000004, 5.073223173, 5.096173536, 5.740239968, 15.38825675, 25.06860504,
      40.85598023, 66.58572018, 108.519222, 176.8610674, 288.2423646, 469.7679483]

NATOMS, L, NCONF = 64, 12.0, 32
GRIDS = [6, 8, 10, 12, 16, 18, 20, 24, 30, 40, 48, 60]
FLOOR = 3e-5
DT = torch.float64
DEV = "cuda"


# ── cardinal B-spline, the recursion of mesh_sum.cu axis_stencil ──────────
def bspline_stencil(t, order):
    """value[k] for grid point (base - order + 1 + k), and its d/dt."""
    a = [t, 1.0 - t]
    for n in range(3, order + 1):
        scale = 1.0 / (n - 1)
        new = [None] * n
        new[n - 1] = scale * (1.0 - t) * a[n - 2]
        for j in range(n - 2, 0, -1):
            new[j] = scale * ((t + j) * a[j] + (n - t - j) * a[j - 1])
        new[0] = scale * t * a[0]
        a = new
    return [a[order - 1 - k] for k in range(order)]


def integer_bspline_weights(order):
    """M_p(1), ..., M_p(p-1): the assignment factor's integer weights."""
    out = []
    for l in range(order - 1):
        x = l + 1.0
        value = 0.0
        binom = 1.0
        for j in range(order + 1):
            if j > 0:
                binom = binom * (order - j + 1) / j
            base = max(x - j, 0.0)
            value += (1.0 if j % 2 == 0 else -1.0) * binom * base ** (order - 1)
        fact = 1.0
        for k in range(2, order):
            fact *= k
        out.append(value / fact)
    return torch.tensor(out, dtype=DT, device=DEV)


def assignment_factor(order, n):
    """|D_d(m)|^-2 for m = 0..n-1."""
    iw = integer_bspline_weights(order)
    m = torch.arange(n, dtype=DT, device=DEV)
    theta = 2.0 * math.pi * m / n
    l = torch.arange(order - 1, dtype=DT, device=DEV)
    ang = theta[:, None] * l[None, :]
    re = (iw[None, :] * torch.cos(ang)).sum(1)
    im = (iw[None, :] * torch.sin(ang)).sum(1)
    return 1.0 / (re * re + im * im)


def kernel_transform(k_sq):
    amp = torch.tensor(KA, dtype=DT, device=DEV)
    sig = torch.tensor(KW, dtype=DT, device=DEV)
    K = torch.zeros_like(k_sq)
    for a, s in zip(amp, sig):
        K = K + a * (2 * math.pi * s * s) ** 1.5 * torch.exp(-0.5 * s * s * k_sq)
    return K


# ── the mesh pipeline ────────────────────────────────────────────────────
def mesh_forces(coord, q, L, n, order, scheme):
    """Energy and force of one periodic frame; scheme in {'ad', 'ik'}."""
    N = coord.shape[0]
    u = coord / L * n                       # scaled coordinate
    base = torch.floor(u)
    t = u - base
    idx0 = (base.to(torch.int64) - (order - 1)) % n

    val = bspline_stencil(t, order)         # order tensors of shape (N,3)
    p = order
    wx = torch.stack([v[:, 0] for v in val], dim=1)      # (N, p) per axis
    wy = torch.stack([v[:, 1] for v in val], dim=1)
    wz = torch.stack([v[:, 2] for v in val], dim=1)

    # spread: rho[g] = sum_i q_i prod_d value_d[k_d]
    off = torch.arange(p, device=DEV)
    gx = (idx0[:, 0:1] + off[None, :]) % n
    gy = (idx0[:, 1:2] + off[None, :]) % n
    gz = (idx0[:, 2:3] + off[None, :]) % n
    w = wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
    fx = (gx[:, :, None, None] * n * n + gy[:, None, :, None] * n
          + gz[:, None, None, :]).reshape(-1)
    rho = torch.zeros(n * n * n, dtype=DT, device=DEV)
    rho.index_put_((fx,), (q[:, None, None, None] * w).reshape(-1), accumulate=True)
    rho = rho.reshape(n, n, n)

    kx = 2 * math.pi * torch.fft.fftfreq(n, d=1.0, device=DEV) * n / L
    kz = 2 * math.pi * torch.fft.rfftfreq(n, d=1.0, device=DEV) * n / L
    KX, KY = torch.meshgrid(kx, kx, indexing="ij")
    KZ = kz[None, None, :]
    k_sq = KX[:, :, None] ** 2 + KY[:, :, None] ** 2 + KZ ** 2

    m = torch.arange(n, dtype=DT, device=DEV)
    mz = torch.arange(n // 2 + 1, dtype=DT, device=DEV)
    keep = ((2 * torch.abs(m) != n)[:, None, None] & (2 * torch.abs(m) != n)[None, :, None]
            & (2 * torch.abs(mz) != n)[None, None, :])
    keep = keep & (k_sq > 0)

    fac = assignment_factor(order, n)
    influence = (COUL / L ** 3) * kernel_transform(k_sq) \
        * fac[:, None, None] * fac[None, :, None] * fac[: n // 2 + 1][None, None, :]
    influence = torch.where(keep, influence, torch.zeros_like(influence))

    rho_hat = torch.fft.rfftn(rho, dim=(0, 1, 2))
    phi_hat = influence * rho_hat
    # cuFFT's inverse is unnormalised; torch's irfftn divides by n^3
    phi_mesh = torch.fft.irfftn(phi_hat, s=(n, n, n), dim=(0, 1, 2)) * n ** 3

    k0 = float(np.sum(KA))

    flat = (gx[:, :, None, None] * n * n + gy[:, None, :, None] * n
            + gz[:, None, None, :]).reshape(-1)

    def gather(field):
        f = field.reshape(-1)[flat].reshape(N, p, p, p)
        return (f * w).sum(dim=(1, 2, 3))

    raw = gather(phi_mesh)                       # includes the mesh self term
    phi_i = raw - COUL * k0 * q
    energy = 0.5 * (q * phi_i).sum()

    if scheme == "ad":
        # the self term is a constant of the positions, so differentiating
        # 0.5*sum(q*raw) is differentiating the energy
        g = torch.autograd.grad((0.5 * q * raw).sum(), coord,
                                create_graph=False)[0]
        return float(energy), -g
    else:
        comps = []
        for (K, ax) in ((KX, 0), (KY, 1), (KZ, 2)):
            uk = torch.zeros(n, n, n, dtype=torch.complex128, device=DEV)
            spec = phi_hat
            if ax == 0:
                spec = 1j * KX[:, :, None] * spec
            elif ax == 1:
                spec = 1j * KY[:, :, None] * spec
            else:
                spec = 1j * KZ * spec
            comps.append(torch.fft.irfftn(spec, s=(n, n, n), dim=(0, 1, 2)) * n ** 3)
        g = torch.stack([gather(c) for c in comps], dim=1)
        return float(energy), -q[:, None] * g


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


def main():
    torch.manual_seed(0)
    coords, charges, refs = [], [], []
    for s in range(NCONF):
        g = torch.Generator().manual_seed(100 + s)
        c = (torch.rand(NATOMS, 3, generator=g, dtype=torch.float64) * L)
        q = torch.randn(NATOMS, generator=g, dtype=torch.float32).double()
        q = (q - q.mean())
        E, F = direct_k_sum(c, q, L)
        coords.append(c.to(DEV))
        charges.append(q.to(DEV))
        refs.append((E, torch.tensor(F, dtype=DT, device=DEV)))
    frms = [float(torch.sqrt((r[1] ** 2).mean())) for r in refs]

    # ── validate the AD branch against the shipped CUDA operator ──────────
    print("=== AD branch vs the shipped operator (bspline backend) ===")
    sys.path.insert(0, "/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt")
    try:
        from deepmd.pt_expt.kernels.long_range import latent_charge_sum, op_available
        from deepmd.pt_expt.long_range.kernel import SOGKernel
        op_available()
        print("  (operator available)")
    except Exception as exc:                                  # noqa: BLE001
        print("  operator not available:", str(exc)[:80])
        return

    ok = torch.load("/data/home/public/jiangzhen/dp/master-merge-test/water/"
                    "run_250k_librule/model.ckpt-250000.pt", map_location="cpu")["model"]
    _ = ok
    order_chk, nchk = 6, 12
    c = coords[0].detach().clone().requires_grad_(True)
    q = charges[0]
    E_ref, F_ref = mesh_forces(c, q, L, nchk, order_chk, "ad")
    k = SOGKernel(amplitude=KA, width=KW, precision="float32")
    box = torch.tensor([[L, 0, 0, 0, L, 0, 0, 0, L]], dtype=torch.float64)
    comm = torch.zeros(1, dtype=torch.int64)
    mask = torch.ones(NATOMS, dtype=torch.bool, device="cuda")
    nn = torch.tensor([NATOMS], dtype=torch.long, device="cuda")
    tot = torch.zeros(1, dtype=torch.float64, device="cuda")
    e_op, _cg, f_op, _av, _vi = latent_charge_sum(
        c.detach().contiguous(), q.reshape(-1, 1).float().contiguous(), mask, nn, tot,
        kernel=k, channel=-1, prefactor=COUL, box=box, order=order_chk,
        spacing=L / nchk * (1 + 1e-12), mesh_backend="bspline", communicator=comm)
    torch.cuda.synchronize()
    f_op = f_op.double()
    print(f"  energy  proto {E_ref:+.9e}  op {float(e_op.sum()):+.9e}  "
          f"rel {abs(E_ref - float(e_op.sum())) / abs(E_ref):.2e}")
    print(f"  force   rms(proto-op)/rms(op) = "
          f"{float(((F_ref - f_op).pow(2).mean()).sqrt() / f_op.pow(2).mean().sqrt()):.2e}")

    # ── the sweep ─────────────────────────────────────────────────────────
    print("\n=== force error, AD vs ik, same pipeline ===")
    print(f"{'order':>5} {'n':>4} {'D/sig':>7} {'AD':>11} {'ik':>11} {'ik/AD':>7}")
    res = {}
    for order in (4, 6):
        res[order] = {"dl": [], "ad": [], "ik": []}
        for n in GRIDS:
            e_ad = []; e_ik = []
            for s in range(NCONF):
                c = coords[s].detach().clone().requires_grad_(True)
                _, F = mesh_forces(c, charges[s], L, n, order, "ad")
                e_ad.append(float(torch.sqrt(((F - refs[s][1]) ** 2).mean())) / frms[s])
                _, F = mesh_forces(c, charges[s], L, n, order, "ik")
                e_ik.append(float(torch.sqrt(((F - refs[s][1]) ** 2).mean())) / frms[s])
            dl = (L / n) / float(np.min(KW))
            ad = float(np.sqrt(np.mean(np.square(e_ad))))
            ik = float(np.sqrt(np.mean(np.square(e_ik))))
            res[order]["dl"].append(dl); res[order]["ad"].append(ad)
            res[order]["ik"].append(ik)
            print(f"{order:>5} {n:>4} {dl:7.3f} {ad:11.3e} {ik:11.3e} {ik/ad:7.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
