"""Check docs/bspline_ad_reference.cpp against the validated Python prototype.

  1. weight derivative: the C++ recursion carried through d/dfrac against
     torch.autograd of the same recursion;
  2. the whole AD contraction: the C++ `bspline_ad_force` against
     `measure_ik_vs_ad_force.mesh_forces(..., 'ad')`, on the same potential
     mesh, positions and charges.

The prototype's AD branch reproduces the shipped CUDA operator to 2.1e-7 (the
script prints that check itself), so agreeing with it is agreeing with
production.

Run:  g++ -O2 -std=c++17 docs/bspline_ad_reference.cpp -o /tmp/bsad
      python docs/check_bspline_ad_reference.py
"""

# Standalone analysis tool, not package code: module-level imports, print and
# unannotated helpers are deliberate.
# ruff: noqa
import math
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_ik_vs_ad_force as P  # noqa: E402

REF = "/tmp/bsad"


def _weights(n, frac):
    """The existing sog.cpp recursion, in torch, differentiable in frac."""
    w = torch.zeros(n, dtype=torch.float64)
    w[0] = 1.0 - frac
    w[1] = frac
    for k in range(3, n + 1):
        inv = 1.0 / (k - 1)
        w_prev = w.clone()
        w[k - 1] = frac * w_prev[k - 2] * inv
        for j in range(1, k - 1):
            w[k - 1 - j] = ((frac + j) * w_prev[k - 2 - j]
                            + (k - j - frac) * w_prev[k - 1 - j]) * inv
        w[0] = (1.0 - frac) * w_prev[0] * inv
    return w


def weight_check():
    for n in (2, 3, 4, 5, 6, 8):
        ts = torch.linspace(0.0, 1.0, 41, dtype=torch.float64)[:-1]
        inp = "".join(f"{t}\n" for t in ts.tolist())
        out = subprocess.run([REF, "weights", str(n), str(len(ts))],
                             input=inp, capture_output=True, text=True, check=True)
        worst_w = worst_d = 0.0
        for line in out.stdout.strip().splitlines():
            parts = line.split()
            i = int(parts[1])
            vals = [float(v) for v in parts[2:]]
            w_cpp = torch.tensor(vals[:n], dtype=torch.float64)
            dw_cpp = torch.tensor(vals[n:], dtype=torch.float64)
            t = ts[i].clone().requires_grad_(True)
            w = _weights(n, t)
            # per-component derivative: dw[k] = d w[k] / d frac
            dw = torch.stack([
                torch.autograd.grad(w[k], t, retain_graph=True)[0] for k in range(n)
            ])
            worst_w = max(worst_w, float((w - w_cpp).abs().max()))
            worst_d = max(worst_d, float((dw - dw_cpp).abs().max()))
        print(f"  n={n:2d}  max|w_cpp - w_torch| = {worst_w:.2e}   "
              f"max|dw_cpp - dw_torch| = {worst_d:.2e}")
        assert worst_w < 1e-13 and worst_d < 1e-11, "weight derivative mismatch"


def _weights_v(n, frac):
    """The sog.cpp recursion, batched over the last axis of `frac`."""
    w = torch.zeros(*frac.shape, n, dtype=torch.float64, device=frac.device)
    w[..., 0] = 1.0 - frac
    w[..., 1] = frac
    for k in range(3, n + 1):
        inv = 1.0 / (k - 1)
        wp = w.clone()
        w[..., k - 1] = frac * wp[..., k - 2] * inv
        for j in range(1, k - 1):
            w[..., k - 1 - j] = ((frac + j) * wp[..., k - 2 - j]
                                 + (k - j - frac) * wp[..., k - 1 - j]) * inv
        w[..., 0] = (1.0 - frac) * wp[..., 0] * inv
    return w


def _mesh_lr(coord, q, L, n, order):
    """The mesh as sog.cpp indexes it: cell (ix0 - half + a) with w[a]."""
    half = (order - 1) // 2
    u = coord / L * n
    ix0 = torch.floor(u)
    t = u - ix0
    w = _weights_v(order, t)                      # (N, 3, order)
    off = torch.arange(order, device=w.device)
    g = (ix0.to(torch.int64)[:, :, None] - half + off[None, None, :]) % n
    w3 = (w[:, 0, :, None, None] * w[:, 1, None, :, None]
          * w[:, 2, None, None, :])
    flat = (g[:, 0, :, None, None] * n * n + g[:, 1, None, :, None] * n
            + g[:, 2, None, None, :]).reshape(-1)
    rho = torch.zeros(n * n * n, dtype=P.DT, device=P.DEV)
    rho.index_put_((flat,), (q[:, None, None, None] * w3).reshape(-1), accumulate=True)
    return _solve(rho.reshape(n, n, n), L, n, order)


def _mesh(coord, q, L, n, order):
    """The potential mesh the prototype builds inside mesh_forces."""
    u = coord / L * n
    base = torch.floor(u)
    t = u - base
    idx0 = (base.to(torch.int64) - (order - 1)) % n
    val = P.bspline_stencil(t, order)
    p = order
    wx = torch.stack([v[:, 0] for v in val], 1)
    wy = torch.stack([v[:, 1] for v in val], 1)
    wz = torch.stack([v[:, 2] for v in val], 1)
    off = torch.arange(p, device=P.DEV)
    gx = (idx0[:, 0:1] + off[None, :]) % n
    gy = (idx0[:, 1:2] + off[None, :]) % n
    gz = (idx0[:, 2:3] + off[None, :]) % n
    w = wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
    flat = (gx[:, :, None, None] * n * n + gy[:, None, :, None] * n
            + gz[:, None, None, :]).reshape(-1)
    rho = torch.zeros(n * n * n, dtype=P.DT, device=P.DEV)
    rho.index_put_((flat,), (q[:, None, None, None] * w).reshape(-1), accumulate=True)
    rho = rho.reshape(n, n, n)
    return _solve(rho, L, n, order)


def _solve(rho, L, n, order):
    kx = 2 * math.pi * torch.fft.fftfreq(n, d=1.0, device=P.DEV) * n / L
    kz = 2 * math.pi * torch.fft.rfftfreq(n, d=1.0, device=P.DEV) * n / L
    KX, KY = torch.meshgrid(kx, kx, indexing="ij")
    KZ = kz[None, None, :]
    k_sq = KX[:, :, None] ** 2 + KY[:, :, None] ** 2 + KZ ** 2
    m = torch.arange(n, dtype=P.DT, device=P.DEV)
    mz = torch.arange(n // 2 + 1, dtype=P.DT, device=P.DEV)
    keep = ((2 * torch.abs(m) != n)[:, None, None]
            & (2 * torch.abs(m) != n)[None, :, None]
            & (2 * torch.abs(mz) != n)[None, None, :]) & (k_sq > 0)
    fac = P.assignment_factor(order, n)
    infl = (P.COUL / L ** 3) * P.kernel_transform(k_sq) * fac[:, None, None] \
        * fac[None, :, None] * fac[: n // 2 + 1][None, None, :]
    infl = torch.where(keep, infl, torch.zeros_like(infl))
    ph = infl * torch.fft.rfftn(rho, dim=(0, 1, 2))
    return torch.fft.irfftn(ph, s=(n, n, n), dim=(0, 1, 2)) * n ** 3


def force_check():
    N, L, order, n = 64, 12.0, 6, 12
    g = torch.Generator().manual_seed(7)
    coord = torch.rand(N, 3, generator=g, dtype=torch.float64).to(P.DEV)
    q = torch.randn(N, generator=g, dtype=torch.float32).double().to(P.DEV)
    q = q - q.mean()

    # sog.cpp centres its stencil at floor(u)-(p-1)/2 and mesh_sum.cu at
    # floor(u)-p+1: the charge mesh is translated by (p-1)/2 cells, which the
    # gather undoes because it uses the same shift.  So the MESH arrays are not
    # comparable; the FORCES are.  The C++ therefore gets the sog.cpp-convention
    # mesh and is compared with the prototype's AD force, which is built in the
    # mesh_sum convention.
    mesh = _mesh_lr(coord, q, L, n, order)
    c = coord.clone().requires_grad_(True)
    _e, f_py = P.mesh_forces(c, q, L, n, order, "ad")
    # The prototype puts COUL inside the influence function, so the C++ must
    # not apply qscale again: in sog.cpp the green table carries no COUL and
    # qscale = force->qqrd2e supplies it.
    qscale = 1.0
    inp = [f"{N} {order} {n} {n} {n} {qscale:.17g} {L} {L} {L} 0 0 0"]
    for i in range(N):
        inp.append(f"{coord[i,0]:.17g} {coord[i,1]:.17g} {coord[i,2]:.17g} {q[i]:.17g}")
    # sog.cpp's mesh_index is ix + nx*(iy + ny*iz) (x fastest); the tensor is
    # [ix, iy, iz] (z fastest), so it is transposed on the way out
    xfast = mesh.detach().permute(2, 1, 0).contiguous().reshape(-1)
    inp.append(" ".join(f"{v:.17g}" for v in xfast.tolist()))
    out = subprocess.run([REF, "frame"], input="\n".join(inp), capture_output=True,
                         text=True, check=True)
    f_cpp = torch.tensor([[float(v) for v in line.split()[:3]]
                          for line in out.stdout.strip().splitlines()],
                         dtype=torch.float64, device=P.DEV)
    rel = float((f_cpp - f_py).abs().max() / f_py.abs().max())
    print(f"  order {order}, n={n}, N={N}: max|F_cpp - F_prototype| / max|F| = {rel:.3e}")
    assert rel < 1e-10, "AD contraction mismatch"


if __name__ == "__main__":
    print("=== weight derivative ===")
    weight_check()
    print("\n=== AD contraction ===")
    force_check()
    print("\nALL OK")
