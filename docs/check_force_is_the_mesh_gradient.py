# Standalone analysis tool, not package code: module-level imports, print and
# unannotated helpers are deliberate.
# ruff: noqa
"""Is the reference wrong, or is "accurate energy, one-order-worse force" intrinsic?

Three checks, all on the same kernel and configurations:

 1. REFERENCE self-consistency: finite-difference the direct-k-sum ENERGY and
    compare with the direct-k-sum FORCE.  If they agree, the reference is a
    consistent (E, F) pair and cannot be the reason the mesh force looks worse.

 2. MESH self-consistency: finite-difference the OPERATOR's mesh energy and
    compare with the operator's AD force.  If they agree, the AD force IS the
    exact gradient of the mesh energy -- the implementation adds nothing.

 3. ERROR LAWS: the force obtained by finite-differencing the mesh energy has
    the error law of the mesh energy's derivative, so comparing it with the
    analytic AD force's law shows where the extra order is lost.

Finite-difference step: h = 1e-2 A (the energy noise from the float32 mesh is
amplified by 1/h, so a larger step trades truncation for noise).  Energy is float64 at the atom level but
comes from float32 meshes, so the FD force carries ~|E| eps32 / h of noise,
which is far below the mesh errors at the grids used here.
"""
import math
import os
import sys

import numpy as np
import torch

COUL = 14.3996454784255
CKPT = ("/data/home/public/jiangzhen/dp/master-merge-test/water/"
        "run_150k_widthfloor_archive/model.ckpt-150000.pt")
KEY = "model.Default.atomic_model.long_range.kernel"
_sd = torch.load(CKPT, map_location="cpu")["model"]
KA = _sd[KEY + ".amplitude"].detach().cpu().double().numpy().tolist()
KW = np.exp(_sd[KEY + ".log_width"].detach().cpu().double().numpy()).tolist()
SMIN = min(KW)

N, L = 64, 12.0
NCONF = 2
GRIDS = [10, 12, 16, 20]
H = 1e-2


def ref_energy(coord, q, L, kcut=14):
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
    return float((E - 0.5 * COUL * float(np.sum(KA)) * float((qq * qq).sum())).item())


def ref_force(coord, q, L, kcut=14):
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
    imag = (rho.conj()[None, :] * torch.exp(-1j * ph)).imag
    F = -(COUL / L ** 3) * (qq[:, None, None] * imag[:, :, None]
                            * K[None, :, None] * k[None, :, :]).sum(1)
    return F.numpy().reshape(-1, 3)


def main():
    from deepmd.pt_expt.kernels.long_range import latent_charge_sum, op_available
    from deepmd.pt_expt.long_range.kernel import SOGKernel
    op_available()
    k = SOGKernel(amplitude=KA, width=KW, precision="float32")
    box = torch.tensor([[L, 0, 0, 0, L, 0, 0, 0, L]], dtype=torch.float64)
    comm = torch.zeros(1, dtype=torch.int64)
    mask = torch.ones(N, dtype=torch.bool, device="cuda")
    nn = torch.tensor([N], dtype=torch.long, device="cuda")
    tot = torch.zeros(1, dtype=torch.float64, device="cuda")

    def op(coord, q, order, spacing, backend, want_force):
        os.environ.pop("SOG_MIDTOWN_FORM_A", None)
        e, _cg, f, _av, _vi = latent_charge_sum(
            coord.cuda(), q.reshape(-1, 1).float().cuda(), mask, nn, tot,
            kernel=k, channel=0, prefactor=COUL, box=box, order=order,
            spacing=spacing, mesh_backend=backend, communicator=comm)
        torch.cuda.synchronize()
        if want_force:
            return float(e.sum().item()), f.double().cpu().numpy().reshape(-1, 3)
        return float(e.sum().item()), None

    def fd_force(fun, coord, h=H):
        """Central difference of a scalar energy in all 3N coordinates."""
        out = np.zeros((N, 3))
        c = coord.clone()
        for i in range(N):
            for a in range(3):
                c[i, a] = coord[i, a] + h
                ep = fun(c)
                c[i, a] = coord[i, a] - h
                em = fun(c)
                c[i, a] = coord[i, a]
                out[i, a] = -(ep - em) / (2 * h)
        return out

    confs = []
    for s in range(NCONF):
        g = torch.Generator().manual_seed(100 + s)
        c = torch.rand(N, 3, generator=g, dtype=torch.float64) * L
        q = torch.randn(N, generator=g, dtype=torch.float32).double()
        confs.append((c, q - q.mean()))

    # ── 1. the reference itself ──────────────────────────────────────────
    print("=== 1. reference self-consistency: FD of E_ref vs F_ref ===")
    for s, (c, q) in enumerate(confs):
        f_ref = ref_force(c, q, L)
        f_fd = fd_force(lambda cc: ref_energy(cc, q, L), c)
        rel = float(np.sqrt(((f_fd - f_ref) ** 2).mean()) / np.sqrt((f_ref ** 2).mean()))
        print(f"  config {s}:  rms(F_fd - F_ref) / rms(F_ref) = {rel:.2e}")
    print("  (both are the same analytic pair, so this is a consistency test of the")
    print("   reference, not of the mesh)")

    # ── 2 & 3. the mesh ──────────────────────────────────────────────────
    print("\n=== 2-3. mesh: FD of the mesh energy vs the operator's AD force ===")
    print(f"{'backend':>9} {'ord':>3} {'n':>4} {'D/sig':>7} "
          f"{'rms(F_ad-F_ref)/|F|':>20} {'rms(F_fd-F_ref)/|F|':>20} "
          f"{'rms(F_fd-F_ad)/|F|':>20}")
    for backend in ("bspline", "midtown"):
        for order in (6,):
            for ng in GRIDS:
                sp = L / ng * (1 + 1e-12)
                e_ad, e_fd, e_dif, fr = [], [], [], []
                for c, q in confs:
                    f_ref = ref_force(c, q, L)
                    frms = float(np.sqrt((f_ref ** 2).mean()))
                    _e, f_ad = op(c, q, order, sp, backend, True)
                    fun = lambda cc, q=q: op(cc, q, order, sp, backend, False)[0]
                    f_fd = fd_force(fun, c)
                    r = lambda x: float(np.sqrt((x ** 2).mean())) / frms
                    e_ad.append(r(f_ad - f_ref))
                    e_fd.append(r(f_fd - f_ref))
                    e_dif.append(r(f_fd - f_ad))
                m = lambda v: float(np.sqrt(np.mean(np.square(v))))
                print(f"{backend:>9} {order:>3} {ng:>4} {(L/ng)/SMIN:7.3f} "
                      f"{m(e_ad):20.3e} {m(e_fd):20.3e} {m(e_dif):20.3e}", flush=True)


if __name__ == "__main__":
    main()
