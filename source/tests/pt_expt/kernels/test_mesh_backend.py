# SPDX-License-Identifier: LGPL-3.0-or-later
"""The `midtown` mesh backend of the latent-charge operator.

`latent_charge_sum` carries the long-range term of periodic frames on a mesh.
The default `bspline` backend is the smooth particle-mesh Ewald sum; the
`midtown` backend is the CubeS2 spline and cuFFT of the SOG library. Both take
the same arguments and return the same quantities, so the same call with one or
the other backend must agree to the accuracy of the mesh, and the narrow first
version of `midtown` must reject the inputs it does not serve instead of
silently changing the physics.
"""

from __future__ import (
    annotations,
)

import pytest
import torch

from deepmd.pt_expt.kernels.long_range import (
    latent_charge_sum,
    op_available,
)
from deepmd.pt_expt.long_range.kernel import (
    SOGKernel,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not op_available(),
    reason="the long-range operator is CUDA only",
)

COULOMB = 14.3996454784255


def _system(atoms: int = 64, length: float = 9.0):
    generator = torch.Generator(device="cpu").manual_seed(0)
    coord = (
        torch.rand(atoms, 3, generator=generator, dtype=torch.float64) * length
    ).cuda()
    charge = torch.randn(atoms, 1, generator=generator, dtype=torch.float32).cuda()
    mask = torch.ones(atoms, dtype=torch.bool, device="cuda")
    n_node = torch.tensor([atoms], dtype=torch.long, device="cuda")
    total = torch.zeros(1, dtype=torch.float64, device="cuda")
    box = torch.tensor(
        [[length, 0.0, 0.0, 0.0, length, 0.0, 0.0, 0.0, length]],
        dtype=torch.float64,
    )
    return coord, charge, mask, n_node, total, box


def _kernel() -> SOGKernel:
    return SOGKernel(
        amplitude=[0.7, -0.2, 0.1],
        width=[0.8, 1.7, 3.2],
        precision="float32",
    )


def _evaluate(mesh_backend: str):
    coord, charge, mask, n_node, total, box = _system()
    return latent_charge_sum(
        coord,
        charge,
        mask,
        n_node,
        total,
        kernel=_kernel(),
        channel=0,
        prefactor=COULOMB,
        box=box,
        order=6,
        spacing=0.3,
        mesh_backend=mesh_backend,
    )


def test_midtown_matches_bspline() -> None:
    """The two mesh backends agree on energy, gradient, force and virial."""
    reference = _evaluate("bspline")
    midtown = _evaluate("midtown")
    names = ("atom_energy", "charge_gradient", "force", "atom_virial", "virial")
    for name, ref, got in zip(names, reference, midtown, strict=False):
        ref = ref.detach().to(torch.float64)
        got = got.detach().to(torch.float64)
        scale = max(float(ref.abs().max()), 1e-12)
        # Both are meshes: they agree to the mesh accuracy of the 0.3 A grid
        # (the measured spread is below 1.5e-3 at this spacing).
        error = float((ref - got).abs().max()) / scale
        assert error < 5e-3, f"{name}: relative error {error:.3e}"


def test_midtown_matches_bspline_on_a_tilted_cell() -> None:
    """A triclinic cell runs in fractional coordinates with the metric k."""
    coord, charge, mask, n_node, total, _ = _system()
    tilted = torch.tensor(
        [[9.0, 0.0, 0.0, 0.8, 9.0, 0.0, 0.3, 0.2, 9.0]], dtype=torch.float64
    )
    common = {
        "kernel": _kernel(),
        "channel": 0,
        "prefactor": COULOMB,
        "box": tilted,
        "order": 6,
        "spacing": 0.25,
    }
    reference = latent_charge_sum(
        coord, charge, mask, n_node, total, mesh_backend="bspline", **common
    )
    midtown = latent_charge_sum(
        coord, charge, mask, n_node, total, mesh_backend="midtown", **common
    )
    names = ("atom_energy", "charge_gradient", "force", "atom_virial", "virial")
    for name, ref, got in zip(names, reference, midtown, strict=False):
        ref = ref.detach().to(torch.float64)
        got = got.detach().to(torch.float64)
        scale = max(float(ref.abs().max()), 1e-12)
        error = float((ref - got).abs().max()) / scale
        assert error < 5e-3, f"{name}: relative error {error:.3e}"


def test_midtown_rejects_a_multi_channel_charge() -> None:
    coord, charge, mask, n_node, total, box = _system()
    two = torch.cat([charge, charge * 0.5], dim=1)
    with pytest.raises(RuntimeError, match="one channel"):
        latent_charge_sum(
            coord,
            two,
            mask,
            n_node,
            total,
            kernel=_kernel(),
            channel=0,
            prefactor=COULOMB,
            box=box,
            order=6,
            spacing=0.6,
            mesh_backend="midtown",
        )


def test_unknown_mesh_backend_is_rejected() -> None:
    coord, charge, mask, n_node, total, box = _system()
    with pytest.raises(ValueError, match="mesh backend"):
        latent_charge_sum(
            coord,
            charge,
            mask,
            n_node,
            total,
            kernel=_kernel(),
            channel=0,
            prefactor=COULOMB,
            box=box,
            order=6,
            spacing=0.6,
            mesh_backend="nope",
        )


def _evaluate_midtown(spacing: float, force_mode: str):
    coord, charge, mask, n_node, total, box = _system()
    return latent_charge_sum(
        coord,
        charge,
        mask,
        n_node,
        total,
        kernel=_kernel(),
        channel=0,
        prefactor=COULOMB,
        box=box,
        order=6,
        spacing=spacing,
        mesh_backend="midtown",
        force_mode=force_mode,
    )


def test_midtown_force_modes_agree_on_a_fine_mesh() -> None:
    """AD and ik differentiate the same mesh; they converge as it refines.

    AD is the exact gradient of the mesh energy (one order lower in the mesh
    spacing); ik differentiates the potential in Fourier space (one order
    higher).  On a fine mesh both reproduce the same force, so the two branches
    of the CUDA gather must agree there -- a mismatch is a broken branch, not
    an accuracy difference.
    """
    ad = _evaluate_midtown(0.2, "ad")
    ik = _evaluate_midtown(0.2, "ik")
    # Energy, the charge gradient and the virial come from the same potential
    # mesh in both modes; they agree to float32 rounding even though the two
    # gathers evaluate the weights through different code paths.
    for name, ref, got in zip(
        ("atom_energy", "charge_gradient", "virial"),
        ad[:2] + ad[4:],
        ik[:2] + ik[4:],
        strict=False,
    ):
        ref = ref.detach().to(torch.float64)
        got = got.detach().to(torch.float64)
        scale = max(float(ref.abs().max()), 1e-12)
        error = float((ref - got).abs().max()) / scale
        assert error < 1e-6, f"{name} must not depend on force_mode: {error:.3e}"
    f_ad = ad[2].detach().to(torch.float64)
    f_ik = ik[2].detach().to(torch.float64)
    scale = max(float(f_ad.abs().max()), 1e-12)
    error = float((f_ad - f_ik).abs().max()) / scale
    assert error < 2e-3, f"force: AD vs ik differ by {error:.3e}"


def test_midtown_ik_is_more_accurate_than_ad() -> None:
    """The ik mode gains an order over AD against the B-spline reference."""
    coord, charge, mask, n_node, total, box = _system()
    common = {
        "kernel": _kernel(),
        "channel": 0,
        "prefactor": COULOMB,
        "box": box,
        "order": 6,
        "spacing": 0.6,
    }
    reference = (
        latent_charge_sum(
            coord, charge, mask, n_node, total, mesh_backend="bspline", **common
        )[2]
        .detach()
        .to(torch.float64)
    )
    scale = max(float(reference.abs().max()), 1e-12)
    errors = {}
    for mode in ("ad", "ik"):
        force = (
            latent_charge_sum(
                coord,
                charge,
                mask,
                n_node,
                total,
                mesh_backend="midtown",
                force_mode=mode,
                **common,
            )[2]
            .detach()
            .to(torch.float64)
        )
        errors[mode] = float((force - reference).abs().max()) / scale
    assert errors["ik"] < errors["ad"], (
        f"ik error {errors['ik']:.3e} should be below AD {errors['ad']:.3e}"
    )


def test_ik_requires_the_midtown_backend() -> None:
    coord, charge, mask, n_node, total, box = _system()
    with pytest.raises(ValueError, match="midtown or quads"):
        latent_charge_sum(
            coord,
            charge,
            mask,
            n_node,
            total,
            kernel=_kernel(),
            channel=0,
            prefactor=COULOMB,
            box=box,
            order=6,
            spacing=0.6,
            mesh_backend="bspline",
            force_mode="ik",
        )


def test_unknown_force_mode_is_rejected() -> None:
    coord, charge, mask, n_node, total, box = _system()
    with pytest.raises(ValueError, match="force mode"):
        latent_charge_sum(
            coord,
            charge,
            mask,
            n_node,
            total,
            kernel=_kernel(),
            channel=0,
            prefactor=COULOMB,
            box=box,
            order=6,
            spacing=0.6,
            mesh_backend="midtown",
            force_mode="nope",
        )


@pytest.mark.parametrize("mesh_backend", ["midtown", "quads", "bspline"])
@pytest.mark.parametrize("order", [4, 6])
def test_slab_solve_matches_replicated(monkeypatch, mesh_backend, order) -> None:
    """The z-slab distributed solve reproduces the replicated solve in every
    spline mode.

    The slab backend carries the CubeS2, QuadS and cardinal-B-spline stencils.
    At a single rank it skips NCCL (the two all-to-alls are the identity), so
    this exercises the spline-specific compact half-spectrum green table
    (`build_green_slab_half`) and the two-transpose FFT without MPI.  Both
    sides run the same single-precision pipeline (float32 mesh, 64-bit
    fixed-point spread, packed `float2` green table) now that the slab shares
    the replicated solve's precision contract; the CubeS2 case pins BOTH sides
    to Form B so the comparison isolates the spline machinery rather than the
    Form-A window measurement (which has its own parity run).  QuadS inverts
    the window exactly and B-spline inverts its assignment factor, both on
    either side.  Any spline-dependent slip in the green table -- the one silent
    failure mode of this path -- shows up as a mismatch here.
    """
    coord, charge, mask, n_node, total, box = _system()
    common = {
        "kernel": _kernel(),
        "channel": 0,
        "prefactor": COULOMB,
        "box": box,
        "order": order,
        "spacing": 0.4,
        "mesh_backend": mesh_backend,
    }
    monkeypatch.delenv("SOG_MIDTOWN_SLAB", raising=False)
    if mesh_backend == "midtown":
        monkeypatch.setenv("SOG_MIDTOWN_FORM_A", "0")
    reference = latent_charge_sum(coord, charge, mask, n_node, total, **common)
    monkeypatch.setenv("SOG_MIDTOWN_SLAB", "1")
    slab = latent_charge_sum(coord, charge, mask, n_node, total, **common)
    names = ("atom_energy", "charge_gradient", "force", "atom_virial", "virial")
    for name, ref, got in zip(names, reference, slab, strict=False):
        ref = ref.detach().to(torch.float64)
        got = got.detach().to(torch.float64)
        scale = max(float(ref.abs().max()), 1e-12)
        error = float((ref - got).abs().max()) / scale
        assert error < 1e-3, f"{mesh_backend}-{order} {name}: {error:.3e}"
