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
