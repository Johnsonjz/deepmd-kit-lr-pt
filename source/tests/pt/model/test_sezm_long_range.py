# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests of the pt (legacy) latent-charge long-range host on DPA4/SeZM.

The equations and the kernel live in :mod:`deepmd.dpmodel.long_range`; these
tests pin the pt integration: a model builds and serializes with a
``long_range`` section, its force and virial include the reciprocal term, and
the charge network and kernel receive gradients.
"""

import unittest

import numpy as np
import torch

from deepmd.dpmodel.long_range.latent_charge import (
    RCUT_TO_SIGMA,
    SOG_DEFAULT_B,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    K_GRID,
    build_topologies,
)
from deepmd.pt.long_range import (
    LatentChargeLongRange,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.model.task.invar_fitting import (
    InvarFitting,
)


def _config(
    *,
    periodic: bool = True,
    mode: str = "fixed",
    k_grid_bounds: list[int] | None = None,
    kspace_tolerance: float = 1e-12,
) -> dict:
    long_range = {
        "type": "latent_charge",
        "channels": 1,
        "charge_net": {"neuron": [4], "precision": "float64"},
        "kernel": {
            "type": "sog",
            "n_gaussian": 3,
            "base_width": 2.18,
            "width_ratio": 1.6,
            "precision": "float64",
        },
        "periodic": periodic,
        "k_grid_mode": mode,
        "kspace_tolerance": kspace_tolerance,
    }
    if k_grid_bounds is not None:
        long_range["k_grid_bounds"] = k_grid_bounds
    return {
        "type": "dpa4",
        "type_map": ["Ni", "O"],
        "descriptor": {
            "type": "dpa4",
            "rcut": 4.0,
            "sel": 8,
            "channels": 4,
            "n_radial": 3,
            "lmax": 1,
            "mmax": 1,
            "n_blocks": 1,
            "precision": "float64",
            "seed": 7,
            "random_gamma": False,
        },
        "fitting_net": {"type": "dpa4_ener", "neuron": [4], "precision": "float64"},
        "long_range": long_range,
    }


def _system() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coord = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.0, 1.5, 0.0],
                [1.5, 1.5, 0.0],
            ]
        ],
        dtype=torch.float64,
        device="cpu",
    )
    atype = torch.tensor([[0, 1, 0, 1]], device="cpu")
    box = torch.tensor(
        [[6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 6.0]],
        dtype=torch.float64,
        device="cpu",
    )
    return coord, atype, box


class TestSeZMLongRange(unittest.TestCase):
    def _model(self, **kwargs) -> torch.nn.Module:
        model = get_model(_config(**kwargs)).to("cpu")
        model.eval()
        return model

    def test_potential_satisfies_half_q_phi(self) -> None:
        class _Descriptor:
            def get_ntypes(self):
                return 2

            def get_dim_out(self):
                return 4

            def mixed_types(self):
                return True

        fitting = InvarFitting(
            var_name="energy",
            ntypes=2,
            dim_descrpt=4,
            dim_out=1,
            neuron=[4],
            type_map=["A", "B"],
            mixed_types=True,
        ).to("cpu")
        cfg = {
            "type": "latent_charge",
            "channels": 1,
            "charge_net": {
                "neuron": [4],
                "precision": "float64",
                "bias_atom_q": [1.0, -1.0],
            },
            "kernel": {
                "type": "sog",
                "n_gaussian": 3,
                "base_width": 2.18,
                "width_ratio": 1.6,
            },
            "periodic": False,
            "k_grid_mode": "ragged",
        }
        module = LatentChargeLongRange.from_config(
            cfg, _Descriptor(), fitting, fitting_seed=1
        ).to("cpu")
        for param in module.charge_net.parameters():
            param.data.zero_()
        module.charge_net.bias_atom_e.data = torch.tensor(
            [[1.0], [-1.0]],
            dtype=module.charge_net.bias_atom_e.dtype,
            device="cpu",
        )
        coord = torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64, device="cpu"
        )
        atype = torch.tensor([0, 1], device="cpu")
        mask = torch.ones(2, dtype=torch.bool, device="cpu")
        n_node = torch.tensor([2], dtype=torch.long, device="cpu")
        desc = torch.zeros(2, 4, dtype=torch.float64, device="cpu")
        topologies = build_topologies(
            module.required_topologies(),
            coord,
            n_node,
            None,
            node_mask=mask,
            atype=atype,
            topology_params=module.topology_params(),
        )
        energy, phi = module.call_graph(
            desc, atype, mask, n_node, topologies, want_potential=True
        )
        charge = module.charge_net.call_graph(desc, atype)["latent_charge"]
        torch.testing.assert_close(energy.reshape(-1), 0.5 * (charge * phi).sum(dim=-1))
        kernel = module.kernel(
            torch.tensor([1.5**2], dtype=torch.float64, device="cpu")
        ).item()
        torch.testing.assert_close(
            phi.reshape(-1),
            torch.tensor(
                [-14.3996454784255 * kernel, 14.3996454784255 * kernel],
                dtype=torch.float64,
                device="cpu",
            ),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "compile parity needs a GPU")
    def test_compiled_static_grid_matches_eager(self) -> None:
        cfg = _config(mode="ellipsoid", k_grid_bounds=[4, 4, 4])
        cfg["use_compile"] = True
        model = get_model(cfg).to("cuda:0")
        model.train()
        model.use_compile = False
        device = torch.device("cuda:0")
        coord = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [1.5, 1.5, 0.0]]],
            dtype=torch.float64,
            device=device,
        )
        atype = torch.tensor([[0, 1, 0, 1]], device=device)
        box = torch.tensor(
            [[6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 6.0]],
            dtype=torch.float64,
            device=device,
        )
        eager = model(coord, atype, box)
        model.use_compile = True
        compiled = model(coord, atype, box)
        self.assertTrue(model.compiled_core_compute_cache)
        torch.testing.assert_close(compiled["energy"], eager["energy"])
        torch.testing.assert_close(compiled["force"], eager["force"])

    def test_legacy_sog_model_type_is_rejected_with_a_migration_hint(self) -> None:
        for legacy in ("sezm_sog", "dpa4_sog", "dpa4c_sog"):
            with self.assertRaisesRegex(ValueError, "convert_sog_to_long_range"):
                get_model({"type": legacy})

    def test_model_builds_and_serializes(self) -> None:
        model = self._model(k_grid_bounds=[3, 3, 3])
        self.assertIsNotNone(model.atomic_model.long_range)
        serialized = model.serialize()
        restored = type(model).deserialize(serialized)
        self.assertIsNotNone(restored.atomic_model.long_range)
        torch.testing.assert_close(
            model.atomic_model.long_range.kernel.amplitude,
            restored.atomic_model.long_range.kernel.amplitude,
        )

    def test_default_sog_width_follows_the_library_rule(self) -> None:
        cfg = _config(k_grid_bounds=[3, 3, 3])
        cfg["long_range"]["kernel"] = {"type": "sog", "n_gaussian": 4}
        model = get_model(cfg).to("cpu")
        kernel = model.atomic_model.long_range.kernel
        widths = torch.exp(kernel.log_width)
        # The geometric initial embedding reaches rcut before the single
        # message-passing block, so the receptive field is (n_blocks + 1)*rcut.
        n_layers = 1 + 1
        sigma = n_layers * 4.0 / RCUT_TO_SIGMA
        expected = sigma * (
            SOG_DEFAULT_B ** torch.arange(4, dtype=torch.float64, device="cpu")
        )
        torch.testing.assert_close(widths, expected)

    def test_ellipsoid_matches_fixed_and_ragged(self) -> None:
        coord, atype, box = _system()
        model = self._model(mode="fixed", k_grid_bounds=[4, 4, 4])
        reference = model(coord, atype, box)
        lr = model.atomic_model.long_range
        for mode in ("ellipsoid", "ragged"):
            lr.k_grid_mode = mode
            out = model(coord, atype, box)
            torch.testing.assert_close(
                out["energy"],
                reference["energy"],
                atol=1e-10,
                rtol=1e-10,
                msg=f"energy under {mode}",
            )
            torch.testing.assert_close(
                out["force"],
                reference["force"],
                atol=1e-8,
                rtol=1e-8,
                msg=f"force under {mode}",
            )

    def test_metric_ellipsoid_matches_ragged_on_tilted_cell(self) -> None:
        coord, atype, _ = _system()
        tilted = torch.tensor(
            [[6.0, 0.0, 0.0, 1.0, 6.0, 0.0, 0.0, 0.0, 6.0]],
            dtype=torch.float64,
            device="cpu",
        )
        model = self._model(mode="ragged", kspace_tolerance=1e-10)
        reference = model(coord, atype, tilted)
        lr = model.atomic_model.long_range
        lr.k_grid_mode = "metric-ellipsoid"
        lr.k_grid_bounds = np.array([6, 6, 6], dtype=np.int64)
        out = model(coord, atype, tilted)
        torch.testing.assert_close(
            out["energy"], reference["energy"], atol=1e-10, rtol=1e-10
        )
        torch.testing.assert_close(
            out["force"], reference["force"], atol=1e-8, rtol=1e-8
        )

    def test_metric_ellipsoid_matches_pme_on_tilted_cell(self) -> None:
        coord, atype, _ = _system()
        tilted = torch.tensor(
            [[6.0, 0.0, 0.0, 1.0, 6.0, 0.0, 0.0, 0.0, 6.0]],
            dtype=torch.float64,
            device="cpu",
        )
        model = self._model(mode="pme", kspace_tolerance=1e-10)
        reference = model(coord, atype, tilted)
        lr = model.atomic_model.long_range
        lr.k_grid_mode = "metric-ellipsoid"
        lr.k_grid_bounds = np.array([12, 12, 12], dtype=np.int64)
        out = model(coord, atype, tilted)
        torch.testing.assert_close(
            out["energy"], reference["energy"], atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            out["force"], reference["force"], atol=1e-5, rtol=1e-5
        )

    def test_ellipsoid_rejects_tilted_cell(self) -> None:
        coord, atype, _ = _system()
        model = self._model(mode="ellipsoid", k_grid_bounds=[4, 4, 4])
        tilted = torch.tensor(
            [[6.0, 0.0, 0.0, 1.0, 6.0, 0.0, 0.0, 0.0, 6.0]],
            dtype=torch.float64,
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "orthogonal"):
            model(coord, atype, tilted)

    def test_charge_bias_is_migrated_to_the_charge_net(self) -> None:
        cfg = _config(k_grid_bounds=[3, 3, 3])
        cfg["long_range"]["charge_net"]["bias_atom_q"] = [-0.6, 0.3]
        model = get_model(cfg).to("cpu")
        bias = model.atomic_model.long_range.charge_net.bias_atom_e
        expected = torch.tensor([[-0.6], [0.3]], dtype=bias.dtype, device="cpu")
        torch.testing.assert_close(bias, expected)

    def test_energy_differs_from_short_range(self) -> None:
        coord, atype, box = _system()
        model = self._model(k_grid_bounds=[3, 3, 3])
        no_lr = self._model(k_grid_bounds=[3, 3, 3])
        no_lr.atomic_model.long_range = None
        e_lr = model(coord, atype, box)["energy"].detach().sum()
        e_sr = no_lr(coord, atype, box)["energy"].detach().sum()
        self.assertNotAlmostEqual(float(e_lr), float(e_sr), places=3)

    def test_periodic_force_matches_finite_difference(self) -> None:
        coord, atype, box = _system()
        model = self._model(k_grid_bounds=[3, 3, 3])
        force = model(coord, atype, box)["force"].detach()
        eps = 1e-5
        for idx in ((0, 0, 0), (0, 1, 0), (0, 2, 1)):
            c_plus = coord.clone()
            c_minus = coord.clone()
            c_plus[idx] += eps
            c_minus[idx] -= eps
            e_plus = model(c_plus, atype, box)["energy"].detach().sum()
            e_minus = model(c_minus, atype, box)["energy"].detach().sum()
            fd = -(e_plus - e_minus) / (2.0 * eps)
            torch.testing.assert_close(
                fd, force[idx], atol=1e-8, rtol=1e-8, msg=f"force at {idx}"
            )

    def test_periodic_virial_matches_finite_difference(self) -> None:
        coord, atype, box = _system()
        model = self._model(k_grid_bounds=[3, 3, 3])
        virial = model(coord, atype, box)["virial"].detach().reshape(3, 3)
        eps = 1e-6
        for a, b in ((0, 0), (1, 1), (2, 2)):
            s_plus = torch.eye(3, dtype=torch.float64, device="cpu")
            s_minus = torch.eye(3, dtype=torch.float64, device="cpu")
            s_plus[a, b] += eps
            s_minus[a, b] -= eps
            # Virial is the strain derivative at fixed fractional coordinates.
            c_plus = (coord.reshape(4, 3) @ s_plus.T).reshape(1, 4, 3)
            c_minus = (coord.reshape(4, 3) @ s_minus.T).reshape(1, 4, 3)
            b_plus = (box.reshape(3, 3) @ s_plus.T).reshape(1, 9)
            b_minus = (box.reshape(3, 3) @ s_minus.T).reshape(1, 9)
            e_plus = model(c_plus, atype, b_plus)["energy"].detach().sum()
            e_minus = model(c_minus, atype, b_minus)["energy"].detach().sum()
            fd = -(e_plus - e_minus) / (2.0 * eps)
            torch.testing.assert_close(
                fd, virial[a, b], atol=1e-8, rtol=1e-8, msg=f"virial at {(a, b)}"
            )

    def test_non_periodic_force_matches_finite_difference(self) -> None:
        coord, atype, _ = _system()
        model = self._model(periodic=False, mode="ragged")
        force = model(coord, atype, None)["force"].detach()
        eps = 1e-5
        for idx in ((0, 0, 0), (0, 1, 0)):
            c_plus = coord.clone()
            c_minus = coord.clone()
            c_plus[idx] += eps
            c_minus[idx] -= eps
            e_plus = model(c_plus, atype, None)["energy"].detach().sum()
            e_minus = model(c_minus, atype, None)["energy"].detach().sum()
            fd = -(e_plus - e_minus) / (2.0 * eps)
            torch.testing.assert_close(
                fd, force[idx], atol=1e-8, rtol=1e-8, msg=f"force at {idx}"
            )

    def _topologies(self, model, coord, atype, box):
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_topologies,
        )

        lr = model.atomic_model.long_range
        n = coord.shape[1]
        n_node = torch.tensor([n], device="cpu")
        node_mask = torch.ones(n, dtype=torch.bool, device="cpu")
        return (
            build_topologies(
                lr.required_topologies(),
                coord.reshape(-1, 3),
                n_node,
                box,
                node_mask=node_mask,
                atype=atype.reshape(-1),
                topology_params=lr.topology_params(),
            ),
            n,
        )

    def test_periodic_potential_is_the_charge_gradient(self) -> None:
        from deepmd.dpmodel.long_range.reciprocal import (
            kspace_energy,
        )

        coord, atype, box = _system()
        model = self._model(k_grid_bounds=[3, 3, 3])
        lr = model.atomic_model.long_range
        topologies, n = self._topologies(model, coord, atype, box)
        charge = torch.tensor(
            [[0.7], [-1.1], [0.4], [0.9]],
            dtype=torch.float64,
            device="cpu",
            requires_grad=True,
        )
        energy, phi = kspace_energy(
            charge, topologies[K_GRID], lr.kernel, want_potential=True
        )
        # E = 1/2 sum_i q_i phi_i, atom by atom.
        self.assertAlmostEqual(
            float(energy.sum() - 0.5 * (charge * phi).sum()), 0.0, places=12
        )
        # phi is the gradient of the energy in the charges.
        grad = torch.autograd.grad(energy.sum(), charge)[0]
        torch.testing.assert_close(phi, grad, atol=1e-10, rtol=1e-10)

    def test_non_periodic_potential_is_the_charge_gradient(self) -> None:
        from deepmd.dpmodel.utils.neighbor_graph import (
            ALL_PAIRS,
        )

        coord, atype, _ = _system()
        model = self._model(periodic=False, mode="ragged")
        lr = model.atomic_model.long_range
        topologies, n = self._topologies(model, coord, atype, None)
        charge = torch.tensor(
            [[0.7], [-1.1], [0.4], [0.9]],
            dtype=torch.float64,
            device="cpu",
            requires_grad=True,
        )
        energy, phi = lr._pair_branch(
            charge, topologies[ALL_PAIRS], n, want_potential=True
        )
        self.assertAlmostEqual(
            float(energy.sum() - 0.5 * (charge * phi).sum()), 0.0, places=12
        )
        grad = torch.autograd.grad(energy.sum(), charge)[0]
        torch.testing.assert_close(phi, grad, atol=1e-10, rtol=1e-10)

    def test_metric_ellipsoid_is_accepted_by_argcheck(self) -> None:
        from deepmd.utils.argcheck import (
            model_args,
        )

        model_args().check_value(_config(mode="metric-ellipsoid"))

    def test_pme_compile_falls_back_with_a_warning(self) -> None:
        coord, atype, box = _system()
        model = self._model(mode="pme")
        model.train()
        model.use_compile = True
        with self.assertLogs(
            "deepmd.pt.model.model.sezm_model", level="WARNING"
        ) as logs:
            model(coord, atype, box)
        self.assertTrue(any("use_compile" in line for line in logs.output))

    def test_k_grid_truncation_warns(self) -> None:
        coord, atype, box = _system()
        model = self._model(mode="ellipsoid", k_grid_bounds=[1, 1, 1])
        with self.assertLogs(
            "deepmd.pt.model.model.sezm_model", level="WARNING"
        ) as logs:
            model(coord, atype, box)
        self.assertTrue(any("truncated" in line for line in logs.output))

    def test_data_statistics_size_the_k_grid(self) -> None:
        """The atomic model forwards the data statistics to the k grid."""
        _, _, box = _system()
        model = self._model(mode="ellipsoid")
        lr = model.atomic_model.long_range
        self.assertFalse(bool(np.any(lr.k_grid_bounds)))
        # An unsized grid cannot build a periodic topology.
        with self.assertRaisesRegex(ValueError, "not sized yet"):
            model.atomic_model.long_range.topology_params()
        model.atomic_model.compute_fitting_input_stat([{"box": box}])
        self.assertTrue(bool(np.any(lr.k_grid_bounds)))
        self.assertEqual(lr.topology_params()[K_GRID]["nk"].shape, (3,))

    def test_spin_model_rejects_long_range(self) -> None:
        cfg = _config(k_grid_bounds=[3, 3, 3])
        cfg["spin"] = {"scheme": "native", "use_spin": [True, False]}
        with self.assertRaisesRegex(NotImplementedError, "spin"):
            get_model(cfg)

    def test_charge_and_kernel_receive_gradients(self) -> None:
        coord, atype, box = _system()
        model = self._model(k_grid_bounds=[3, 3, 3])
        model.train()
        energy = model(coord, atype, box)["energy"].sum()
        energy.backward()
        lr = model.atomic_model.long_range
        charge_grads = [
            p.grad for p in lr.charge_net.parameters() if p.grad is not None
        ]
        self.assertTrue(charge_grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in charge_grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in charge_grads), 0.0)
        for name, param in lr.kernel.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} received no gradient")
            self.assertGreater(float(param.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
