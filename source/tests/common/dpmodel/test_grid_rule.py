# SPDX-License-Identifier: LGPL-3.0-or-later
"""The mesh-spacing rule that turns a target accuracy into a grid.

The constants here are the single source of truth for the in-graph operator:
``deepmd/dpmodel/long_range/grid_rule.py`` sizes the mesh, and the operator
only rounds the extent up to a transformable size.  The LAMMPS plugin carries
its own transcription in ``source/lmp/sog.cpp``; these tests pin the values so
that a change on either side is visible.
"""

import math

import pytest

from deepmd.dpmodel.long_range import grid_rule


class TestRuleTable:
    """The measured table itself."""

    def test_force_exponents_lose_one_order_for_the_window(self):
        # A C0 assignment window loses one order in the FORCE; the B-spline
        # keeps its order because order p is C^(p-2).
        assert grid_rule.rule("midtown", 6) == (1.52e-1, 5.0, "measured")
        assert grid_rule.rule("midtown", 4) == (1.96e-1, 3.0, "measured")
        assert grid_rule.rule("bspline", 6) == (5.12e-2, 6.0, "measured")
        assert grid_rule.rule("bspline", 4)[:2] == (1.70e-1, 3.5)

    def test_bpline_interpolates_and_extrapolates(self):
        c5, p5, prov5 = grid_rule.rule("bspline", 5)
        c7, p7, prov7 = grid_rule.rule("bspline", 7)
        c8, p8, prov8 = grid_rule.rule("bspline", 8)
        assert (c5, p5, prov5) == (9.32e-2, 4.5, "interpolated")
        assert (c7, p7, prov7) == (2.82e-2, 7.0, "extrapolated")
        assert (c8, p8, prov8) == (1.55e-2, 8.0, "extrapolated")

    def test_no_rule_for_an_order_the_midtown_mesh_does_not_have(self):
        with pytest.raises(ValueError, match="order 4 or 6 only"):
            grid_rule.rule("midtown", 8)


class TestIkForceMode:
    """The Fourier-differentiated (``force_mode="ik"``) rule for the midtown mesh.

    The ik force is the inverse transform of ``i k_a phi_hat``; it does not
    differentiate the assignment window, so the force keeps the energy's
    exponent ``2 nu`` and its constant is an order of magnitude below AD.
    """

    sigma, kmax = 2.0, 3.0349  # the reference water kernel

    def test_ik_rule_carries_the_energy_exponent(self):
        assert grid_rule.rule("midtown", 4, "ik") == (1.89e-2, 4.0, "measured")
        assert grid_rule.rule("midtown", 6, "ik") == (1.83e-2, 6.0, "measured")

    def test_ik_is_midtown_only(self):
        with pytest.raises(ValueError, match="only defined for the midtown"):
            grid_rule.rule("bspline", 6, "ik")

    def test_the_default_force_mode_is_ad(self):
        # Every existing caller keeps the weight-differentiated rule.
        assert grid_rule.rule("midtown", 6) == grid_rule.rule("midtown", 6, "ad")

    def test_spacing_uses_the_energy_law_for_every_mode(self):
        # The grid is sized by the energy law, which is independent of the
        # force mode, so AD and ik give the same spacing.
        for order in (4, 6):
            ce, pe, _ = grid_rule.energy_rule("midtown", order)
            d_energy = self.sigma * (1e-4 / ce) ** (1.0 / pe)
            for mode in ("ad", "ik"):
                d = grid_rule.spacing(
                    "midtown", order, self.sigma, self.kmax, 1e-4, force_mode=mode
                )
                assert d == pytest.approx(d_energy, rel=1e-9)

    def test_spacing_is_independent_of_the_force_mode(self):
        # The energy law does not care how the force is differentiated.
        for order in (4, 6):
            d_ad = grid_rule.spacing("midtown", order, self.sigma, self.kmax, 1e-4)
            d_ik = grid_rule.spacing(
                "midtown", order, self.sigma, self.kmax, 1e-4, force_mode="ik"
            )
            assert d_ik == pytest.approx(d_ad, rel=1e-12)

    def test_energy_rule_rounds_down_to_2nu(self):
        assert grid_rule.energy_rule("midtown", 4)[:2] == (3.81e-3, 4.0)
        assert grid_rule.energy_rule("midtown", 6)[:2] == (4.03e-3, 6.0)

    @pytest.mark.parametrize(
        "order,spacing,points",
        [(4, 0.805005, 152), (6, 1.080138, 114)],
    )
    def test_ik_reference_kernel_grid(self, order, spacing, points):
        d = grid_rule.spacing("midtown", order, 2.0000, 3.0349, 1e-4, force_mode="ik")
        assert d == pytest.approx(spacing, abs=1e-4)
        assert math.ceil(122.31 / d) == points


class TestSpacing:
    """Inversion of ``eps = C (Delta / sigma_min) ** p``."""

    sigma, kmax = 2.0, 3.0349  # the reference water kernel

    def test_inverts_the_energy_law_inside_the_validity_band(self):
        for backend, order in (
            ("bspline", 4),
            ("bspline", 5),
            ("bspline", 6),
            ("midtown", 4),
            ("midtown", 6),
        ):
            ce, pe, _ = grid_rule.energy_rule(backend, order)
            d = grid_rule.spacing(backend, order, self.sigma, self.kmax, 1e-4)
            assert ce * (d / self.sigma) ** pe == pytest.approx(1e-4, rel=1e-9)

    def test_a_tighter_target_is_a_finer_mesh(self):
        loose = grid_rule.spacing("midtown", 6, self.sigma, self.kmax, 1e-3)
        tight = grid_rule.spacing("midtown", 6, self.sigma, self.kmax, 1e-5)
        assert tight < loose

    def test_the_measured_law_alone_sizes_the_grid(self):
        # There is no extra Nyquist ceiling: every backend, including the
        # high-order B-spline, uses exactly the spacing its energy law gives.
        for backend, order in (("bspline", 6), ("bspline", 8), ("midtown", 6)):
            ce, pe, _ = grid_rule.energy_rule(backend, order)
            d = grid_rule.spacing(backend, order, self.sigma, self.kmax, 1e-4)
            assert ce * (d / self.sigma) ** pe == pytest.approx(1e-4, rel=1e-9)

    def test_midtown_is_capped_by_the_cubes2_validity_ceiling(self):
        # Beyond sigma_min / (sqrt(2) xi) the variance subtraction of the
        # CubeS2 window has a non-positive on-grid variance, so the validity
        # cap binds on a loose target.
        xi = grid_rule.CUBES2_XI[6]
        cap = 0.95 * self.sigma / (math.sqrt(2.0) * xi)
        d = grid_rule.spacing("midtown", 6, self.sigma, 0.5, 0.2)
        assert d == pytest.approx(cap)

    def test_rejects_a_target_that_is_not_a_positive_number(self):
        for bad in (0.0, -1e-4, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="finite and positive"):
                grid_rule.spacing("bspline", 6, self.sigma, self.kmax, bad)


class TestReferenceKernelGrid:
    """The numbers the shipped rule gives for the reference water kernel.

    Recording them makes a constant change visible in review: the grid, and
    with it the cost of every periodic evaluation, moves with them.
    """

    sigma_min = 2.0000
    k_max = 3.0349
    box = 122.31

    @pytest.mark.parametrize(
        "backend,order,spacing,points",
        [
            ("bspline", 8, 1.42806, 86),
            ("bspline", 6, 0.98494, 125),
            ("midtown", 6, 1.08014, 114),
        ],
    )
    def test_spacing_and_grid(self, backend, order, spacing, points):
        d = grid_rule.spacing(backend, order, self.sigma_min, self.k_max, 1e-4)
        assert d == pytest.approx(spacing, abs=1e-4)
        assert math.ceil(self.box / d) == points

    def test_at_equal_energy_accuracy_midtown_needs_the_fewer_points(self):
        # At equal ENERGY accuracy the midtown order-6 constant is smaller than
        # the B-spline's, so its grid is coarser (fewer points).
        d_b = grid_rule.spacing("bspline", 6, self.sigma_min, self.k_max)
        d_m = grid_rule.spacing("midtown", 6, self.sigma_min, self.k_max)
        assert d_m > d_b
        ratio = (d_m / d_b) ** 3
        assert 1.2 < ratio < 1.4
