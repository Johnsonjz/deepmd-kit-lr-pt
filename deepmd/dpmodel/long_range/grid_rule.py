# SPDX-License-Identifier: LGPL-3.0-or-later
"""Mesh spacing from the measured force-relative error law.

The mesh error of a periodic frame has three terms,

    eps_mesh = eps_tail(k_max) + (C_assign + C_infl) (Delta / sigma_min)**p,

where ``sigma_min`` is the width of the narrowest Gaussian of the SOG kernel,
``C_assign`` is the assignment (interpolation/alias) residual of the spline and
``C_infl`` is the residual of the influence function -- identically zero when
the window symbol is inverted exactly (the B-spline divides by the analytic
``|D_d(m)|^-2``), nonzero for the variance subtraction of the CubeS2 path,
which is why the two Green forms carry different constants.

The exponent is not always ``2 nu``: the force needs the *derivative* of the
assignment window, so a window that is only ``C0`` loses one order,

    p = 2 nu       energy metric
    p = 2 nu - 1   force metric, C0 window

The midtown paper (Predescu, Bergdorf & Shaw, J. Chem. Phys. 153, 224117
(2020), Table I) lists CubeS2 order 4 as ``C2`` and order 6 as ``C0``; the
deployed-operator audit reproduces that (force slopes 3.35-3.40 / 6.23-6.36 for
the B-spline of orders 4 and 6, 3.16-3.21 / 5.32-5.34 for CubeS2 of orders 4
and 6, against 4.3 / 7.1 for the energy of the same meshes).  **Both** CubeS2
orders lose about an order in the force, not only the C0 order-6 window, and
the B-spline of order 6 loses more than one -- which is why CubeS2 Form A is
5-50x more accurate in ENERGY and 1.2-9.6x less accurate in FORCE than the
B-spline at the same grid.

The constants below are FIXED-CHARGE FORCE-RELATIVE values, measured against a
converged direct k-sum of the same kernel (analytic force), on TWO trained
kernels (sigma_min = 1.13 and 2.00 A) and 32 independent 64-atom
configurations each, with both backends weight differentiated and the abscissa
taken as the spacing the mesh actually has.  ``C`` is refit at the shipped
``p`` on the worse of the two kernels, so a grid meets the target for both.
The B-spline order dependence between the measured orders 4 and 6 is
interpolated in log, and anything above 6 is extrapolated.  A caller that needs
a number for its own kernel should re-derive it -- the point of this module is
that the *shape* of the rule is enforced, not that one constant is universal.

The CubeS2 backend has a second force mode, ``force_mode="ik"``, that
differentiates the potential in Fourier space instead of differentiating the
assignment weights.  It removes the extra order that the C0 window costs AD, so
the force exponent rises to ``2 nu`` and the force constant falls by an order of
magnitude; :data:`IK_FORCE_RULE` holds that law.  Because the force and the
energy then share one exponent, the energy can be the binding constraint, so
:func:`spacing` sizes an ik grid by the worse of :data:`IK_FORCE_RULE` and
:data:`ENERGY_RULE`.  AD and the B-spline keep their force-only rule: their
force exponent is strictly lower than the energy's, so the force already
dominates every grid they size.

The measured law alone sizes the grid.  There is deliberately **no** Nyquist
ceiling ``PME_SPACING_FRACTION * pi / k_max`` on top of it: the law is measured
at the target spacings themselves, so the aliasing of the assignment window at
that spacing is already inside the measurement, and a high spline order then
buys accuracy at the same grid rather than shrinking it.  The one cap left is
the CubeS2 validity ceiling of :func:`spacing`.
"""

import math

#: Force-relative accuracy target of the spacing rule.
DEFAULT_ACCURACY = 1e-4

#: CubeS2 optimal assignment parameters (midtown paper SI, notebook 2).
CUBES2_XI = {4: 0.5773502691896258, 6: 0.6503998764035732}

#: ``(backend, order) -> (C, p, provenance)``, force-relative, measured against
#: a converged direct k-sum with an analytic force on two trained SOG kernels
#: (sigma_min = 1.13 and 2.00 A) and 32 independent 64-atom configurations each,
#: both backends weight differentiated (AD), abscissa = the spacing the mesh
#: actually has (Delta = L/n).
#:
#: ``p`` is the measured force slope of the shipped operator, kept where it was
#: (3.5 / 6.0 for the B-spline of orders 4 and 6, 3.0 / 5.0 for CubeS2); the
#: slopes measured across the two kernels are 3.35-3.40, 6.23-6.36, 3.16-3.21
#: and 5.32-5.34, so every entry is within 0.2 of both kernels.  **Both CubeS2
#: orders lose about an order in the force** -- not only the C0 order-6 window --
#: whereas the energy keeps 2 nu.
#:
#: ``C`` is refit at the shipped ``p`` on the WORSE of the two kernels and is
#: therefore the value a grid must satisfy for both.  Measured at p = 3.5 /
#: 6.0 / 3.0 / 5.0: B-spline-4 8.49e-2 (sigma 2.00) vs 1.70e-1 (sigma 1.13);
#: B-spline-6 2.56e-2 vs 5.12e-2; CubeS2-4 8.88e-2 vs 1.96e-1; CubeS2-6
#: 6.76e-2 vs 1.52e-1.  Every constant is larger than the previous ones
#: (1.17e-1 / 3.34e-2 / 1.66e-1 / 1.24e-1), i.e. the grids had been 4-10 % too
#: coarse for the sigma_min = 1.13 kernel.
FORCE_RULE = {
    ("bspline", 4): (1.70e-1, 3.5, "measured"),
    ("bspline", 5): (9.32e-2, 4.5, "interpolated"),
    ("bspline", 6): (5.12e-2, 6.0, "measured"),
    ("bspline", 7): (2.82e-2, 7.0, "extrapolated"),
    ("bspline", 8): (1.55e-2, 8.0, "extrapolated"),
    ("midtown", 4): (1.96e-1, 3.0, "measured"),
    ("midtown", 6): (1.52e-1, 5.0, "measured"),
    ("quads", 4): (1.71e-1, 3.5, "measured"),
    ("quads", 6): (9.29e-2, 5.5, "measured"),
}

#: ``(backend, order) -> (C, p, provenance)``, force-relative, for the CubeS2
#: backend with Fourier differentiation (``force_mode="ik"``).  The ik force is
#: the inverse transform of ``i k_a phi_hat``, so it differentiates the mesh
#: potential spectrally and does NOT differentiate the assignment window: the
#: force keeps the energy's exponent ``2 nu`` and the constant is an order of
#: magnitude below the AD one.  Measured on the same two kernels and ensemble as
#: :data:`FORCE_RULE` through the shipped operator; the slopes are 4.37-4.38
#: (order 4) and 6.59-6.64 (order 6), rounded down to ``2 nu`` = 4.0 / 6.0, and
#: ``C`` is refit at that ``p`` on the worse kernel (1.890e-2 / 1.830e-2).  The
#: two kernels agree to <1 %, so unlike AD the ik constant is nearly universal.
IK_FORCE_RULE = {
    ("midtown", 4): (1.89e-2, 4.0, "measured"),
    ("midtown", 6): (1.83e-2, 6.0, "measured"),
    ("quads", 4): (1.55e-2, 4.0, "measured"),
    ("quads", 6): (9.92e-3, 6.0, "measured"),
}

#: ``(backend, order) -> (C, p, provenance)`` for the ENERGY relative error,
#: ``eps_E = C (Delta / sigma_min)**p``.  Only the ``ik`` force mode consults it
#: (there the energy and the force share an exponent and either can bind); the
#: AD and B-spline grids stay force-only.  Same kernels and ensemble as
#: :data:`FORCE_RULE`; ``p`` is ``2 nu`` = 4.0 / 6.0 (the measured energy slopes
#: are 4.28-5.09 / 6.40-7.18, above ``2 nu``, so the rounding is conservative)
#: and ``C`` is refit at it on the worse kernel.  Measured constants: B-spline-4
#: 7.336e-1 (sigma 2.00) vs 5.390e-1 (1.13); B-spline-6 1.577e-1 vs 1.145e-1;
#: CubeS2-4 5.117e-2 vs 4.731e-2; CubeS2-6 4.596e-2 vs 5.835e-2.
ENERGY_RULE = {
    ("bspline", 4): (7.34e-1, 4.0, "measured"),
    ("bspline", 5): (3.40e-1, 5.0, "interpolated"),
    ("bspline", 6): (1.58e-1, 6.0, "measured"),
    ("bspline", 7): (7.28e-2, 7.0, "extrapolated"),
    ("bspline", 8): (3.37e-2, 8.0, "extrapolated"),
    ("midtown", 4): (5.12e-2, 4.0, "measured"),
    ("midtown", 6): (5.84e-2, 6.0, "measured"),
    ("quads", 4): (4.54e-2, 4.0, "measured"),
    ("quads", 6): (2.48e-2, 6.0, "measured"),
}

#: The B-spline constant falls by this factor per additional order (measured
#: 0.170 -> 0.0512 between orders 4 and 6 at their shipped exponents).
_BSPLINE_C4 = 1.70e-1
_BSPLINE_P4 = 3.5
_BSPLINE_C_RATIO = 1.82
#: Energy fall-off of the B-spline, 0.734 -> 0.158 between orders 4 and 6.
_BSPLINE_E_C4 = 7.34e-1
_BSPLINE_E_C_RATIO = 2.16


def rule(backend: str, order: int, force_mode: str = "ad") -> tuple[float, int, str]:
    """``(C, p, provenance)`` for a backend, spline order and force mode.

    ``force_mode="ad"`` (the default) is the weight-differentiated force; for
    the midtown backend ``force_mode="ik"`` takes :data:`IK_FORCE_RULE`, which
    is only defined there.
    """
    key = (str(backend), int(order))
    if str(force_mode) == "ik":
        if str(backend) not in ("midtown", "quads"):
            raise ValueError(
                "`force_mode='ik'` (Fourier differentiation) is only defined "
                "for the midtown and quads backends"
            )
        if key not in IK_FORCE_RULE:
            raise ValueError(
                f"no ik grid rule for the {backend!r} backend at spline order "
                f"{order}; the midtown mesh has a CubeS2 stencil of order 4 or "
                "6 only"
            )
        return IK_FORCE_RULE[key]
    if key in FORCE_RULE:
        return FORCE_RULE[key]
    if backend == "bspline":
        c = _BSPLINE_C4 * _BSPLINE_C_RATIO ** -(int(order) - 4)
        p = _BSPLINE_P4 + (int(order) - 4)
        return c, p, "extrapolated"
    raise ValueError(
        f"no grid rule for the {backend!r} backend at spline order {order}; "
        "the midtown mesh has a CubeS2 stencil of order 4 or 6 only"
    )


def energy_rule(backend: str, order: int) -> tuple[float, int, str]:
    """``(C, p, provenance)`` for the energy-relative error law."""
    key = (str(backend), int(order))
    if key in ENERGY_RULE:
        return ENERGY_RULE[key]
    if backend == "bspline":
        c = _BSPLINE_E_C4 * _BSPLINE_E_C_RATIO ** -(int(order) - 4)
        return c, int(order), "extrapolated"
    raise ValueError(
        f"no energy rule for the {backend!r} backend at spline order {order}"
    )


def spacing(
    backend: str,
    order: int,
    sigma_min: float,
    k_max: float,
    accuracy: float = DEFAULT_ACCURACY,
    force_mode: str = "ad",
) -> float:
    """Largest mesh spacing, in angstrom, that reaches ``accuracy``.

    ``sigma_min`` is the width of the narrowest SOG Gaussian and ``k_max`` the
    kernel's wave-number cutoff.  The measured law alone sizes the grid; there
    is no extra Nyquist ceiling (the law is measured at the 1e-4 spacings
    themselves, where aliasing of the window is already part of the measurement,
    so a high spline order buys accuracy instead of a coarser mesh).  For the
    midtown backend the result is capped by the CubeS2 validity ceiling
    ``sigma_min / (sqrt(2) xi)``, beyond which the variance subtraction has an
    on-grid variance that is no longer positive.

    ``accuracy`` is the target of the **force** law of :func:`rule`. For
    ``force_mode="ik"`` the energy and the force share the exponent ``2 nu``,
    and the energy constant is the larger of the two, so the grid is sized by
    whichever law is binding (:data:`ENERGY_RULE` versus :data:`IK_FORCE_RULE`);
    a caller that only needs the force keeps the target, and a caller that also
    needs the energy gets it at the same target. AD and the B-spline are sized
    by the force alone.
    """
    c, p, _ = rule(backend, order, force_mode)
    if not (accuracy > 0.0) or not math.isfinite(accuracy):
        raise ValueError(f"accuracy must be finite and positive, got {accuracy}")
    d_table = float(sigma_min) * (float(accuracy) / c) ** (1.0 / float(p))
    if str(force_mode) == "ik":
        ce, pe, _ = energy_rule(backend, order)
        d_energy = float(sigma_min) * (float(accuracy) / ce) ** (1.0 / float(pe))
        d_table = min(d_table, d_energy)
    d = d_table
    if str(backend) == "midtown":
        xi = CUBES2_XI.get(int(order))
        if xi is not None:
            d = min(d, 0.95 * float(sigma_min) / (math.sqrt(2.0) * xi))
    return d
