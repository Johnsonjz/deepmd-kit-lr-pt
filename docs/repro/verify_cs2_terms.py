#!/usr/bin/env python3
"""Verify the branch-free four-term form of the CubeS2 weight and gradient.

The CubeS2 assignment window is not separable, but every node weight is a sum
of at most four products of 1-D functions of (tx, ty, tz).  That is the property
the proposed gather rewrite relies on: move the per-axis polynomial work out of
the per-node loop, then contract at most four separable terms per node with no
class branch.

This script checks the property against the reference implementation in
``source/op/pt/long_range/midtown/sog_spline.h`` for every node of both orders,
for the value and all three derivatives, at random sub-cell positions.  It runs
on the CPU and needs no GPU, so it can guard the table generator in CI.

Basis kinds (index into the per-atom 1-D table):

    0  zero          (unused term slot)
    1  E   = t
    2  L111 (order 6) / L (order 4)
    3  L211 (order 6 only; L211 = -L111/4 - 5*L311/2)
    4  L311 (order 6 only)
    5  S6  (order 6 only)
    6  R

A term slot is ``(kind, axis, e)``: the value is ``f_kind(e ? 1-t : t)`` and the
derivative with respect to ``t`` is ``f_kind'(e ? 1-t : t) * (e ? -1 : +1)``.
The per-atom table stores value and derivative for all (kind, axis, e); the
per-node table stores four such slots per axis, with kind 0 for unused slots.

Usage:
    python docs/repro/verify_cs2_terms.py

Exit code 0 if every value and derivative matches to ~1e-15.
"""

from __future__ import annotations

import random
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPLINE = REPO / "source/op/pt/long_range/midtown/sog_spline.h"

XI4 = 0.5773502691896258
XI6 = 0.6503998764035732


def parse_nodes(text: str, name: str) -> list[tuple[int, int, int, int, int, int]]:
    """The ``{dx, dy, dz, cls, sp_axis, sp_is_neg}`` rows of one node table."""
    start = text.index(name)
    end = text.index("};", start)
    rows = re.findall(
        r"\{\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(\d+)\s*,"
        r"\s*(-?\d+)\s*,\s*(-?\d+)\s*\}",
        text[start:end],
    )
    return [tuple(int(v) for v in row) for row in rows]


def basis(xi: float, order: int):
    """Value ``v[kind](t)`` and derivative ``d[kind](t)`` of the 1-D basis."""
    x2 = xi * xi
    x4 = x2 * x2

    def l111(t: float) -> float:
        return (
            t**5 / 12
            - t**4 / 6
            + (10 * x2 - 1) * t**3 / 12
            - (6 * x2 - 1) * t**2 / 6
            + (5 * x4 - x2) * t / 4
            - (3 * x4 - x2) / 6
        )

    def dl111(t: float) -> float:
        return (
            5 * t**4 / 12
            - 2 * t**3 / 3
            + (10 * x2 - 1) * t**2 / 4
            - (6 * x2 - 1) * t / 3
            + (5 * x4 - x2) / 4
        )

    def l311(t: float) -> float:
        return t**5 / 120 + (2 * x2 - 1) * t**3 / 24 + (15 * x4 - 15 * x2 + 4) * t / 120

    def dl311(t: float) -> float:
        return (
            5 * t**4 / 120
            + (2 * x2 - 1) * t**2 / 8
            + (15 * x4 - 15 * x2 + 4) / 120
        )

    def l(t: float) -> float:
        return -0.5 * t**3 + 0.5 * t**2 - (9 * x2 - 2) * t / 6 + 0.5 * x2

    def dl(t: float) -> float:
        return -1.5 * t**2 + t - (9 * x2 - 2) / 6

    def s6(t: float) -> float:
        return -0.5 * t**3 + 0.5 * t**2 - (3 * x2 - 2) * t / 2 + 0.5 * x2

    def ds6(t: float) -> float:
        return -1.5 * t**2 + t - (3 * x2 - 2) / 2

    def r(t: float) -> float:
        return t**3 / 6 + (3 * x2 - 1) * t / 6

    def dr(t: float) -> float:
        return 0.5 * t**2 + (3 * x2 - 1) / 6

    zero = lambda t: 0.0  # noqa: E731
    if order == 4:
        return {0: zero, 1: lambda t: t, 2: l, 6: r}, {
            0: zero,
            1: lambda t: 1.0,
            2: dl,
            6: dr,
        }
    return (
        {
            0: zero,
            1: lambda t: t,
            2: l111,
            3: lambda t: -0.25 * l111(t) - 2.5 * l311(t),
            4: l311,
            5: s6,
            6: r,
        },
        {
            0: zero,
            1: lambda t: 1.0,
            2: dl111,
            3: lambda t: -0.25 * dl111(t) - 2.5 * dl311(t),
            4: dl311,
            5: ds6,
            6: dr,
        },
    )


def term_table(node, order: int):
    """The at-most-four separable terms of one node, as (kind, axis, e) triples."""
    dx, dy, dz, cls, sp, spn = node
    terms: list[list[tuple[int, int, int]]] = []

    def e_of(axis: int) -> int:
        # eta / ex use t when the offset is 1 and 1-t otherwise.
        return 0 if node[axis] == 1 else 1

    def e_eta(axis: int) -> int:
        return e_of(axis)

    def e_refl(axis: int) -> int:
        # A reflected special axis uses 1-t when its sign bit is set.
        return 1 if (spn >> axis) & 1 else 0

    if order == 4:
        if cls == 0:
            for special in range(3):
                terms.append(
                    [(2, b, e_of(b)) if b == special else (1, b, e_of(b)) for b in range(3)]
                )
        else:
            n1, n2 = sorted(((sp + 1) % 3, (sp + 2) % 3))
            terms.append(
                [(6, sp, 1 if spn else 0), (1, n1, e_eta(n1)), (1, n2, e_eta(n2))]
            )
        return terms

    if cls == 0:
        # L111_x ey ez + L111_y ez ex + L111_z ex ey + S6x S6y S6z
        for special in range(3):
            terms.append(
                [(2, b, e_of(b)) if b == special else (1, b, e_of(b)) for b in range(3)]
            )
        terms.append([(5, b, e_of(b)) for b in range(3)])
    elif cls == 1:
        n1, n2 = sorted(((sp + 1) % 3, (sp + 2) % 3))
        terms.append(
            [(3, sp, 1 if spn else 0), (1, n1, e_eta(n1)), (1, n2, e_eta(n2))]
        )
        terms.append(
            [(6, sp, 1 if spn else 0), (5, n1, e_eta(n1)), (5, n2, e_eta(n2))]
        )
    elif cls == 2:
        n1, n2 = sorted(((sp + 1) % 3, (sp + 2) % 3))
        terms.append(
            [(4, sp, 1 if spn else 0), (1, n1, e_eta(n1)), (1, n2, e_eta(n2))]
        )
    elif cls == 3:
        nm = sp
        s1, s2 = sorted(((nm + 1) % 3, (nm + 2) % 3))
        terms.append([(5, nm, e_eta(nm)), (6, s1, 1 if spn & 1 else 0), (6, s2, 1 if spn & 2 else 0)])
    else:
        terms.append([(6, 0, 1 if spn & 1 else 0), (6, 1, 1 if spn & 2 else 0), (6, 2, 1 if spn & 4 else 0)])
    return terms


def reference(node, t, xi: float, order: int):
    """Value and gradient from sog_spline.h's class-based expressions."""
    dx, dy, dz, cls, sp, spn = node
    c = t
    off = [dx, dy, dz]
    v, d = basis(xi, order)

    def eta(a: int) -> float:
        return c[a] if off[a] == 1 else 1.0 - c[a]

    def deta(a: int) -> float:
        return 1.0 if off[a] == 1 else -1.0

    if order == 4:
        if cls == 0:
            ex = c[0] if dx == 1 else 1 - t[0]
            ey = c[1] if dy == 1 else 1 - t[1]
            ez = c[2] if dz == 1 else 1 - t[2]
            sx = 1.0 if dx == 1 else -1.0
            sy = 1.0 if dy == 1 else -1.0
            sz = 1.0 if dz == 1 else -1.0
            lf = v[2]
            dlf = d[2]
            return (
                lf(ex) * ey * ez + lf(ey) * ez * ex + lf(ez) * ex * ey,
                [
                    dlf(ex) * sx * ey * ez + lf(ey) * ez * sx + lf(ez) * sx * ey,
                    lf(ex) * sy * ez + dlf(ey) * sy * ez * ex + lf(ez) * ex * sy,
                    lf(ex) * ey * sz + lf(ey) * sz * ex + dlf(ez) * sz * ex * ey,
                ],
            )
        es = 1 - c[sp] if spn else c[sp]
        ss = -1.0 if spn else 1.0
        n1, n2 = sorted(((sp + 1) % 3, (sp + 2) % 3))
        g = [0.0, 0.0, 0.0]
        g[sp] = d[6](es) * ss * eta(n1) * eta(n2)
        g[n1] = v[6](es) * deta(n1) * eta(n2)
        g[n2] = v[6](es) * eta(n1) * deta(n2)
        return v[6](es) * eta(n1) * eta(n2), g

    if cls == 0:
        ex = c[0] if dx == 1 else 1 - t[0]
        ey = c[1] if dy == 1 else 1 - t[1]
        ez = c[2] if dz == 1 else 1 - t[2]
        sx = 1.0 if dx == 1 else -1.0
        sy = 1.0 if dy == 1 else -1.0
        sz = 1.0 if dz == 1 else -1.0
        lf, dlf = v[2], d[2]
        sf, dsf = v[5], d[5]
        val = (
            lf(ex) * ey * ez
            + lf(ey) * ez * ex
            + lf(ez) * ex * ey
            + sf(ex) * sf(ey) * sf(ez)
        )
        grad = [
            dlf(ex) * sx * ey * ez
            + lf(ey) * ez * sx
            + lf(ez) * sx * ey
            + dsf(ex) * sx * sf(ey) * sf(ez),
            lf(ex) * sy * ez
            + dlf(ey) * sy * ez * ex
            + lf(ez) * ex * sy
            + sf(ex) * dsf(ey) * sy * sf(ez),
            lf(ex) * ey * sz
            + lf(ey) * sz * ex
            + dlf(ez) * sz * ex * ey
            + sf(ex) * sf(ey) * dsf(ez) * sz,
        ]
        return val, grad

    if cls in (1, 2):
        es = 1 - c[sp] if spn else c[sp]
        ss = -1.0 if spn else 1.0
        n1, n2 = sorted(((sp + 1) % 3, (sp + 2) % 3))
        g = [0.0, 0.0, 0.0]
        if cls == 1:
            val = v[3](es) * eta(n1) * eta(n2) + v[6](es) * v[5](eta(n1)) * v[5](eta(n2))
            g[sp] = d[3](es) * ss * eta(n1) * eta(n2) + d[6](es) * ss * v[5](eta(n1)) * v[5](eta(n2))
            g[n1] = v[3](es) * deta(n1) * eta(n2) + v[6](es) * d[5](eta(n1)) * deta(n1) * v[5](eta(n2))
            g[n2] = v[3](es) * eta(n1) * deta(n2) + v[6](es) * v[5](eta(n1)) * d[5](eta(n2)) * deta(n2)
            return val, g
        val = v[4](es) * eta(n1) * eta(n2)
        g[sp] = d[4](es) * ss * eta(n1) * eta(n2)
        g[n1] = v[4](es) * deta(n1) * eta(n2)
        g[n2] = v[4](es) * eta(n1) * deta(n2)
        return val, g

    if cls == 3:
        nm = sp
        s1, s2 = sorted(((nm + 1) % 3, (nm + 2) % 3))
        e1 = 1 - c[s1] if spn & 1 else c[s1]
        e2 = 1 - c[s2] if spn & 2 else c[s2]
        ss1 = -1.0 if spn & 1 else 1.0
        ss2 = -1.0 if spn & 2 else 1.0
        val = v[5](eta(nm)) * v[6](e1) * v[6](e2)
        g = [0.0, 0.0, 0.0]
        g[nm] = d[5](eta(nm)) * deta(nm) * v[6](e1) * v[6](e2)
        g[s1] = v[5](eta(nm)) * d[6](e1) * ss1 * v[6](e2)
        g[s2] = v[5](eta(nm)) * v[6](e1) * d[6](e2) * ss2
        return val, g

    ex = 1 - t[0] if spn & 1 else t[0]
    ey = 1 - t[1] if spn & 2 else t[1]
    ez = 1 - t[2] if spn & 4 else t[2]
    ss = [-1.0 if spn & 1 else 1.0, -1.0 if spn & 2 else 1.0, -1.0 if spn & 4 else 1.0]
    return v[6](ex) * v[6](ey) * v[6](ez), [
        d[6](ex) * ss[0] * v[6](ey) * v[6](ez),
        v[6](ex) * d[6](ey) * ss[1] * v[6](ez),
        v[6](ex) * v[6](ey) * d[6](ez) * ss[2],
    ]


def contract(node, t, xi: float, order: int):
    """The proposed contraction: four terms, no class branch."""
    v, d = basis(xi, order)
    terms = term_table(node, order)

    def value(kind: int, axis: int, e: int) -> float:
        return v[kind](1.0 - t[axis] if e else t[axis])

    def deriv(kind: int, axis: int, e: int) -> float:
        return d[kind](1.0 - t[axis] if e else t[axis]) * (-1.0 if e else 1.0)

    total = 0.0
    for tr in terms:
        total += value(*tr[0]) * value(*tr[1]) * value(*tr[2])
    grad = [0.0, 0.0, 0.0]
    for axis in range(3):
        for tr in terms:
            product = 1.0
            for kind, a, e in tr:
                product *= deriv(kind, a, e) if a == axis else value(kind, a, e)
            grad[axis] += product
    return total, grad


def main() -> int:
    text = SPLINE.read_text()
    nodes = {
        4: parse_nodes(text, "kCubes2Nodes4[32]"),
        6: parse_nodes(text, "kCubes2Nodes6[88]"),
    }
    rng = random.Random(20261113)
    worst = 0.0
    for order, table in nodes.items():
        xi = XI4 if order == 4 else XI6
        terms_per_node = Counter()
        classes = Counter()
        for node in table:
            terms = term_table(node, order)
            assert len(terms) <= 4, (order, node, terms)
            terms_per_node[len(terms)] += 1
            classes[node[3]] += 1
            for _ in range(500):
                t = [rng.random(), rng.random(), rng.random()]
                got = contract(node, t, xi, order)
                want = reference(node, t, xi, order)
                worst = max(
                    worst,
                    abs(got[0] - want[0]),
                    *(abs(got[1][a] - want[1][a]) for a in range(3)),
                )
        print(
            f"order {order}: {len(table)} nodes, classes {dict(sorted(classes.items()))}, "
            f"terms {dict(sorted(terms_per_node.items()))}"
        )
    print(f"max |value/gradient difference| = {worst:.3e}")
    if worst > 1e-12:
        print("FAIL: the four-term form does not reproduce the reference")
        return 1
    print("PASS: the four-term form reproduces the reference for every node")
    return 0


if __name__ == "__main__":
    sys.exit(main())
