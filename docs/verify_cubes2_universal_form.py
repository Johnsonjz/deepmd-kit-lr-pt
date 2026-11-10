"""Validate the universal 4-term form of the CubeS2 node weight.

Companion to docs/MIDTOWN_AD_DESIGN.md section 3.  Needs the SOG library's
Python reference on sys.path (the path below), so it does not run as part of
this repository's test suite; run it by hand after touching the node tables.

Result at the time of writing: max deviation 1.1e-16 over 120 nodes x 50
random (tx,ty,tz) x 32 points, for the value and all three gradients.
The one trap: an UNUSED term must use the zero basis (kind 0 = 0).  Filling it
with the constant 1 adds +1 per empty term -- exactly the +3.0 seen first.

Claim: every node class is a sum of <=4 products A(x)*B(y)*C(z) of 1-D basis
functions, so a per-node index table + a per-atom 1-D table replaces the
5-way class branch.  Checks value AND gradient (autograd) against the
reference implementation for every node and many random positions.
"""

# Standalone analysis tool, not package code: module-level imports, print and
# unannotated helpers are deliberate.
# ruff: noqa
import sys
import torch
sys.path.insert(0, '/data/home/public/jiangzhen/sog/src/sog/module')
import cubes2_spline as cs2

torch.set_default_dtype(torch.float64)

K_ZERO, K_CONST, K_ELT = 7, 0, 1
XI = {4: cs2._get_xi(4), 6: cs2._get_xi(6)}


def basis(order, kind, axis, e, t):
    """1-D basis: returns tensor of shape t[0].shape."""
    x = t[axis]
    arg = (1 - x) if e else x
    if kind == K_ZERO:
        return torch.zeros_like(arg)
    if kind == K_CONST:
        return torch.ones_like(arg)
    if kind == K_ELT:
        return arg
    if order == 6:
        return {2: lambda z: cs2._cubes2_L111_6(z, XI[6]),
                3: lambda z: cs2._cubes2_L211_6(z, XI[6]),
                4: lambda z: cs2._cubes2_L311_6(z, XI[6]),
                5: lambda z: cs2._cubes2_S_6(z, XI[6]),
                6: lambda z: cs2._cubes2_R(z, XI[6])}[kind](arg)
    return {2: lambda z: cs2._cubes2_L(z, XI[4]),
            6: lambda z: cs2._cubes2_R(z, XI[4])}[kind](arg)


def terms_of(order, nd):
    """[[(kind, e) per axis] per term] -- the static index table."""
    T = [[(K_ZERO, 0)] * 3 for _ in range(4)]
    if order == 6:
        if nd.cls == 0:
            ex, ey, ez = 1 - nd.dx, 1 - nd.dy, 1 - nd.dz
            T[0] = [(2, ex), (K_ELT, ey), (K_ELT, ez)]
            T[1] = [(K_ELT, ex), (2, ey), (K_ELT, ez)]
            T[2] = [(K_ELT, ex), (K_ELT, ey), (2, ez)]
            T[3] = [(5, ex), (5, ey), (5, ez)]
        elif nd.cls in (1, 2):
            sp = nd.sp_axis
            n1, n2 = (sp + 1) % 3, (sp + 2) % 3
            if n1 > n2:
                n1, n2 = n2, n1
            e1 = 0 if (nd.dx, nd.dy, nd.dz)[n1] + 1 == 2 else 1
            e2 = 0 if (nd.dx, nd.dy, nd.dz)[n2] + 1 == 2 else 1
            neg = 1 if nd.sp_is_neg else 0
            first = [(K_CONST, 0)] * 3
            first[sp] = (3 if nd.cls == 1 else 4, neg)
            first[n1] = (K_ELT, e1)
            first[n2] = (K_ELT, e2)
            T[0] = first
            if nd.cls == 1:
                second = [(K_CONST, 0)] * 3
                second[sp] = (6, neg)
                second[n1] = (5, e1)
                second[n2] = (5, e2)
                T[1] = second
        elif nd.cls == 3:
            nm = nd.sp_axis
            s1, s2 = (nm + 1) % 3, (nm + 2) % 3
            if s1 > s2:
                s1, s2 = s2, s1
            onm = (nd.dx, nd.dy, nd.dz)[nm] + 1
            t0 = [(K_CONST, 0)] * 3
            t0[nm] = (5, 0 if onm == 2 else 1)
            t0[s1] = (6, 1 if (nd.sp_is_neg & 1) else 0)
            t0[s2] = (6, 1 if (nd.sp_is_neg & 2) else 0)
            T[0] = t0
        else:
            T[0] = [(6, 1 if (nd.sp_is_neg & 1) else 0),
                    (6, 1 if (nd.sp_is_neg & 2) else 0),
                    (6, 1 if (nd.sp_is_neg & 4) else 0)]
    else:  # order 4
        if nd.cls == 0:
            ex, ey, ez = 1 - nd.dx, 1 - nd.dy, 1 - nd.dz
            T[0] = [(2, ex), (K_ELT, ey), (K_ELT, ez)]
            T[1] = [(K_ELT, ex), (2, ey), (K_ELT, ez)]
            T[2] = [(K_ELT, ex), (K_ELT, ey), (2, ez)]
        else:
            sp = nd.sp_axis
            n1, n2 = (sp + 1) % 3, (sp + 2) % 3
            e1 = 0 if (nd.dx, nd.dy, nd.dz)[n1] + 1 == 2 else 1
            e2 = 0 if (nd.dx, nd.dy, nd.dz)[n2] + 1 == 2 else 1
            t0 = [(K_CONST, 0)] * 3
            t0[sp] = (6, 1 if nd.sp_is_neg else 0)
            t0[n1] = (K_ELT, e1)
            t0[n2] = (K_ELT, e2)
            T[0] = t0
    return T


def universal(order, nd, t):
    T = terms_of(order, nd)
    w = 0
    for term in T:
        p = 1
        for a in range(3):
            k, e = term[a]
            p = p * basis(order, k, a, e, t)
        w = w + p
    return w


worst = 0.0
for order in (4, 6):
    xi = XI[order]
    ref_fn = cs2.cubes2_weight_6 if order == 6 else cs2.cubes2_weight_4
    nodes = cs2.get_nodes(order)
    g = torch.Generator().manual_seed(5)
    for trial in range(50):
        tx = torch.rand(32, generator=g); ty = torch.rand(32, generator=g)
        tz = torch.rand(32, generator=g)
        t = [tx.requires_grad_(True), ty.requires_grad_(True), tz.requires_grad_(True)]
        for nd in nodes:
            got = universal(order, nd, t)
            ref = ref_fn(t[0], t[1], t[2], nd, xi)
            worst = max(worst, float((got - ref).abs().max()))
            gr = torch.autograd.grad(got.sum(), t, retain_graph=True, allow_unused=True)
            gf = torch.autograd.grad(ref.sum(), t, retain_graph=True, allow_unused=True)
            for a in range(3):
                ga = gr[a] if gr[a] is not None else torch.zeros_like(t[a])
                gb = gf[a] if gf[a] is not None else torch.zeros_like(t[a])
                worst = max(worst, float((ga - gb).abs().max()))
print(f"max |universal - reference| over value and all 3 gradients: {worst:.3e}")
print("nodes checked:", sum(len(cs2.get_nodes(o)) for o in (4, 6)), "x 50 random (tx,ty,tz) x 32 points")

print("\n--- per-node worst (order 6) ---")
g = torch.Generator().manual_seed(11)
tx = torch.rand(64, generator=g); ty = torch.rand(64, generator=g); tz = torch.rand(64, generator=g)
t = [tx, ty, tz]
rows = []
for i, nd in enumerate(cs2.get_nodes(6)):
    got = universal(6, nd, t)
    ref = cs2.cubes2_weight_6(tx, ty, tz, nd, XI[6])
    rows.append((float((got - ref).abs().max()), i, nd.cls, nd.sp_axis, nd.sp_is_neg,
                 nd.dx, nd.dy, nd.dz))
rows.sort(reverse=True)
for r in rows[:6]:
    print("  |d|=%.3e  node %2d cls=%d sp=%d neg=%d d=(%d,%d,%d)" % r)
