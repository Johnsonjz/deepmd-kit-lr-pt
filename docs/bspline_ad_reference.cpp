// Standalone reference of the two functions that go into source/lmp/sog.cpp
// for the AD branch, plus a self-check main.
//
// Why standalone: sog.cpp depends on the LAMMPS headers and this machine has no
// LAMMPS build of that tree, but the risky part of the change is pure
// arithmetic -- the weight derivative and the tensor-product contraction.  Both
// are reproduced here verbatim so they can be checked against the validated
// Python prototype (docs/measure_ik_vs_ad_force.py) before they land.
//
// Build:  g++ -O2 -std=c++17 docs/bspline_ad_reference.cpp -o /tmp/bsad
// Check:  python docs/check_bspline_ad_reference.py
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// ── The existing function, copied from sog.cpp:102 verbatim ───────────────
static void sog_bspline_weights_1d_order(const int n, const double frac,
                                         std::vector<double> &w) {
  w.resize(n);
  std::fill(w.begin(), w.end(), 0.0);
  w[0] = 1.0 - frac;
  w[1] = frac;
  for (int k = 3; k <= n; ++k) {
    const double inv = 1.0 / static_cast<double>(k - 1);
    std::vector<double> w_prev = w;
    w[static_cast<size_t>(k - 1)] =
        frac * w_prev[static_cast<size_t>(k - 2)] * inv;
    for (int j = 1; j <= k - 2; ++j) {
      w[static_cast<size_t>(k - 1 - j)] =
          ((frac + static_cast<double>(j)) *
               w_prev[static_cast<size_t>(k - 2 - j)] +
           (static_cast<double>(k - j) - frac) *
               w_prev[static_cast<size_t>(k - 1 - j)]) *
          inv;
    }
    w[0] = (1.0 - frac) * w_prev[0] * inv;
  }
}

// ── NEW: the same recursion carried through the derivative ────────────────
//
// Every line of the weight recursion is linear in `frac`, so differentiating
// the whole recursion with respect to `frac` gives the derivative of the weight
// array in the SAME index convention, with no offset bookkeeping and no
// assumption about how the array is centred.  That is the whole reason to do it
// this way: the alternative (the identity M_p'(x) = M_{p-1}(x) - M_{p-1}(x-1))
// is shorter but has to be mapped onto the index convention by hand, and a
// one-off error there is a silent wrong force.
//
// `dw[a]` is d w[a] / d frac, i.e. the derivative with respect to the scaled
// coordinate u = n * s; the physical factor n_d / L_d is applied by the caller.
static void sog_bspline_weights_1d_order_d(const int n, const double frac,
                                           std::vector<double> &w,
                                           std::vector<double> &dw) {
  w.assign(n, 0.0);
  dw.assign(n, 0.0);
  w[0] = 1.0 - frac;
  dw[0] = -1.0;
  w[1] = frac;
  dw[1] = 1.0;
  for (int k = 3; k <= n; ++k) {
    const double inv = 1.0 / static_cast<double>(k - 1);
    const std::vector<double> w_prev = w;
    const std::vector<double> dw_prev = dw;
    w[static_cast<size_t>(k - 1)] =
        frac * w_prev[static_cast<size_t>(k - 2)] * inv;
    dw[static_cast<size_t>(k - 1)] =
        (w_prev[static_cast<size_t>(k - 2)] +
         frac * dw_prev[static_cast<size_t>(k - 2)]) * inv;
    for (int j = 1; j <= k - 2; ++j) {
      const double fj = static_cast<double>(j);
      const double fk = static_cast<double>(k);
      w[static_cast<size_t>(k - 1 - j)] =
          ((frac + fj) * w_prev[static_cast<size_t>(k - 2 - j)] +
           (fk - fj - frac) * w_prev[static_cast<size_t>(k - 1 - j)]) * inv;
      dw[static_cast<size_t>(k - 1 - j)] =
          (w_prev[static_cast<size_t>(k - 2 - j)] +
           (frac + fj) * dw_prev[static_cast<size_t>(k - 2 - j)] -
           w_prev[static_cast<size_t>(k - 1 - j)] +
           (fk - fj - frac) * dw_prev[static_cast<size_t>(k - 1 - j)]) * inv;
    }
    w[0] = (1.0 - frac) * w_prev[0] * inv;
    dw[0] = (-w_prev[0] + (1.0 - frac) * dw_prev[0]) * inv;
  }
}

// ── NEW: the AD force of one frame from the potential mesh ────────────────
//
//   phi_i  = sum_{abc} wx[a] wy[b] wz[c] u[ix[a],iy[b],iz[c]]
//   dphi/du_d = sum (dw_d ...) u
//   F_j    = -qscale * q_i * sum_d (dphi/du_d) * (n_d / L_d)
//
// `u` is the potential mesh laid out ix-fastest (mesh_index), the same layout
// the ik branch's mesh_grad* used.
static void bspline_ad_force(const int nlocal, const double *x, const double *q,
                             const double *u, const int n1d, const int nx,
                             const int ny, const int nz, const double lx,
                             const double ly, const double lz, const double boxlo[3],
                             const double qscale, double *force, double *pot) {
  const int half = (n1d - 1) / 2;
  std::vector<double> wx, wy, wz, dwx, dwy, dwz;
  std::vector<int> ix(n1d), iy(n1d), iz(n1d);
  for (int i = 0; i < nlocal; ++i) {
    const double fx = (x[3 * i + 0] - boxlo[0]) / lx;
    const double fy = (x[3 * i + 1] - boxlo[1]) / ly;
    const double fz = (x[3 * i + 2] - boxlo[2]) / lz;
    const double ux = (fx - std::floor(fx)) * nx;
    const double uy = (fy - std::floor(fy)) * ny;
    const double uz = (fz - std::floor(fz)) * nz;
    const int ix0 = static_cast<int>(std::floor(ux));
    const int iy0 = static_cast<int>(std::floor(uy));
    const int iz0 = static_cast<int>(std::floor(uz));
    sog_bspline_weights_1d_order_d(n1d, ux - ix0, wx, dwx);
    sog_bspline_weights_1d_order_d(n1d, uy - iy0, wy, dwy);
    sog_bspline_weights_1d_order_d(n1d, uz - iz0, wz, dwz);
    for (int a = 0; a < n1d; ++a) {
      ix[a] = ((ix0 - half + a) % nx + nx) % nx;
      iy[a] = ((iy0 - half + a) % ny + ny) % ny;
      iz[a] = ((iz0 - half + a) % nz + nz) % nz;
    }
    double phi = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
    for (int a = 0; a < n1d; ++a) {
      for (int b = 0; b < n1d; ++b) {
        for (int c = 0; c < n1d; ++c) {
          const size_t idx =
              static_cast<size_t>(ix[a]) +
              static_cast<size_t>(nx) * (static_cast<size_t>(iy[b]) +
                                         static_cast<size_t>(ny) * iz[c]);
          const double uu = u[idx];
          const double wyz = wy[b] * wz[c];
          phi += wx[a] * wyz * uu;
          gx += dwx[a] * wyz * uu;
          gy += wx[a] * (dwy[b] * wz[c]) * uu;
          gz += wx[a] * (wy[b] * dwz[c]) * uu;
        }
      }
    }
    const double qi = q[i];
    force[3 * i + 0] = -qscale * qi * gx * nx / lx;
    force[3 * i + 1] = -qscale * qi * gy * ny / ly;
    force[3 * i + 2] = -qscale * qi * gz * nz / lz;
    if (pot != nullptr) pot[i] = qscale * phi;
  }
}

// ── self-check driver: reads n, order, t values; prints w and dw ──────────
int main(int argc, char **argv) {
  if (argc >= 4 && std::string(argv[1]) == "weights") {
    const int n = std::atoi(argv[2]);
    const int count = std::atoi(argv[3]);
    std::vector<double> w, dw;
    for (int i = 0; i < count; ++i) {
      double t;
      if (std::scanf("%lf", &t) != 1) return 1;
      sog_bspline_weights_1d_order_d(n, t, w, dw);
      std::printf("%d %d", n, i);
      for (int a = 0; a < n; ++a) std::printf(" %.17g", w[a]);
      for (int a = 0; a < n; ++a) std::printf(" %.17g", dw[a]);
      std::printf("\n");
    }
    return 0;
  }
  if (argc >= 2 && std::string(argv[1]) == "frame") {
    int nlocal, n1d, nx, ny, nz;
    double qscale, lx, ly, lz, boxlo[3];
    if (std::scanf("%d %d %d %d %d %lf %lf %lf %lf %lf %lf %lf", &nlocal, &n1d,
                   &nx, &ny, &nz, &qscale, &lx, &ly, &lz, &boxlo[0], &boxlo[1],
                   &boxlo[2]) != 12)
      return 1;
    std::vector<double> x(3 * nlocal), q(nlocal);
    for (int i = 0; i < nlocal; ++i)
      if (std::scanf("%lf %lf %lf %lf", &x[3 * i], &x[3 * i + 1], &x[3 * i + 2],
                     &q[i]) != 4)
        return 1;
    std::vector<double> u(static_cast<size_t>(nx) * ny * nz);
    for (size_t i = 0; i < u.size(); ++i)
      if (std::scanf("%lf", &u[i]) != 1) return 1;
    std::vector<double> force(3 * nlocal), pot(nlocal);
    bspline_ad_force(nlocal, x.data(), q.data(), u.data(), n1d, nx, ny, nz, lx, ly,
                     lz, boxlo, qscale, force.data(), pot.data());
    for (int i = 0; i < nlocal; ++i)
      std::printf("%.17g %.17g %.17g %.17g\n", force[3 * i], force[3 * i + 1],
                  force[3 * i + 2], pot[i]);
    return 0;
  }
  std::fprintf(stderr,
               "usage: %s weights <n> <count>   |   %s frame\n", argv[0], argv[0]);
  return 1;
}
