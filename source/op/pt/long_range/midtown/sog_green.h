// sog_green.h — CubeS₂ green tables for the SOG kspace custom op.
//
// Ported line-for-line from source/lmp/sog.cpp so the op reproduces the
// DEPLOYED chain's discretization, not an approximation of it:
//   * green tables       : SOGKSpace::precompute_green_functions (sog.cpp:1413-1671,
//                          spline_type >= 4 / CubeS₂ branch)
// Keeping the two implementations separate is deliberate: sog_op.py's python
// green builder is an independent second implementation, and validate192.py
// checks the op against the LAMMPS run-0 energy, so a transcription error in
// either one shows up as a disagreement.
//
// GRID SIZING IS NOT HERE.  The number of mesh points is decided in Python, by
// deepmd/dpmodel/long_range/grid_rule.py, and reaches the operator as the
// `spacing` argument of deepmd::latent_charge_sum; the operator only rounds
// ceil(L/spacing) up to a transformable size (midtown_sum.cu next_factorable).
// A second, C++ transcription of the error law used to live in this header --
// as the fallback of a `spacing <= 0` branch that nothing reaches -- and it had
// drifted from the Python rule (it lacked the 0.75 pi / k_max ceiling), so it
// was removed rather than kept as a trap.  Anything that needs a grid from a
// target accuracy calls grid_rule.spacing().
//
// The CubeS₂ family (spline 4/6) is the production spline.  QuadS is encoded as
// spline = 100 + order (104/106): its separable 1-D window symbol is inverted
// exactly (Form A), so the variance-subtraction D factor is replaced by
// 1/quads_influence_1d per axis.  The legacy B-spline (spline_type == 0) branch
// needs the sinc alias tables that live in the SOGKSpace class and is not here.
#ifndef SOG_GREEN_H
#define SOG_GREEN_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "sog_spline.h"

namespace sog_green {

constexpr double kCubes2Xi4 = 0.5773502691896258;   // sog_spline.h:46
constexpr double kCubes2Xi6 = 0.6503998764035732;   // sog_spline.h:48

// The integer weights M_p(1), ..., M_p(p-1) of the cardinal-B-spline assignment
// factor (identical to mesh_sum.cu's integer_bspline_weights): the B-spline
// value at the integer points, from the truncated-power form
//   M_p(x) = 1/(p-1)! sum_j (-1)^j C(p,j) (x-j)_+^{p-1}.
// The B-spline deconvolution is |D_d(m)|^-2 with D_d(m) = sum_l w_l e^{i theta l}.
inline void integer_bspline_weights(int order, double *w) {
  for (int l = 0; l < order - 1; ++l) {
    const double x = l + 1.0;
    double value = 0.0, binomial = 1.0;
    for (int j = 0; j <= order; ++j) {
      if (j > 0) binomial = binomial * (order - j + 1) / j;
      const double base = std::max(x - static_cast<double>(j), 0.0);
      value += (j % 2 == 0 ? 1.0 : -1.0) * binomial *
               std::pow(base, order - 1);
    }
    double factorial = 1.0;
    for (int k = 2; k < order; ++k) factorial *= k;
    w[l] = value / factorial;
  }
}

// SOGKSpace::precompute_green_functions, CubeS₂ branch (sog.cpp:1413-1565).
// Fills the five tables flat in mesh_index(ix,iy,iz) = ix + nx*(iy + ny*iz).
// Only the three deconvolved tables are produced.  The bare K and K_v "self"
// tables that used to live alongside them are gone: the self-energy term is
// closed-form (see sog_kspace_op.cc / sog_gpu.cu), needs no per-mode table,
// and the mesh sum it was used for was not volume-extensive.
// Shared body for the full and slab (y-sliced) tables.  `ylo`/`YL` select the
// y-range to fill and `ystride` is the storage extent along y: the full table
// uses (0, ny, ny), the per-rank slab uses (ylo, YL, YL) and stores
// idx = ix + nx*(iyl + YL*iz).  Both variants run the SAME arithmetic, so a
// slab value at a given (ix,iy,iz) is bit-identical to the full table.
inline void build_green_range(int nx, int ny, int nz, double lx, double ly,
                              double lz, const std::vector<double> &amp,
                              const std::vector<double> &bandwidth, int spline,
                              int ylo, int YL, int ystride,
                              std::vector<double> &ge, std::vector<double> &gf,
                              std::vector<double> &gv) {
  const size_t ng = static_cast<size_t>(nx) * YL * nz;
  ge.assign(ng, 0.0);
  gf.assign(ng, 0.0);
  gv.assign(ng, 0.0);

  const double dx = lx / nx, dy = ly / ny, dz = lz / nz;
  // Grid Nyquist for the CubeS₂ family (sog.cpp:1428-1433).
  const double k_sq_max = M_PI * M_PI *
                          (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz));
  const double tx = 2.0 * M_PI / lx, ty = 2.0 * M_PI / ly, tz = 2.0 * M_PI / lz;

  // QuadS is spline = 100 + order: its separable window is inverted exactly
  // (Form A), so D_alpha is 1/|W_alpha|^2 instead of the Gaussian variance
  // subtraction.  sog.cpp:1559-1600 is the deployed reference.
  const bool is_quads = spline >= 100 && spline < 200;
  const int nord = is_quads ? spline - 100 : (spline == 6 ? 6 : 4);
  const double xi_q = is_quads ? LAMMPS_NS::quads_xi(nord) : 0.0;
  const double xi = (spline == 6) ? kCubes2Xi6 : kCubes2Xi4;
  const double sig_x2 = (xi * dx) * (xi * dx);
  const double sig_y2 = (xi * dy) * (xi * dy);
  const double sig_z2 = (xi * dz) * (xi * dz);

  const size_t M = amp.size();
  std::vector<double> Bx(M * nx), By(M * ny), Bz(M * nz);
  std::vector<double> Dx(nx), Dy(ny), Dz(nz);
  for (int ix = 0; ix < nx; ++ix) {
    const int mode = ix - nx * (2 * ix / nx);
    const double k = tx * mode, k2 = k * k;
    if (is_quads) {
      const double I = LAMMPS_NS::quads_influence_1d(mode, nx, xi_q, nord);
      Dx[ix] = I > 1e-30 ? 1.0 / I : 0.0;
    } else {
      Dx[ix] = std::exp(sig_x2 * k2);
    }
    for (size_t m = 0; m < M; ++m)
      Bx[m * nx + ix] = std::exp(-0.5 * bandwidth[m] * k2);
  }
  for (int iy = 0; iy < ny; ++iy) {
    const int mode = iy - ny * (2 * iy / ny);
    const double k = ty * mode, k2 = k * k;
    if (is_quads) {
      const double I = LAMMPS_NS::quads_influence_1d(mode, ny, xi_q, nord);
      Dy[iy] = I > 1e-30 ? 1.0 / I : 0.0;
    } else {
      Dy[iy] = std::exp(sig_y2 * k2);
    }
    for (size_t m = 0; m < M; ++m)
      By[m * ny + iy] = std::exp(-0.5 * bandwidth[m] * k2);
  }
  for (int iz = 0; iz < nz; ++iz) {
    const int mode = iz - nz * (2 * iz / nz);
    const double k = tz * mode, k2 = k * k;
    if (is_quads) {
      const double I = LAMMPS_NS::quads_influence_1d(mode, nz, xi_q, nord);
      Dz[iz] = I > 1e-30 ? 1.0 / I : 0.0;
    } else {
      Dz[iz] = std::exp(sig_z2 * k2);
    }
    for (size_t m = 0; m < M; ++m)
      Bz[m * nz + iz] = std::exp(-0.5 * bandwidth[m] * k2);
  }

#pragma omp parallel for schedule(static)
  for (int iz = 0; iz < nz; ++iz) {
    const int mz = iz - nz * (2 * iz / nz);
    const double kz = tz * mz;
    for (int iyl = 0; iyl < YL; ++iyl) {
      const int iy = ylo + iyl;
      const int my = iy - ny * (2 * iy / ny);
      const double ky = ty * my;
      for (int ix = 0; ix < nx; ++ix) {
        const int mx = ix - nx * (2 * ix / nx);
        const double kx = tx * mx;
        const double sqk = kx * kx + ky * ky + kz * kz;
        if (!(sqk > 0.0 && sqk <= k_sq_max)) continue;   // k == 0 dropped
        const size_t idx = static_cast<size_t>(ix) +
                           nx * (static_cast<size_t>(iyl) +
                                 static_cast<size_t>(ystride) * static_cast<size_t>(iz));
        double kfac = 0.0, kfacv = 0.0;
        for (size_t m = 0; m < M; ++m) {
          const double p = Bx[m * nx + ix] * By[m * ny + iy] * Bz[m * nz + iz];
          kfac += amp[m] * p;
          kfacv += amp[m] * bandwidth[m] * p;   // amp_virial[m] = amp[m]*beta[m]
        }
        const double dc = Dx[ix] * Dy[iy] * Dz[iz];
        ge[idx] = kfac * dc;
        gf[idx] = kfac * dc;
        gv[idx] = kfacv * dc;
      }
    }
  }
}

// Half-spectrum slab tables: only the compact x column (Nh = nx/2+1) is
// stored, because the (x,y) transform is real-to-complex and the green
// function depends on kx^2 only.  Layout: idx = kh + Nh*(iyl + YL*iz).
// Values are identical to the full table's column ix = kh.
inline void build_green_slab_half(int nx, int ny, int nz, double lx,
                                  double ly, double lz,
                                  const std::vector<double> &amp,
                                  const std::vector<double> &bandwidth,
                                  int spline, int ylo, int YL,
                                  std::vector<double> &ge,
                                  std::vector<double> &gf,
                                  std::vector<double> &gv,
                                  const double *window = nullptr,
                                  double window_floor = 1e-4) {
  const int Nh = nx / 2 + 1;
  const size_t ng = static_cast<size_t>(Nh) * YL * nz;
  ge.assign(ng, 0.0);
  gf.assign(ng, 0.0);
  gv.assign(ng, 0.0);

  const double dx = lx / nx, dy = ly / ny, dz = lz / nz;
  const double k_sq_max = M_PI * M_PI *
                          (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz));
  const double tx = 2.0 * M_PI / lx, ty = 2.0 * M_PI / ly, tz = 2.0 * M_PI / lz;
  // QuadS is spline = 100 + order; it inverts the separable window exactly
  // (Form A), like build_green_range.  The x axis is stored as the compact
  // half-spectrum, whose representative kh has |kx| = |mode| and the influence
  // is even in the mode, so quads_influence_1d(kh, nx) is the table entry.
  //
  // B-spline is spline = 200 + order; its deconvolution is the inverse squared
  // cardinal-B-spline assignment factor |D_d(m)|^-2 (the same factor mesh_sum
  // inverts), so the slab solve reproduces the replicated SPME.
  const bool is_quads = spline >= 100 && spline < 200;
  const bool is_bspline = spline >= 200;
  const int nord = is_quads ? spline - 100
                            : (is_bspline ? spline - 200 : (spline == 6 ? 6 : 4));
  const double xi_q = is_quads ? LAMMPS_NS::quads_xi(nord) : 0.0;
  const double xi = (spline == 6) ? kCubes2Xi6 : kCubes2Xi4;
  const double sig_y2 = (xi * dy) * (xi * dy);
  const double sig_z2 = (xi * dz) * (xi * dz);
  // The B-spline integer weights M_p(1..p-1) and the per-mode |D(m)|^-2.
  double iw[16];
  if (is_bspline) integer_bspline_weights(nord, iw);
  auto bspline_d = [&](int m, int n) -> double {
    const double theta = 2.0 * M_PI * static_cast<double>(m) /
                         static_cast<double>(n);
    double re = 0.0, im = 0.0;
    for (int l = 0; l < nord - 1; ++l) {
      re += iw[l] * std::cos(theta * l);
      im += iw[l] * std::sin(theta * l);
    }
    const double d2 = re * re + im * im;
    return d2 > 1e-30 ? 1.0 / d2 : 0.0;
  };
  const size_t M = amp.size();
  std::vector<double> Bx(M * Nh), By(M * ny), Bz(M * nz);
  std::vector<double> Dx(Nh), Dy(ny), Dz(nz);
  for (int kh = 0; kh < Nh; ++kh) {
    const double k = tx * kh, k2 = k * k;   // +kx representative
    if (is_bspline) {
      Dx[kh] = bspline_d(kh, nx);
    } else if (is_quads) {
      const double I = LAMMPS_NS::quads_influence_1d(kh, nx, xi_q, nord);
      Dx[kh] = I > 1e-30 ? 1.0 / I : 0.0;
    } else {
      Dx[kh] = std::exp((xi * dx) * (xi * dx) * k2);
    }
    for (size_t m = 0; m < M; ++m) Bx[m * Nh + kh] = std::exp(-0.5 * bandwidth[m] * k2);
  }
  for (int iy = 0; iy < ny; ++iy) {
    const int mode = iy - ny * (2 * iy / ny);
    const double k = ty * mode, k2 = k * k;
    if (is_bspline) {
      Dy[iy] = bspline_d(mode, ny);
    } else if (is_quads) {
      const double I = LAMMPS_NS::quads_influence_1d(mode, ny, xi_q, nord);
      Dy[iy] = I > 1e-30 ? 1.0 / I : 0.0;
    } else {
      Dy[iy] = std::exp(sig_y2 * k2);
    }
    for (size_t m = 0; m < M; ++m) By[m * ny + iy] = std::exp(-0.5 * bandwidth[m] * k2);
  }
  for (int iz = 0; iz < nz; ++iz) {
    const int mode = iz - nz * (2 * iz / nz);
    const double k = tz * mode, k2 = k * k;
    if (is_bspline) {
      Dz[iz] = bspline_d(mode, nz);
    } else if (is_quads) {
      const double I = LAMMPS_NS::quads_influence_1d(mode, nz, xi_q, nord);
      Dz[iz] = I > 1e-30 ? 1.0 / I : 0.0;
    } else {
      Dz[iz] = std::exp(sig_z2 * k2);
    }
    for (size_t m = 0; m < M; ++m) Bz[m * nz + iz] = std::exp(-0.5 * bandwidth[m] * k2);
  }

#pragma omp parallel for schedule(static)
  for (int iz = 0; iz < nz; ++iz) {
    const int mz = iz - nz * (2 * iz / nz);
    const double kz = tz * mz;
    for (int iyl = 0; iyl < YL; ++iyl) {
      const int iy = ylo + iyl;
      const int my = iy - ny * (2 * iy / ny);
      const double ky = ty * my;
      for (int kh = 0; kh < Nh; ++kh) {
        const double kx = tx * kh;
        const double sqk = kx * kx + ky * ky + kz * kz;
        if (!(sqk > 0.0 && sqk <= k_sq_max)) continue;   // k == 0 dropped
        const size_t idx = static_cast<size_t>(kh) +
                           Nh * (static_cast<size_t>(iyl) +
                                 static_cast<size_t>(YL) * static_cast<size_t>(iz));
        double kfac = 0.0, kfacv = 0.0;
        for (size_t m = 0; m < M; ++m) {
          const double p = Bx[m * Nh + kh] * By[m * ny + iy] * Bz[m * nz + iz];
          kfac += amp[m] * p;
          kfacv += amp[m] * bandwidth[m] * p;
        }
        if (window != nullptr) {
          // Form A: replace the variance-subtraction deconvolution D by the
          // exact measured window <|W|^2>, so ge = kfac / <|W|^2>.  The slab
          // entry (kh, iyl, iz) maps to the global compact index
          // kh + nh*(iy + ny*iz) of the measured window (iy = ylo + iyl).
          const size_t gi = static_cast<size_t>(kh) +
                            static_cast<size_t>(Nh) *
                                (static_cast<size_t>(iy) +
                                 static_cast<size_t>(ny) * static_cast<size_t>(iz));
          const double w = window[gi];
          if (!(w > window_floor)) {
            continue;   // near-zero window at the Nyquist corner: drop the mode
          }
          ge[idx] = kfac / w;
          gf[idx] = kfac / w;
          gv[idx] = kfacv / w;
        } else {
          const double dc = Dx[kh] * Dy[iy] * Dz[iz];
          ge[idx] = kfac * dc;
          gf[idx] = kfac * dc;
          gv[idx] = kfacv * dc;
        }
      }
    }
  }
}

// Full-grid tables (replicated path and LAMMPS kspace).
inline void build_green(int nx, int ny, int nz, double lx, double ly, double lz,
                        const std::vector<double> &amp,
                        const std::vector<double> &bandwidth, int spline,
                        std::vector<double> &ge, std::vector<double> &gf,
                        std::vector<double> &gv) {
  build_green_range(nx, ny, nz, lx, ly, lz, amp, bandwidth, spline, 0, ny, ny,
                    ge, gf, gv);
}

// Full-grid tables of a triclinic cell, in fractional coordinates.
//
// The grid lives in the fractional space of the cell, so the deconvolution
// factor is per axis of the *mode index* and independent of the cell,
// D_d = exp((xi m_d / n_d)^2) with m_d the integer mode; this is the same
// identity that makes B-spline deconvolution cell free.  Only the Gaussian
// factor needs the metric:
//
//     k^2 = (2 pi)^2 m^T W m,   W = A^{-1} A^{-T},
//
// with A the matrix whose rows are the lattice vectors.  The Gaussian no
// longer factors over the axes, so the tables cost one exp per band and mode:
// a rebuild is O(M nx ny nz).  It happens only when the cell or the kernel
// changes, not per step.
inline void build_green_metric(int nx, int ny, int nz, const double W[9],
                               const std::vector<double> &amp,
                               const std::vector<double> &bandwidth, int spline,
                               std::vector<double> &ge, std::vector<double> &gf,
                               std::vector<double> &gv) {
  const size_t ng = static_cast<size_t>(nx) * ny * nz;
  ge.assign(ng, 0.0);
  gf.assign(ng, 0.0);
  gv.assign(ng, 0.0);
  const double xi = (spline == 6) ? kCubes2Xi6 : kCubes2Xi4;
  const double two_pi_sq = (2.0 * M_PI) * (2.0 * M_PI);
  // Nyquist corner of the mode box, mapped through the metric.
  const double cm[3] = {0.5 * nx, 0.5 * ny, 0.5 * nz};
  double k_sq_max = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      k_sq_max += two_pi_sq * cm[a] * W[3 * a + b] * cm[b];
    }
  }
  std::vector<double> Dx(nx), Dy(ny), Dz(nz);
  for (int ix = 0; ix < nx; ++ix) {
    const double m = ix - nx * (2 * ix / nx);
    Dx[ix] = std::exp(std::pow(2.0 * M_PI * xi * m / nx, 2.0));
  }
  for (int iy = 0; iy < ny; ++iy) {
    const double m = iy - ny * (2 * iy / ny);
    Dy[iy] = std::exp(std::pow(2.0 * M_PI * xi * m / ny, 2.0));
  }
  for (int iz = 0; iz < nz; ++iz) {
    const double m = iz - nz * (2 * iz / nz);
    Dz[iz] = std::exp(std::pow(2.0 * M_PI * xi * m / nz, 2.0));
  }
  const size_t M = amp.size();
#pragma omp parallel for schedule(static)
  for (int iz = 0; iz < nz; ++iz) {
    const double mz = iz - nz * (2 * iz / nz);
    for (int iy = 0; iy < ny; ++iy) {
      const double my = iy - ny * (2 * iy / ny);
      for (int ix = 0; ix < nx; ++ix) {
        const double mx = ix - nx * (2 * ix / nx);
        const double mm[3] = {mx, my, mz};
        double k2 = 0.0;
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            k2 += mm[a] * W[3 * a + b] * mm[b];
          }
        }
        k2 *= two_pi_sq;
        if (!(k2 > 0.0 && k2 <= k_sq_max)) {
          continue;
        }
        double kfac = 0.0, kfacv = 0.0;
        for (size_t m = 0; m < M; ++m) {
          const double e = amp[m] * std::exp(-0.5 * bandwidth[m] * k2);
          kfac += e;
          kfacv += e * bandwidth[m];
        }
        const double dc = Dx[ix] * Dy[iy] * Dz[iz];
        const size_t idx = static_cast<size_t>(ix) +
                           static_cast<size_t>(nx) *
                               (static_cast<size_t>(iy) +
                                static_cast<size_t>(ny) * static_cast<size_t>(iz));
        ge[idx] = kfac * dc;
        gf[idx] = kfac * dc;
        gv[idx] = kfacv * dc;
      }
    }
  }
}

// This rank's y-slab of the tables (distributed path): YL*nx*nz entries,
// indexed ix + nx*(iyl + YL*iz).  The green pencil kernel consumes exactly
// this layout, so per-rank green memory is 1/np of the full tables.
inline void build_green_slab(int nx, int ny, int nz, double lx, double ly,
                             double lz, const std::vector<double> &amp,
                             const std::vector<double> &bandwidth, int spline,
                             int ylo, int YL, std::vector<double> &ge,
                             std::vector<double> &gf,
                             std::vector<double> &gv) {
  build_green_range(nx, ny, nz, lx, ly, lz, amp, bandwidth, spline, ylo, YL,
                    YL, ge, gf, gv);
}

}  // namespace sog_green

#endif  // SOG_GREEN_H
