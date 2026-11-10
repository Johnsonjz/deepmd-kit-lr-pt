// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Midtown mesh backend of deepmd::latent_charge_sum (schema in ops.cc).
//
// This backend evaluates a periodic batch with the CubeS2 / QuadS midtown
// splines and cuFFT of the SOG library instead of the B-spline SPME of
// mesh_sum.cu.  It shares the contract of ``mesh_sum``: the same arguments,
// the same outputs and the same formulas, so the caller does not change.
//
// The first version is deliberately narrow: one channel, the SOG kernel,
// orthogonal cells and a single rank.  Every unsupported combination raises
// instead of silently falling back, so a caller never gets a different
// physics than it asked for.
//
// Coordinate systems: the op takes absolute positions; the midtown kernels
// take positions in [0, L) with a zero origin, so the positions are wrapped
// on the device.  The real-space kernel of the op is
//
//     K_op(r) = sum_k a_k exp(-r^2 / (2 sigma_k^2)),
//
// while the midtown device code carries the normalized form
//
//     K_mid(r) = sum_m amp_m (2 pi bw_m)^{-3/2} exp(-r^2 / (2 bw_m)),
//
// so ``amp_m = a_k (2 pi bw_k)^{3/2}`` and ``bw_k = sigma_k^2``.
//
// Every per-node array stays on the device: the charges, the wrapped
// positions, the potential and the forces are torch tensors, and only the two
// scalars the device entry point wants on the host (the squared charge sum
// and the charged-node count) make the round trip.

#include <torch/library.h>
#include <torch/types.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <string>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "charge_sum.h"
#include "midtown/sog_gpu.cuh"
#include "midtown/sog_green.h"

namespace deepmd {
namespace long_range {
namespace {

constexpr double kLn2 = 0.69314718055994530942;
constexpr double kTwoPi = 6.28318530717958647692;

// One device state per process; the op serializes its calls.
struct MidtownState {
  SogGpuState* state = nullptr;
  int nx = 0, ny = 0, nz = 0;
  double cell[9] = {0.0};
  bool tilted = false;
  std::vector<double> amp;
  std::vector<double> bw;
  int spline = 0;
  bool ready = false;
  // The replicated path calls sog_gpu_setup (full-grid buffers and a full
  // green table) while the slab path calls sog_gpu_setup_slab (1/np scratch and
  // this rank's compact y-slice).  The two layouts must never be shared, so the
  // cache records which one built `ready`.  Ignoring this silently reuses a
  // full-grid table for a compact-stride lookup, or sizes the scratch below
  // what a stage indexes.
  bool slab = false;
  int slab_ylo = -1, slab_YL = 0;
  // Which green form is resident in the device tables: 0 = Form B
  // (variance subtraction), 1 = Form A (exact measured window symbol).
  // Only the replicated path converts, so this must be part of every
  // cache key or a rebuild in another path silently changes the physics.
  int form = 0;

  ~MidtownState() {
    if (state != nullptr) {
      sog_gpu_destroy(state);
    }
  }
};

// The replicated path can convert its green table to Form A; the local-spread
// and slab paths cannot.  Refuse loudly rather than silently running Form B,
// which would read as an MPI bug (the two forms differ by up to 66x in the
// reciprocal error at order 4).
bool form_a_requested() {
  const char* e = std::getenv("SOG_MIDTOWN_FORM_A");
  return e != nullptr && std::atoi(e) != 0;
}

MidtownState& midtown_state() {
  static MidtownState state;
  return state;
}

std::mutex& midtown_mutex() {
  static std::mutex mutex;
  return mutex;
}

int midtown_spline(int64_t order, bool quads = false) {
  TORCH_CHECK(order == 4 || order == 6,
              "midtown mesh backend supports spline order 4 or 6, got ",
              order);
  if (quads) return order == 6 ? SOG_QUADS_6 : SOG_QUADS_4;
  return order == 6 ? SOG_CUBES2_6 : SOG_CUBES2_4;
}

// Whether n factors over 2, 3 and 5.
//
// 7 is deliberately EXCLUDED even though cuFFT can transform it and the
// B-spline backend sizes its mesh over 2-3-5-7 (mesh_sum.cu even_fft_size):
// the plans are not equal in cost.  At L = 122.31 A and spacing 0.5 the
// requested extent is 245 points, which a 2-3-5 mesh rounds to 250 (2 * 5^3)
// but a 2-3-5-7 mesh accepts as 245 = 5 * 7^2.  Measured, the 245^3 plan runs
// regular_fft_factor<49> + regular_fft_factor<5> + vector_fft (1.42 ms of
// transforms plus 0.39 ms of pre/post processing) against 0.59 ms for 250^3 --
// the radix-7 pass makes the smaller mesh 3x more expensive.  A finer grid is
// cheaper than a factor-7 one.
bool factorable_235(int n) {
  for (const int p : {2, 3, 5}) {
    while (n % p == 0) {
      n /= p;
    }
  }
  return n == 1;
}

int next_factorable(int n) {
  const int minimum = 4;
  n = std::max(n, minimum);
  while (!factorable_235(n)) {
    ++n;
  }
  return n;
}

// The per-axis grid. The operator contract is a mesh of spacing at most
// `spacing`, and the same rule serves both backends so that one call means the
// same grid whichever mesh backend evaluates it.
//
// The spacing itself is NOT derived here: it comes from the caller, which is
// deepmd/dpmodel/long_range/grid_rule.py inverting the measured error law for a
// target force-relative accuracy.  The operator only rounds the extent up to a
// size cuFFT transforms well.  A non-positive spacing is rejected rather than
// silently sized by a second copy of the accuracy rule: the B-spline backend
// rejects it too (mesh_sum.cu), so accepting it here would make the same call
// mean different things per backend.
int midtown_axis(double length, double spacing) {
  TORCH_CHECK(spacing > 0.0,
              "the midtown mesh backend takes a positive mesh spacing, got ",
              spacing,
              "; the grid of a target accuracy comes from "
              "deepmd.dpmodel.long_range.grid_rule.spacing()");
  return next_factorable(static_cast<int>(std::ceil(length / spacing)));
}

}  // namespace

// The charges of a frame: q_i = w_i c_i, shifted uniformly over the charged
// nodes when a channel is constrained.  Phase 1 writes the masked charges and
// accumulates the count and the sum; phase 2 applies the shift.  Two kernels
// replace the six tensor operations (cast, multiply, sum, clamp, subtract,
// multiply) this used to take.
__global__ void k_midtown_charges1(int n, const float* raw, const bool* mask,
                                   double* q, double* sums) {
  __shared__ double ssum[2][8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double local[2] = {0.0, 0.0};
  if (i < n) {
    const double m = mask[i] ? 1.0 : 0.0;
    const double v = static_cast<double>(raw[i]) * m;
    q[i] = v;
    local[0] = 1.0;
    local[1] = v;
  }
  #pragma unroll
  for (int t = 0; t < 2; ++t) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      local[t] += __shfl_down_sync(0xffffffffu, local[t], off);
    }
  }
  __syncthreads();
  if (lane == 0) {
    ssum[0][warp] = local[0];
    ssum[1][warp] = local[1];
  }
  __syncthreads();
  const int nwarps = static_cast<int>(blockDim.x >> 5);
  if (static_cast<int>(threadIdx.x) < 2) {
    double v = ssum[threadIdx.x][0];
    for (int w = 1; w < nwarps; ++w) v += ssum[threadIdx.x][w];
    atomicAdd(&sums[threadIdx.x], v);
  }
}

__global__ void k_midtown_charges2(int n, const bool* mask, const double* sums,
                                   const double* total_charge, int channel,
                                   double* q) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (channel < 0) return;
  const double cnt = sums[0] > 0.0 ? sums[0] : 1.0;
  const double shift = (sums[1] - total_charge[0]) / cnt;
  q[i] -= (mask[i] ? 1.0 : 0.0) * shift;
}

// The per-atom assembly, in two kernels instead of a dozen tensor ops.
//
// Phase 1 writes the energy and the r (x) F moments and accumulates the two
// scalars the frame needs (the potential sum of the constrained channel and
// the moment sum).  Phase 2 applies them: the charge gradient, the atomic
// virial and the share of the k-space virial.  Both read the same arrays, so
// nothing crosses to the host between them.
__global__ void k_midtown_finish1(int n, const double* vpot, const double* f3,
                                  const double* q, const bool* mask,
                                  const double* coord, double prefactor,
                                  double kernel_zero, int channel,
                                  double* energy, double* sums,
                                  double* moments) {
  __shared__ double ssum[10][8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double local[10];
  for (int t = 0; t < 10; ++t) local[t] = 0.0;
  if (i < n) {
    const double m = mask[i] ? 1.0 : 0.0;
    const double phi = (vpot[i] - prefactor * kernel_zero * q[i]) * m;
    energy[i] = 0.5 * q[i] * phi;
    if (channel >= 0) local[0] = phi;
    const int k = i * 9;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        const double mom = f3[3 * i + a] * coord[3 * i + b];
        moments[k + 3 * a + b] = mom;
        local[1 + 3 * a + b] = mom;
      }
    }
  }
  // Warp reduction, then one atomic per block and counter: the per-atom
  // atomics all hit the same ten doubles and serialize.
  #pragma unroll
  for (int t = 0; t < 10; ++t) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      local[t] += __shfl_down_sync(0xffffffffu, local[t], off);
    }
  }
  __syncthreads();
  if (lane == 0) {
    #pragma unroll
    for (int t = 0; t < 10; ++t) ssum[t][warp] = local[t];
  }
  __syncthreads();
  const int nwarps = static_cast<int>(blockDim.x >> 5);
  if (static_cast<int>(threadIdx.x) < 10) {
    double v = ssum[threadIdx.x][0];
    for (int w = 1; w < nwarps; ++w) v += ssum[threadIdx.x][w];
    atomicAdd(&sums[threadIdx.x], v);
  }
}

__global__ void k_midtown_finish2(int n, const double* vpot, const double* q,
                                  const bool* mask, double prefactor,
                                  double kernel_zero, int channel,
                                  const double* sums, const double* count,
                                  const double* moments, const double* red,
                                  double virial_scale, float* grad,
                                  float* virial9) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double m = mask[i] ? 1.0 : 0.0;
  const double phi = (vpot[i] - prefactor * kernel_zero * q[i]) * m;
  const double cnt = count[0] > 0.0 ? count[0] : 1.0;
  const double mean = channel >= 0 ? sums[0] / cnt : 0.0;
  grad[i] = static_cast<float>(phi - m * mean);
  // red[1..6] holds xx, yy, zz, xy, xz, yz of the k-space virial.
  const double v = virial_scale;
  const double W[9] = {red[1] * v, red[4] * v, red[5] * v,
                       red[4] * v, red[2] * v, red[6] * v,
                       red[5] * v, red[6] * v, red[3] * v};
  const int k = i * 9;
  for (int t = 0; t < 9; ++t) {
    virial9[k + t] = static_cast<float>(moments[k + t] + (W[t] - sums[1 + t]) / cnt);
  }
}

// The frame virial is nine values of the six k-space sums: one thread each.
__global__ void k_midtown_virial(const double* red, double scale, float* out) {
  static const int kSym[9] = {0, 3, 4, 3, 1, 5, 4, 5, 2};
  const int t = threadIdx.x;
  if (t >= 9) {
    return;
  }
  out[t] = static_cast<float>(red[1 + kSym[t]] * scale);
}

namespace {
// Per-stage CPU timing of the operator, off unless SOG_MIDTOWN_CPU is set.
struct CpuTimer {
  static bool on() {
    static const bool on_ = std::getenv("SOG_MIDTOWN_CPU") != nullptr;
    return on_;
  }
  // Entry-to-entry interval: everything the marks cannot see (the return
  // path, the tensor destructors, the caller) shows up here.
  static void cycle() {
    if (!on()) {
      return;
    }
    static std::chrono::steady_clock::time_point prev;
    static bool first = true;
    const auto now = std::chrono::steady_clock::now();
    if (!first) {
      acc()["cycle"] +=
          std::chrono::duration<double, std::micro>(now - prev).count();
    }
    first = false;
    prev = now;
  }
  static std::map<std::string, double>& acc() {
    static std::map<std::string, double> m;
    return m;
  }
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  void mark(const char* name) {
    if (!on()) return;
    auto t1 = std::chrono::steady_clock::now();
    acc()[name] += std::chrono::duration<double, std::micro>(t1 - t0).count();
    t0 = t1;
  }
  static void flush() {
    if (!on()) return;
    static int n = 0;
    if (++n % 50 != 0) return;
    for (auto& kv : acc()) {
      fprintf(stderr, "[midtown cpu] %-16s %8.1f us/call\n", kv.first.c_str(),
              kv.second / 50.0);
      kv.second = 0.0;
    }
  }
};
}  // namespace

ChargeSumOutputs midtown_sum(const torch::Tensor& coord,
                             const torch::Tensor& charge,
                             const torch::Tensor& node_mask,
                             const torch::Tensor& n_node_per_frame,
                             const torch::Tensor& total_charge,
                             const torch::Tensor& box,
                             const torch::Tensor& coefficient,
                             int64_t kernel,
                             int64_t channel,
                             double prefactor,
                             int64_t order,
                             double spacing,
                             int owner_begin,
                             int owner_count,
                             const torch::Tensor* communicator,
                             int64_t force_mode,
                             bool quads) {
  CpuTimer::cycle();
  TORCH_CHECK(kernel == kSog,
              "the midtown mesh backend evaluates the SOG kernel only");
  TORCH_CHECK(charge.size(1) == 1,
              "the midtown mesh backend supports one channel, got ",
              charge.size(1));
  TORCH_CHECK(channel == -1 || channel == 0,
              "the midtown mesh backend supports a constraint on channel 0 "
              "only");
  const int64_t n = coord.size(0);
  const int64_t frame_count = n_node_per_frame.size(0);
  // Every rank of a decomposed system receives the complete frame from the
  // caller (domain_exchange), so this backend solves the frame on each rank and
  // returns the owner slice.  The mesh is replicated across ranks; a genuinely
  // distributed mesh is a later step.
  TORCH_CHECK(owner_begin >= 0 && owner_begin + owner_count <= n,
              "the midtown mesh backend got an owner range outside the frame");
  (void)communicator;
  TORCH_CHECK(coefficient.size(0) > 1 && coefficient.size(0) % 2 == 0,
              "the midtown mesh backend needs amplitude and rate pairs, got ",
              coefficient.size(0), " coefficients");

  const int spline = midtown_spline(order, quads);
  const bool is_quads = (spline == SOG_QUADS_4 || spline == SOG_QUADS_6);
  const int64_t bands = coefficient.size(0) / 2;

  // The coefficients are metadata: the caller passes them on the host, so no
  // device synchronization happens here.  The frame boundaries are only read
  // when there is more than one frame; the total charge never leaves the
  // device (the charge constraint reads it there).
  CpuTimer tm;
  const bool coef_host = coefficient.device().is_cpu();
  torch::Tensor coef_h =
      coef_host ? coefficient.detach().flatten().contiguous()
                : coefficient.detach().flatten().to(at::kCPU).contiguous();
  const double* cf = coef_h.data_ptr<double>();
  std::vector<double> amp(bands), bw(bands);
  double kernel_zero = 0.0;
  for (int64_t m = 0; m < bands; ++m) {
    const double a = cf[m];
    const double rate = cf[bands + m];
    TORCH_CHECK(rate > 0.0, "the midtown mesh backend needs positive rates");
    const double sigma_sq = 1.0 / (2.0 * rate * kLn2);
    bw[m] = sigma_sq;
    // The op's kernel is un-normalized; the device green tables are normalized.
    amp[m] = a * std::pow(kTwoPi * sigma_sq, 1.5);
    kernel_zero += a;
  }
  tm.mark("pack");
  torch::Tensor cells = box.detach().to(at::kCPU).to(at::kDouble).contiguous();
  const double* cell = cells.data_ptr<double>();
  std::vector<int64_t> nf_host(frame_count);
  if (frame_count == 1) {
    // One frame holds every node, so the count is the batch size and the
    // operator never has to wait for the device to read it.  A multi-frame
    // batch still reads the boundaries on the host (one call per batch, not
    // per step, and the model passes a single frame in MD).
    nf_host[0] = n;
  } else {
    torch::Tensor nf = n_node_per_frame.device().is_cpu()
                           ? n_node_per_frame.detach().flatten().to(at::kLong)
                           : n_node_per_frame.detach().flatten().to(at::kCPU).to(
                                 at::kLong);
    for (int64_t f = 0; f < frame_count; ++f) {
      nf_host[f] = static_cast<int64_t>(nf.data_ptr<int64_t>()[f]);
    }
  }
  const int64_t* nf_all = nf_host.data();
  // total_charge is a CUDA tensor by contract; the constraint kernel reads it
  // in place, one entry per frame.
  TORCH_CHECK(total_charge.device().is_cuda(),
              "the midtown mesh backend reads the total charge on the device");
  const double* qtot = total_charge.data_ptr<double>();

  const auto double_options = coord.options().dtype(at::kDouble);
  const auto float_options = charge.options().dtype(at::kFloat);

  // The finish kernels write every node of the owner range and every frame of
  // the virial, so the outputs need no zero fill: five at::zeros calls were
  // five launches and five memsets per step.
  auto atom_energy = torch::empty({n, 1}, double_options);
  auto charge_gradient = torch::empty({n, 1}, float_options);
  auto force = torch::empty({n, 3}, float_options);
  auto atom_virial = torch::empty({n, 9}, float_options);
  // A frame with no charged nodes is skipped, so its virial row is never
  // written: keep the zero fill here (one tiny memset, not per-node).
  auto virial = torch::zeros({frame_count, 9}, float_options);

  tm.mark("hostparse");
  std::lock_guard<std::mutex> lock(midtown_mutex());
  MidtownState& local = midtown_state();
  if (local.state == nullptr) {
    local.state = sog_gpu_create();
    TORCH_CHECK(local.state != nullptr, "midtown: sog_gpu_create failed");
  }

  int64_t begin = 0;
  for (int64_t f = 0; f < frame_count; ++f) {
    const int64_t count = nf_all[f];
    if (count <= 0) {
      continue;
    }
    const double* c = cell + 9 * f;
    const bool tilted = std::fabs(c[1]) >= 1e-12 || std::fabs(c[2]) >= 1e-12 ||
                        std::fabs(c[3]) >= 1e-12 || std::fabs(c[5]) >= 1e-12 ||
                        std::fabs(c[6]) >= 1e-12 || std::fabs(c[7]) >= 1e-12;
    TORCH_CHECK(!(is_quads && tilted),
                "the QuadS backend supports orthogonal cells only");
    const double lx = c[0], ly = c[4], lz = c[8];

    // The metric of a triclinic cell: W = A^{-1} A^{-T} with A the rows.  The
    // grid lives in fractional coordinates, so its spacing is set by the
    // lattice-vector lengths, and the volume by |det A|.
    double Ainv[9] = {0.0};
    double W[9] = {0.0};
    double lens[3];
    double volume;
    for (int d = 0; d < 3; ++d) {
      lens[d] = std::sqrt(c[3 * d] * c[3 * d] + c[3 * d + 1] * c[3 * d + 1] +
                          c[3 * d + 2] * c[3 * d + 2]);
      TORCH_CHECK(lens[d] > 0.0,
                  "the midtown mesh backend needs non-degenerate lattice "
                  "vectors");
    }
    if (tilted) {
      const double det = c[0] * (c[4] * c[8] - c[5] * c[7]) -
                         c[1] * (c[3] * c[8] - c[5] * c[6]) +
                         c[2] * (c[3] * c[7] - c[4] * c[6]);
      TORCH_CHECK(std::fabs(det) > 1e-8 * lens[0] * lens[1] * lens[2],
                  "the midtown mesh backend rejects a singular cell");
      volume = std::fabs(det);
      const double id = 1.0 / det;
      Ainv[0] = (c[4] * c[8] - c[5] * c[7]) * id;
      Ainv[1] = (c[2] * c[7] - c[1] * c[8]) * id;
      Ainv[2] = (c[1] * c[5] - c[2] * c[4]) * id;
      Ainv[3] = (c[5] * c[6] - c[3] * c[8]) * id;
      Ainv[4] = (c[0] * c[8] - c[2] * c[6]) * id;
      Ainv[5] = (c[2] * c[3] - c[0] * c[5]) * id;
      Ainv[6] = (c[3] * c[7] - c[4] * c[6]) * id;
      Ainv[7] = (c[1] * c[6] - c[0] * c[7]) * id;
      Ainv[8] = (c[0] * c[4] - c[1] * c[3]) * id;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          double s = 0.0;
          for (int d = 0; d < 3; ++d) {
            s += Ainv[3 * d + a] * Ainv[3 * d + b];
          }
          W[3 * a + b] = s;
        }
      }
    } else {
      volume = lx * ly * lz;
    }

    tm.mark("state");
    // ── charges, with the uniform constraint of the op (on the device) ──
    auto mask_f = node_mask.narrow(0, begin, count);
    auto raw = charge.narrow(0, begin, count).select(1, 0);
    auto q = at::empty({count}, double_options);
    auto csums = at::zeros({2}, double_options);
    const int qblocks = static_cast<int>((count + 255) / 256);
    k_midtown_charges1<<<qblocks, 256>>>(
        static_cast<int>(count), raw.data_ptr<float>(), mask_f.data_ptr<bool>(),
        q.data_ptr<double>(), csums.data_ptr<double>());
    k_midtown_charges2<<<qblocks, 256>>>(
        static_cast<int>(count), mask_f.data_ptr<bool>(),
        csums.data_ptr<double>(), qtot + f, static_cast<int>(channel),
        q.data_ptr<double>());

    // ── positions: the midtown kernels wrap into [0, L) themselves ──
    auto c_f = coord.narrow(0, begin, count);

    // ── grid, green tables and device state ──
    const int nx = midtown_axis(lens[0], spacing);
    const int ny = midtown_axis(lens[1], spacing);
    const int nz = midtown_axis(lens[2], spacing);
    // The green tables and the metric depend on the cell, but an NPT step
    // changes it by ~1e-4: rebuilding them every step costs milliseconds.
    // Rebuild only when the cell drifts past a tolerance (the box_tol idea of
    // the old solver) or the grid, the spline or the kernel changes.
    double cell_drift = 0.0;
    if (local.ready) {
      for (int k = 0; k < 9; ++k) {
        const double scale = std::max(std::fabs(local.cell[k]), 1.0);
        cell_drift = std::max(cell_drift, std::fabs(local.cell[k] - c[k]) / scale);
      }
    }
    // Must mirror the default chosen at the rebuild below, or a rebuild in one
    // form would be reported as satisfying a request for the other.
    //
    // Default: Form A on for every non-tilted midtown mesh, at both CubeS2
    // orders.  `SOG_MIDTOWN_FORM_A=0/1` overrides either way; tilted cells have
    // no measured window symbol and stay on Form B.
    const char* fa_env_want = std::getenv("SOG_MIDTOWN_FORM_A");
    // QuadS builds the exact window inverse directly (sog_green.h), so its
    // table is Form A by construction; the env switch below only governs the
    // CubeS2 variance subtraction.
    const int want_form =
        (is_quads || (!tilted &&
                      (fa_env_want != nullptr ? std::atoi(fa_env_want) != 0
                                              : true)))
            ? 1
            : 0;
    const bool same = local.ready && !local.slab && local.form == want_form &&
                      local.nx == nx && local.ny == ny && local.nz == nz &&
                      local.spline == spline && local.amp == amp &&
                      local.bw == bw && local.tilted == tilted &&
                      cell_drift <= 1e-3;
    if (!same) {
      if (std::getenv("SOG_MIDTOWN_DEBUG") != nullptr) {
        fprintf(stderr,
                "[midtown] rebuild grid %dx%dx%d tilted=%d bands=%zu\n", nx, ny,
                nz, int(tilted), amp.size());
      }
      std::vector<double> ge, gf, gv;
      // Refuse before the host-side green build: at a large box the replicated
      // mesh costs several GiB per rank, and a bare `cudaMalloc: out of memory`
      // arrives only after tens of GB of host tables and about a minute of
      // single-threaded work.  `setup_replicated` allocates f_gp (8 B), f_rho_r
      // (4 B) and f_rho_fx (8 B) per grid point; keep 10% headroom for the
      // cuFFT work area.
      {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const double ng = static_cast<double>(nx) * ny * nz;
        const double need = ng * 20.0;
        TORCH_CHECK(static_cast<double>(free_b) >= need * 1.10,
                    "midtown: the replicated mesh needs ",
                    need / 1073741824.0, " GiB of device memory for a ", nx,
                    "x", ny, "x", nz, " grid, but only ",
                    static_cast<double>(free_b) / 1073741824.0,
                    " GiB is free. Use a coarser mesh (a larger `pme_spacing` "
                    "or a less demanding `pme_accuracy`), a smaller box, or a "
                    "distributed backend where the force mode allows one.");
      }
      if (tilted) {
        sog_green::build_green_metric(nx, ny, nz, W, amp, bw, spline, ge, gf,
                                      gv);
      } else {
        sog_green::build_green(nx, ny, nz, lx, ly, lz, amp, bw, spline, ge, gf,
                               gv);
      }
      // The replicated solve reads only the single-precision meshes, so the
      // fp64 set (56 B per grid point) would never be touched.
      sog_gpu_setup_f32(local.state, nx, ny, nz, ge.data(), gf.data(),
                        gv.data());
      // Form A: replace the Form-B variance subtraction by the exact measured
      // assignment-window symbol (see docs/MESH_ERROR_MODEL_AND_FORM_A.md).
      //
      // Default: ON at both CubeS2 orders, for every non-tilted mesh.  The
      // variance-subtraction residual dominates the order-4 energy error
      // (18-78x on the same grid) and is small but nonzero at order 6; exact
      // inversion is the reference form and `SOG_MIDTOWN_FORM_A=0/1` overrides.
      const char* fa_env = std::getenv("SOG_MIDTOWN_FORM_A");
      const int nord = (spline == SOG_CUBES2_6) ? 6 : 4;
      // QuadS is already the exact window inverse; converting it again with
      // the CubeS2 symbol would apply the wrong deconvolution.
      const bool form_a =
          !tilted && !is_quads &&
          (fa_env != nullptr ? std::atoi(fa_env) != 0 : true);
      if (form_a) {
        // nth must be >= 1: with 0 the theta loops never run, acc stays zero
        // and w = 0 * inf is NaN, which the floor test then silently swallows
        // while zeroing the whole reciprocal interaction.
        const char* nth_env = std::getenv("SOG_MIDTOWN_FORM_A_NTH");
        const int nth = std::max(1, nth_env ? std::atoi(nth_env) : 8);
        const int fa_rc =
            sog_gpu_green_form_a(local.state, nord, nth, lx, ly, lz);
        TORCH_CHECK(fa_rc == 0, "midtown: Form-A window symbol failed rc=",
                    fa_rc);
      }
      local.form = want_form;
      sog_gpu_set_metric(local.state, Ainv, volume, tilted ? 1 : 0);
      local.nx = nx;
      local.ny = ny;
      local.nz = nz;
      local.spline = spline;
      local.amp = amp;
      local.bw = bw;
      local.tilted = tilted;
      local.slab = false;
      for (int k = 0; k < 9; ++k) {
        local.cell[k] = c[k];
      }
      local.ready = true;
    }

    // A tilted cell runs on the unit cell in fractional coordinates.
    torch::Tensor Ainv_mat;
    torch::Tensor x = c_f;
    if (tilted) {
      Ainv_mat = at::from_blob(Ainv, {3, 3}, at::kDouble)
                     .clone()
                     .to(coord.device());
      // s = r A^{-1} (row vectors): s_d = sum_j r_j (A^{-1})_{j d}.
      x = c_f.matmul(Ainv_mat);
    }
    const double ux = tilted ? 1.0 : lx;
    const double uy = tilted ? 1.0 : ly;
    const double uz = tilted ? 1.0 : lz;

    // ── the midtown solve, device to device ──
    auto d_vpot = torch::empty({count}, double_options);
    auto d_force = torch::empty({3 * count}, double_options);
    const double boxlo[3] = {0.0, 0.0, 0.0};
    sog_gpu_compute_dev_defer(
        local.state, static_cast<int>(count), x.data_ptr<double>(), boxlo, ux,
        uy, uz, q.data_ptr<double>(), prefactor, spline,
        /*want_potential=*/1, d_force.data_ptr<double>(),
        d_vpot.data_ptr<double>(), /*self_coeff=*/0.0, /*qsqsum=*/0.0,
        /*remove_self=*/0, static_cast<int>(force_mode));
    // The 20 reduction sums stay on the device: the finish kernels read them
    // in place, so no scalar and no copy crosses to the host.
    const double* red = sog_gpu_red_device_ptr(local.state);

    tm.mark("solve");
    // ── outputs, with the conventions of mesh_sum.cu, in two kernels ──
    auto f_s = d_force.reshape({count, 3});
    // A tilted cell gathers -dE/ds in fractional space; the physical force is
    // F = F_s A^{-T}.
    auto f3 = tilted ? f_s.matmul(Ainv_mat.transpose(0, 1)) : f_s;
    force.narrow(0, begin, count).copy_(f3);

    auto sums = at::zeros({10}, double_options);
    auto moments = at::empty({count * 9}, double_options);
    const int fblocks = static_cast<int>((count + 255) / 256);
    // The energy lands in the output directly: no buffer, no copy.
    double* energy_out = atom_energy.data_ptr<double>() + begin;
    k_midtown_finish1<<<fblocks, 256>>>(
        static_cast<int>(count), d_vpot.data_ptr<double>(),
        f3.data_ptr<double>(), q.data_ptr<double>(), mask_f.data_ptr<bool>(),
        c_f.data_ptr<double>(), prefactor, kernel_zero,
        static_cast<int>(channel), energy_out, sums.data_ptr<double>(),
        moments.data_ptr<double>());
    k_midtown_finish2<<<fblocks, 256>>>(
        static_cast<int>(count), d_vpot.data_ptr<double>(),
        q.data_ptr<double>(), mask_f.data_ptr<bool>(), prefactor, kernel_zero,
        static_cast<int>(channel), sums.data_ptr<double>(),
        csums.data_ptr<double>(), moments.data_ptr<double>(), red,
        0.5 * volume * prefactor,
        charge_gradient.narrow(0, begin, count).data_ptr<float>(),
        atom_virial.narrow(0, begin, count).data_ptr<float>());

    // The frame virial, from the same six sums, in one nine-thread kernel.
    k_midtown_virial<<<1, 32>>>(red, 0.5 * volume * prefactor,
                                virial.data_ptr<float>() + 9 * f);

    tm.mark("finish");
    begin += count;
  }
  tm.mark("postloop");
  // The caller keeps only the nodes this rank owns.
  //
  // The flush is NOT part of the ownership branch: with np > 1 every rank owns a
  // slice, so gating the report on "this rank owns the whole frame" meant the
  // timer printed nothing exactly where the per-step cost matters.  flush()
  // itself already throttles to one report per 50 calls.
  CpuTimer::flush();
  if (owner_begin == 0 && owner_count == n) {
  return {atom_energy, charge_gradient, force, atom_virial, virial};
  }
  auto cut = [&](const torch::Tensor& t) {
    return t.narrow(0, owner_begin, owner_count).contiguous();
  };
  return {cut(atom_energy), cut(charge_gradient), cut(force), cut(atom_virial),
          virial};
}

// The distributed solve: every rank spreads only its own atoms into the full
// mesh, the meshes are summed over the ranks, and then each rank runs the
// transform and gathers its own atoms.  The mesh itself is still replicated
// (one full grid per rank); what is distributed is the particle work.
ChargeSumOutputs midtown_sum_local(
    const torch::Tensor& coord, const torch::Tensor& charge,
    const torch::Tensor& node_mask, const torch::Tensor& n_node_per_frame,
    const torch::Tensor& total_charge, const torch::Tensor& box,
    const torch::Tensor& coefficient, int64_t kernel, int64_t channel,
    double prefactor, int64_t order, double spacing,
    const torch::Tensor* communicator) {
  TORCH_CHECK(!form_a_requested(),
              "midtown: SOG_MIDTOWN_FORM_A is implemented on the replicated "
              "solve only; unset it for the local-spread path");
  TORCH_CHECK(kernel == kSog,
              "the midtown mesh backend evaluates the SOG kernel only");
  TORCH_CHECK(charge.size(1) == 1,
              "the midtown mesh backend supports one channel");
  TORCH_CHECK(channel == -1 || channel == 0,
              "the midtown mesh backend supports a constraint on channel 0");
  TORCH_CHECK(n_node_per_frame.size(0) == 1,
              "the distributed midtown backend takes one frame");
  const int64_t count = coord.size(0);

  const bool coef_host = coefficient.device().is_cpu();
  torch::Tensor coef_h =
      coef_host ? coefficient.detach().flatten().contiguous()
                : coefficient.detach().flatten().to(at::kCPU).contiguous();
  const double* pk = coef_h.data_ptr<double>();
  const int64_t bands = coefficient.size(0) / 2;
  std::vector<double> amp(bands), bw(bands);
  double kernel_zero = 0.0;
  for (int64_t m = 0; m < bands; ++m) {
    const double a = pk[m];
    const double rate = pk[bands + m];
    TORCH_CHECK(rate > 0.0, "the midtown mesh backend needs positive rates");
    const double sigma_sq = 1.0 / (2.0 * rate * kLn2);
    bw[m] = sigma_sq;
    amp[m] = a * std::pow(kTwoPi * sigma_sq, 1.5);
    kernel_zero += a;
  }
  TORCH_CHECK(total_charge.device().is_cuda(),
              "the midtown mesh backend reads the total charge on the device");
  const double* qtot = total_charge.data_ptr<double>();
  const int spline = midtown_spline(order);

  torch::Tensor cells = box.detach().to(at::kCPU).to(at::kDouble).contiguous();
  const double* c = cells.data_ptr<double>();
  const bool tilted = std::fabs(c[1]) >= 1e-12 || std::fabs(c[2]) >= 1e-12 ||
                      std::fabs(c[3]) >= 1e-12 || std::fabs(c[5]) >= 1e-12 ||
                      std::fabs(c[6]) >= 1e-12 || std::fabs(c[7]) >= 1e-12;
  TORCH_CHECK(!tilted,
              "the distributed midtown backend supports orthogonal cells only");
  const double lx = c[0], ly = c[4], lz = c[8];
  const double volume = lx * ly * lz;

  const auto double_options = coord.options().dtype(at::kDouble);
  const auto float_options = charge.options().dtype(at::kFloat);
  auto atom_energy = torch::zeros({count, 1}, double_options);
  auto charge_gradient = torch::zeros({count, 1}, float_options);
  auto force = torch::zeros({count, 3}, float_options);
  auto atom_virial = torch::zeros({count, 9}, float_options);
  auto virial = torch::zeros({1, 9}, float_options);

  auto mask_f = node_mask.narrow(0, 0, count);
  auto raw = charge.narrow(0, 0, count).select(1, 0);
  auto q = at::empty({count}, double_options);
  auto csums = at::zeros({2}, double_options);
  const int qblocks = static_cast<int>((count + 255) / 256);
  k_midtown_charges1<<<qblocks, 256>>>(static_cast<int>(count),
                                       raw.data_ptr<float>(),
                                       mask_f.data_ptr<bool>(),
                                       q.data_ptr<double>(),
                                       csums.data_ptr<double>());
  // The uniform constraint of the operator is a property of the whole frame,
  // not of this rank's slice: shifting each rank to neutrality on its own
  // changes the charges by a different constant per rank.
  csums.copy_(sum_over_domains(*communicator, csums));
  k_midtown_charges2<<<qblocks, 256>>>(
      static_cast<int>(count), mask_f.data_ptr<bool>(),
      csums.data_ptr<double>(), qtot, static_cast<int>(channel),
      q.data_ptr<double>());
  auto count_d = csums.narrow(0, 0, 1).clamp_min(1.0);

  std::lock_guard<std::mutex> lock(midtown_mutex());
  MidtownState& local = midtown_state();
  if (local.state == nullptr) {
    local.state = sog_gpu_create();
    TORCH_CHECK(local.state != nullptr, "midtown: sog_gpu_create failed");
  }
  const int nx = midtown_axis(lx, spacing);
  const int ny = midtown_axis(ly, spacing);
  const int nz = midtown_axis(lz, spacing);
  // The green tables depend on the cell, and an MD barostat moves it by ~1e-4
  // per step: rebuilding them every call cost 11 ms at a 100^3 mesh, which is
  // more than the whole solve.  Rebuild only when the cell drifts past the
  // tolerance, or the grid, the spline or the kernel changes.
  (void)lx;
  double cell_drift = 0.0;
  if (local.ready) {
    for (int k = 0; k < 9; ++k) {
      const double scale = std::max(std::fabs(local.cell[k]), 1.0);
      cell_drift = std::max(cell_drift, std::fabs(local.cell[k] - c[k]) / scale);
    }
  }
  const bool same = local.ready && !local.slab && local.form == 0 &&
                    local.nx == nx && local.ny == ny && local.nz == nz &&
                    local.spline == spline && local.amp == amp &&
                    local.bw == bw && !local.tilted && cell_drift <= 1e-3;
  if (!same) {
    static const double kZero[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<double> ge, gf, gv;
    sog_green::build_green(nx, ny, nz, lx, ly, lz, amp, bw, spline, ge, gf, gv);
    // Local spread runs sog_gpu_solve_f, i.e. the single-precision meshes.
    sog_gpu_setup_f32(local.state, nx, ny, nz, ge.data(), gf.data(),
                      gv.data());
    sog_gpu_set_metric(local.state, kZero, volume, 0);
    local.nx = nx;
    local.ny = ny;
    local.nz = nz;
    local.spline = spline;
    local.amp = amp;
    local.bw = bw;
    local.tilted = false;
    local.slab = false;
    for (int k = 0; k < 9; ++k) {
      local.cell[k] = c[k];
    }
    local.ready = true;
  }

  auto c_f = coord.narrow(0, 0, count);
  auto d_vpot = torch::empty({count}, double_options);
  auto d_force = torch::empty({3 * count}, double_options);
  const double boxlo[3] = {0.0, 0.0, 0.0};
  sog_gpu_spread_f(local.state, static_cast<int>(count),
                   c_f.data_ptr<double>(), boxlo, lx, ly, lz,
                   q.data_ptr<double>(), (double)(nx * ny * nz) / volume,
                   spline);
  sog_gpu_fx_to_real(local.state);
  {
    const int64_t ng = int64_t{nx} * ny * nz;
    auto real = at::from_blob(sog_gpu_real_device_ptr(local.state), {ng},
                              float_options);
    // The density is float32 and grid-sized: it stays on the device when the
    // MPI library reads device memory.  The float64 allgather it used to take
    // moved 8 bytes per grid point through the host on every step.
    const bool red_dbg = (std::getenv("SOG_DIST_CPU") != nullptr);
    auto t0 = std::chrono::steady_clock::now();
    real.copy_(sum_over_domains_f32(*communicator, real));
    if (red_dbg) {
      cudaDeviceSynchronize();
      auto t1 = std::chrono::steady_clock::now();
      fprintf(stderr, "[dist] density reduce %.2f ms\n",
              std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
  }
  sog_gpu_solve_f(local.state, static_cast<int>(count), c_f.data_ptr<double>(),
                  boxlo, lx, ly, lz, q.data_ptr<double>(), prefactor,
                  /*want_pot=*/1, /*density_ready=*/1, d_force.data_ptr<double>(),
                  d_vpot.data_ptr<double>(), spline);
  // red[0..6] comes from the k-space pass over the SUMMED density, so it is
  // already the whole frame's energy and virial; summing it again would double
  // it (it is a mode sum, not an atom sum).
  auto red_dev = at::from_blob(sog_gpu_red_device_ptr(local.state), {20},
                               double_options);

  auto f_s = d_force.reshape({count, 3});
  force.copy_(f_s);
  auto sums = at::zeros({10}, double_options);
  auto moments = at::empty({count * 9}, double_options);
  auto energy_f = at::empty({count}, double_options);
  const int fblocks = static_cast<int>((count + 255) / 256);
  k_midtown_finish1<<<fblocks, 256>>>(
      static_cast<int>(count), d_vpot.data_ptr<double>(),
      f_s.data_ptr<double>(), q.data_ptr<double>(), mask_f.data_ptr<bool>(),
      c_f.data_ptr<double>(), prefactor, kernel_zero, static_cast<int>(channel),
      energy_f.data_ptr<double>(), sums.data_ptr<double>(),
      moments.data_ptr<double>());
  // The frame sums (the constrained-channel potential sum and the nine moment
  // sums) are what finish2 shares over the charged nodes, so they are global.
  sums.copy_(sum_over_domains(*communicator, sums));
  auto red = red_dev.clone();
  auto vscale = red.narrow(0, 1, 6) * (0.5 * volume * prefactor);
  k_midtown_finish2<<<fblocks, 256>>>(
      static_cast<int>(count), d_vpot.data_ptr<double>(),
      q.data_ptr<double>(), mask_f.data_ptr<bool>(), prefactor, kernel_zero,
      static_cast<int>(channel), sums.data_ptr<double>(),
      count_d.data_ptr<double>(), moments.data_ptr<double>(),
      red.data_ptr<double>(), 0.5 * volume * prefactor,
      charge_gradient.data_ptr<float>(), atom_virial.data_ptr<float>());
  atom_energy.copy_(energy_f.reshape({count, 1}));
  static const int64_t kSym[9] = {0, 3, 4, 3, 1, 5, 4, 5, 2};
  auto sym_index = at::from_blob(const_cast<int64_t*>(kSym), {9}, at::kLong)
                       .clone()
                       .to(coord.device());
  auto w = vscale.index_select(0, sym_index).reshape({3, 3});
  virial.copy_(w.reshape({1, 9}).to(at::kFloat));
  return {atom_energy, charge_gradient, force, atom_virial, virial};
}


// ── Slab (z-decomposed) distributed solve ────────────────────────────────
// Each rank holds nz/np planes of the mesh plus a ghost margin, spreads only
// its own atoms, and the transform is a two-transpose 3-D FFT: (x,y) FFT on
// the z-slab, an all-to-all into y-pencils, the z FFT, the green multiply, and
// the same two transposes back.  With NCCL every buffer between the messages
// stays on the device.  Memory is 1/np of the replicated path; the density
// never crosses a rank boundary at all (unlike the local-spread variant, which
// sums the whole mesh).
namespace {

// NCCL bring-up: rank 0 makes the unique id and every rank receives it.
bool ensure_nccl(const torch::Tensor& communicator, int np, int me) {
  static int done_for = -1;
  if (done_for == np) {
    return true;
  }
  const bool tr = std::getenv("SOG_SLAB_TRACE") != nullptr;
  // NCCL boots over the first non-loopback interface by default and segfaults
  // inside ncclCommInitRank on this machine; the loopback + no-IB settings are
  // what single-node runs need.  An explicit setting of either wins.
  if (std::getenv("NCCL_SOCKET_IFNAME") == nullptr) {
    setenv("NCCL_SOCKET_IFNAME", "lo", 1);
  }
  if (std::getenv("NCCL_IB_DISABLE") == nullptr) {
    setenv("NCCL_IB_DISABLE", "1", 1);
  }
  if (tr) fprintf(stderr, "[slab] nccl: probing\n");
  if (sog_gpu_nccl_available() == 0) {
    return false;
  }
  if (tr) fprintf(stderr, "[slab] nccl: available\n");
  char id[128];
  if (me == 0) {
    TORCH_CHECK(sog_gpu_nccl_get_id(id) == 0, "midtown slab: ncclGetUniqueId failed");
  }
  if (tr) fprintf(stderr, "[slab] nccl: id made\n");
  broadcast_bytes(communicator, id, 128);
  if (tr) fprintf(stderr, "[slab] nccl: id broadcast\n");
  // The communicator has a device-memory price and it is charged whether or not
  // the mesh is large enough to need the slab, so report it next to the mesh
  // accounting (SOG_DIST_DEBUG prints that one).
  size_t free_before = 0, total_gpu = 0;
  const bool mem = std::getenv("SOG_DIST_DEBUG") != nullptr;
  if (mem) {
    cudaMemGetInfo(&free_before, &total_gpu);
  }
  TORCH_CHECK(sog_gpu_nccl_init(np, me, id) == 0,
              "midtown slab: ncclCommInitRank failed");
  if (mem) {
    size_t free_after = 0, t2 = 0;
    cudaMemGetInfo(&free_after, &t2);
    fprintf(stderr, "[sog-mem] nccl comm (np=%d rank=%d): %.1f MiB\n", np, me,
            (double)(free_before - free_after) / 1048576.0);
    fflush(stderr);
  }
  if (tr) fprintf(stderr, "[slab] nccl: comm ready\n");
  done_for = np;
  return true;
}

}  // namespace

ChargeSumOutputs midtown_sum_slab(
    const torch::Tensor& coord, const torch::Tensor& charge,
    const torch::Tensor& node_mask, const torch::Tensor& n_node_per_frame,
    const torch::Tensor& total_charge, const torch::Tensor& box,
    const torch::Tensor& coefficient, int64_t kernel, int64_t channel,
    double prefactor, int64_t order, double spacing,
    const torch::Tensor* communicator, bool quads, bool bspline) {
  TORCH_CHECK(kernel == kSog, "the slab midtown backend evaluates the SOG kernel only");
  TORCH_CHECK(charge.size(1) == 1, "the slab midtown backend supports one channel");
  TORCH_CHECK(channel == -1 || channel == 0,
              "the slab midtown backend supports a constraint on channel 0");
  TORCH_CHECK(n_node_per_frame.size(0) == 1,
              "the slab midtown backend takes one frame");
  const int64_t count = coord.size(0);

  torch::Tensor coef_h = coefficient.device().is_cpu()
                             ? coefficient.detach().flatten().contiguous()
                             : coefficient.detach().flatten().to(at::kCPU).contiguous();
  const double* pk = coef_h.data_ptr<double>();
  const int64_t bands = coefficient.size(0) / 2;
  std::vector<double> amp(bands), bw(bands);
  double kernel_zero = 0.0;
  for (int64_t m = 0; m < bands; ++m) {
    const double rate = pk[bands + m];
    TORCH_CHECK(rate > 0.0, "the midtown mesh backend needs positive rates");
    const double sigma_sq = 1.0 / (2.0 * rate * kLn2);
    bw[m] = sigma_sq;
    amp[m] = pk[m] * std::pow(kTwoPi * sigma_sq, 1.5);
    kernel_zero += pk[m];
  }
  torch::Tensor cells = box.detach().to(at::kCPU).to(at::kDouble).contiguous();
  const double* c = cells.data_ptr<double>();
  const bool tilted = std::fabs(c[1]) >= 1e-12 || std::fabs(c[2]) >= 1e-12 ||
                      std::fabs(c[3]) >= 1e-12 || std::fabs(c[5]) >= 1e-12 ||
                      std::fabs(c[6]) >= 1e-12 || std::fabs(c[7]) >= 1e-12;
  TORCH_CHECK(!tilted, "the slab midtown backend supports orthogonal cells only");
  const double lx = c[0], ly = c[4], lz = c[8];
  const double volume = lx * ly * lz;
  TORCH_CHECK(!bspline || (order == 4 || order == 5 || order == 6),
              "the slab B-spline backend supports order 4, 5 or 6, got ", order);
  const int spline =
      bspline ? (order == 6 ? SOG_BSPLINE_6
                            : (order == 5 ? SOG_BSPLINE_5 : SOG_BSPLINE_4))
              : midtown_spline(order, quads);

  const int np = deepmd::long_range::domain_count(*communicator);
  const int me = deepmd::long_range::domain_rank(*communicator);
  auto TR = [](){ return std::getenv("SOG_SLAB_TRACE") != nullptr; };
  if (TR()) fprintf(stderr, "[slab] enter np=%d me=%d\n", np, me);
  // NCCL only moves the transposed blocks BETWEEN ranks.  At np=1 both
  // all-to-alls are an identity permutation and the accumulator sum is already
  // complete, so no communicator is created and the backend runs without
  // libnccl.  That also avoids a rank-sends-to-itself NCCL path that has no
  // validated counterpart (the reference never uses NCCL at np=1).
  const bool use_nccl = (np > 1);
  if (use_nccl) {
    TORCH_CHECK(ensure_nccl(*communicator, np, me),
                "the slab midtown backend needs NCCL for more than one rank "
                "(libnccl.so.2, or the CUDA wheel's nvidia/nccl)");
  }

  const int nx = midtown_axis(lx, spacing);
  const int ny = midtown_axis(ly, spacing);
  const int nz = midtown_axis(lz, spacing);
  const int zlo = me * nz / np, zhi = (me + 1) * nz / np - 1;
  const int ZL = zhi - zlo + 1;
  const int ylo = me * ny / np, YL = (me + 1) * ny / np - ylo;
  // Ghost margin = the stencil width.  CubeS2/QuadS reach `order` points per
  // axis (offsets -1..2 for 4, -2..3 for 6); B-spline of order p reaches
  // p points from floor(p-1)/2 to the left, so p covers it too.
  const int gh = bspline ? static_cast<int>(order) : ((order == 6) ? 6 : 4);
  TORCH_CHECK(ZL >= gh, "midtown slab: nz/np = ", ZL,
              " is thinner than the stencil (", gh,
              "); use fewer ranks or a finer grid");
  // full_brick: the atom decomposition is NOT rank == z-slab, so a rank owns
  // atoms at any z.  Every rank then spreads into a full-height brick and the
  // density/potential are made global with one device all-reduce each; the
  // transforms below still work on this rank's nz/np z-slab, so the mesh
  // memory and the collective traffic of the transposes are unchanged.  The
  // trade is the spread brick: it holds all nz planes (+ghosts) instead of
  // ZL, i.e. the full mesh rather than 1/np of it.
  //
  // Selected automatically when a single z band owns the box (np == 1), or
  // forced with SOG_MIDTOWN_FULLBRICK=1 for a processor grid that splits x
  // and/or y (the LAMMPS default).  Without the force the z-band guard below
  // rejects such a grid rather than silently spreading atoms onto the wrong
  // slab.
  const char* fb_env = std::getenv("SOG_MIDTOWN_FULLBRICK");
  const bool full_brick =
      (ZL == nz) || (fb_env != nullptr && std::atoi(fb_env) != 0);
  const int zb_lo = full_brick ? -gh : zlo - gh;
  const int zb_hi = full_brick ? nz - 1 + gh : zhi + gh;
  const int ZB = full_brick ? nz + 2 * gh : ZL + 2 * gh;
  const int zoff = zlo - zb_lo;
  const size_t plane = (size_t)nx * ny;
  const double scaleinv = 1.0 / (double)((size_t)nx * ny * nz);
  const double rho_scale = (double)((size_t)nx * ny * nz) / volume;

  const auto float_options = charge.options().dtype(at::kFloat);
  const auto double_options = coord.options().dtype(at::kDouble);

  std::lock_guard<std::mutex> lock(midtown_mutex());
  MidtownState& local = midtown_state();
  if (local.state == nullptr) {
    local.state = sog_gpu_create();
    TORCH_CHECK(local.state != nullptr, "midtown: sog_gpu_create failed");
  }

  // Green tables: rebuilt only when the cell drifts, as in the shared-memory
  // path, and sliced to this rank's y-pencils.
  double cell_drift = 0.0;
  if (local.ready) {
    for (int k = 0; k < 9; ++k) {
      const double scale = std::max(std::fabs(local.cell[k]), 1.0);
      cell_drift = std::max(cell_drift, std::fabs(local.cell[k] - c[k]) / scale);
    }
  }
  const bool is_quads = (spline == SOG_QUADS_4 || spline == SOG_QUADS_6);
  const bool is_bspline = (spline >= 200);
  const char* fa_env_slab = std::getenv("SOG_MIDTOWN_FORM_A");
  const int want_form =
      (is_quads || is_bspline ||
       (fa_env_slab != nullptr ? std::atoi(fa_env_slab) != 0 : true))
          ? 1
          : 0;
  // Only the CubeS2 Form A needs the measured window; QuadS inverts its window
  // exactly and B-spline inverts its assignment factor exactly, both inside
  // build_green_slab_half.
  const bool use_window = (!is_quads && !is_bspline && want_form == 1);
  const bool same = local.ready && local.slab && local.form == want_form &&
                    local.nx == nx &&
                    local.ny == ny && local.nz == nz &&
                    local.spline == spline && local.amp == amp &&
                    local.bw == bw && local.slab_ylo == ylo &&
                    local.slab_YL == YL && cell_drift <= 1e-3;
  if (!same) {
    // The slab tables are the COMPACT x half-spectrum (Nh = nx/2+1) of this
    // rank's y-slice, laid out as kh + Nh*(iyl + YL*iz) -- exactly the index
    // k_green_pencil reads.  Building the full x table and slicing it with an
    // x-stride of nx instead makes the kernel read a value belonging to another
    // (kh, iy, iz) cell, which silently produces a plausible but wrong solve.
    //
    // Form A (CubeS2): measure the assignment window <|W|^2> on the full grid
    // (self-contained, temporary full-grid plan) and hand it to the builder,
    // which then divides the Form-B slice by D*<|W|^2>.  The window is a global
    // property, identical on every rank, so no communication is needed.
    std::vector<double> window;
    const double* window_ptr = nullptr;
    if (use_window) {
      const size_t nw = (size_t)(nx / 2 + 1) * ny * nz;
      window.resize(nw);
      const int nord = (spline == SOG_CUBES2_6) ? 6 : 4;
      const char* nth_env = std::getenv("SOG_MIDTOWN_FORM_A_NTH");
      const int nth = std::max(1, nth_env ? std::atoi(nth_env) : 8);
      const int rc = sog_gpu_measure_window(nx, ny, nz, nord, nth, lx, ly, lz,
                                            window.data());
      TORCH_CHECK(rc == 0, "midtown slab: Form-A window measurement failed rc=",
                  rc);
      window_ptr = window.data();
    }
    std::vector<double> sge, sgf, sgv;
    sog_green::build_green_slab_half(nx, ny, nz, lx, ly, lz, amp, bw, spline,
                                     ylo, YL, sge, sgf, sgv, window_ptr);
    const size_t gn = (size_t)(nx / 2 + 1) * YL * nz;
    TORCH_CHECK(sge.size() == gn, "midtown slab: green table size mismatch");
    sog_gpu_setup_slab(local.state, nx, ny, nz, YL, ZL, sge.data(), sgv.data(),
                       gn);
    local.nx = nx;
    local.ny = ny;
    local.nz = nz;
    local.spline = spline;
    local.amp = amp;
    local.bw = bw;
    local.slab = true;
    local.form = want_form;
    local.slab_ylo = ylo;
    local.slab_YL = YL;
    for (int k = 0; k < 9; ++k) {
      local.cell[k] = c[k];
    }
    local.ready = true;
  }
  TORCH_CHECK(sog_gpu_slab_setup(local.state, nx, ny, nz, ZL, YL) == 0,
              "midtown slab: the cuFFT plans failed");

  // Charges, with the frame-wide uniform constraint.
  auto mask_f = node_mask.narrow(0, 0, count);
  auto raw = charge.narrow(0, 0, count).select(1, 0);
  auto q = at::empty({count}, double_options);
  auto csums = at::zeros({2}, double_options);
  const int qblocks = static_cast<int>((count + 255) / 256);
  k_midtown_charges1<<<qblocks, 256>>>(static_cast<int>(count),
                                       raw.data_ptr<float>(),
                                       mask_f.data_ptr<bool>(),
                                       q.data_ptr<double>(),
                                       csums.data_ptr<double>());
  csums.copy_(sum_over_domains(*communicator, csums));
  const double* qtot = total_charge.data_ptr<double>();
  k_midtown_charges2<<<qblocks, 256>>>(static_cast<int>(count),
                                       mask_f.data_ptr<bool>(),
                                       csums.data_ptr<double>(), qtot,
                                       static_cast<int>(channel),
                                       q.data_ptr<double>());

  auto c_f = coord.narrow(0, 0, count);
  const double boxlo[3] = {0.0, 0.0, 0.0};
  const int prev = (me + np - 1) % np, next = (me + 1) % np;

  // Stage timing (SOG_SLAB_TIME=1).  Events rather than wall-clock: almost every
  // stage enqueues asynchronously, so only the last one would be honest.
  const bool tim = std::getenv("SOG_SLAB_TIME") != nullptr;
  static cudaEvent_t ev[11];
  static bool ev_ready = false;
  if (tim && !ev_ready) {
    for (int i = 0; i < 11; ++i) cudaEventCreate(&ev[i]);
    ev_ready = true;
  }
  struct StageClock {
    bool on;
    cudaEvent_t* e;
  } clock{tim, ev};
  auto mark = [&](int i) {
    if (clock.on) cudaEventRecord(clock.e[i], 0);
  };
  // The mesh is split by MPI rank, which is only correct when the framework's
  // atom decomposition is 1 x 1 x np (rank == z slab).  Without this check a
  // rank that owns atoms anywhere in the box spreads them onto its own slab and
  // the solve silently returns a plausible-looking wrong number.  Ghost atoms
  // carry q == 0 and are skipped: a rank's ghosts span the whole box.
  //
  // Checked once per (np, nz): the decomposition is a property of the caller,
  // not of the frame, so a per-call check only costs three device->host syncs
  // and a scalar loop over every atom -- which measured 0.25 ms of a 1.0 ms
  // call before it was hoisted.  The bounds (not the atoms) are what a
  // mis-decomposed run violates, and it violates them on its first frame.
  if (np > 1 && !full_brick) {
    static int checked_np = -1, checked_nz = -1;
    if (checked_np != np || checked_nz != nz) {
      // Device side: only two integers come back, in one sync.
      torch::Tensor sel = mask_f & (q != 0.0);
      const bool any = sel.any().item<bool>();
      if (any) {
        torch::Tensor zs = c_f.select(1, 2).masked_select(sel);
        const double zmin = zs.min().item<double>() / lz * nz;
        const double zmax = zs.max().item<double>() / lz * nz;
        // Compare against the CONTINUOUS z-subdomain [me*nz/np, (me+1)*nz/np],
        // not the mesh slab, so this only fires for a decomposition that does
        // not split z at all (one grid cell of tolerance covers the rounding).
        const double zgb = static_cast<double>(me) * nz / np;
        const double zgt = static_cast<double>(me + 1) * nz / np;
        TORCH_CHECK(zmin >= zgb - 1.0 && zmax <= zgt + 1.0,
                    "midtown slab: rank ", me,
                    " owns atoms spanning z-grid [", zmin, ", ", zmax,
                    "] but its slab is [", zlo, ", ", zhi,
                    "]. The distributed solver needs the framework's processor "
                    "grid to be 1 x 1 x nprocs (rank == z slab).");
      }
      checked_np = np;
      checked_nz = nz;
    }
  }

  if (TR()) fprintf(stderr, "[slab] setup done n=%d,%d,%d ZL=%d YL=%d gh=%d ZB=%d zoff=%d\n", nx, ny, nz, ZL, YL, gh, ZB, zoff);
  mark(0);  // start: everything above is setup, charged to no stage
  sog_gpu_spread_brick_dev(local.state, static_cast<int>(count),
                           c_f.data_ptr<double>(), boxlo, lx, ly, lz,
                           q.data_ptr<double>(), rho_scale, spline, nx, ny, nz,
                           0, nx - 1, 0, ny - 1, zb_lo, zb_hi);
  if (full_brick) {
    // No z neighbour: the ghost planes are periodic images inside this rank's
    // own full-height brick.  Fold them first, then sum the bricks across
    // ranks so every rank holds the GLOBAL density.  The 2^40 integers add
    // exactly, so the global mesh is independent of the reduction order and of
    // the atom decomposition.
    for (int j = 0; j < gh; ++j) {
      sog_gpu_brick_add_planes(local.state, plane * (size_t)j,
                               plane * (size_t)(nz + j), plane);
      sog_gpu_brick_add_planes(local.state, plane * (size_t)(nz + gh + j),
                               plane * (size_t)(gh + j), plane);
    }
    if (np > 1) {
      TORCH_CHECK(sog_gpu_brick_allreduce_fx(local.state, plane * (size_t)gh,
                                             plane * (size_t)nz) == 0,
                  "midtown slab: full-brick density all-reduce failed");
    }
  } else {
    TORCH_CHECK(sog_gpu_nccl_halo_rev(local.state, prev, next, plane, zoff, ZL,
                                      gh, gh) == 0,
                "midtown slab: NCCL reverse halo failed");
  }

  mark(1);  // spread + reverse halo
  if (TR()) fprintf(stderr, "[slab] spread+halo_rev done\n");
  std::vector<int> scnt(np), sdisp(np), rcnt(np), rdisp(np);
  const int Nh = nx / 2 + 1;
  for (int d = 0; d < np; ++d) {
    scnt[d] = Nh * ((d + 1) * ny / np - d * ny / np) * ZL * 2;
  }
  for (int r = 0; r < np; ++r) {
    rcnt[r] = Nh * YL * ((r + 1) * nz / np - r * nz / np) * 2;
  }
  sdisp[0] = 0;
  for (int d = 1; d < np; ++d) sdisp[d] = sdisp[d - 1] + scnt[d - 1];
  rdisp[0] = 0;
  for (int r = 1; r < np; ++r) rdisp[r] = rdisp[r - 1] + rcnt[r - 1];

  if (TR()) fprintf(stderr, "[slab] counts done\n");
  TORCH_CHECK(sog_gpu_slab_xyfft_pack_brick_dev(local.state, nx, ny, nz, ZL, np,
                                                zoff) == 0,
              "midtown slab: (x,y) FFT + pack failed");
  mark(2);  // (x,y) FFT + pack
  if (TR()) fprintf(stderr, "[slab] xyfft pack done\n");
  if (use_nccl) {
    TORCH_CHECK(sog_gpu_nccl_alltoall_fwd(local.state, scnt.data(),
                                          sdisp.data(), rcnt.data(),
                                          rdisp.data(), np) == 0,
                "midtown slab: NCCL forward all-to-all failed");
  } else {
    // np == 1: one block, d_pot -> d_rho, no collective.
    TORCH_CHECK(sog_gpu_slab_copy_packed(local.state, /*forward=*/1,
                                         (size_t)scnt[0]) == 0,
                "midtown slab: packed block copy failed");
  }
  mark(3);  // forward all-to-all
  if (TR()) fprintf(stderr, "[slab] alltoall fwd done\n");
  TORCH_CHECK(sog_gpu_slab_unpack_green_izpack_dev(
                  local.state, nx, ny, nz, ylo, YL, lx, ly, lz, scaleinv, np) == 0,
              "midtown slab: unpack / z FFT / green failed");
  mark(4);  // unpack + z FFT + green + inverse
  if (TR()) fprintf(stderr, "[slab] zfft+green done\n");
  // At np=1 the accumulators this rank wrote are already the frame totals.
  if (use_nccl) {
    TORCH_CHECK(sog_gpu_nccl_allreduce_red(local.state) == 0,
                "midtown slab: NCCL accumulator reduction failed");
  }
  mark(5);  // accumulator all-reduce
  if (TR()) fprintf(stderr, "[slab] red reduce done\n");
  if (use_nccl) {
    TORCH_CHECK(sog_gpu_nccl_alltoall_bwd(local.state, rcnt.data(),
                                          rdisp.data(), scnt.data(),
                                          sdisp.data(), np) == 0,
                "midtown slab: NCCL inverse all-to-all failed");
  } else {
    TORCH_CHECK(sog_gpu_slab_copy_packed(local.state, /*forward=*/0,
                                         (size_t)rcnt[0]) == 0,
                "midtown slab: packed block copy failed");
  }
  mark(6);  // inverse all-to-all
  if (TR()) fprintf(stderr, "[slab] alltoall bwd done\n");
  TORCH_CHECK(sog_gpu_slab_unpack_ixyfft_to_brick_dev(local.state, nx, ny, nz,
                                                      ZL, np, zoff, ZB) == 0,
              "midtown slab: unpack + inverse (x,y) FFT failed");
  mark(7);  // unpack + inverse (x,y) FFT
  if (full_brick) {
    // The inverse transform wrote this rank's own ZL planes; the rest of the
    // brick was zeroed by the unpack.  Sum across ranks to get the full
    // potential, then fill the periodic ghosts from the owned edge planes.
    if (np > 1) {
      TORCH_CHECK(sog_gpu_brick_allreduce_f(local.state, plane * (size_t)gh,
                                            plane * (size_t)nz) == 0,
                  "midtown slab: full-brick potential all-reduce failed");
    }
    for (int j = 0; j < gh; ++j) {
      // lower ghost <- the last gh owned planes (periodic wrap)
      sog_gpu_brick_copy_planes(local.state, plane * (size_t)(nz + j),
                                plane * (size_t)j, plane);
      // upper ghost <- the first gh owned planes
      sog_gpu_brick_copy_planes(local.state, plane * (size_t)(gh + j),
                                plane * (size_t)(nz + gh + j), plane);
    }
  } else {
    TORCH_CHECK(sog_gpu_nccl_halo_fwd(local.state, prev, next, plane, zoff, ZL,
                                      gh, gh) == 0,
                "midtown slab: NCCL forward halo failed");
  }

  mark(8);  // forward halo
  if (TR()) fprintf(stderr, "[slab] ixyfft done\n");
  std::vector<double> force(3 * (size_t)count), vpot((size_t)count);
  sog_gpu_gather_brick_dev(local.state, static_cast<int>(count),
                           c_f.data_ptr<double>(), boxlo, lx, ly, lz,
                           q.data_ptr<double>(), prefactor, spline, nx, ny, nz,
                           0, nx - 1, 0, ny - 1, zb_lo, zb_hi, force.data(),
                           vpot.data());
  mark(9);  // gather
  if (TR()) fprintf(stderr, "[slab] gather done\n");
  if (tim) {
    static const char* kNames[9] = {
        "spread+halo_rev", "xyfft+pack",    "alltoall fwd", "zfft+green",
        "red allreduce",   "alltoall bwd",  "ixyfft",       "halo fwd",
        "gather"};
    cudaEventSynchronize(ev[9]);
    float ms[9];
    for (int i = 0; i < 9; ++i) {
      cudaEventElapsedTime(&ms[i], ev[i], ev[i + 1]);
    }
    float tot = 0.0f;
    for (int i = 0; i < 9; ++i) tot += ms[i];
    fprintf(stderr, "[slab-time] rank %d np %d total %.3f ms |", me, np, tot);
    for (int i = 0; i < 9; ++i) {
      fprintf(stderr, " %s %.3f", kNames[i], ms[i]);
    }
    fprintf(stderr, "\n");
    fflush(stderr);
  }
  double acc[20] = {0};
  sog_gpu_red_download(local.state, acc);

  // double_options carries the CUDA device: from_blob on a host pointer needs
  // plain CPU options or the tensor claims a device it does not live on.
  auto d_vpot = at::from_blob(vpot.data(), {count}, at::kDouble)
                    .to(coord.device()).clone();
  auto d_force = at::from_blob(force.data(), {3 * count}, at::kDouble)
                     .to(coord.device()).clone();
  auto red = at::from_blob(acc, {7}, at::kDouble).to(coord.device()).clone();

  auto atom_energy = torch::zeros({count, 1}, double_options);
  auto charge_gradient = torch::zeros({count, 1}, float_options);
  auto fp = charge.options().dtype(at::kFloat);
  auto force_out = torch::zeros({count, 3}, fp);
  auto atom_virial = torch::zeros({count, 9}, fp);
  auto virial = torch::zeros({1, 9}, fp);
  force_out.copy_(d_force.reshape({count, 3}));

  auto sums = at::zeros({10}, double_options);
  auto moments = at::empty({count * 9}, double_options);
  const int fblocks = static_cast<int>((count + 255) / 256);
  k_midtown_finish1<<<fblocks, 256>>>(
      static_cast<int>(count), d_vpot.data_ptr<double>(),
      d_force.data_ptr<double>(), q.data_ptr<double>(), mask_f.data_ptr<bool>(),
      c_f.data_ptr<double>(), prefactor, kernel_zero, static_cast<int>(channel),
      atom_energy.data_ptr<double>(), sums.data_ptr<double>(),
      moments.data_ptr<double>());
  sums.copy_(sum_over_domains(*communicator, sums));
  auto count_d = csums.narrow(0, 0, 1).clamp_min(1.0);
  k_midtown_finish2<<<fblocks, 256>>>(
      static_cast<int>(count), d_vpot.data_ptr<double>(),
      q.data_ptr<double>(), mask_f.data_ptr<bool>(), prefactor, kernel_zero,
      static_cast<int>(channel), sums.data_ptr<double>(),
      count_d.data_ptr<double>(), moments.data_ptr<double>(),
      red.data_ptr<double>(), 0.5 * volume * prefactor,
      charge_gradient.data_ptr<float>(), atom_virial.data_ptr<float>());
  k_midtown_virial<<<1, 32>>>(red.data_ptr<double>(), 0.5 * volume * prefactor,
                              virial.data_ptr<float>());
  return {atom_energy, charge_gradient, force_out, atom_virial, virial};
}

}  // namespace long_range
}  // namespace deepmd
