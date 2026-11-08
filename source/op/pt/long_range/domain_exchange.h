// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Exchange of the latent-charge long range between the domains of a
// decomposed system.
//
// A domain is one MPI rank of the communicator that a caller hands to
// deepmd::latent_charge_sum as an int64 tensor holding the address of its
// MPI_Comm handle. A zero address stands for a single domain. Both exchanges
// follow the rank order, so every rank receives the same rows and the same
// sums, independent of the reduction algorithm of the MPI library.

#pragma once

#include <torch/types.h>

#include <cstdint>
#include <utility>

namespace deepmd {
namespace long_range {

// The number of ranks of ``communicator``; one for a zero address.
int domain_count(const torch::Tensor& communicator);

// The rank of the caller in ``communicator``; zero for a zero address.
int domain_rank(const torch::Tensor& communicator);

// Copies ``count`` bytes of ``data`` from rank 0 to every rank.
void broadcast_bytes(const torch::Tensor& communicator, void* data, int count);

// Concatenates, in rank order, the rows of every rank. ``rows`` is a
// contiguous float64 matrix of shape (M_r, W) on the CUDA device, with the
// same W on every rank. Returns the rows of all ranks, of shape (sum_r M_r,
// W) on the device of ``rows``, and the index of the first row of this rank.
std::pair<torch::Tensor, std::int64_t> gather_domain_rows(
    const torch::Tensor& communicator, const torch::Tensor& rows);

// Sums the float64 vector ``values`` over the ranks, adding the vectors of
// the ranks in rank order. Returns the sum on the device of ``values``.
torch::Tensor sum_over_domains(const torch::Tensor& communicator,
                               const torch::Tensor& values);

// The same sum for a large float32 vector -- the density mesh.  It stays on
// the device when the MPI library reads device memory and otherwise stages
// through pinned host buffers; unlike ``sum_over_domains`` it uses an
// Allreduce, so the ranks never materialise ``size`` copies of the mesh.
torch::Tensor sum_over_domains_f32(const torch::Tensor& communicator,
                                   const torch::Tensor& values);

}  // namespace long_range
}  // namespace deepmd
