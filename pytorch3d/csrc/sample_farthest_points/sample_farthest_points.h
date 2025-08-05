/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/extension.h>
#include <tuple>
#include "utils/pytorch3d_cutils.h"
#include <c10/util/Optional.h>   // gives you c10::optional

// Iterative farthest point sampling algorithm [1] to subsample a set of
// K points from a given pointcloud. At each iteration, a point is selected
// which has the largest nearest neighbor distance to any of the
// already selected points.

// Farthest point sampling provides more uniform coverage of the input
// point cloud compared to uniform random sampling.

// [1] Charles R. Qi et al, "PointNet++: Deep Hierarchical Feature Learning
//     on Point Sets in a Metric Space", NeurIPS 2017.

// Args:
//     points: (N, P, D) float32 Tensor containing the batch of pointclouds.
//     lengths: (N,) long Tensor giving the number of points in each pointcloud
//        (to support heterogeneous batches of pointclouds).
//     K: a tensor of length (N,) giving the number of
//        samples to select for each element in the batch.
//        The number of samples is typically << P.
//     start_idxs: (N,) long Tensor giving the index of the first point to
//        sample. Default is all 0. When a random start point is required,
//        start_idxs should be set to a random value between [0, lengths[n]]
//        for batch element n.
// Returns:
//     selected_indices: (N, K) array of selected indices. If the values in
//        K are not all the same, then the shape will be (N, max(K), D), and
//        padded with -1 for batch elements where k_i < max(K). The selected
//        points are gathered in the pytorch autograd wrapper.

at::Tensor FarthestPointSamplingCuda(
    const at::Tensor& points,
    const at::Tensor& lengths,
    const at::Tensor& K,
    const at::Tensor& start_idxs);

at::Tensor FarthestPointSamplingCpu(
    const at::Tensor& points,
    const at::Tensor& lengths,
    const at::Tensor& K,
    const at::Tensor& start_idxs);

at::Tensor FarthestPointSamplingGraphCuda(
    const at::Tensor& points,
    const at::Tensor& lengths,
    const at::Tensor& K,
    const at::Tensor& start_idxs,
    const at::Tensor& start_length);

at::Tensor FarthestPointSamplingGraphCpu(
    const at::Tensor& points,
    const at::Tensor& lengths,
    const at::Tensor& K,
    const at::Tensor& start_idxs,
    const at::Tensor& start_length);  // <-- new parameter

// Exposed implementation.
at::Tensor FarthestPointSampling(
    const at::Tensor&                      points,
    const at::Tensor&                      lengths,
    const at::Tensor&                      K,
    const at::Tensor&                      start_idxs,
    c10::optional<at::Tensor>              start_length = c10::nullopt)
{
  /* decide on device path first */
  const bool use_cuda = points.is_cuda() || lengths.is_cuda() || K.is_cuda();

  /* ── CUDA path ─────────────────────────────────────────────────────────── */
  if (use_cuda) {
#ifdef WITH_CUDA
    /* fast compile-time checks */
    CHECK_CUDA(points);
    CHECK_CUDA(lengths);
    CHECK_CUDA(K);
    CHECK_CUDA(start_idxs);

    if (start_length.has_value()) {
        /* variable-seed variant */
        const at::Tensor& start_len = *start_length;   // ← one clean alias

        CHECK_CUDA(start_len);
        // If you also use CHECK_CONTIGUOUS / shape checks, do it on start_len too.

        return FarthestPointSamplingGraphCuda(
            points, lengths, K, start_idxs, start_len);
    }
    /* classic single-seed variant */
    return FarthestPointSamplingCuda(points, lengths, K, start_idxs);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }

  /* ── CPU path ──────────────────────────────────────────────────────────── */
  if (start_length.has_value()) {
    return FarthestPointSamplingGraphCpu(
        points, lengths, K, start_idxs, *start_length);
  }
  return FarthestPointSamplingCpu(points, lengths, K, start_idxs);
}
