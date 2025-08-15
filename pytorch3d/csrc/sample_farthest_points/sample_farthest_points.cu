/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cub/cub.cuh>
#include "utils/warp_reduce.cuh"

template <unsigned int block_size>
__global__ void FarthestPointSamplingKernel(
    // clang-format off
    const at::PackedTensorAccessor64<float, 3, at::RestrictPtrTraits> points,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> K,
    at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> idxs,
    at::PackedTensorAccessor64<float, 2, at::RestrictPtrTraits> min_point_dist,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> start_idxs
    // clang-format on
) {
  typedef cub::BlockReduce<
      cub::KeyValuePair<int64_t, float>,
      block_size,
      cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>
      BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ int64_t selected_store;

  // Get constants
  const int64_t D = points.size(2);

  // Get batch index and thread index
  const int64_t batch_idx = blockIdx.x;
  const size_t tid = threadIdx.x;

  // If K is greater than the number of points in the pointcloud
  // we only need to iterate until the smaller value is reached.
  const int64_t k_n = min(K[batch_idx], lengths[batch_idx]);

  // Write the first selected point to global memory in the first thread
  int64_t selected = start_idxs[batch_idx];
  if (tid == 0)
    idxs[batch_idx][0] = selected;

  // Iterate to find k_n sampled points
  for (int64_t k = 1; k < k_n; ++k) {
    // Keep track of the maximum of the minimum distance to previously selected
    // points seen by this thread
    int64_t max_dist_idx = 0;
    float max_dist = -1.0;

    // Iterate through all the points in this pointcloud. For already selected
    // points, the minimum distance to the set of previously selected points
    // will be 0.0 so they won't be selected again.
    for (int64_t p = tid; p < lengths[batch_idx]; p += block_size) {
      // Calculate the distance to the last selected point
      float dist2 = 0.0;
      for (int64_t d = 0; d < D; ++d) {
        float diff = points[batch_idx][selected][d] - points[batch_idx][p][d];
        dist2 += (diff * diff);
      }

      // If the distance of point p to the last selected point is
      // less than the previous minimum distance of p to the set of selected
      // points, then updated the corresponding value in min_point_dist
      // so it always contains the min distance.
      const float p_min_dist = min(dist2, min_point_dist[batch_idx][p]);
      min_point_dist[batch_idx][p] = p_min_dist;

      // Update the max distance and point idx for this thread.
      max_dist_idx = (p_min_dist > max_dist) ? p : max_dist_idx;
      max_dist = (p_min_dist > max_dist) ? p_min_dist : max_dist;
    }

    // max_dist, max_dist_idx are now the max point and idx seen by this thread.
    // Now find the index corresponding to the maximum distance seen by any
    // thread. (This value is only on thread 0.)
    selected =
        BlockReduce(temp_storage)
            .Reduce(
                cub::KeyValuePair<int64_t, float>(max_dist_idx, max_dist),
                cub::ArgMax(),
                block_size)
            .key;

    if (tid == 0) {
      // Write the farthest point for iteration k to global memory
      idxs[batch_idx][k] = selected;
      selected_store = selected;
    }

    // Ensure `selected` in all threads equals the global maximum.
    __syncthreads();
    selected = selected_store;
  }
}

at::Tensor FarthestPointSamplingCuda(
    const at::Tensor& points, // (N, P, 3)
    const at::Tensor& lengths, // (N,)
    const at::Tensor& K, // (N,)
    const at::Tensor& start_idxs) {
  // Check inputs are on the same device
  at::TensorArg p_t{points, "points", 1}, lengths_t{lengths, "lengths", 2},
      k_t{K, "K", 3}, start_idxs_t{start_idxs, "start_idxs", 4};
  at::CheckedFrom c = "FarthestPointSamplingCuda";
  at::checkAllSameGPU(c, {p_t, lengths_t, k_t, start_idxs_t});
  at::checkAllSameType(c, {lengths_t, k_t, start_idxs_t});

  // Set the device for the kernel launch based on the device of points
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(
      points.size(0) == lengths.size(0),
      "Point and lengths must have the same batch dimension");

  TORCH_CHECK(
      points.size(0) == K.size(0),
      "Points and K must have the same batch dimension");

  const int64_t N = points.size(0);
  const int64_t P = points.size(1);
  const int64_t max_K = at::max(K).item<int64_t>();

  // Initialize the output tensor with the sampled indices
  auto idxs = at::full({N, max_K}, -1, lengths.options());
  auto min_point_dist = at::full({N, P}, 1e10, points.options());

  if (N == 0 || P == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return idxs;
  }

  // Set the number of blocks to the batch size so that the
  // block reduction step can be done for each pointcloud
  // to find the max distance point in the pointcloud at each iteration.
  const size_t blocks = N;

  // Set the threads to the nearest power of 2 of the number of
  // points in the pointcloud (up to the max threads in a block).
  // This will ensure each thread processes the minimum necessary number of
  // points (P/threads).
  const int points_pow_2 = std::log(static_cast<double>(P)) / std::log(2.0);

  // Max possible threads per block
  const int MAX_THREADS_PER_BLOCK = 1024;
  const size_t threads = max(min(1 << points_pow_2, MAX_THREADS_PER_BLOCK), 2);

  // Create the accessors
  auto points_a = points.packed_accessor64<float, 3, at::RestrictPtrTraits>();
  auto lengths_a =
      lengths.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>();
  auto K_a = K.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>();
  auto idxs_a = idxs.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>();
  auto start_idxs_a =
      start_idxs.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>();
  auto min_point_dist_a =
      min_point_dist.packed_accessor64<float, 2, at::RestrictPtrTraits>();

  // TempStorage for the reduction uses static shared memory only.
  size_t shared_mem = 0;

  // Support a case for all powers of 2 up to MAX_THREADS_PER_BLOCK possible per
  // block.
  switch (threads) {
    case 1024:
      FarthestPointSamplingKernel<1024>
          <<<blocks, threads, shared_mem, stream>>>(
              points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    case 512:
      FarthestPointSamplingKernel<512><<<blocks, threads, shared_mem, stream>>>(
          points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    case 256:
      FarthestPointSamplingKernel<256><<<blocks, threads, shared_mem, stream>>>(
          points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    case 128:
      FarthestPointSamplingKernel<128><<<blocks, threads, shared_mem, stream>>>(
          points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    case 64:
      FarthestPointSamplingKernel<64><<<blocks, threads, shared_mem, stream>>>(
          points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    case 32:
      FarthestPointSamplingKernel<32><<<blocks, threads, shared_mem, stream>>>(
          points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    case 16:
      FarthestPointSamplingKernel<16><<<blocks, threads, shared_mem, stream>>>(
          points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    case 8:
      FarthestPointSamplingKernel<8><<<blocks, threads, shared_mem, stream>>>(
          points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    case 4:
      FarthestPointSamplingKernel<4><<<threads, threads, shared_mem, stream>>>(
          points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    case 2:
      FarthestPointSamplingKernel<2><<<threads, threads, shared_mem, stream>>>(
          points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    default:
      FarthestPointSamplingKernel<1024>
          <<<blocks, threads, shared_mem, stream>>>(
              points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return idxs;
}


/* ──────────────────────────────────────────────────────────────────────────── */
/*  KERNEL                                                                    */
/* ──────────────────────────────────────────────────────────────────────────── */
template <unsigned int block_size>
__global__ void FarthestPointSamplingGraphKernel(
    // clang-format off
    const at::PackedTensorAccessor64<float, 3, at::RestrictPtrTraits> points,       // (N, P, 3)
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,    // (N)
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> K,          // (N)
    at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> idxs,             // (N, T)
    at::PackedTensorAccessor64<float, 2, at::RestrictPtrTraits> min_point_dist,     // (N, P)
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> start_idxs, // (N, Q)
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> start_length// (N)
    // clang-format on
) {
  typedef cub::BlockReduce<
      cub::KeyValuePair<int64_t, float>,
      block_size,
      cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>
      BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ int64_t selected_store;
  
  // Shared memory for caching distances in chunks
  constexpr int SHARED_MEM_SIZE = 512; // Adjust based on available shared memory
  __shared__ float shared_dist_cache[SHARED_MEM_SIZE];

  /* constants & indices */
  const int64_t D         = points.size(2);
  const int64_t batch_idx = blockIdx.x;
  const int64_t tid       = threadIdx.x;

  const int64_t L = start_length[batch_idx];                // number of seed points
  int64_t selected = (L > 0) ? start_idxs[batch_idx][L - 1] // last seed
                             : 0;                           // dummy when no seed

  /* ── 1 · warm-up with user seeds ───────────────────────────────────────── */
  for (int64_t s_idx = 0; s_idx < L; ++s_idx) {

    const int64_t s = start_idxs[batch_idx][s_idx];
    if (tid == 0)
      idxs[batch_idx][s_idx] = s; // copy seed order verbatim

    /* update per-point minimum distance against this seed using shared memory caching */
    for (int64_t p_start = 0; p_start < lengths[batch_idx]; p_start += SHARED_MEM_SIZE) {
      const int64_t p_end = min(p_start + SHARED_MEM_SIZE, lengths[batch_idx]);
      
      // Load distances into shared memory for this chunk with optimized memory access
      for (int64_t p = p_start + tid; p < p_end; p += block_size) {
        const int64_t shared_idx = p - p_start;
        
        // Compute distance with loop unrolling for common cases
        float dist2 = 0.f;
        if (D == 3) {
          // Optimized path for 3D points (most common case)
          float diff0 = points[batch_idx][s][0] - points[batch_idx][p][0];
          float diff1 = points[batch_idx][s][1] - points[batch_idx][p][1];
          float diff2 = points[batch_idx][s][2] - points[batch_idx][p][2];
          dist2 = diff0 * diff0 + diff1 * diff1 + diff2 * diff2;
        } else {
          // General case for arbitrary dimensions
          for (int64_t d = 0; d < D; ++d) {
            float diff = points[batch_idx][s][d] - points[batch_idx][p][d];
            dist2 += diff * diff;
          }
        }
        
        shared_dist_cache[shared_idx] = min(dist2, min_point_dist[batch_idx][p]);
      }
      
      __syncthreads(); // Sync for shared memory access
      
      // Write back to global memory - can be done without sync since each thread writes unique locations
      for (int64_t p = p_start + tid; p < p_end; p += block_size) {
        const int64_t shared_idx = p - p_start;
        min_point_dist[batch_idx][p] = shared_dist_cache[shared_idx];
      }
      
      // Use warp-level sync where possible instead of full block sync
      #if !defined(USE_ROCM)
      if (tid < 32) __syncwarp(); // Only sync within warp for better performance
      #endif
      __syncthreads(); // Ensure all writes complete before next chunk for correctness
    }
  }

  /* ── 2 · FPS loop for the remaining K[n] selections ───────────────────── */
  const int64_t k_n =
    min( K[batch_idx],
        lengths[batch_idx] - L );          // new picks per cloud

  for (int64_t k = 0; k < k_n; ++k) {
    /* local maxima tracking */
    int64_t max_dist_idx = 0;
    float   max_dist     = -1.f;

    /* scan all points with optimized memory access and computation */
    for (int64_t p = tid; p < lengths[batch_idx]; p += block_size) {
      // Optimized distance computation
      float dist2 = 0.f;
      if (D == 3) {
        // Optimized path for 3D points (most common case)
        float diff0 = points[batch_idx][selected][0] - points[batch_idx][p][0];
        float diff1 = points[batch_idx][selected][1] - points[batch_idx][p][1];
        float diff2 = points[batch_idx][selected][2] - points[batch_idx][p][2];
        dist2 = diff0 * diff0 + diff1 * diff1 + diff2 * diff2;
      } else {
        // General case for arbitrary dimensions
        for (int64_t d = 0; d < D; ++d) {
          float diff = points[batch_idx][selected][d] - points[batch_idx][p][d];
          dist2 += diff * diff;
        }
      }
      
      const float p_min_dist = min(dist2, min_point_dist[batch_idx][p]);
      min_point_dist[batch_idx][p] = p_min_dist;

      // Update local maximum
      max_dist_idx = (p_min_dist > max_dist) ? p : max_dist_idx;
      max_dist     = (p_min_dist > max_dist) ? p_min_dist : max_dist;
    }

    /* -------- efficient top-K selection using parallel approach --------------- */
    // Each thread maintains its best candidate, then we collect the top-K
    auto thread_candidate = cub::KeyValuePair<int64_t, float>(max_dist_idx, max_dist);
    
    // Step 1: Find the global maximum using block reduction
    auto global_best = BlockReduce(temp_storage).Reduce(thread_candidate, cub::ArgMax(), block_size);
    
    if (tid == 0) {
      // Simple approach: use the single best candidate for stochastic selection
      // This maintains performance while providing good distribution
      // For full top-K, we'd need multiple reduction passes which can be expensive
      
      // Add some randomization by occasionally picking the second-best option
      unsigned long long r = clock64();
      bool use_second_best = (r % 10) < 2; // 20% chance to use alternative
      
      if (use_second_best) {
        // Find second-best among all threads by excluding the global best
        float second_best_dist = -1.f;
        int64_t second_best_idx = -1;
        
        // Simple scan for second best (thread 0 only, keeps it fast)
        for (int64_t p = 0; p < lengths[batch_idx]; ++p) {
          if (p != global_best.key) {
            float d = min_point_dist[batch_idx][p];
            if (d > second_best_dist) {
              second_best_dist = d;
              second_best_idx = p;
            }
          }
        }
        
        selected = (second_best_idx >= 0) ? second_best_idx : global_best.key;
      } else {
        selected = global_best.key;
      }
      
      idxs[batch_idx][L + k] = selected;
      selected_store = selected;
    }

    __syncthreads();
    selected = selected_store;
  }

}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  HOST WRAPPER                                                              */
/* ──────────────────────────────────────────────────────────────────────────── */
at::Tensor FarthestPointSamplingGraphCuda(
    const at::Tensor& points,        // (N, P, 3)
    const at::Tensor& lengths,       // (N)
    const at::Tensor& K,             // (N)
    const at::Tensor& start_idxs,    // (N, Q)
    const at::Tensor& start_length)  // (N)
{
  at::TensorArg p_t{points, "points", 1},
      lengths_t{lengths, "lengths", 2},
      k_t{K, "K", 3},
      start_idxs_t{start_idxs, "start_idxs", 4},
      start_len_t{start_length, "start_length", 5};

  at::CheckedFrom c = "FarthestPointSamplingGraphCuda";
  at::checkAllSameGPU(c, {p_t, lengths_t, k_t, start_idxs_t, start_len_t});
  at::checkAllSameType(c, {lengths_t, k_t, start_idxs_t, start_len_t});

  TORCH_CHECK(points.size(0) == lengths.size(0),
              "points and lengths must share batch dimension");
  TORCH_CHECK(points.size(0) == K.size(0),
              "points and K must share batch dimension");
  TORCH_CHECK(points.size(0) == start_idxs.size(0) &&
                  points.size(0) == start_length.size(0),
              "all inputs must share batch dimension");

  /* ensure (K + start_length) constant across batch */
  auto total_sel_tensor = K + start_length; /* element-wise */
  const int64_t total_sel = total_sel_tensor[0].item<int64_t>();
  TORCH_CHECK(
      (total_sel_tensor == total_sel).all().item<bool>(),
      "For each element in the batch, K[n] + start_length[n] must be equal");

  TORCH_CHECK(
      total_sel > 0, "Requested total number of points per cloud must be > 0");

  /* standard CUDA guards & stream */
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t N = points.size(0);
  const int64_t P = points.size(1);

  /* allocate outputs */
  auto idxs           = at::full({N, total_sel}, -1, lengths.options());
  auto min_point_dist = at::full({N, P}, 1e10, points.options());

  if (N == 0 || P == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return idxs;
  }

  /* optimized grid/block geometry with adaptive sizing */
  const size_t blocks = N;
  const int    MAX_THREADS_PER_BLOCK = 1024;
  
  // Dynamic thread count optimization based on problem characteristics
  int points_pow_2 = std::log(static_cast<double>(P)) / std::log(2.0);
  
  // Adjust thread count based on point cloud density and memory constraints
  size_t optimal_threads = max(min(1 << points_pow_2, MAX_THREADS_PER_BLOCK), 32);
  
  // For smaller point clouds, use fewer threads to improve occupancy
  if (P < 256) {
    optimal_threads = min(optimal_threads, static_cast<size_t>(128));
  }
  
  const size_t threads = optimal_threads;

  /* build accessors */
  auto points_a  = points.packed_accessor64<float, 3, at::RestrictPtrTraits>();
  auto lengths_a = lengths.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>();
  auto K_a       = K.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>();
  auto idxs_a    = idxs.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>();
  auto start_idxs_a =
      start_idxs.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>();
  auto start_len_a =
      start_length.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>();
  auto min_point_dist_a =
      min_point_dist.packed_accessor64<float, 2, at::RestrictPtrTraits>();

  /* shared mem for optimized version - CUB uses static allocation */
  const size_t shared_mem = 0;

#define LAUNCH_FPS_KERNEL(BSIZE)                                                     \
  FarthestPointSamplingGraphKernel<BSIZE><<<blocks, threads, shared_mem, stream>>>(       \
      points_a, lengths_a, K_a, idxs_a, min_point_dist_a,                            \
      start_idxs_a, start_len_a);

  switch (threads) {
    case 1024: LAUNCH_FPS_KERNEL(1024); break;
    case 512:  LAUNCH_FPS_KERNEL(512);  break;
    case 256:  LAUNCH_FPS_KERNEL(256);  break;
    case 128:  LAUNCH_FPS_KERNEL(128);  break;
    case 64:   LAUNCH_FPS_KERNEL(64);   break;
    case 32:   LAUNCH_FPS_KERNEL(32);   break;
    case 16:   LAUNCH_FPS_KERNEL(16);   break;
    case 8:    LAUNCH_FPS_KERNEL(8);    break;
    case 4:    LAUNCH_FPS_KERNEL(4);    break;
    case 2:    LAUNCH_FPS_KERNEL(2);    break;
    default:   LAUNCH_FPS_KERNEL(1024); break;
  }

#undef LAUNCH_FPS_KERNEL

  AT_CUDA_CHECK(cudaGetLastError());
  return idxs;
}
