# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from random import randint
from typing import List, Optional, Tuple, Union

import torch
from pytorch3d import _C

from .utils import masked_gather


def sample_farthest_points(
    points: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    K: Union[int, List, torch.Tensor] = 50,
    random_start_point: bool = False,
    start_idxs: Optional[torch.Tensor] = None,
    start_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Iterative farthest-point sampling.

    New args
    --------
    start_idxs : (N, Q) optional
        Pre-selected seed indices for each point cloud.
    start_length : (N,) optional
        Number of valid seeds in each row of `start_idxs`.

    If *both* `start_idxs` **and** `start_length` are given the C++ routine that
    supports variable-length seeds is invoked. Otherwise the original path is
    used and, unless `random_start_point=True`, all clouds start from index 0.
    """
    N, P, D = points.shape
    device = points.device

    # --------------------------------------------------------------------- #
    # Validate / canonicalise `lengths`
    # --------------------------------------------------------------------- #
    if lengths is None:
        lengths = torch.full((N,), P, dtype=torch.int64, device=device)
    else:
        if lengths.shape != (N,):
            raise ValueError("points and lengths must have same batch dimension.")
        if lengths.max() > P:
            raise ValueError("A value in lengths was too large.")

    # --------------------------------------------------------------------- #
    # Canonicalise `K`
    # --------------------------------------------------------------------- #
    if isinstance(K, int):
        K = torch.full((N,), K, dtype=torch.int64, device=device)
    elif isinstance(K, list):
        K = torch.tensor(K, dtype=torch.int64, device=device)

    if K.shape[0] != N:
        raise ValueError("K and points must have the same batch dimension")

    # --------------------------------------------------------------------- #
    # Dtype checks
    # --------------------------------------------------------------------- #
    if points.dtype != torch.float32:
        points = points.to(torch.float32)
    if lengths.dtype != torch.int64:
        lengths = lengths.to(torch.int64)
    if K.dtype != torch.int64:
        K = K.to(torch.int64)

    # --------------------------------------------------------------------- #
    # Decide which C++ backend to call
    # --------------------------------------------------------------------- #
    seeds_provided = (start_idxs is not None) and (start_length is not None)
    if seeds_provided and ((start_idxs is None) ^ (start_length is None)):
        raise ValueError("start_idxs and start_length must be both provided or both None.")

    if seeds_provided:
        # ------------ variable-length seed path -------------------------- #
        if start_length.shape != (N,):
            raise ValueError("start_length must have shape (N,)")
        if start_idxs.shape[0] != N:
            raise ValueError("start_idxs must have batch dimension N")

        # dtypes
        if start_idxs.dtype != torch.int64:
            start_idxs = start_idxs.to(torch.int64)
        if start_length.dtype != torch.int64:
            start_length = start_length.to(torch.int64)

        with torch.no_grad():
            idx = _C.sample_farthest_points(
                points, lengths, K, start_idxs, start_length
            )

    else:
        # ------------ original single-seed path -------------------------- #
        if start_idxs is None:
            start_idxs = torch.zeros_like(lengths)
            if random_start_point:
                for n in range(N):
                    start_idxs[n] = torch.randint(high=lengths[n], size=(1,)).item()

        if start_idxs.dtype != torch.int64:
            start_idxs = start_idxs.to(torch.int64)

        with torch.no_grad():
            idx = _C.sample_farthest_points(
                points, lengths, K, start_idxs
            )

    # --------------------------------------------------------------------- #
    sampled_points = masked_gather(points, idx)

    return sampled_points, idx


def sample_farthest_points_naive(
    points: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    K: Union[int, List, torch.Tensor] = 50,
    random_start_point: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Same Args/Returns as sample_farthest_points
    """
    N, P, D = points.shape
    device = points.device

    # Validate inputs
    if lengths is None:
        lengths = torch.full((N,), P, dtype=torch.int64, device=device)
    else:
        if lengths.shape != (N,):
            raise ValueError("points and lengths must have same batch dimension.")
        if lengths.max() > P:
            raise ValueError("Invalid lengths.")

    # TODO: support providing K as a ratio of the total number of points instead of as an int
    if isinstance(K, int):
        K = torch.full((N,), K, dtype=torch.int64, device=device)
    elif isinstance(K, list):
        K = torch.tensor(K, dtype=torch.int64, device=device)

    if K.shape[0] != N:
        raise ValueError("K and points must have the same batch dimension")

    # Find max value of K
    max_K = torch.max(K)

    # List of selected indices from each batch element
    all_sampled_indices = []

    for n in range(N):
        # Initialize an array for the sampled indices, shape: (max_K,)
        sample_idx_batch = torch.full(
            # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
            #  typing.Tuple[int, ...]]` but got `Tuple[Tensor]`.
            (max_K,),
            fill_value=-1,
            dtype=torch.int64,
            device=device,
        )

        # Initialize closest distances to inf, shape: (P,)
        # This will be updated at each iteration to track the closest distance of the
        # remaining points to any of the selected points
        closest_dists = points.new_full(
            # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
            #  typing.Tuple[int, ...]]` but got `Tuple[Tensor]`.
            (lengths[n],),
            float("inf"),
            dtype=torch.float32,
        )

        # Select a random point index and save it as the starting point
        # pyre-fixme[6]: For 2nd argument expected `int` but got `Tensor`.
        selected_idx = randint(0, lengths[n] - 1) if random_start_point else 0
        sample_idx_batch[0] = selected_idx

        # If the pointcloud has fewer than K points then only iterate over the min
        # pyre-fixme[6]: For 1st param expected `SupportsRichComparisonT` but got
        #  `Tensor`.
        # pyre-fixme[6]: For 2nd param expected `SupportsRichComparisonT` but got
        #  `Tensor`.
        k_n = min(lengths[n], K[n])

        # Iteratively select points for a maximum of k_n
        for i in range(1, k_n):
            # Find the distance between the last selected point
            # and all the other points. If a point has already been selected
            # it's distance will be 0.0 so it will not be selected again as the max.
            dist = points[n, selected_idx, :] - points[n, : lengths[n], :]
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            dist_to_last_selected = (dist**2).sum(-1)  # (P - i)

            # If closer than currently saved distance to one of the selected
            # points, then updated closest_dists
            closest_dists = torch.min(dist_to_last_selected, closest_dists)  # (P - i)

            # The aim is to pick the point that has the largest
            # nearest neighbour distance to any of the already selected points
            selected_idx = torch.argmax(closest_dists)
            sample_idx_batch[i] = selected_idx

        # Add the list of points for this batch to the final list
        all_sampled_indices.append(sample_idx_batch)

    all_sampled_indices = torch.stack(all_sampled_indices, dim=0)

    # Gather the points
    all_sampled_points = masked_gather(points, all_sampled_indices)

    # Return (N, max_K, D) subsampled points and indices
    return all_sampled_points, all_sampled_indices
