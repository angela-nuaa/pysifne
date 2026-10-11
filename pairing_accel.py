"""Deterministic CUDA primitives for the pairing distance precomputation."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .tips import check_dir_between2pts

try:  # Optional CPU JIT for the order-sensitive final selection loop.
    from numba import njit
except ImportError:  # pragma: no cover - environment dependent
    njit = None


@lru_cache(maxsize=1)
def _pairwise_distance_kernel():
    from numba import cuda

    @cuda.jit(fastmath=False)
    def kernel(points, distances):
        linear = cuda.grid(1)
        count = points.shape[0]
        total = count * count
        if linear >= total:
            return
        base_index = linear // count
        candidate_index = linear - base_index * count
        delta_row = points[base_index, 0] - points[candidate_index, 0]
        delta_column = points[base_index, 1] - points[candidate_index, 1]
        distances[base_index, candidate_index] = math.sqrt(
            delta_row * delta_row + delta_column * delta_column
        )

    return kernel


def cuda_pairwise_distances(
    coordinates: np.ndarray,
    *,
    device: int = 0,
    threads_per_block: int = 256,
) -> np.ndarray:
    """Return an ``(N,N)`` distance matrix in source row order.

    One CUDA thread computes one ``(base_tip, candidate_tip)`` entry.  The
    result is only a numeric precomputation; MATLAB-order filtering and all
    candidate insertion/tie decisions remain on the CPU reference path.
    """

    if int(threads_per_block) <= 0:
        raise ValueError("threads_per_block must be positive")
    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"coordinates must have shape (N, 2), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("coordinates must contain finite values")
    count = int(points.shape[0])
    if count == 0:
        return np.empty((0, 0), dtype=np.float64)

    try:
        from numba import cuda
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("CUDA backend requires numba") from exc
    if not cuda.is_available():
        raise RuntimeError("Numba CUDA is not available")

    points_host = np.ascontiguousarray(points)
    distances_host = np.empty((count, count), dtype=np.float64)
    cuda.select_device(int(device))
    d_points = cuda.to_device(points_host)
    d_distances = cuda.to_device(distances_host)
    kernel = _pairwise_distance_kernel()
    total = count * count
    blocks = (total + int(threads_per_block) - 1) // int(threads_per_block)
    kernel[blocks, int(threads_per_block)](d_points, d_distances)
    d_distances.copy_to_host(distances_host)
    return distances_host


def cpu_pairwise_directions_exact(coordinates: np.ndarray, *, chunk_rows: int = 128) -> np.ndarray:
    """Batch ``check_dir_between2pts`` with the MATLAB-compatible CPU atan.

    The CUDA ``atan`` implementation differs from the frozen Windows MATLAB
    values by a few ULPs for some slopes.  PyTorch's CPU double ``atan``
    matches the scalar compatibility backend used by :mod:`pysifne.tips`;
    this routine only batches those independent scalar operations and keeps
    the source's branch formula, image-coordinate convention, and matrix
    row/column order.
    """

    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"coordinates must have shape (N, 2), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("coordinates must contain finite values")
    if int(chunk_rows) <= 0:
        raise ValueError("chunk_rows must be positive")
    count = int(points.shape[0])
    if count == 0:
        return np.empty((0, 0), dtype=np.float64)

    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency fallback
        result = np.empty((count, count), dtype=np.float64)
        for row in range(count):
            for column in range(count):
                result[row, column] = check_dir_between2pts(
                    points[row], points[column]
                )
        return result

    point_tensor = torch.as_tensor(np.ascontiguousarray(points), dtype=torch.float64)
    result = np.empty((count, count), dtype=np.float64)
    scale = 180.0
    pi = math.pi
    for row_start in range(0, count, int(chunk_rows)):
        row_end = min(count, row_start + int(chunk_rows))
        rows = point_tensor[row_start:row_end]
        row_values = rows[:, 0:1]
        column_values = rows[:, 1:2]
        candidate_rows = point_tensor[:, 0].reshape(1, -1)
        candidate_columns = point_tensor[:, 1].reshape(1, -1)
        numerator = row_values - candidate_rows
        denominator = candidate_columns - column_values
        directions = torch.empty(
            (row_end - row_start, count), dtype=torch.float64
        )

        horizontal_right = (candidate_columns > column_values) & (
            candidate_rows == row_values
        )
        horizontal_left = (candidate_columns < column_values) & (
            candidate_rows == row_values
        )
        vertical_down = (candidate_columns == column_values) & (
            candidate_rows > row_values
        )
        vertical_up = (candidate_columns == column_values) & (
            candidate_rows < row_values
        )
        directions[horizontal_right] = 0.0
        directions[horizontal_left] = -180.0
        directions[vertical_down] = -90.0
        directions[vertical_up] = 90.0

        diagonal = (candidate_columns != column_values) & (
            candidate_rows != row_values
        )
        right_diagonal = diagonal & (candidate_columns >= column_values)
        left_diagonal = diagonal & (candidate_columns < column_values)
        ratio = numerator / denominator
        directions[right_diagonal] = scale * torch.atan(ratio[right_diagonal]) / pi
        directions[left_diagonal] = (
            scale * torch.atan(ratio[left_diagonal]) / pi
            + torch.sign(numerator[left_diagonal]) * 180.0
        )
        directions[(candidate_rows == row_values) & (candidate_columns == column_values)] = (
            float("nan")
        )
        result[row_start:row_end, :] = directions.numpy()
    return result


def cpu_pair_directions_exact(
    first_coordinates: np.ndarray, second_coordinates: np.ndarray
) -> np.ndarray:
    """Batch the independent TipReg direction pairs with exact CPU ``atan``."""

    first = np.asarray(first_coordinates, dtype=np.float64)
    second = np.asarray(second_coordinates, dtype=np.float64)
    if first.ndim != 2 or first.shape[1] != 2 or second.shape != first.shape:
        raise ValueError("both coordinate arrays must have shape (N, 2)")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("coordinates must contain finite values")
    count = int(first.shape[0])
    if count == 0:
        return np.empty((0,), dtype=np.float64)
    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency fallback
        return np.asarray(
            [check_dir_between2pts(a, b) for a, b in zip(first, second)],
            dtype=np.float64,
        )

    a = torch.as_tensor(np.ascontiguousarray(first), dtype=torch.float64)
    b = torch.as_tensor(np.ascontiguousarray(second), dtype=torch.float64)
    row1 = a[:, 0]
    column1 = a[:, 1]
    row2 = b[:, 0]
    column2 = b[:, 1]
    numerator = row1 - row2
    denominator = column2 - column1
    directions = torch.empty(count, dtype=torch.float64)
    directions[(column2 > column1) & (row2 == row1)] = 0.0
    directions[(column2 < column1) & (row2 == row1)] = -180.0
    directions[(column2 == column1) & (row2 > row1)] = -90.0
    directions[(column2 == column1) & (row2 < row1)] = 90.0
    diagonal = (column2 != column1) & (row2 != row1)
    right = diagonal & (column2 >= column1)
    left = diagonal & (column2 < column1)
    ratio = numerator / denominator
    directions[right] = 180.0 * torch.atan(ratio[right]) / math.pi
    directions[left] = (
        180.0 * torch.atan(ratio[left]) / math.pi
        + torch.sign(numerator[left]) * 180.0
    )
    directions[(row2 == row1) & (column2 == column1)] = float("nan")
    return directions.numpy()


if njit is not None:

    @njit(cache=True, fastmath=False)
    def _wrap_angle_numba(value: float) -> float:
        if value > 180.0:
            return 360.0 - value
        return value


    @njit(cache=True, fastmath=False)
    def _tip_direction_numba(orientation: float) -> float:
        if orientation == 0.0:
            return -180.0
        if orientation > 0.0:
            return orientation - 180.0
        return orientation + 180.0


    @njit(cache=True, fastmath=False)
    def _normalise_score_numba(numerator: float, denominator: float) -> float:
        if denominator == 0.0:
            if numerator == 0.0:
                return np.nan
            if numerator < 0.0:
                return -np.inf
            return np.inf
        return numerator / denominator


    @njit(cache=True, fastmath=False)
    def _score_less_numba(first: float, second: float) -> bool:
        first_nan = np.isnan(first)
        second_nan = np.isnan(second)
        if first_nan:
            return False
        if second_nan:
            return True
        return first < second


    @njit(cache=True, fastmath=False)
    def _pairing_pool_kernel(
        tips: np.ndarray,
        distances: np.ndarray,
        directions: np.ndarray,
        fan_angle: float,
        min_angle_difference: float,
        min_distance: float,
        min_gap_angle_difference: float,
        c1_weight: float,
        c3_weight: float,
        partner_slots: int,
    ) -> np.ndarray:
        """Batch the source's final local_search selection, serially per row.

        This is intentionally not a nearest-neighbor or graph kernel.  Each
        row scans candidates in increasing MATLAB global-ID order, visits
        duplicate-label groups in the same ascending ``np.unique`` order, and
        applies the original strict comparisons before a stable score sort.
        """

        tip_count = tips.shape[0]
        result = np.zeros((tip_count, partner_slots), dtype=np.float64)
        candidate_ids = np.empty(tip_count, dtype=np.int64)
        candidate_distances = np.empty(tip_count, dtype=np.float64)
        candidate_angle_diff = np.empty(tip_count, dtype=np.float64)
        candidate_gap_diff = np.empty(tip_count, dtype=np.float64)
        shortlist_ids = np.empty(tip_count, dtype=np.int64)
        shortlist_angle_diff = np.empty(tip_count, dtype=np.float64)
        shortlist_gap_diff = np.empty(tip_count, dtype=np.float64)
        delete_flags = np.zeros(tip_count, dtype=np.bool_)
        unique_labels = np.empty(tip_count, dtype=np.float64)
        surviving_ids = np.empty(tip_count, dtype=np.int64)
        surviving_angle_diff = np.empty(tip_count, dtype=np.float64)
        surviving_gap_diff = np.empty(tip_count, dtype=np.float64)
        scores = np.empty(tip_count, dtype=np.float64)
        used = np.zeros(tip_count, dtype=np.bool_)

        for base_index in range(tip_count):
            base_orientation = tips[base_index, 3]
            candidate_count = 0
            for candidate_index in range(tip_count):
                distance = distances[base_index, candidate_index]
                if distance == 0.0:
                    distance = 3.0 * min_distance
                if distance > min_distance:
                    continue
                difference = abs(
                    directions[base_index, candidate_index] - base_orientation
                )
                difference = _wrap_angle_numba(difference)
                if difference <= fan_angle / 2.0:
                    candidate_ids[candidate_count] = candidate_index
                    candidate_distances[candidate_count] = distance
                    candidate_angle_diff[candidate_count] = difference
                    gap_difference = abs(
                        directions[base_index, candidate_index] - base_orientation
                    )
                    candidate_gap_diff[candidate_count] = _wrap_angle_numba(
                        gap_difference
                    )
                    candidate_count += 1

            shortlist_count = 0
            for candidate_position in range(candidate_count):
                candidate_index = candidate_ids[candidate_position]
                candidate_direction = _tip_direction_numba(
                    tips[candidate_index, 3]
                )
                candidate_direction_difference = abs(
                    candidate_direction - base_orientation
                )
                candidate_direction_difference = _wrap_angle_numba(
                    candidate_direction_difference
                )
                # The source uses GapOrient (the base-to-candidate direction)
                # for the second threshold and MIN3 for its upper bound.
                gap_difference = candidate_gap_diff[candidate_position]
                if candidate_direction_difference <= min_angle_difference:
                    if gap_difference <= min_gap_angle_difference:
                        min3 = min_gap_angle_difference
                        if candidate_direction_difference < min_gap_angle_difference:
                            min3 = (
                                candidate_direction_difference
                                + min_gap_angle_difference
                            ) / 2.0
                        if gap_difference <= min3:
                            shortlist_ids[shortlist_count] = candidate_index
                            shortlist_angle_diff[shortlist_count] = (
                                candidate_direction_difference
                            )
                            shortlist_gap_diff[shortlist_count] = gap_difference
                            delete_flags[shortlist_count] = False
                            shortlist_count += 1

            # Inner-fragment suppression.  ``unique`` in the MATLAB source
            # is ascending, while pair_indices retains candidate order.
            for short_position in range(shortlist_count):
                short_index = shortlist_ids[short_position]
                short_label = tips[short_index, 2]
                unique_count = 0
                for candidate_position in range(candidate_count):
                    candidate_index = candidate_ids[candidate_position]
                    candidate_label = tips[candidate_index, 2]
                    if candidate_label == short_label:
                        continue
                    seen = False
                    for unique_position in range(unique_count):
                        if unique_labels[unique_position] == candidate_label:
                            seen = True
                            break
                    if seen:
                        continue
                    insert_position = unique_count
                    while (
                        insert_position > 0
                        and unique_labels[insert_position - 1] > candidate_label
                    ):
                        unique_labels[insert_position] = unique_labels[
                            insert_position - 1
                        ]
                        insert_position -= 1
                    unique_labels[insert_position] = candidate_label
                    unique_count += 1

                for unique_position in range(unique_count):
                    unique_label = unique_labels[unique_position]
                    first_position = -1
                    second_position = -1
                    for candidate_position in range(candidate_count):
                        candidate_index = candidate_ids[candidate_position]
                        if tips[candidate_index, 2] != unique_label:
                            continue
                        if first_position < 0:
                            first_position = candidate_position
                        elif second_position < 0:
                            second_position = candidate_position
                            break
                    if second_position < 0:
                        continue

                    short_distance = distances[base_index, short_index]
                    if short_distance == 0.0:
                        short_distance = 3.0 * min_distance
                    first_distance = candidate_distances[first_position]
                    second_distance = candidate_distances[second_position]
                    if not (
                        short_distance > first_distance
                        and short_distance > second_distance
                    ):
                        continue

                    if (
                        candidate_distances[second_position]
                        < candidate_distances[first_position]
                    ):
                        first_inner_position = second_position
                    else:
                        first_inner_position = first_position
                    first_inner_index = candidate_ids[first_inner_position]
                    first_inner_direction = _tip_direction_numba(
                        tips[first_inner_index, 3]
                    )
                    first_inner_angle_difference = _wrap_angle_numba(
                        abs(first_inner_direction - base_orientation)
                    )
                    first_inner_gap_difference = _wrap_angle_numba(
                        abs(
                            directions[base_index, first_inner_index]
                            - base_orientation
                        )
                    )

                    short_first_distance = distances[short_index, candidate_ids[first_position]]
                    short_second_distance = distances[short_index, candidate_ids[second_position]]
                    if short_second_distance < short_first_distance:
                        second_inner_position = second_position
                    else:
                        second_inner_position = first_position
                    second_inner_index = candidate_ids[second_inner_position]
                    short_orientation = tips[short_index, 3]
                    second_inner_direction = _tip_direction_numba(
                        tips[second_inner_index, 3]
                    )
                    second_inner_angle_difference = _wrap_angle_numba(
                        abs(second_inner_direction - short_orientation)
                    )
                    second_inner_gap_difference = _wrap_angle_numba(
                        abs(
                            directions[short_index, second_inner_index]
                            - short_orientation
                        )
                    )

                    if (
                        first_inner_angle_difference <= min_angle_difference
                        and first_inner_gap_difference <= min_gap_angle_difference
                        and second_inner_angle_difference <= min_angle_difference
                        and second_inner_gap_difference <= min_gap_angle_difference
                    ):
                        delete_flags[short_position] = True

            surviving_count = 0
            for short_position in range(shortlist_count):
                if delete_flags[short_position]:
                    continue
                surviving_ids[surviving_count] = shortlist_ids[short_position]
                surviving_angle_diff[surviving_count] = shortlist_angle_diff[
                    short_position
                ]
                surviving_gap_diff[surviving_count] = shortlist_gap_diff[
                    short_position
                ]
                surviving_count += 1

            if surviving_count == 0:
                continue
            if surviving_count > partner_slots:
                raise ValueError("local search result has more slots than partner_slots")
            if surviving_count == 1:
                result[base_index, 0] = float(surviving_ids[0] + 1)
                continue

            has_nan_angle = False
            has_nan_gap = False
            max_angle = surviving_angle_diff[0]
            max_gap = surviving_gap_diff[0]
            if np.isnan(max_angle):
                has_nan_angle = True
            if np.isnan(max_gap):
                has_nan_gap = True
            for surviving_position in range(1, surviving_count):
                angle = surviving_angle_diff[surviving_position]
                gap = surviving_gap_diff[surviving_position]
                if np.isnan(angle):
                    has_nan_angle = True
                elif not has_nan_angle and angle > max_angle:
                    max_angle = angle
                if np.isnan(gap):
                    has_nan_gap = True
                elif not has_nan_gap and gap > max_gap:
                    max_gap = gap
            if has_nan_angle:
                max_angle = np.nan
            if has_nan_gap:
                max_gap = np.nan

            for surviving_position in range(surviving_count):
                c1_score = _normalise_score_numba(
                    surviving_angle_diff[surviving_position], max_angle
                )
                c3_score = _normalise_score_numba(
                    surviving_gap_diff[surviving_position], max_gap
                )
                scores[surviving_position] = (
                    c1_score * c1_weight + c3_score * c3_weight
                )
                used[surviving_position] = False

            for output_position in range(surviving_count):
                best_position = -1
                for surviving_position in range(surviving_count):
                    if used[surviving_position]:
                        continue
                    if best_position < 0 or _score_less_numba(
                        scores[surviving_position], scores[best_position]
                    ):
                        best_position = surviving_position
                used[best_position] = True
                result[base_index, output_position] = float(
                    surviving_ids[best_position] + 1
                )

        return result
else:  # pragma: no cover - environment dependent.
    _pairing_pool_kernel = None


def numba_pairing_pools(
    tips: np.ndarray,
    distance_matrix: np.ndarray,
    direction_matrix: np.ndarray,
    fan_angle: float,
    min_info: np.ndarray,
    c1_weight: float,
    c3_weight: float,
    partner_slots: int = 100,
) -> np.ndarray:
    """Return all final ``local_search`` partner pools in one JIT call."""

    if _pairing_pool_kernel is None:
        raise ImportError("numba is required for numba_pairing_pools")
    value = np.asarray(tips, dtype=np.float64)
    distances = np.asarray(distance_matrix, dtype=np.float64)
    directions = np.asarray(direction_matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] < 4:
        raise ValueError("tips must have shape (N, >=4)")
    count = value.shape[0]
    if distances.shape != (count, count) or directions.shape != (count, count):
        raise ValueError("distance and direction matrices must be square tip tables")
    info = np.asarray(min_info, dtype=np.float64).reshape(-1)
    if info.size < 3:
        raise ValueError("min_info must contain three values")
    if int(partner_slots) <= 0:
        raise ValueError("partner_slots must be positive")
    return _pairing_pool_kernel(
        np.ascontiguousarray(value),
        np.ascontiguousarray(distances),
        np.ascontiguousarray(directions),
        float(fan_angle),
        float(info[0]),
        float(info[1]),
        float(info[2]),
        float(c1_weight),
        float(c3_weight),
        int(partner_slots),
    )


__all__ = [
    "cuda_pairwise_distances",
    "cpu_pairwise_directions_exact",
    "cpu_pair_directions_exact",
    "numba_pairing_pools",
]
