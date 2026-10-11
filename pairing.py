"""Readable MATLAB-reference implementation of SIFNE tip pairing.

The implementation follows the active ``local_search.m`` and
``biDirPairing.m`` source paths.  Tip coordinates exposed here use Python's
zero-based ``(row, column)`` convention.  Fragment labels, global tip IDs,
and partner IDs remain MATLAB-style one-based identifiers because those are
values, not array coordinates.

The code intentionally keeps scalar loops, MATLAB ``find`` ordering, strict
comparisons, and the source's in-place update order.  It is a reference
implementation, not a nearest-neighbor optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Callable, Sequence

import numpy as np

from .tips import check_dir_between2pts


MAX_PARTNER_SLOTS = 100


def _as_tip_table(all_tips: np.ndarray, *, name: str = "all_tips") -> np.ndarray:
    value = np.asarray(all_tips, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] < 6:
        raise ValueError(f"{name} must have shape (N, >=6), got {value.shape}")
    return value


def _as_image(value: np.ndarray, *, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got shape {image.shape}")
    return image


def _as_point(point: Sequence[Real], *, name: str) -> np.ndarray:
    value = np.asarray(point, dtype=np.float64).reshape(-1)
    if value.size != 2 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain two finite coordinates")
    return value


def _wrap_positive_angle_difference(value: float) -> float:
    """Reproduce the source's one-sided ``if Diff>180`` wrap."""

    if value > 180.0:
        return 360.0 - value
    return value


def _tip_direction_from_orientation(orientation: float) -> float:
    """Reproduce local_search's CandidateDir transformation."""

    if orientation == 0.0:
        return -180.0
    return orientation - float(np.sign(orientation)) * 180.0


def _image_coordinate_to_index(
    l_global_index: np.ndarray, coordinate_zero_based: Sequence[Real]
) -> int:
    """Read a one-based global tip ID at a zero-based image coordinate."""

    point = np.asarray(coordinate_zero_based, dtype=np.float64).reshape(-1)
    if point.size != 2 or not np.all(np.isfinite(point)):
        raise ValueError("image coordinate must contain two finite values")
    if not np.all(point == np.floor(point)):
        raise ValueError(f"image coordinate must be integral, got {point}")
    row, column = int(point[0]), int(point[1])
    image = _as_image(l_global_index, name="L_GlobalIndex")
    if not (0 <= row < image.shape[0] and 0 <= column < image.shape[1]):
        raise IndexError(f"image coordinate is outside L_GlobalIndex: {point}")
    global_index = float(image[row, column])
    if not math.isfinite(global_index) or global_index != math.floor(global_index):
        raise ValueError(
            f"L_GlobalIndex at ({row}, {column}) is not an integer ID: {global_index!r}"
        )
    return int(global_index)


def _euclidean_distance(point1: Sequence[Real], point2: Sequence[Real]) -> float:
    """Reproduce ``sqrt((r1-r2)^2 + (c1-c2)^2)`` rather than ``hypot``."""

    first = _as_point(point1, name="point1")
    second = _as_point(point2, name="point2")
    row_difference = float(first[0] - second[0])
    column_difference = float(first[1] - second[1])
    return math.sqrt(row_difference * row_difference + column_difference * column_difference)


def _nearest_point_index(
    points: np.ndarray, query: Sequence[Real]
) -> int:
    """Reference ``dsearchn`` for a two-row candidate matrix.

    The strict update keeps the first candidate on an equal-distance tie,
    which is the ordering needed by the MATLAB direct fixtures.
    """

    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or value.shape[0] == 0:
        raise ValueError(f"points must have shape (N, 2), got {value.shape}")
    best_index = 0
    best_distance = _euclidean_distance(value[0], query)
    for index in range(1, value.shape[0]):
        distance = _euclidean_distance(value[index], query)
        if distance < best_distance:
            best_index = index
            best_distance = distance
    return best_index


def _stable_matlab_sort_indices(values: np.ndarray) -> np.ndarray:
    """Sort finite scores stably and place NaNs after numeric values."""

    value = np.asarray(values, dtype=np.float64).reshape(-1)

    # MATLAB's sort retains input order for equal values in the relevant
    # release.  Keep the implementation scalar and stable; do not replace it
    # with a graph or nearest-neighbor ordering.
    order = list(range(value.size))
    order.sort(
        key=lambda index: (
            bool(math.isnan(float(value[index]))),
            0.0 if math.isnan(float(value[index])) else float(value[index]),
        )
    )
    return np.asarray(order, dtype=np.int64)


def _normalise_score(numerator: float, denominator: float) -> float:
    """Apply MATLAB double division, including 0/0 -> NaN and x/0 -> Inf."""

    if denominator == 0.0:
        if numerator == 0.0:
            return float("nan")
        return math.copysign(float("inf"), numerator)
    return numerator / denominator


def generate_distance_candidates(
    all_tips: np.ndarray,
    base_loc: Sequence[Real],
    min_dist: Real,
    *,
    distance_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce local_search lines 16--19.

    Returns ``(All_Dist, index_global_pool)``.  The index pool is one-based,
    matching MATLAB tip IDs/global row indices; coordinates in ``all_tips``
    are still zero-based.
    """

    tips = _as_tip_table(all_tips)
    base = _as_point(base_loc, name="base_loc")
    radius = float(min_dist)
    if not math.isfinite(radius):
        raise ValueError("min_dist must be finite")

    if distance_values is None:
        distances = np.empty(tips.shape[0], dtype=np.float64)
        for index in range(tips.shape[0]):
            distances[index] = _euclidean_distance(tips[index, 0:2], base)
    else:
        distances = np.asarray(distance_values, dtype=np.float64).reshape(-1).copy()
        if distances.size != tips.shape[0]:
            raise ValueError(
                "distance_values must contain one value per tip, "
                f"got {distances.size} for {tips.shape[0]} tips"
            )
    for index in range(distances.size):
        if distances[index] == 0.0:
            distances[index] = 3.0 * radius

    pool = [
        float(index + 1)
        for index in range(distances.size)
        if distances[index] <= radius
    ]
    return distances, np.asarray(pool, dtype=np.float64)


def generate_distance_candidates_fast(
    all_tips: np.ndarray,
    base_loc: Sequence[Real],
    min_dist: Real,
    *,
    distance_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized form of the source candidate scan with identical order."""

    tips = _as_tip_table(all_tips)
    _as_point(base_loc, name="base_loc")
    radius = float(min_dist)
    if not math.isfinite(radius):
        raise ValueError("min_dist must be finite")
    if distance_values is None:
        # This branch is kept for direct use outside the CUDA pipeline.  The
        # distance expression is the same scalar formula as the source; only
        # the independent rows are evaluated in a NumPy expression.
        base = np.asarray(base_loc, dtype=np.float64).reshape(2)
        delta_row = tips[:, 0] - base[0]
        delta_column = tips[:, 1] - base[1]
        distances = np.sqrt(delta_row * delta_row + delta_column * delta_column)
    else:
        distances = np.asarray(distance_values, dtype=np.float64).reshape(-1).copy()
        if distances.size != tips.shape[0]:
            raise ValueError(
                "distance_values must contain one value per tip, "
                f"got {distances.size} for {tips.shape[0]}"
            )
    distances[distances == 0.0] = 3.0 * radius
    pool = np.flatnonzero(distances <= radius).astype(np.float64) + 1.0
    return distances, pool


@dataclass(frozen=True)
class LocalSearchResult:
    """All observable intermediate arrays from one MATLAB local search."""

    all_dist: np.ndarray
    distance_pool: np.ndarray
    fan_pool: np.ndarray
    candidate_tips: np.ndarray
    candidate_distances: np.ndarray
    candidate_dir: np.ndarray
    candidate_angle_diff: np.ndarray
    angle_diff: np.ndarray
    min3_list: np.ndarray
    gap_orient: np.ndarray
    candidate_gap_orient_diff: np.ndarray
    gap_orient_diff: np.ndarray
    post_inner_angle_diff: np.ndarray
    post_inner_gap_orient_diff: np.ndarray
    shortlist_candidate_tips: np.ndarray
    shortlist_pool: np.ndarray
    surviving_candidate_tips: np.ndarray
    delete_list: np.ndarray
    surviving_pool: np.ndarray
    scores: np.ndarray
    sort_indices: np.ndarray
    index_global_pool: np.ndarray


def _empty_candidate_tips(column_count: int) -> np.ndarray:
    return np.empty((0, column_count), dtype=np.float64)


def local_search_detailed(
    l_global_index: np.ndarray,
    label_list: np.ndarray,
    fan_angle: Real,
    all_tips: np.ndarray,
    r: Real,
    base_ori: Real,
    base_loc: Sequence[Real],
    min_info: Sequence[Real],
    c1_weight: Real,
    c3_weight: Real,
    *,
    distance_values: np.ndarray | None = None,
    direction_values: np.ndarray | None = None,
    direction_matrix: np.ndarray | None = None,
    distance_matrix: np.ndarray | None = None,
    base_global_index: int | None = None,
    vectorized: bool = False,
) -> LocalSearchResult:
    """Reproduce the complete active ``local_search.m`` body.

    ``r`` is accepted because it is part of the MATLAB signature; the active
    source does not read it after entry.  ``label_list`` is likewise passed
    for call compatibility and used only by the source's signature path.
    """

    del label_list  # The checked-in MATLAB body does not read this argument.
    del r

    tips = _as_tip_table(all_tips)
    l_global = _as_image(l_global_index, name="L_GlobalIndex")
    base = _as_point(base_loc, name="base_loc")
    fan_angle_value = float(fan_angle)
    base_orientation = float(base_ori)
    info = np.asarray(min_info, dtype=np.float64).reshape(-1)
    if info.size < 3:
        raise ValueError("min_info must contain MIN_Angle_Diff, MIN_Dist, MIN_GapAngle_Diff")
    min_angle_difference = float(info[0])
    min_distance = float(info[1])
    min_gap_angle_difference = float(info[2])
    c1_weight_value = float(c1_weight)
    c3_weight_value = float(c3_weight)
    tip_count = tips.shape[0]
    column_count = tips.shape[1]

    candidate_generator = (
        generate_distance_candidates_fast
        if vectorized
        else generate_distance_candidates
    )
    all_dist, distance_pool = candidate_generator(
        tips, base, min_distance, distance_values=distance_values
    )
    direction_row = None
    if direction_values is not None:
        direction_row = np.asarray(direction_values, dtype=np.float64).reshape(-1)
        if direction_row.size != tip_count:
            raise ValueError(
                "direction_values must contain one value per tip, "
                f"got {direction_row.size} for {tip_count}"
            )
    direction_table = None
    if direction_matrix is not None:
        direction_table = np.asarray(direction_matrix, dtype=np.float64)
        if direction_table.shape != (tip_count, tip_count):
            raise ValueError(
                "direction_matrix must have shape (tip_count, tip_count), "
                f"got {direction_table.shape} for {tip_count}"
            )
        if base_global_index is None:
            raise ValueError("base_global_index is required with direction_matrix")
        if not (0 <= int(base_global_index) < tip_count):
            raise ValueError("base_global_index is outside the tip table")
    distance_table = None
    if distance_matrix is not None:
        distance_table = np.asarray(distance_matrix, dtype=np.float64)
        if distance_table.shape != (tip_count, tip_count):
            raise ValueError(
                "distance_matrix must have shape (tip_count, tip_count), "
                f"got {distance_table.shape} for {tip_count}"
            )
    if distance_pool.size == 0:
        return LocalSearchResult(
            all_dist=all_dist,
            distance_pool=distance_pool,
            fan_pool=np.empty(0, dtype=np.float64),
            candidate_tips=_empty_candidate_tips(column_count + 1),
            candidate_distances=np.empty(0, dtype=np.float64),
            candidate_dir=np.empty(0, dtype=np.float64),
            candidate_angle_diff=np.empty(0, dtype=np.float64),
            angle_diff=np.empty(0, dtype=np.float64),
            min3_list=np.empty(0, dtype=np.float64),
            gap_orient=np.empty(0, dtype=np.float64),
            candidate_gap_orient_diff=np.empty(0, dtype=np.float64),
            gap_orient_diff=np.empty(0, dtype=np.float64),
            post_inner_angle_diff=np.empty(0, dtype=np.float64),
            post_inner_gap_orient_diff=np.empty(0, dtype=np.float64),
            shortlist_candidate_tips=_empty_candidate_tips(column_count + 1),
            shortlist_pool=np.empty(0, dtype=np.float64),
            surviving_candidate_tips=_empty_candidate_tips(column_count + 1),
            delete_list=np.empty(0, dtype=np.float64),
            surviving_pool=np.empty(0, dtype=np.float64),
            scores=np.empty(0, dtype=np.float64),
            sort_indices=np.empty(0, dtype=np.int64),
            index_global_pool=np.zeros(MAX_PARTNER_SLOTS, dtype=np.float64),
        )

    # Fan-angle check.  The source mutates entries to zero and then removes
    # zeros; filtering a separate list preserves the same order.
    if vectorized and direction_row is not None:
        distance_indices = distance_pool.astype(np.int64) - 1
        candidate_angles = direction_row[distance_indices]
        difference = np.abs(candidate_angles - base_orientation)
        difference = np.where(difference > 180.0, 360.0 - difference, difference)
        fan_pool = distance_pool[difference <= fan_angle_value / 2.0].copy()
    else:
        fan_pool_list: list[float] = []
        for global_index in distance_pool:
            tip_index = int(global_index) - 1
            candidate_angle = check_dir_between2pts(base, tips[tip_index, 0:2])
            difference = abs(candidate_angle - base_orientation)
            if difference > 180.0:
                difference = 360.0 - difference
            if difference <= fan_angle_value / 2.0:
                fan_pool_list.append(float(global_index))
        fan_pool = np.asarray(fan_pool_list, dtype=np.float64)

    if fan_pool.size == 0:
        return LocalSearchResult(
            all_dist=all_dist,
            distance_pool=distance_pool,
            fan_pool=fan_pool,
            candidate_tips=_empty_candidate_tips(column_count + 1),
            candidate_distances=np.empty(0, dtype=np.float64),
            candidate_dir=np.empty(0, dtype=np.float64),
            candidate_angle_diff=np.empty(0, dtype=np.float64),
            angle_diff=np.empty(0, dtype=np.float64),
            min3_list=np.empty(0, dtype=np.float64),
            gap_orient=np.empty(0, dtype=np.float64),
            candidate_gap_orient_diff=np.empty(0, dtype=np.float64),
            gap_orient_diff=np.empty(0, dtype=np.float64),
            post_inner_angle_diff=np.empty(0, dtype=np.float64),
            post_inner_gap_orient_diff=np.empty(0, dtype=np.float64),
            shortlist_candidate_tips=_empty_candidate_tips(column_count + 1),
            shortlist_pool=np.empty(0, dtype=np.float64),
            surviving_candidate_tips=_empty_candidate_tips(column_count + 1),
            delete_list=np.empty(0, dtype=np.float64),
            surviving_pool=np.empty(0, dtype=np.float64),
            scores=np.empty(0, dtype=np.float64),
            sort_indices=np.empty(0, dtype=np.int64),
            index_global_pool=np.zeros(MAX_PARTNER_SLOTS, dtype=np.float64),
        )

    # CandidateTips = all_tips(index_global_pool,:); CandidateTips(:,7)=pool.
    candidate_tips = tips[fan_pool.astype(np.int64) - 1, :].copy()
    if candidate_tips.shape[1] < 7:
        candidate_tips = np.column_stack(
            (candidate_tips, np.zeros((candidate_tips.shape[0], 7 - candidate_tips.shape[1])))
        )
    candidate_tips[:, 6] = fan_pool
    candidate_distances = all_dist[fan_pool.astype(np.int64) - 1]

    if vectorized:
        orientations = candidate_tips[:, 3]
        candidate_dir = np.where(
            orientations == 0.0,
            -180.0,
            orientations - np.sign(orientations) * 180.0,
        )
        angle_diff = np.abs(candidate_dir - base_orientation)
        angle_diff = np.where(angle_diff > 180.0, 360.0 - angle_diff, angle_diff)
    else:
        candidate_dir = np.empty(candidate_tips.shape[0], dtype=np.float64)
        for index in range(candidate_tips.shape[0]):
            candidate_dir[index] = _tip_direction_from_orientation(
                float(candidate_tips[index, 3])
            )
        angle_diff = np.empty(candidate_tips.shape[0], dtype=np.float64)
        for index in range(candidate_dir.size):
            angle_diff[index] = _wrap_positive_angle_difference(
                abs(candidate_dir[index] - base_orientation)
            )

    min3_list = np.empty(candidate_tips.shape[0], dtype=np.float64)
    if vectorized:
        min3_list = np.empty(candidate_tips.shape[0], dtype=np.float64)
        for index in range(candidate_tips.shape[0]):
            if angle_diff[index] < min_gap_angle_difference:
                min3_list[index] = float(
                    np.mean([angle_diff[index], min_gap_angle_difference])
                )
            else:
                min3_list[index] = min_gap_angle_difference
        if direction_row is None:
            raise ValueError("vectorized local search requires direction_values")
        fan_indices = fan_pool.astype(np.int64) - 1
        gap_orient = direction_row[fan_indices].copy()
        gap_orient_diff = np.abs(gap_orient - base_orientation)
        gap_orient_diff = np.where(
            gap_orient_diff > 180.0,
            360.0 - gap_orient_diff,
            gap_orient_diff,
        )
    else:
        min3_list = np.empty(candidate_tips.shape[0], dtype=np.float64)
        gap_orient = np.empty(candidate_tips.shape[0], dtype=np.float64)
        for index in range(candidate_tips.shape[0]):
            if angle_diff[index] < min_gap_angle_difference:
                min3_list[index] = float(
                    np.mean([angle_diff[index], min_gap_angle_difference])
                )
            else:
                min3_list[index] = min_gap_angle_difference
            gap_orient[index] = check_dir_between2pts(
                base, candidate_tips[index, 0:2]
            )
        gap_orient_diff = np.empty(candidate_tips.shape[0], dtype=np.float64)
        for index in range(gap_orient.size):
            gap_orient_diff[index] = _wrap_positive_angle_difference(
                abs(gap_orient[index] - base_orientation)
            )

    candidate_angle_diff = angle_diff.copy()
    candidate_gap_orient_diff = gap_orient_diff.copy()

    accepted_positions: list[int] = []
    shortlist_rows: list[np.ndarray] = []
    for index in range(candidate_tips.shape[0]):
        min1 = angle_diff[index] <= min_angle_difference
        min3 = gap_orient_diff[index] <= min3_list[index]
        if min1 and min3:
            accepted_positions.append(index)
            shortlist_rows.append(candidate_tips[index, :].copy())

    shortlist_pool = fan_pool[np.asarray(accepted_positions, dtype=np.int64)]
    if shortlist_rows:
        shortlist_candidate_tips = np.vstack(shortlist_rows)
    else:
        # MATLAB initializes ShortListCandidateTips=[] (0-by-0) when the
        # condition loop rejects every candidate.
        shortlist_candidate_tips = _empty_candidate_tips(0)

    # Keep the values that correspond to the current shortlist.  They are
    # deleted in the same k-order as ShortListCandidateTips below.
    surviving_pool = shortlist_pool.copy()
    surviving_candidate_tips = shortlist_candidate_tips.copy()
    surviving_angle_diff = angle_diff[np.asarray(accepted_positions, dtype=np.int64)]
    surviving_gap_diff = gap_orient_diff[np.asarray(accepted_positions, dtype=np.int64)]
    shortlist_angle_diff = surviving_angle_diff.copy()
    shortlist_gap_diff = surviving_gap_diff.copy()

    delete_positions: list[int] = []
    for k in range(shortlist_candidate_tips.shape[0]):
        short_label = shortlist_candidate_tips[k, 2]
        # CandidateTips, not the already filtered shortlist, is used here.
        temp_label_list = candidate_tips[:, 2].copy()
        temp_label_list = temp_label_list[temp_label_list != short_label]
        unique_labels = np.unique(temp_label_list)
        for unique_label in unique_labels:
            occurrences = np.flatnonzero(temp_label_list == unique_label)
            if occurrences.size == 1:
                continue
            pair_indices = np.flatnonzero(candidate_tips[:, 2] == unique_label)
            if pair_indices.size < 2:
                continue
            if vectorized and distance_table is not None:
                short_global = int(shortlist_candidate_tips[k, 6])
                first_global = int(candidate_tips[pair_indices[0], 6])
                second_global = int(candidate_tips[pair_indices[1], 6])
                short_distance = float(all_dist[short_global - 1])
                first_distance = float(all_dist[first_global - 1])
                second_distance = float(all_dist[second_global - 1])
            else:
                short_distance = _euclidean_distance(base, shortlist_candidate_tips[k, 0:2])
                first_distance = _euclidean_distance(base, candidate_tips[pair_indices[0], 0:2])
                second_distance = _euclidean_distance(base, candidate_tips[pair_indices[1], 0:2])
            if not (short_distance > first_distance and short_distance > second_distance):
                continue

            pair_points = candidate_tips[pair_indices[:2], 0:2]
            if vectorized and distance_table is not None:
                first_global = int(candidate_tips[pair_indices[0], 6])
                second_global = int(candidate_tips[pair_indices[1], 6])
                in_tip_idx1 = int(
                    1
                    if float(all_dist[second_global - 1])
                    < float(all_dist[first_global - 1])
                    else 0
                )
            else:
                in_tip_idx1 = _nearest_point_index(pair_points, base)
            in_tip1 = pair_points[in_tip_idx1]
            in_tip1_global = (
                int(candidate_tips[pair_indices[in_tip_idx1], 6])
                if vectorized and distance_table is not None
                else _image_coordinate_to_index(l_global, in_tip1)
            )
            in_tip_angle_temp1 = float(tips[in_tip1_global - 1, 3])
            in_tip_angle1 = _tip_direction_from_orientation(in_tip_angle_temp1)
            in_angle_diff1 = _wrap_positive_angle_difference(
                abs(in_tip_angle1 - base_orientation)
            )
            if direction_table is None:
                in_gap_angle1 = check_dir_between2pts(base, in_tip1)
            else:
                in_gap_angle1 = float(
                    direction_table[int(base_global_index), in_tip1_global - 1]
                )
            in_gap_orient_diff1 = _wrap_positive_angle_difference(
                abs(in_gap_angle1 - base_orientation)
            )

            if vectorized and distance_table is not None:
                short_distances = distance_table[
                    short_global - 1,
                    [first_global - 1, second_global - 1],
                ]
                in_tip_idx2 = int(
                    1 if short_distances[1] < short_distances[0] else 0
                )
            else:
                in_tip_idx2 = _nearest_point_index(
                    pair_points, shortlist_candidate_tips[k, 0:2]
                )
            in_tip2 = pair_points[in_tip_idx2]
            in_tip2_global = (
                int(candidate_tips[pair_indices[in_tip_idx2], 6])
                if vectorized and distance_table is not None
                else _image_coordinate_to_index(l_global, in_tip2)
            )
            in_tip_angle_temp2 = float(tips[in_tip2_global - 1, 3])
            in_tip_angle2 = _tip_direction_from_orientation(in_tip_angle_temp2)
            if not (vectorized and distance_table is not None):
                short_global = _image_coordinate_to_index(
                    l_global, shortlist_candidate_tips[k, 0:2]
                )
            short_orientation = float(tips[short_global - 1, 3])
            in_angle_diff2 = _wrap_positive_angle_difference(
                abs(in_tip_angle2 - short_orientation)
            )
            if direction_table is None:
                in_gap_angle2 = check_dir_between2pts(
                    shortlist_candidate_tips[k, 0:2], in_tip2
                )
            else:
                in_gap_angle2 = float(
                    direction_table[short_global - 1, in_tip2_global - 1]
                )
            in_gap_orient_diff2 = _wrap_positive_angle_difference(
                abs(in_gap_angle2 - short_orientation)
            )

            if (
                in_angle_diff1 <= min_angle_difference
                and in_gap_orient_diff1 <= min_gap_angle_difference
                and in_angle_diff2 <= min_angle_difference
                and in_gap_orient_diff2 <= min_gap_angle_difference
            ):
                delete_positions.append(k)
                continue

    if delete_positions:
        delete_set = set(delete_positions)
        surviving_angle_diff = np.asarray(
            [value for index, value in enumerate(surviving_angle_diff) if index not in delete_set],
            dtype=np.float64,
        )
        surviving_gap_diff = np.asarray(
            [value for index, value in enumerate(surviving_gap_diff) if index not in delete_set],
            dtype=np.float64,
        )
        surviving_pool = np.asarray(
            [value for index, value in enumerate(surviving_pool) if index not in delete_set],
            dtype=np.float64,
        )
        surviving_candidate_tips = np.asarray(
            [
                row
                for index, row in enumerate(surviving_candidate_tips)
                if index not in delete_set
            ],
            dtype=np.float64,
        ).reshape(-1, candidate_tips.shape[1])

    delete_list = np.asarray([float(index + 1) for index in delete_positions], dtype=np.float64)

    scores = np.empty(0, dtype=np.float64)
    sort_indices = np.empty(0, dtype=np.int64)
    if surviving_pool.size == 0:
        result_pool = np.zeros(MAX_PARTNER_SLOTS, dtype=np.float64)
    elif surviving_pool.size == 1:
        result_pool = np.concatenate(
            (surviving_pool.copy(), np.zeros(MAX_PARTNER_SLOTS - 1, dtype=np.float64))
        )
    else:
        max_angle = float(np.max(surviving_angle_diff))
        max_gap = float(np.max(surviving_gap_diff))
        c1_score = np.asarray(
            [_normalise_score(float(value), max_angle) for value in surviving_angle_diff],
            dtype=np.float64,
        )
        c3_score = np.asarray(
            [_normalise_score(float(value), max_gap) for value in surviving_gap_diff],
            dtype=np.float64,
        )
        scores = c1_score * c1_weight_value + c3_score * c3_weight_value
        sort_indices = _stable_matlab_sort_indices(scores)
        sorted_pool = surviving_pool[sort_indices]
        result_pool = np.concatenate(
            (
                sorted_pool,
                np.zeros(MAX_PARTNER_SLOTS - sorted_pool.size, dtype=np.float64),
            )
        )

    return LocalSearchResult(
        all_dist=all_dist,
        distance_pool=distance_pool,
        fan_pool=fan_pool,
        candidate_tips=candidate_tips,
        candidate_distances=candidate_distances,
        candidate_dir=candidate_dir,
        candidate_angle_diff=candidate_angle_diff,
        angle_diff=shortlist_angle_diff,
        min3_list=min3_list,
        gap_orient=gap_orient,
        candidate_gap_orient_diff=candidate_gap_orient_diff,
        gap_orient_diff=shortlist_gap_diff,
        post_inner_angle_diff=surviving_angle_diff.copy(),
        post_inner_gap_orient_diff=surviving_gap_diff.copy(),
        shortlist_candidate_tips=shortlist_candidate_tips,
        shortlist_pool=shortlist_pool,
        surviving_candidate_tips=surviving_candidate_tips,
        delete_list=delete_list,
        surviving_pool=surviving_pool,
        scores=scores,
        sort_indices=sort_indices,
        index_global_pool=result_pool,
    )


def local_search(
    l_global_index: np.ndarray,
    label_list: np.ndarray,
    fan_angle: Real,
    all_tips: np.ndarray,
    r: Real,
    base_ori: Real,
    base_loc: Sequence[Real],
    min_info: Sequence[Real],
    c1_weight: Real,
    c3_weight: Real,
) -> np.ndarray:
    """Return the MATLAB-shaped 100-slot partner index row."""

    return local_search_detailed(
        l_global_index,
        label_list,
        fan_angle,
        all_tips,
        r,
        base_ori,
        base_loc,
        min_info,
        c1_weight,
        c3_weight,
    ).index_global_pool


def enforce_bidirectional_pairing(all_tips: np.ndarray) -> np.ndarray:
    """Reproduce ``biDirPairing.m``'s in-place row traversal."""

    result = _as_tip_table(all_tips).copy()
    if result.shape[1] < 8:
        raise ValueError("all_tips must contain columns 1..8 before partner columns")

    for row_index in range(result.shape[0]):
        original_partner_length = result.shape[1] - 8
        partner_list = [
            int(value)
            for value in result[row_index, 8:]
            if value != 0.0
        ]
        tip_life = len(partner_list)

        for partner_position in range(len(partner_list)):
            partner_id = partner_list[partner_position]
            partner_row = result[partner_id - 1, 8:]
            partner_partners = [
                int(value) for value in partner_row if value != 0.0
            ]
            if row_index + 1 not in partner_partners:
                partner_list[partner_position] = 0
                if tip_life > 1:
                    tip_life -= 1

        partner_list = [value for value in partner_list if value != 0]
        tip_new = np.zeros(original_partner_length, dtype=np.float64)
        tip_new[: len(partner_list)] = partner_list
        result[row_index, 8:] = tip_new
        result[row_index, 7] = float(tip_life)

    return result


@dataclass(frozen=True)
class PairingResult:
    """Stage-06 pairing outputs in Python's zero-based coordinate convention."""

    l_global_index: np.ndarray
    label_list: np.ndarray
    local_search_results: tuple[LocalSearchResult, ...]
    new_partner_list_untrimmed: np.ndarray
    new_partner_list: np.ndarray
    all_tips_pre_bidir: np.ndarray
    all_tips_post_bidir: np.ndarray
    all_tips: np.ndarray

    @staticmethod
    def _matlab_coordinates(table: np.ndarray) -> np.ndarray:
        value = table.copy()
        if value.shape[0]:
            value[:, 0:2] += 1.0
            value[:, 4:6] += 1.0
        return value

    @property
    def all_tips_pre_bidir_matlab(self) -> np.ndarray:
        return self._matlab_coordinates(self.all_tips_pre_bidir)

    @property
    def all_tips_post_bidir_matlab(self) -> np.ndarray:
        return self._matlab_coordinates(self.all_tips_post_bidir)

    @property
    def all_tips_matlab(self) -> np.ndarray:
        return self._matlab_coordinates(self.all_tips)


def pair_filament_tips(
    stage05_all_tips: np.ndarray,
    label_image: np.ndarray,
    fan_angle: Real,
    fan_r: Real,
    min_info: Sequence[Real],
    c1_weight: Real = 1.0,
    c3_weight: Real = 1.0,
    *,
    partner_slots: int = MAX_PARTNER_SLOTS,
    distance_matrix: np.ndarray | None = None,
    direction_matrix: np.ndarray | None = None,
    vectorized_local_search: bool = False,
    pool_impl: Callable[
        [np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, float, float, int],
        np.ndarray,
    ]
    | None = None,
    retain_local_results: bool = True,
) -> PairingResult:
    """Reproduce the active ``func_analysis_mt.m`` pairing block."""

    if partner_slots <= 0:
        raise ValueError("partner_slots must be positive")
    tips = _as_tip_table(stage05_all_tips).copy()
    labels = _as_image(label_image, name="label_image")
    tip_count = tips.shape[0]
    if distance_matrix is not None:
        distance_matrix_value = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix_value.shape != (tip_count, tip_count):
            raise ValueError(
                "distance_matrix must have shape (tip_count, tip_count), "
                f"got {distance_matrix_value.shape} for {tip_count} tips"
            )
    else:
        distance_matrix_value = None
    if direction_matrix is not None:
        direction_matrix_value = np.asarray(direction_matrix, dtype=np.float64)
        if direction_matrix_value.shape != (tip_count, tip_count):
            raise ValueError(
                "direction_matrix must have shape (tip_count, tip_count), "
                f"got {direction_matrix_value.shape} for {tip_count} tips"
            )
    else:
        direction_matrix_value = None
    label_list = tips[:, 2:3].copy()

    # all_tips(:,7) = 1:size(all_tips,1), retaining one-based IDs.
    tips_with_global = np.column_stack(
        (tips, np.arange(1, tip_count + 1, dtype=np.float64))
    )
    l_global_index = np.zeros(labels.shape, dtype=np.float64)
    for index in range(tip_count):
        row = int(tips_with_global[index, 0])
        column = int(tips_with_global[index, 1])
        if not (0 <= row < labels.shape[0] and 0 <= column < labels.shape[1]):
            raise IndexError(f"tip coordinate is outside label_image: {tips_with_global[index, 0:2]}")
        l_global_index[row, column] = float(index + 1)

    new_partner_list_untrimmed = np.zeros((tip_count, partner_slots), dtype=np.float64)
    local_results: list[LocalSearchResult] = []
    if pool_impl is not None:
        if distance_matrix_value is None or direction_matrix_value is None:
            raise ValueError(
                "pool_impl requires both distance_matrix and direction_matrix"
            )
        pools = np.asarray(
            pool_impl(
                tips_with_global[:, :6],
                distance_matrix_value,
                direction_matrix_value,
                float(fan_angle),
                np.asarray(min_info, dtype=np.float64).reshape(-1),
                float(c1_weight),
                float(c3_weight),
                int(partner_slots),
            ),
            dtype=np.float64,
        )
        if pools.shape != (tip_count, partner_slots):
            raise ValueError(
                "pool_impl must return one partner row per tip with the requested "
                f"slot count, got {pools.shape} for {(tip_count, partner_slots)}"
            )
        new_partner_list_untrimmed[:, :] = pools
    else:
        for index in range(tip_count):
            detailed = local_search_detailed(
                l_global_index,
                label_list,
                fan_angle,
                tips_with_global[:, :6],
                fan_r,
                tips_with_global[index, 3],
                tips_with_global[index, 0:2],
                min_info,
                c1_weight,
                c3_weight,
                distance_values=(
                    None
                    if distance_matrix_value is None
                    else distance_matrix_value[index, :]
                ),
                direction_values=(
                    None
                    if direction_matrix_value is None
                    else direction_matrix_value[index, :]
                ),
                direction_matrix=direction_matrix_value,
                distance_matrix=distance_matrix_value,
                base_global_index=index,
                vectorized=vectorized_local_search,
            )
            if retain_local_results:
                local_results.append(detailed)
            if detailed.index_global_pool.size > partner_slots:
                raise ValueError("local search result has more slots than partner_slots")
            new_partner_list_untrimmed[index, :] = detailed.index_global_pool[:partner_slots]

    nonzero_columns = [
        column
        for column in range(new_partner_list_untrimmed.shape[1])
        if np.any(new_partner_list_untrimmed[:, column] != 0.0)
    ]
    if nonzero_columns:
        last_partner_column = nonzero_columns[-1] + 1
        new_partner_list = new_partner_list_untrimmed[:, :last_partner_column].copy()
    else:
        new_partner_list = np.empty((tip_count, 0), dtype=np.float64)

    # all_tips(:,8)=1 and then only nonzero partner IDs are inserted.
    # MATLAB grows ``all_tips`` when assigning columns 9:end.  Allocate the
    # same rectangular partner region explicitly before filling it.
    all_tips_pre_bidir = np.zeros(
        (tip_count, 8 + new_partner_list.shape[1]), dtype=np.float64
    )
    all_tips_pre_bidir[:, 0:7] = tips_with_global
    all_tips_pre_bidir[:, 7] = 1.0
    for row_index in range(tip_count):
        partners = [
            value
            for value in new_partner_list[row_index, :]
            if value != 0.0
        ]
        if partners:
            partner_array = np.asarray(partners, dtype=np.float64)
            all_tips_pre_bidir[row_index, 7] = float(partner_array.size)
            all_tips_pre_bidir[row_index, 8 : 8 + partner_array.size] = partner_array

    all_tips_post_bidir = enforce_bidirectional_pairing(all_tips_pre_bidir)
    all_tips = all_tips_post_bidir.copy()
    # Active func_analysis_mt.m uses OverlapList=1 at this point.
    all_tips[:, 7] = 1.0

    return PairingResult(
        l_global_index=l_global_index,
        label_list=label_list,
        local_search_results=tuple(local_results),
        new_partner_list_untrimmed=new_partner_list_untrimmed,
        new_partner_list=new_partner_list,
        all_tips_pre_bidir=all_tips_pre_bidir,
        all_tips_post_bidir=all_tips_post_bidir,
        all_tips=all_tips,
    )
