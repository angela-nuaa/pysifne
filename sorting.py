"""Readable MATLAB-reference implementation of SIFNE filament sorting.

The implementation follows the checked-in ``filline.m`` and
``SortFilament.m`` literally.  It deliberately does not use a generic path
ordering algorithm: the source builds a labelled image, fills connection gaps
in the order supplied by ``ConnectPts``, and repeatedly removes the first
nearest point returned by the MATLAB ``dsearchn`` call.

Python-facing coordinates are zero-based ``(row, column)``.  The MATLAB
compatibility properties add one only to materialised coordinate rows and
preserve zero padding.  Fragment labels and other numeric IDs remain
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Callable, Sequence

import numpy as np

from .fragments import label_components, matlab_find


try:
    from numba import njit
except ImportError:  # pragma: no cover - environment dependent
    njit = None


def _as_image(image: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got {value.shape}")
    return value


def _as_points(points: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    if value.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {value.shape}")
    return value


def _matlab_colon(start: float, step: float, stop: float) -> np.ndarray:
    """The scalar colon cases used by ``filline``.

    MATLAB returns an empty vector for a zero step, including ``a:0:a``;
    this is important because the complementary loop handles horizontal or
    vertical lines and identical endpoints write no pixel.
    """

    start_value = float(start)
    step_value = float(step)
    stop_value = float(stop)
    if step_value == 0.0:
        return np.empty(0, dtype=np.float64)
    values: list[float] = []
    current = start_value
    if step_value > 0.0:
        while current <= stop_value:
            values.append(current)
            current += step_value
    else:
        while current >= stop_value:
            values.append(current)
            current += step_value
    return np.asarray(values, dtype=np.float64)


def matlab_round(value: Real) -> int:
    """MATLAB's default ``round`` for finite scalar image indices."""

    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"round index must be finite, got {value!r}")
    if numeric >= 0.0:
        return int(math.floor(numeric + 0.5))
    return int(math.ceil(numeric - 0.5))


def fill_line(
    image: np.ndarray,
    value: Real,
    x1: Real,
    y1: Real,
    x2: Real,
    y2: Real,
    *,
    _copy: bool = True,
) -> np.ndarray:
    """Transcribe ``filline.m`` using zero-based ``(row, column)`` points.

    The two loops, their endpoint order, and the MATLAB rounding rule are
    retained.  This is not Bresenham: for nonzero slopes both loops write to
    the output, while a zero row/column difference makes one colon expression
    empty and lets the other loop handle the line.
    """

    result = np.asarray(image).copy() if _copy else np.asarray(image)
    if result.ndim != 2:
        raise ValueError(f"image must be 2-D, got {result.shape}")

    row_start = float(x1)
    column_start = float(y1)
    row_end = float(x2)
    column_end = float(y2)
    row_difference = row_end - row_start
    column_difference = column_end - column_start

    row_step = float(np.sign(row_difference))
    for row_value in _matlab_colon(row_start, row_step, row_end):
        if row_difference == 0.0:
            # The zero-step colon is empty in MATLAB; this branch is only a
            # guard for non-integral synthetic inputs.
            continue
        column_value = (
            (column_difference / row_difference) * (row_value - row_start)
            + column_start
        )
        row_index = matlab_round(row_value)
        column_index = matlab_round(column_value)
        if not (
            0 <= row_index < result.shape[0]
            and 0 <= column_index < result.shape[1]
        ):
            raise IndexError(
                f"filline index ({row_index}, {column_index}) is outside {result.shape}"
            )
        result[row_index, column_index] = value

    column_step = float(np.sign(column_difference))
    for column_value in _matlab_colon(column_start, column_step, column_end):
        if column_difference == 0.0:
            continue
        row_value = (
            (row_difference / column_difference) * (column_value - column_start)
            + row_start
        )
        row_index = matlab_round(row_value)
        column_index = matlab_round(column_value)
        if not (
            0 <= row_index < result.shape[0]
            and 0 <= column_index < result.shape[1]
        ):
            raise IndexError(
                f"filline index ({row_index}, {column_index}) is outside {result.shape}"
            )
        result[row_index, column_index] = value
    return result


def fill_line_matlab(
    image: np.ndarray,
    value: Real,
    x1: Real,
    y1: Real,
    x2: Real,
    y2: Real,
) -> np.ndarray:
    """Compatibility wrapper for one-based MATLAB coordinates."""

    return fill_line(
        image,
        value,
        float(x1) - 1.0,
        float(y1) - 1.0,
        float(x2) - 1.0,
        float(y2) - 1.0,
    )


def _nearest_index(points: np.ndarray, query: np.ndarray) -> int:
    """Reference ``dsearchn`` for the 2-D rows used by SortFilament."""

    best_index = 0
    best_distance_squared = math.inf
    for index in range(points.shape[0]):
        delta_row = float(points[index, 0] - query[0])
        delta_column = float(points[index, 1] - query[1])
        distance_squared = delta_row * delta_row + delta_column * delta_column
        # Strict comparison preserves the first row on equal-distance ties.
        if distance_squared < best_distance_squared:
            best_index = index
            best_distance_squared = distance_squared
    return best_index


def _nearest_active_index(
    points: np.ndarray, active: np.ndarray, query: np.ndarray
) -> int:
    """Nearest-point scan over undeleted rows, preserving source order.

    ``SortFilament`` repeatedly deletes one row from ``temp``.  A boolean
    activity vector is an allocation-free representation of the same compact
    row sequence: rows are still visited in their original order, and the
    strict comparison keeps the first minimum-distance row.
    """

    best_index = -1
    best_distance_squared = math.inf
    for index in range(points.shape[0]):
        if not bool(active[index]):
            continue
        if best_index < 0:
            # The reference starts with the first compact row and an Inf
            # score.  Keeping the Inf sentinel also preserves its NaN case.
            best_index = index
        delta_row = float(points[index, 0] - query[0])
        delta_column = float(points[index, 1] - query[1])
        distance_squared = delta_row * delta_row + delta_column * delta_column
        if distance_squared < best_distance_squared:
            best_index = index
            best_distance_squared = distance_squared
    if best_index < 0:
        raise ValueError("active nearest-point scan received no active rows")
    return best_index


if njit is not None:

    @njit(cache=True, fastmath=False)
    def _nearest_active_index_numba_kernel(points, active, query):
        best_index = -1
        best_distance_squared = np.inf
        for index in range(points.shape[0]):
            if not active[index]:
                continue
            if best_index < 0:
                # Match the reference's initial compact-row sentinel.  Do
                # not initialize the score from this row: NaN behavior then
                # remains identical to the Python loop.
                best_index = index
            delta_row = points[index, 0] - query[0]
            delta_column = points[index, 1] - query[1]
            distance_squared = delta_row * delta_row + delta_column * delta_column
            if distance_squared < best_distance_squared:
                best_index = index
                best_distance_squared = distance_squared
        return best_index

    @njit(cache=True, fastmath=False)
    def _order_points_numba_kernel(points, start):
        point_count = points.shape[0]
        ordered = np.empty((point_count, 2), dtype=np.float64)
        active = np.ones(point_count, dtype=np.bool_)
        changing_row = start[0]
        changing_column = start[1]
        for output_index in range(point_count):
            best_index = -1
            best_distance_squared = np.inf
            for index in range(point_count):
                if not active[index]:
                    continue
                if best_index < 0:
                    # Match the compact temporary array's first-row
                    # sentinel, including the case where its distance is NaN.
                    best_index = index
                delta_row = points[index, 0] - changing_row
                delta_column = points[index, 1] - changing_column
                distance_squared = delta_row * delta_row + delta_column * delta_column
                if distance_squared < best_distance_squared:
                    best_index = index
                    best_distance_squared = distance_squared
            ordered[output_index, 0] = points[best_index, 0]
            ordered[output_index, 1] = points[best_index, 1]
            changing_row = points[best_index, 0]
            changing_column = points[best_index, 1]
            active[best_index] = False
        return ordered

else:  # pragma: no cover - environment dependent
    _nearest_active_index_numba_kernel = None
    _order_points_numba_kernel = None


def numba_nearest_active_index(
    points: np.ndarray, active: np.ndarray, query: np.ndarray
) -> int:
    """JIT the same scalar nearest scan used by ``SortFilament``.

    This is an optional CUDA-pipeline companion.  It keeps the source loop,
    strict comparison, and no-fastmath setting; when Numba is unavailable the
    exact Python scan is used.
    """

    if _nearest_active_index_numba_kernel is None:
        return _nearest_active_index(points, active, query)
    return int(
        _nearest_active_index_numba_kernel(
            np.asarray(points, dtype=np.float64),
            np.asarray(active, dtype=np.bool_),
            np.asarray(query, dtype=np.float64),
        )
    )


def numba_order_points(
    points: np.ndarray, start: np.ndarray
) -> np.ndarray:
    """Return the source nearest-removal order in one deterministic JIT call."""

    if _order_points_numba_kernel is None:
        value = np.asarray(points, dtype=np.float64)
        active = np.ones(value.shape[0], dtype=bool)
        ordered_rows: list[np.ndarray] = []
        changing = np.asarray(start, dtype=np.float64)
        while np.any(active):
            nearest = _nearest_active_index(value, active, changing)
            next_point = value[nearest, :].copy()
            ordered_rows.append(next_point)
            changing = next_point
            active[nearest] = False
        return (
            np.vstack(ordered_rows)
            if ordered_rows
            else np.empty((0, 2), dtype=np.float64)
        )
    return _order_points_numba_kernel(
        np.asarray(points, dtype=np.float64),
        np.asarray(start, dtype=np.float64),
    )


def _distance(point1: np.ndarray, point2: np.ndarray) -> float:
    delta_row = float(point1[0] - point2[0])
    delta_column = float(point1[1] - point2[1])
    return math.sqrt(delta_row * delta_row + delta_column * delta_column)


def _find_component_points(
    label_image: np.ndarray,
    label: float,
    *,
    bbox: tuple[int, int, int, int] | None = None,
    origin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    origin_row, origin_column = origin
    if bbox is None:
        coordinates_one = matlab_find(label_image == label)
        row_offset = origin_row
        column_offset = origin_column
    else:
        row_min, row_max, column_min, column_max = bbox
        coordinates_one = matlab_find(
            label_image[
                row_min - origin_row : row_max - origin_row + 1,
                column_min - origin_column : column_max - origin_column + 1,
            ]
            == label
        )
        row_offset = row_min
        column_offset = column_min
    if coordinates_one.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    coordinates = coordinates_one.astype(np.float64)
    coordinates[:, 0] += float(row_offset)
    coordinates[:, 1] += float(column_offset)
    return coordinates - 1.0


def _column_major_unique_points(points: np.ndarray) -> np.ndarray:
    """Return integer points in the order used by MATLAB ``find``.

    This helper is used only when the current SortFilament mask is known to
    contain exactly the supplied filament points.  Sorting by column first
    is the same order as scanning a MATLAB image column by column; removing
    repeated coordinates reproduces the set semantics of ``find``.
    """

    value = _as_points(points, name="points")
    if value.shape[0] == 0:
        return value
    order = np.lexsort((value[:, 0], value[:, 1]))
    ordered = value[order]
    keep = np.ones(ordered.shape[0], dtype=bool)
    if ordered.shape[0] > 1:
        keep[1:] = np.any(ordered[1:, :] != ordered[:-1, :], axis=1)
    return ordered[keep]


def _label_sparse_filament(
    image_shape: tuple[int, int],
    points: np.ndarray,
    component_labeler: Callable[..., tuple[np.ndarray, int]],
) -> tuple[np.ndarray, int]:
    """Label a sparse filament in its bounding box without changing labels.

    The active SortFilament caller starts from an all-zero image and writes
    only the current filament.  A global MATLAB column-major scan and a scan
    of the translated bounding box visit foreground pixels in exactly the
    same order, and the relative 8-neighborhood is unchanged.  Reconstruct
    the full-size label image afterwards because later source logic still
    observes the original image shape and boundary coordinates.
    """

    value = _as_points(points, name="points")
    if value.shape[0] == 0:
        empty = np.zeros(image_shape, dtype=np.float64)
        return empty, 0
    rows = value[:, 0].astype(np.int64)
    columns = value[:, 1].astype(np.int64)
    row_min = int(np.min(rows))
    row_max = int(np.max(rows))
    column_min = int(np.min(columns))
    column_max = int(np.max(columns))
    local = np.zeros(
        (row_max - row_min + 1, column_max - column_min + 1),
        dtype=bool,
    )
    local[rows - row_min, columns - column_min] = True
    local_labels, count = component_labeler(local, connectivity=8)
    labels = np.zeros(image_shape, dtype=np.float64)
    labels[row_min : row_max + 1, column_min : column_max + 1] = np.asarray(
        local_labels,
        dtype=np.float64,
    )
    return labels, int(count)


def _label_sparse_filament_window(
    points: np.ndarray,
    bbox: tuple[int, int, int, int],
    component_labeler: Callable[..., tuple[np.ndarray, int]],
) -> tuple[np.ndarray, int]:
    """Label the source sparse mask in its exact global bounding window.

    The global MATLAB image is all zero outside ``bbox``.  A translated
    window therefore has the same column-major foreground scan and the same
    8-neighbor connectivity.  Unlike ``_label_sparse_filament`` this helper
    does not expand the result back to the full acquisition image; callers
    that need the public full image do that explicitly at the boundary.
    """

    value = _as_points(points, name="points")
    row_min, row_max, column_min, column_max = bbox
    local = np.zeros(
        (row_max - row_min + 1, column_max - column_min + 1),
        dtype=bool,
    )
    if value.shape[0]:
        rows = value[:, 0].astype(np.int64)
        columns = value[:, 1].astype(np.int64)
        local[rows - row_min, columns - column_min] = True
    labels, count = component_labeler(local, connectivity=8)
    return np.asarray(labels, dtype=np.float64), int(count)


def _matlab_regionprops_sum(values: np.ndarray) -> float:
    """Reproduce the local MATLAB R2023b double-column ``sum`` reduction.

    ``regionprops`` delegates its second-moment calculation to MATLAB's
    vector ``sum``.  On the Windows R2023b runtime used to freeze the SIFNE
    references, short double columns use a scalar left-to-right reduction;
    columns of length 44 or more use four interleaved accumulators followed
    by a left-to-right reduction of those accumulators.  The distinction is
    observable at the last bit for curved/junction fixtures.  Keeping this
    explicit compatibility reduction avoids replacing MATLAB's numerical
    semantics with NumPy's platform-dependent reduction tree.
    """

    value = np.asarray(values, dtype=np.float64).reshape(-1)
    if value.size < 44:
        total = 0.0
        for item in value:
            total += float(item)
        return total

    lane0 = 0.0
    lane1 = 0.0
    lane2 = 0.0
    lane3 = 0.0
    for index, item in enumerate(value):
        if index & 3 == 0:
            lane0 += float(item)
        elif index & 3 == 1:
            lane1 += float(item)
        elif index & 3 == 2:
            lane2 += float(item)
        else:
            lane3 += float(item)
    return ((lane0 + lane1) + lane2) + lane3


def _detect_tips(binary_label_image: np.ndarray) -> np.ndarray:
    """Transcribe the explicit 3-by-3 endpoint loop in SortFilament."""

    binary = np.asarray(binary_label_image)
    coordinates_one = matlab_find(binary == 1)
    height, width = binary.shape
    tips: list[tuple[float, float]] = []
    for row_one, column_one in coordinates_one:
        row = int(row_one) - 1
        column = int(column_one) - 1
        if row == 0 or row == height - 1 or column == 0 or column == width - 1:
            raise IndexError(
                "SortFilament endpoint window exceeds image boundary at "
                f"MATLAB coordinate ({row_one}, {column_one})"
            )
        temp = binary[row - 1 : row + 2, column - 1 : column + 2]
        if float(np.sum(temp)) == 2.0 and temp[1, 1] != 0:
            tips.append((float(row), float(column)))
    return np.asarray(tips, dtype=np.float64).reshape(-1, 2)


def _detect_tips_sparse(
    label_image: np.ndarray,
    points_column_major: np.ndarray,
    *,
    origin: tuple[int, int] = (0, 0),
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Equivalent endpoint loop for a sparse, already-labelled image."""

    if image_shape is None:
        height, width = label_image.shape
    else:
        height, width = image_shape
    origin_row, origin_column = origin
    points = [
        (int(row), int(column))
        for row, column in points_column_major
        if label_image[int(row) - origin_row, int(column) - origin_column] == 1
    ]
    point_set = set(points)
    tips: list[tuple[float, float]] = []
    for row, column in points:
        if row == 0 or row == height - 1 or column == 0 or column == width - 1:
            raise IndexError(
                "SortFilament endpoint window exceeds image boundary at "
                f"MATLAB coordinate ({row + 1}, {column + 1})"
            )
        neighbors = 0
        for row_offset in (-1, 0, 1):
            for column_offset in (-1, 0, 1):
                if (row + row_offset, column + column_offset) in point_set:
                    neighbors += 1
        if neighbors == 2:
            tips.append((float(row), float(column)))
    return np.asarray(tips, dtype=np.float64).reshape(-1, 2)


def _detect_tips_sparse_label(
    label_image: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    origin: tuple[int, int] = (0, 0),
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Endpoint loop on a sparse labelled image, preserving ``find`` order."""

    row_min, row_max, column_min, column_max = bbox
    origin_row, origin_column = origin
    local = np.asarray(label_image)[
        row_min - origin_row : row_max - origin_row + 1,
        column_min - origin_column : column_max - origin_column + 1,
    ]
    coordinates_one = matlab_find(local != 0)
    points = coordinates_one.astype(np.int64, copy=False)
    points = points - 1
    points[:, 0] += row_min
    points[:, 1] += column_min
    if image_shape is None:
        height, width = label_image.shape
    else:
        height, width = image_shape
    point_set = {(int(row), int(column)) for row, column in points}
    tips: list[tuple[float, float]] = []
    for row_value, column_value in points:
        row = int(row_value)
        column = int(column_value)
        if row == 0 or row == height - 1 or column == 0 or column == width - 1:
            raise IndexError(
                "SortFilament endpoint window exceeds image boundary at "
                f"MATLAB coordinate ({row + 1}, {column + 1})"
            )
        neighbors = 0
        for row_offset in (-1, 0, 1):
            for column_offset in (-1, 0, 1):
                if (row + row_offset, column + column_offset) in point_set:
                    neighbors += 1
        if neighbors == 2:
            tips.append((float(row), float(column)))
    return np.asarray(tips, dtype=np.float64).reshape(-1, 2)


def detect_tips_sparse_label_vectorized(
    label_image: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    origin: tuple[int, int] = (0, 0),
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Exact endpoint detector with a vectorized 3-by-3 count.

    This is an optional companion for the CUDA pipeline.  It preserves the
    source detector's observable semantics: the sparse window, MATLAB
    column-major ``find`` order, the boundary check order, and the endpoint
    condition ``neighbour_count == 2``.  Only the independent 3-by-3 count is
    evaluated with NumPy; no path ordering or floating-point metric is moved
    into this helper.  The scalar ``_detect_tips_sparse_label`` remains the
    default/reference implementation.
    """

    row_min, row_max, column_min, column_max = bbox
    origin_row, origin_column = origin
    local = np.asarray(label_image)[
        row_min - origin_row : row_max - origin_row + 1,
        column_min - origin_column : column_max - origin_column + 1,
    ]
    coordinates_one = matlab_find(local != 0)
    if coordinates_one.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    points = coordinates_one.astype(np.int64, copy=False) - 1
    points[:, 0] += row_min
    points[:, 1] += column_min
    if image_shape is None:
        height, width = np.asarray(label_image).shape
    else:
        height, width = image_shape

    # Keep the scalar source loop for error ordering.  SortFilament checks
    # every foreground point before calculating endpoint neighbourhoods, and
    # reports the first offending point in MATLAB find order.
    for row_value, column_value in points:
        row = int(row_value)
        column = int(column_value)
        if row == 0 or row == height - 1 or column == 0 or column == width - 1:
            raise IndexError(
                "SortFilament endpoint window exceeds image boundary at "
                f"MATLAB coordinate ({row + 1}, {column + 1})"
            )

    binary = np.asarray(local != 0, dtype=np.int16)
    padded = np.pad(binary, 1, mode="constant", constant_values=0)
    neighbour_count = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    tip_mask = (binary != 0) & (neighbour_count == 2)
    tip_coordinates_one = matlab_find(tip_mask).astype(np.float64, copy=False)
    tips = tip_coordinates_one - 1.0
    tips[:, 0] += float(row_min)
    tips[:, 1] += float(column_min)
    return tips


def _compute_centroid_from_points(points_column_major: np.ndarray) -> np.ndarray:
    """Compute the same one-based centroid reduction from ordered points."""

    points = _as_points(points_column_major, name="points_column_major")
    if points.shape[0] == 0:
        raise ValueError("regionprops requires at least one nonzero object")
    row_sum = 0.0
    column_sum = 0.0
    for row, column in points:
        row_sum += float(row + 1.0)
        column_sum += float(column + 1.0)
    count = float(points.shape[0])
    return np.asarray([row_sum / count, column_sum / count], dtype=np.float64)


def _compute_orientation_from_points(points_column_major: np.ndarray) -> float:
    """Compute regionprops orientation without rescanning a full sparse mask."""

    points = _as_points(points_column_major, name="points_column_major")
    if points.shape[0] == 0:
        raise ValueError("regionprops requires at least one nonzero object")
    columns = points[:, 1].astype(np.float64) + 1.0
    rows = points[:, 0].astype(np.float64) + 1.0
    count = float(points.shape[0])
    xbar = float(np.sum(columns) / count)
    ybar = float(np.sum(rows) / count)
    x = columns - xbar
    y = -(rows - ybar)
    uxx = float(_matlab_regionprops_sum(x * x) / count + 1.0 / 12.0)
    uyy = float(_matlab_regionprops_sum(y * y) / count + 1.0 / 12.0)
    uxy = float(_matlab_regionprops_sum(x * y) / count)
    if uyy > uxx:
        num = uyy - uxx + math.sqrt((uyy - uxx) ** 2 + 4.0 * uxy * uxy)
        den = 2.0 * uxy
    else:
        num = 2.0 * uxy
        den = uxx - uyy + math.sqrt((uxx - uyy) ** 2 + 4.0 * uxy * uxy)
    if num == 0.0 and den == 0.0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = float(np.divide(np.float64(num), np.float64(den)))
    return float((180.0 / math.pi) * math.atan(ratio))


def _regionprops_objects(mask_image: np.ndarray) -> list[np.ndarray]:
    """Return object coordinate sets with MATLAB regionprops input semantics.

    ``SortFilament`` receives ``maskI=zeros(...)`` as a numeric matrix and
    writes ones into it.  MATLAB therefore treats it as a label matrix: all
    disconnected pixels with value 1 form the first object.  Logical inputs
    use connected components, matching regionprops' separate logical-image
    mode.
    """

    mask = np.asarray(mask_image)
    if mask.dtype.kind == "b":
        labels, count = label_components(mask, connectivity=8)
        return [
            _find_component_points(labels, float(label))
            for label in range(1, count + 1)
        ]
    values = np.unique(mask[mask != 0])
    return [_find_component_points(mask, float(value)) for value in values]


def compute_centroid(mask_image: np.ndarray) -> np.ndarray:
    """Return MATLAB ``regionprops(...,'Centroid')`` as ``(row, column)``."""

    objects = _regionprops_objects(mask_image)
    if not objects:
        raise ValueError("regionprops requires at least one nonzero object")
    points = objects[0]
    if points.shape[0] == 0:
        raise ValueError("first regionprops object is empty")
    row_sum = 0.0
    column_sum = 0.0
    for row, column in points:
        row_sum += float(row + 1.0)
        column_sum += float(column + 1.0)
    count = float(points.shape[0])
    return np.asarray([row_sum / count, column_sum / count], dtype=np.float64)


def compute_orientation(mask_image: np.ndarray) -> float:
    """Return MATLAB regionprops orientation in degrees."""

    objects = _regionprops_objects(mask_image)
    if not objects:
        raise ValueError("regionprops requires at least one nonzero object")
    points = objects[0]
    if points.shape[0] == 0:
        raise ValueError("first regionprops object is empty")

    # This is the actual R2023b ``regionprops`` implementation in
    # ``ComputeEllipseParams``.  In particular, it is not the common
    # covariance/atan2 expression: MATLAB chooses ``num``/``den`` according
    # to the principal-axis branch and then evaluates ``atan(num/den)``.
    # ``PixelList`` is ``[x, y] == [column, row]`` and is one-based, while
    # ``points`` is our zero-based ``(row, column)`` representation.
    columns = points[:, 1].astype(np.float64) + 1.0
    rows = points[:, 0].astype(np.float64) + 1.0
    count = float(points.shape[0])

    xbar = float(np.sum(columns) / count)
    ybar = float(np.sum(rows) / count)
    x = columns - xbar
    y = -(rows - ybar)

    # Keep the vectorized reductions and the source expression grouping.  The
    # ``+1/12`` terms are part of regionprops' pixel-area model, not an
    # optional smoothing or a covariance correction.
    uxx = float(_matlab_regionprops_sum(x * x) / count + 1.0 / 12.0)
    uyy = float(_matlab_regionprops_sum(y * y) / count + 1.0 / 12.0)
    uxy = float(_matlab_regionprops_sum(x * y) / count)

    if uyy > uxx:
        num = uyy - uxx + math.sqrt((uyy - uxx) ** 2 + 4.0 * uxy * uxy)
        den = 2.0 * uxy
    else:
        num = 2.0 * uxy
        den = uxx - uyy + math.sqrt((uxx - uyy) ** 2 + 4.0 * uxy * uxy)

    if num == 0.0 and den == 0.0:
        return 0.0
    # MATLAB evaluates ``num / den`` in IEEE-754 arithmetic.  For a vertical
    # object ``den`` can be signed zero, so the quotient is signed Inf and
    # ``atan`` returns +/-pi/2; Python's scalar division would raise instead.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = float(np.divide(np.float64(num), np.float64(den)))
    return float((180.0 / math.pi) * math.atan(ratio))


def end_to_end_distance(tips: np.ndarray) -> float:
    points = _as_points(tips, name="tips")
    if points.shape[0] < 2:
        raise ValueError("at least two endpoint rows are required")
    return _distance(points[0], points[1])


def accumulate_path_length(ordered_pixels: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Accumulate ``pdist`` step distances in source loop order."""

    points = _as_points(ordered_pixels, name="ordered_pixels")
    if points.shape[0] < 2:
        return 0.0, np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64)
    deltas = np.empty((points.shape[0] - 1, 2), dtype=np.float64)
    steps = np.empty(points.shape[0] - 1, dtype=np.float64)
    total = 0.0
    for index in range(1, points.shape[0]):
        deltas[index - 1, :] = points[index, :] - points[index - 1, :]
        steps[index - 1] = _distance(points[index], points[index - 1])
        total += float(steps[index - 1])
    return total, deltas, steps


def build_analysis_info(
    orientation: Real,
    total_length: Real,
    end_to_end: Real,
    centroid_row_column: Sequence[Real],
) -> np.ndarray:
    """Build MATLAB ``[orient TotalLength EndToEndDist ctrs(2) ctrs(1)]``."""

    centroid = np.asarray(centroid_row_column, dtype=np.float64).reshape(-1)
    if centroid.size != 2:
        raise ValueError("centroid_row_column must contain (row, column)")
    return np.asarray(
        [
            float(orientation),
            float(total_length),
            float(end_to_end),
            float(centroid[0]),
            float(centroid[1]),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class SortFilamentResult:
    """One MATLAB ``SortFilament`` call and its observable intermediates."""

    sorted_filament: np.ndarray
    ordered_pixels: np.ndarray
    filled_label_image: np.ndarray
    tips: np.ndarray
    l_unique_list: np.ndarray
    l_list: np.ndarray
    step_deltas: np.ndarray
    step_distances: np.ndarray
    cumulative_distances: np.ndarray
    total_length: float
    end_to_end_distance: float
    orientation: float
    centroid_row_column: np.ndarray
    analysis_info: np.ndarray

    @property
    def sorted_filament_matlab(self) -> np.ndarray:
        result = np.zeros_like(self.sorted_filament, dtype=np.float64)
        if self.ordered_pixels.shape[0]:
            result[: self.ordered_pixels.shape[0], :] = self.ordered_pixels + 1.0
        return result

    @property
    def tips_matlab(self) -> np.ndarray:
        return self.tips + 1.0


def _prepare_filament_pixels(filament: np.ndarray) -> np.ndarray:
    value = np.asarray(filament, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"filament must have shape (N, 2), got {value.shape}")
    # The core API is zero-based, so row/column zero is a valid image
    # coordinate.  MATLAB's zero-padding removal is performed by the explicit
    # ``*_matlab`` compatibility wrappers before this function is called.
    return value.copy()


@dataclass(frozen=True)
class _PreparedSortFilament:
    """State produced before the MATLAB nearest-point loop."""

    mask: np.ndarray
    label_image: np.ndarray
    all_points: np.ndarray
    l_list: np.ndarray
    l_unique: np.ndarray
    tips: np.ndarray
    changing: np.ndarray
    sparse_fast_path: bool
    sparse_points: np.ndarray
    label_origin: tuple[int, int]
    image_shape: tuple[int, int]
    retain_intermediates: bool


def _prepare_sort_filament(
    filament: np.ndarray,
    mask_image: np.ndarray,
    connect_points: np.ndarray,
    *,
    label_components_impl: Callable[..., tuple[np.ndarray, int]] | None = None,
    tip_detector_impl: Callable[..., np.ndarray] | None = None,
    _mask_known_empty: bool = False,
    _retain_intermediates: bool = True,
) -> _PreparedSortFilament:
    """Run the source setup through ``changing = tips(1,:)`` exactly once."""

    pixels = _prepare_filament_pixels(filament)
    mask_source = _as_image(mask_image, name="mask_image")
    height, width = mask_source.shape
    mask_was_empty = bool(_mask_known_empty)
    if not mask_was_empty:
        mask_was_empty = not bool(np.any(mask_source))
    # In the CUDA batch path the caller supplies a known-empty mask and no
    # public intermediate image is requested.  Do not allocate/copy a full
    # image for each composite; the source dimensions remain available for
    # all boundary checks below.  The default/public path still materializes
    # and fills the exact MATLAB-style mask.
    if _mask_known_empty and _retain_intermediates and mask_was_empty:
        mask = mask_source.copy()
    elif _mask_known_empty and mask_was_empty and not _retain_intermediates:
        mask = np.empty((0, 0), dtype=np.float64)
    else:
        mask = mask_source.copy()
    if not (_mask_known_empty and mask_was_empty and not _retain_intermediates):
        for row, column in pixels:
            row_index = int(row)
            column_index = int(column)
            if not (0 <= row_index < height and 0 <= column_index < width):
                raise IndexError(
                    f"filament coordinate ({row}, {column}) is outside {mask_source.shape}"
                )
            mask[row_index, column_index] = 1

    component_labeler = (
        label_components if label_components_impl is None else label_components_impl
    )
    connects = _as_points(connect_points, name="connect_points")
    if connects.shape[0] and np.any(
        (connects[:, 0] < 0)
        | (connects[:, 0] >= height)
        | (connects[:, 1] < 0)
        | (connects[:, 1] >= width)
    ):
        raise IndexError("connect_points contain an out-of-bounds coordinate")

    sparse_label_path = bool(
        mask_was_empty
        and pixels.shape[0] > 0
        and np.all(np.isfinite(pixels))
        and np.all(pixels == np.floor(pixels))
    )
    # A tight-bounding-box label is equivalent to a full MATLAB column-major
    # scan when the caller supplied an otherwise empty mask.  The same sparse
    # point list also reproduces regionprops for a numeric all-one mask even
    # when connection lines are present.
    sparse_fast_path = sparse_label_path
    sparse_points = np.empty((0, 2), dtype=np.float64)
    sparse_bbox: tuple[int, int, int, int] | None = None
    label_origin = (0, 0)
    if sparse_label_path:
        sparse_points = _column_major_unique_points(pixels)
        bbox_points = pixels
        if connects.shape[0]:
            bbox_points = np.vstack((bbox_points, connects))
        row_min = max(0, int(math.floor(float(np.min(bbox_points[:, 0])))))
        row_max = min(height - 1, int(math.ceil(float(np.max(bbox_points[:, 0])))))
        column_min = max(0, int(math.floor(float(np.min(bbox_points[:, 1])))))
        column_max = min(width - 1, int(math.ceil(float(np.max(bbox_points[:, 1])))))
        sparse_bbox = (row_min, row_max, column_min, column_max)
        label_origin = (row_min, column_min)
        label_image, _ = _label_sparse_filament_window(
            pixels,
            sparse_bbox,
            component_labeler,
        )
    else:
        labels, _ = component_labeler(mask != 0, connectivity=8)
        label_image = labels.astype(np.float64, copy=False)

    l_unique: list[float] = []
    all_point_parts: list[np.ndarray] = []
    all_points = np.empty((0, 2), dtype=np.float64)
    l_list: list[float] = []

    if connects.shape[0] != 2:
        for j in range(1, int(math.floor(connects.shape[0] / 2.0 - 1.0)) + 1):
            start = 2 * j - 1
            end = 2 * j
            value = float(
                label_image[
                    int(connects[start, 0]) - label_origin[0],
                    int(connects[start, 1]) - label_origin[1],
                ]
            )
            label_image = fill_line(
                label_image,
                value,
                connects[start, 0] - label_origin[0],
                connects[start, 1] - label_origin[1],
                connects[end, 0] - label_origin[0],
                connects[end, 1] - label_origin[1],
                _copy=False,
            )
            component = _find_component_points(
                label_image,
                value,
                bbox=sparse_bbox,
                origin=label_origin,
            )
            all_point_parts.append(component)
            l_list.extend([value] * component.shape[0])
            l_unique.append(value)

        end_point = connects[-1]
        value = float(
            label_image[
                int(end_point[0]) - label_origin[0],
                int(end_point[1]) - label_origin[1],
            ]
        )
        component = _find_component_points(
            label_image,
            value,
            bbox=sparse_bbox,
            origin=label_origin,
        )
        all_point_parts.append(component)
        l_list.extend([value] * component.shape[0])
        l_unique.append(value)
    else:
        l_unique = [1.0]
        all_points = pixels.copy()
        l_list = [
            float(
                label_image[
                    int(row) - label_origin[0],
                    int(column) - label_origin[1],
                ]
            )
            for row, column in pixels
        ]
    if connects.shape[0] != 2:
        all_points = (
            np.vstack(all_point_parts).astype(np.float64, copy=False)
            if all_point_parts
            else np.empty((0, 2), dtype=np.float64)
        )

    if sparse_label_path and connects.shape[0] == 2:
        tips = _detect_tips_sparse(
            label_image,
            sparse_points,
            origin=label_origin,
            image_shape=(height, width),
        )
    elif sparse_label_path:
        if sparse_bbox is None:
            raise RuntimeError("sparse SortFilament path did not build a bounding box")
        detect_tips_impl = (
            _detect_tips_sparse_label
            if tip_detector_impl is None
            else tip_detector_impl
        )
        tips = detect_tips_impl(
            label_image,
            sparse_bbox,
            origin=label_origin,
            image_shape=(height, width),
        )
    else:
        binary = label_image.copy()
        binary[binary != 0] = 1
        tips = _detect_tips(binary)
    if tips.shape[0] < 2:
        raise ValueError("SortFilament requires at least two detected tips")

    first_tip_label = label_image[
        int(tips[0, 0]) - label_origin[0],
        int(tips[0, 1]) - label_origin[1],
    ]
    if first_tip_label != l_unique[0]:
        all_points = np.flipud(all_points)
        l_list = list(reversed(l_list))
        l_unique = list(reversed(l_unique))

    return _PreparedSortFilament(
        mask=mask,
        label_image=label_image,
        all_points=all_points,
        l_list=np.asarray(l_list, dtype=np.float64),
        l_unique=np.asarray(l_unique, dtype=np.float64),
        tips=tips,
        changing=tips[0, :].copy(),
        sparse_fast_path=sparse_fast_path,
        sparse_points=sparse_points,
        label_origin=label_origin,
        image_shape=(height, width),
        retain_intermediates=bool(_retain_intermediates),
    )


def _finish_sort_filament(
    prepared: _PreparedSortFilament,
    full_length: int,
    ordered_pixels: np.ndarray,
    *,
    reuse_path_metrics: bool = False,
) -> SortFilamentResult:
    """Build the observable result after an exact source-ordering loop."""

    ordered = _as_points(ordered_pixels, name="ordered_pixels")
    sorted_filament = np.zeros((int(full_length), 2), dtype=np.float64)
    sorted_filament[: ordered.shape[0], :] = ordered

    if reuse_path_metrics:
        # The CUDA batch ordering has already visited every point in source
        # order.  Reuse that same scalar distance for the public path audit
        # instead of running a second, identical distance loop.  The first
        # distance is still measured from ``tips(1,:)`` exactly as in the
        # source; subsequent rows are the same previous/current pairs used by
        # ``accumulate_path_length``.  The default/reference path below is
        # intentionally unchanged.
        changing = prepared.changing.copy()
        total_length = 0.0
        step_deltas = np.empty(
            (max(ordered.shape[0] - 1, 0), 2), dtype=np.float64
        )
        step_distances = np.empty(
            max(ordered.shape[0] - 1, 0), dtype=np.float64
        )
        cumulative = np.zeros(ordered.shape[0], dtype=np.float64)
        for index, next_point in enumerate(ordered):
            distance = _distance(changing, next_point)
            total_length += distance
            if index != 0:
                step_deltas[index - 1, :] = next_point - changing
                step_distances[index - 1] = distance
                cumulative[index] = cumulative[index - 1] + distance
            changing = next_point.copy()
    else:
        changing = prepared.changing.copy()
        total_length = 0.0
        for next_point in ordered:
            total_length += _distance(changing, next_point)
            changing = next_point.copy()

        total_from_steps, step_deltas, step_distances = accumulate_path_length(ordered)
        cumulative = np.zeros(ordered.shape[0], dtype=np.float64)
        for index in range(1, ordered.shape[0]):
            cumulative[index] = cumulative[index - 1] + step_distances[index - 1]
        if total_from_steps != total_length:
            total_length = float(total_length)

    if prepared.sparse_fast_path:
        orientation = _compute_orientation_from_points(prepared.sparse_points)
        centroid = _compute_centroid_from_points(prepared.sparse_points)
    else:
        orientation = compute_orientation(prepared.mask)
        centroid = compute_centroid(prepared.mask)
    end_distance = end_to_end_distance(prepared.tips)
    analysis_info = build_analysis_info(
        orientation, total_length, end_distance, centroid
    )

    if prepared.retain_intermediates:
        if prepared.label_origin == (0, 0) and prepared.label_image.shape == prepared.image_shape:
            filled_label_image = prepared.label_image
        else:
            filled_label_image = np.zeros(prepared.image_shape, dtype=np.float64)
            row_min, column_min = prepared.label_origin
            row_max = row_min + prepared.label_image.shape[0]
            column_max = column_min + prepared.label_image.shape[1]
            filled_label_image[row_min:row_max, column_min:column_max] = prepared.label_image
    else:
        # The CUDA pipeline only exposes ordered pixels and AnalysisInfo.  Do
        # not retain a per-composite full acquisition-sized label image.
        filled_label_image = np.empty((0, 0), dtype=np.float64)

    return SortFilamentResult(
        sorted_filament=sorted_filament,
        ordered_pixels=ordered,
        filled_label_image=filled_label_image,
        tips=prepared.tips,
        l_unique_list=prepared.l_unique,
        l_list=prepared.l_list,
        step_deltas=step_deltas,
        step_distances=step_distances,
        cumulative_distances=cumulative,
        total_length=float(total_length),
        end_to_end_distance=float(end_distance),
        orientation=float(orientation),
        centroid_row_column=centroid,
        analysis_info=analysis_info,
    )


def sort_filament(
    filament: np.ndarray,
    mask_image: np.ndarray,
    connect_points: np.ndarray,
    full_length: int,
    *,
    label_components_impl: Callable[..., tuple[np.ndarray, int]] | None = None,
    tip_detector_impl: Callable[..., np.ndarray] | None = None,
    nearest_index_impl: Callable[[np.ndarray, np.ndarray, np.ndarray], int] | None = None,
    order_points_impl: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    order_filament_impl: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    | None = None,
    reuse_path_metrics: bool = False,
    _mask_known_empty: bool = False,
    _retain_intermediates: bool = True,
) -> SortFilamentResult:
    """Transcribe ``SortFilament.m`` with zero-based input coordinates."""

    if full_length < 0:
        raise ValueError("full_length must be non-negative")
    full_length_int = int(full_length)
    if full_length_int != float(full_length):
        raise ValueError("full_length must be an integer")

    prepared = _prepare_sort_filament(
        filament,
        mask_image,
        connect_points,
        label_components_impl=label_components_impl,
        tip_detector_impl=tip_detector_impl,
        _mask_known_empty=_mask_known_empty,
        _retain_intermediates=_retain_intermediates,
    )

    changing = prepared.changing.copy()
    total_length = 0.0
    ordered_rows: list[np.ndarray] = []
    if order_filament_impl is not None:
        ordered_all = order_filament_impl(
            prepared.all_points,
            prepared.l_list,
            prepared.l_unique,
            changing,
        )
        expected_rows = sum(
            int(np.count_nonzero(prepared.l_list == component_label))
            for component_label in prepared.l_unique
        )
        if ordered_all.shape != (expected_rows, 2):
            raise ValueError(
                "order_filament_impl must preserve the source point count, "
                f"got {ordered_all.shape} for {(expected_rows, 2)}"
            )
        for next_point in ordered_all:
            total_length += _distance(changing, next_point)
            changing = next_point.copy()
            ordered_rows.append(changing.copy())
    else:
        for component_label in prepared.l_unique:
            temp = prepared.all_points[prepared.l_list == component_label].copy()
            if order_points_impl is not None and temp.shape[0] != 0:
                ordered_component = order_points_impl(temp, changing)
                if ordered_component.shape != temp.shape:
                    raise ValueError(
                        "order_points_impl must preserve the temporary point shape, "
                        f"got {ordered_component.shape} for {temp.shape}"
                    )
                for next_point in ordered_component:
                    total_length += _distance(changing, next_point)
                    changing = next_point.copy()
                    ordered_rows.append(changing.copy())
                continue
            active = np.ones(temp.shape[0], dtype=bool)
            remaining = int(temp.shape[0])
            while remaining != 0:
                nearest = (
                    _nearest_active_index(temp, active, changing)
                    if nearest_index_impl is None
                    else nearest_index_impl(temp, active, changing)
                )
                next_point = temp[nearest, :].copy()
                total_length += _distance(changing, next_point)
                changing = next_point
                ordered_rows.append(next_point)
                active[nearest] = False
                remaining -= 1

    ordered_pixels = (
        np.vstack(ordered_rows).astype(np.float64, copy=False)
        if ordered_rows
        else np.empty((0, 2), dtype=np.float64)
    )
    # ``total_length`` is deliberately accumulated above in source loop
    # order.  ``_finish_sort_filament`` repeats the same scalar loop to keep
    # all public metrics and step-audit arrays in one result builder.
    return _finish_sort_filament(
        prepared,
        full_length_int,
        ordered_pixels,
        reuse_path_metrics=reuse_path_metrics,
    )


def sort_filament_matlab(
    filament: np.ndarray,
    mask_image: np.ndarray,
    connect_points: np.ndarray,
    full_length: int,
    *,
    label_components_impl: Callable[..., tuple[np.ndarray, int]] | None = None,
    tip_detector_impl: Callable[..., np.ndarray] | None = None,
    nearest_index_impl: Callable[[np.ndarray, np.ndarray, np.ndarray], int] | None = None,
    order_points_impl: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    order_filament_impl: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    | None = None,
    reuse_path_metrics: bool = False,
) -> SortFilamentResult:
    """Compatibility wrapper accepting one-based MATLAB coordinates/padding."""

    filament_value = np.asarray(filament, dtype=np.float64)
    connect_value = np.asarray(connect_points, dtype=np.float64)
    filament_value = filament_value[filament_value[:, 0] != 0.0].copy()
    connect_value = connect_value[connect_value[:, 0] != 0.0].copy()
    if filament_value.shape[0]:
        filament_value -= 1.0
    if connect_value.shape[0]:
        connect_value -= 1.0
    return sort_filament(
        filament_value,
        mask_image,
        connect_value,
        full_length,
        label_components_impl=label_components_impl,
        tip_detector_impl=tip_detector_impl,
        nearest_index_impl=nearest_index_impl,
        order_points_impl=order_points_impl,
        order_filament_impl=order_filament_impl,
        reuse_path_metrics=reuse_path_metrics,
    )


@dataclass(frozen=True)
class SortFilamentsResult:
    per_filament: tuple[SortFilamentResult, ...]
    all_sorted_filament: np.ndarray
    analysis_info: np.ndarray
    full_length: int

    @property
    def all_sorted_filament_matlab(self) -> np.ndarray:
        if not self.per_filament:
            return np.empty((0, 2), dtype=np.float64)
        max_length = self.all_sorted_filament.shape[0]
        result = np.zeros((max_length, 2, len(self.per_filament)), dtype=np.float64)
        for index, trace in enumerate(self.per_filament):
            if trace.ordered_pixels.shape[0]:
                result[: trace.ordered_pixels.shape[0], :, index] = (
                    trace.ordered_pixels + 1.0
                )
        if len(self.per_filament) == 1:
            return result[:, :, 0]
        return result


def _as_composite_array(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2:
        return array[:, :, None]
    if array.ndim == 3 and array.shape[1] == 2:
        return array
    raise ValueError(f"{name} must have shape (N,2) or (N,2,M), got {array.shape}")


def _trim_sorted_rows(value: np.ndarray) -> np.ndarray:
    if value.shape[0] == 0:
        return value
    used = np.any(value != 0.0, axis=(1, 2))
    if not np.any(used):
        return value[:0, :, :]
    last = int(np.flatnonzero(used)[-1]) + 1
    return value[:last, :, :]


def sort_filaments(
    all_filament: np.ndarray,
    all_connects: np.ndarray,
    image_shape: tuple[int, int],
    full_length: int,
    *,
    filament_lengths: Sequence[int] | None = None,
    connection_lengths: Sequence[int] | None = None,
    label_components_impl: Callable[..., tuple[np.ndarray, int]] | None = None,
    tip_detector_impl: Callable[..., np.ndarray] | None = None,
    nearest_index_impl: Callable[[np.ndarray, np.ndarray, np.ndarray], int] | None = None,
    order_points_impl: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    order_filament_impl: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    | None = None,
    order_filaments_impl: Callable[
        [Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray]],
        Sequence[np.ndarray],
    ]
    | None = None,
    reuse_path_metrics: bool = False,
    retain_intermediates: bool = True,
) -> SortFilamentsResult:
    """Run the active ``func_analysis_mt.m`` SortFilament loop.

    Arrays use Python zero-based coordinates and may be supplied as a single
    ``(N,2)`` composite or ``(N,2,M)`` composites.  Explicit lengths are
    preferred when a real coordinate can be zero; absent lengths use the
    source-compatible first-column zero padding convention.
    """

    if full_length < 0:
        raise ValueError("full_length must be non-negative")
    full_length_int = int(full_length)
    if full_length_int != float(full_length):
        raise ValueError("full_length must be an integer")

    filaments = _as_composite_array(all_filament, name="all_filament")
    connects = _as_composite_array(all_connects, name="all_connects")
    if filaments.shape[2] != connects.shape[2]:
        raise ValueError("all_filament and all_connects must have the same composite count")
    composite_count = filaments.shape[2]
    empty_mask = np.zeros(image_shape, dtype=np.float64)
    if filament_lengths is None:
        filament_lengths_value = [
            int(np.count_nonzero(filaments[:, 0, index]))
            for index in range(composite_count)
        ]
    else:
        filament_lengths_value = [int(item) for item in filament_lengths]
    if connection_lengths is None:
        connection_lengths_value = [
            int(np.count_nonzero(connects[:, 0, index]))
            for index in range(composite_count)
        ]
    else:
        connection_lengths_value = [int(item) for item in connection_lengths]
    if len(filament_lengths_value) != composite_count or len(connection_lengths_value) != composite_count:
        raise ValueError("length metadata must match the composite count")

    traces: list[SortFilamentResult] = []
    if order_filaments_impl is None:
        for index in range(composite_count):
            filament_rows = filaments[: filament_lengths_value[index], :, index]
            connect_rows = connects[: connection_lengths_value[index], :, index]
            traces.append(
                sort_filament(
                    filament_rows,
                    empty_mask,
                    connect_rows,
                    full_length_int,
                    label_components_impl=label_components_impl,
                    tip_detector_impl=tip_detector_impl,
                    nearest_index_impl=nearest_index_impl,
                    order_points_impl=order_points_impl,
                    order_filament_impl=order_filament_impl,
                    reuse_path_metrics=reuse_path_metrics,
                    _mask_known_empty=True,
                    _retain_intermediates=retain_intermediates,
                )
            )
    else:
        prepared: list[_PreparedSortFilament] = []
        for index in range(composite_count):
            filament_rows = filaments[: filament_lengths_value[index], :, index]
            connect_rows = connects[: connection_lengths_value[index], :, index]
            prepared.append(
                _prepare_sort_filament(
                    filament_rows,
                    empty_mask,
                    connect_rows,
                    label_components_impl=label_components_impl,
                    tip_detector_impl=tip_detector_impl,
                    _mask_known_empty=True,
                    _retain_intermediates=retain_intermediates,
                )
            )
        ordered_values = order_filaments_impl(
            [item.all_points for item in prepared],
            [item.l_list for item in prepared],
            [item.l_unique for item in prepared],
            [item.changing for item in prepared],
        )
        if len(ordered_values) != composite_count:
            raise ValueError(
                "order_filaments_impl must return one ordered array per composite"
            )
        for item, ordered_value in zip(prepared, ordered_values):
            expected_rows = sum(
                int(np.count_nonzero(item.l_list == component_label))
                for component_label in item.l_unique
            )
            ordered_array = _as_points(ordered_value, name="ordered_value")
            if ordered_array.shape != (expected_rows, 2):
                raise ValueError(
                    "order_filaments_impl must preserve each source point count, "
                    f"got {ordered_array.shape} for {(expected_rows, 2)}"
                )
            traces.append(
                _finish_sort_filament(
                    item,
                    full_length_int,
                    ordered_array,
                    reuse_path_metrics=reuse_path_metrics,
                )
            )

    raw = np.zeros((full_length_int, 2, composite_count), dtype=np.float64)
    for index, trace in enumerate(traces):
        raw[:, :, index] = trace.sorted_filament
    trimmed = _trim_sorted_rows(raw)
    analysis = (
        np.vstack([trace.analysis_info for trace in traces])
        if traces
        else np.empty((0, 5), dtype=np.float64)
    )
    return SortFilamentsResult(
        per_filament=tuple(traces),
        all_sorted_filament=trimmed,
        analysis_info=analysis,
        full_length=full_length_int,
    )


def sort_filaments_matlab(
    all_filament: np.ndarray,
    all_connects: np.ndarray,
    image_shape: tuple[int, int],
    full_length: int,
    *,
    label_components_impl: Callable[..., tuple[np.ndarray, int]] | None = None,
    tip_detector_impl: Callable[..., np.ndarray] | None = None,
    nearest_index_impl: Callable[[np.ndarray, np.ndarray, np.ndarray], int] | None = None,
    order_points_impl: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    order_filament_impl: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    | None = None,
    order_filaments_impl: Callable[
        [Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray]],
        Sequence[np.ndarray],
    ]
    | None = None,
    reuse_path_metrics: bool = False,
) -> SortFilamentsResult:
    """Compatibility wrapper for one-based padded stage-07 arrays."""

    filaments = _as_composite_array(all_filament, name="all_filament")
    connects = _as_composite_array(all_connects, name="all_connects")
    converted_filaments: list[np.ndarray] = []
    converted_connects: list[np.ndarray] = []
    filament_lengths: list[int] = []
    connection_lengths: list[int] = []
    for index in range(filaments.shape[2]):
        filament_rows = filaments[:, :, index]
        connect_rows = connects[:, :, index]
        filament_rows = filament_rows[filament_rows[:, 0] != 0.0].copy()
        connect_rows = connect_rows[connect_rows[:, 0] != 0.0].copy()
        filament_lengths.append(filament_rows.shape[0])
        connection_lengths.append(connect_rows.shape[0])
        if filament_rows.shape[0]:
            filament_rows -= 1.0
        if connect_rows.shape[0]:
            connect_rows -= 1.0
        converted_filaments.append(filament_rows)
        converted_connects.append(connect_rows)

    max_filament = max(filament_lengths, default=0)
    max_connect = max(connection_lengths, default=0)
    converted_f = np.zeros((max_filament, 2, len(converted_filaments)), dtype=np.float64)
    converted_c = np.zeros((max_connect, 2, len(converted_connects)), dtype=np.float64)
    for index, rows in enumerate(converted_filaments):
        converted_f[: rows.shape[0], :, index] = rows
    for index, rows in enumerate(converted_connects):
        converted_c[: rows.shape[0], :, index] = rows
    return sort_filaments(
        converted_f,
        converted_c,
        image_shape,
        full_length,
        filament_lengths=filament_lengths,
        connection_lengths=connection_lengths,
        label_components_impl=label_components_impl,
        tip_detector_impl=tip_detector_impl,
        nearest_index_impl=nearest_index_impl,
        order_points_impl=order_points_impl,
        order_filament_impl=order_filament_impl,
        order_filaments_impl=order_filaments_impl,
        reuse_path_metrics=reuse_path_metrics,
    )
