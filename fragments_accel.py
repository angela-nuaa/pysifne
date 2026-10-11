"""Deterministic CUDA primitives used by the fragment stage.

The independent neighborhood, square-mask, thinning, and ``LargeCurFlag``
counting work is accelerated here.  The input coordinate order is preserved
exactly; component labeling and all MATLAB-order state transitions remain in
``fragments.py``.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import numpy as np


try:
    from numba import njit
except ImportError:  # pragma: no cover - environment dependent
    njit = None


if njit is not None:

    @njit(cache=True)
    def _label_components_kernel(binary, row_offsets, column_offsets):
        height, width = binary.shape
        labels = np.zeros((height, width), dtype=np.float64)
        stack_rows = np.empty(height * width, dtype=np.int64)
        stack_columns = np.empty(height * width, dtype=np.int64)
        next_label = 0

        # This is the same column-major seed scan and LIFO neighbor traversal
        # as fragments.label_components.  It is JIT-compiled, not replaced by
        # a different connected-component algorithm.
        for column in range(width):
            for row in range(height):
                if not binary[row, column] or labels[row, column] != 0.0:
                    continue
                next_label += 1
                labels[row, column] = next_label
                top = 1
                stack_rows[0] = row
                stack_columns[0] = column
                while top > 0:
                    top -= 1
                    current_row = stack_rows[top]
                    current_column = stack_columns[top]
                    for offset_index in range(row_offsets.size):
                        neighbor_row = current_row + row_offsets[offset_index]
                        neighbor_column = current_column + column_offsets[offset_index]
                        if (
                            0 <= neighbor_row < height
                            and 0 <= neighbor_column < width
                            and binary[neighbor_row, neighbor_column]
                            and labels[neighbor_row, neighbor_column] == 0.0
                        ):
                            labels[neighbor_row, neighbor_column] = next_label
                            stack_rows[top] = neighbor_row
                            stack_columns[top] = neighbor_column
                            top += 1
        return labels, next_label
else:  # pragma: no cover - environment dependent
    _label_components_kernel = None


@lru_cache(maxsize=1)
def _neighbor_count_kernel():
    from numba import cuda

    @cuda.jit(fastmath=False)
    def kernel(binary, coordinates, counts):
        index = cuda.grid(1)
        if index >= coordinates.shape[0]:
            return
        row = int(coordinates[index, 0]) - 1
        column = int(coordinates[index, 1]) - 1
        count = 0
        for row_offset in range(-1, 2):
            for column_offset in range(-1, 2):
                if row_offset != 0 or column_offset != 0:
                    count += binary[row + row_offset, column + column_offset]
        counts[index] = count

    return kernel


def cuda_neighbor_counts(
    binary: np.ndarray,
    coordinates: np.ndarray,
    *,
    device: int = 0,
    threads_per_block: int = 256,
) -> np.ndarray:
    """Count 8-neighbors in the supplied MATLAB coordinate order.

    The host validates the same interior-pixel precondition as the reference
    implementation. One CUDA thread handles one coordinate, so no reduction
    or atomic operation can reorder or combine neighboring results.
    """

    if int(threads_per_block) <= 0:
        raise ValueError("threads_per_block must be positive")
    value = np.asarray(binary)
    if value.ndim != 2:
        raise ValueError(f"binary must be a 2-D matrix, got shape {value.shape}")
    points = np.asarray(coordinates)
    if points.size == 0:
        return np.empty((0,), dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2)")
    if not np.all(np.isfinite(points)) or not np.all(points == np.floor(points)):
        raise ValueError("coordinates must contain finite integer values")
    points = points.astype(np.int64, copy=False)
    height, width = value.shape
    rows = points[:, 0] - 1
    columns = points[:, 1] - 1
    if (
        np.any(rows <= 0)
        or np.any(rows >= height - 1)
        or np.any(columns <= 0)
        or np.any(columns >= width - 1)
    ):
        raise IndexError("junction neighbor lookup requires an interior pixel")

    try:
        from numba import cuda
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("CUDA backend requires numba") from exc
    if not cuda.is_available():
        raise RuntimeError("Numba CUDA is not available")

    binary_host = np.ascontiguousarray(value != 0, dtype=np.uint8)
    coordinates_host = np.ascontiguousarray(points, dtype=np.int32)
    counts_host = np.zeros(points.shape[0], dtype=np.int64)
    cuda.select_device(int(device))
    d_binary = cuda.to_device(binary_host)
    d_coordinates = cuda.to_device(coordinates_host)
    d_counts = cuda.to_device(counts_host)
    kernel = _neighbor_count_kernel()
    blocks = (points.shape[0] + int(threads_per_block) - 1) // int(threads_per_block)
    kernel[blocks, int(threads_per_block)](d_binary, d_coordinates, d_counts)
    d_counts.copy_to_host(counts_host)
    return counts_host


@lru_cache(maxsize=1)
def _square_mask_kernel():
    from numba import cuda

    @cuda.jit(fastmath=False)
    def kernel(binary, radius, expanded):
        linear = cuda.grid(1)
        height, width = binary.shape
        if linear >= height * width:
            return
        row = linear // width
        column = linear - row * width
        found = 0
        for row_offset in range(-radius, radius + 1):
            source_row = row + row_offset
            if source_row < 0 or source_row >= height:
                continue
            for column_offset in range(-radius, radius + 1):
                source_column = column + column_offset
                if source_column < 0 or source_column >= width:
                    continue
                if binary[source_row, source_column] != 0:
                    found = 1
        expanded[linear] = found

    return kernel


def cuda_square_mask(
    binary: np.ndarray,
    radius: int,
    *,
    device: int = 0,
    threads_per_block: int = 256,
) -> np.ndarray:
    """Compute IterGenFragment's inclusive square mask on the GPU."""

    value = np.asarray(binary)
    if value.ndim != 2:
        raise ValueError(f"binary must be a 2-D matrix, got {value.shape}")
    radius_int = int(radius)
    if radius_int < 0 or radius_int != float(radius):
        raise ValueError("radius must be a non-negative integer")
    if int(threads_per_block) <= 0:
        raise ValueError("threads_per_block must be positive")
    try:
        from numba import cuda
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("CUDA backend requires numba") from exc
    if not cuda.is_available():
        raise RuntimeError("Numba CUDA is not available")

    height, width = value.shape
    binary_host = np.ascontiguousarray(value != 0, dtype=np.uint8)
    expanded_host = np.zeros(height * width, dtype=np.uint8)
    cuda.select_device(int(device))
    d_binary = cuda.to_device(binary_host)
    d_expanded = cuda.to_device(expanded_host)
    kernel = _square_mask_kernel()
    total = height * width
    blocks = (total + int(threads_per_block) - 1) // int(threads_per_block)
    kernel[blocks, int(threads_per_block)](d_binary, radius_int, d_expanded)
    d_expanded.copy_to_host(expanded_host)
    return expanded_host.reshape((height, width)).astype(bool, copy=False)


def _thin_luts() -> tuple[np.ndarray, np.ndarray]:
    """Load the exact Guo--Hall LUTs used by ``skimage.morphology.thin``."""

    try:
        from skimage.morphology._skeletonize import G123P_LUT, G123_LUT
    except ImportError as exc:  # pragma: no cover - dependency/version guard
        raise RuntimeError(
            "CUDA thinning requires the scikit-image thin LUTs"
        ) from exc
    return (
        np.asarray(G123_LUT, dtype=np.uint8),
        np.asarray(G123P_LUT, dtype=np.uint8),
    )


@lru_cache(maxsize=1)
def _thin_subiteration_kernel():
    from numba import cuda

    @cuda.jit(fastmath=False)
    def kernel(current, next_image, lut, changed):
        linear = cuda.grid(1)
        height, width = current.shape
        if linear >= height * width:
            return
        row = linear // width
        column = linear - row * width
        if current[row, column] == 0:
            next_image[row, column] = 0
            return

        # This is scipy.ndimage.correlate(skel, mask, mode="constant") with
        # the exact mask used by skimage.morphology.thin.  Explicit bounds
        # checks provide the same zero boundary, including image edges.
        neighborhood = 0
        if row > 0:
            if column > 0:
                neighborhood += int(current[row - 1, column - 1]) * 8
            neighborhood += int(current[row - 1, column]) * 4
            if column + 1 < width:
                neighborhood += int(current[row - 1, column + 1]) * 2
        if column > 0:
            neighborhood += int(current[row, column - 1]) * 16
        if column + 1 < width:
            neighborhood += int(current[row, column + 1])
        if row + 1 < height:
            if column > 0:
                neighborhood += int(current[row + 1, column - 1]) * 32
            neighborhood += int(current[row + 1, column]) * 64
            if column + 1 < width:
                neighborhood += int(current[row + 1, column + 1]) * 128

        if lut[neighborhood] != 0:
            next_image[row, column] = 0
            cuda.atomic.max(changed, 0, 1)
        else:
            next_image[row, column] = 1

    return kernel


def cuda_thin(
    binary: np.ndarray,
    *,
    device: int = 0,
    threads_per_block: int = 256,
    max_num_iter: int | None = None,
) -> np.ndarray:
    """Run the same two-subiteration Guo--Hall thinning loop on CUDA.

    The kernel is a direct execution backend for the already validated
    ``bwmorph_thin`` implementation: it uses the same 3x3 correlation mask,
    the same two lookup tables, zero boundary, deletion rule, and convergence
    test.  No connected-component or skeletonization algorithm is substituted.
    """

    value = np.asarray(binary)
    if value.ndim != 2:
        raise ValueError(f"binary must be a 2-D matrix, got shape {value.shape}")
    if int(threads_per_block) <= 0:
        raise ValueError("threads_per_block must be positive")
    if max_num_iter is not None:
        max_value = int(max_num_iter)
        if max_value != max_num_iter:
            raise ValueError("max_num_iter must be an integer or None")
    else:
        max_value = 0
    if value.size == 0:
        return value.astype(bool, copy=True)

    try:
        from numba import cuda
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("CUDA backend requires numba") from exc
    if not cuda.is_available():
        raise RuntimeError("Numba CUDA is not available")

    current_host = np.ascontiguousarray(value != 0, dtype=np.uint8)
    next_host = np.empty_like(current_host)
    lut_first, lut_second = _thin_luts()
    cuda.select_device(int(device))
    d_current = cuda.to_device(current_host)
    d_next = cuda.device_array_like(current_host)
    d_lut_first = cuda.to_device(lut_first)
    d_lut_second = cuda.to_device(lut_second)
    d_changed = cuda.to_device(np.zeros(1, dtype=np.uint32))
    kernel = _thin_subiteration_kernel()
    total = current_host.size
    blocks = (total + int(threads_per_block) - 1) // int(threads_per_block)

    iteration = 0
    any_change = True
    while any_change and (max_value == 0 or iteration < max_value):
        any_change = False
        for lut in (d_lut_first, d_lut_second):
            d_changed.copy_to_device(np.zeros(1, dtype=np.uint32))
            kernel[blocks, int(threads_per_block)](
                d_current,
                d_next,
                lut,
                d_changed,
            )
            changed_host = np.zeros(1, dtype=np.uint32)
            d_changed.copy_to_host(changed_host)
            d_current, d_next = d_next, d_current
            if changed_host[0] != 0:
                any_change = True
        iteration += 1

    d_current.copy_to_host(next_host)
    return next_host.astype(bool, copy=False)


@lru_cache(maxsize=1)
def _large_cur_neighbor_count_kernel():
    from numba import cuda

    @cuda.jit(fastmath=False)
    def kernel(label_image, coordinates, group_labels, counts):
        index = cuda.grid(1)
        if index >= coordinates.shape[0]:
            return
        row = int(coordinates[index, 0])
        column = int(coordinates[index, 1])
        group_label = group_labels[index]
        height, width = label_image.shape
        count = 0
        for row_offset in range(-1, 2):
            neighbor_row = row + row_offset
            if neighbor_row < 0 or neighbor_row >= height:
                continue
            for column_offset in range(-1, 2):
                neighbor_column = column + column_offset
                if neighbor_column < 0 or neighbor_column >= width:
                    continue
                if label_image[neighbor_row, neighbor_column] == group_label:
                    count += 1
        counts[index] = count

    return kernel


if njit is not None:

    @njit(cache=True, fastmath=False)
    def _large_cur_flag_path_kernel(points, neighbor_counts):
        """JIT the order-sensitive tail of ``LargeCurFlag``.

        The active array is a compact-array equivalent: scanning the original
        rows while skipping deleted entries has the same source order and
        first-minimum tie behavior as repeated MATLAB ``dsearchn``/delete.
        ``fastmath`` is deliberately disabled because the final strict
        ``1.2 * straight_distance < distance`` decision is part of the
        faithful backend contract.
        """

        point_count = points.shape[0]
        first_tip = -1
        second_tip = -1
        tip_count = 0
        for index in range(point_count):
            if neighbor_counts[index] == 2:
                if tip_count == 0:
                    first_tip = index
                elif tip_count == 1:
                    second_tip = index
                tip_count += 1
        if tip_count != 2:
            return 1

        minimum_row = points[0, 0]
        minimum_column = points[0, 1]
        for index in range(1, point_count):
            if points[index, 0] < minimum_row:
                minimum_row = points[index, 0]
            if points[index, 1] < minimum_column:
                minimum_column = points[index, 1]

        # LargeCurFlag creates x/y, shifts both by one again, and then casts
        # them to double for the nearest-point loop.  The combined +2 is
        # intentional and mirrors those two source statements exactly.
        x = np.empty(point_count, dtype=np.float64)
        y = np.empty(point_count, dtype=np.float64)
        for index in range(point_count):
            x[index] = float(points[index, 0] - minimum_row) + 2.0
            y[index] = float(points[index, 1] - minimum_column) + 2.0

        active = np.ones(point_count, dtype=np.bool_)
        tip_row = x[first_tip]
        tip_column = y[first_tip]
        path_distance = 0.0
        for _ in range(point_count):
            best_index = 0
            best_distance_squared = np.inf
            for index in range(point_count):
                if not active[index]:
                    continue
                delta_row = tip_row - x[index]
                delta_column = tip_column - y[index]
                distance_squared = (
                    delta_row * delta_row + delta_column * delta_column
                )
                # Strict comparison preserves the first row on equal ties.
                if distance_squared < best_distance_squared:
                    best_distance_squared = distance_squared
                    best_index = index
            path_distance += math.sqrt(best_distance_squared)
            tip_row = x[best_index]
            tip_column = y[best_index]
            active[best_index] = False

        delta_row = x[first_tip] - x[second_tip]
        delta_column = y[first_tip] - y[second_tip]
        straight_distance = math.hypot(delta_row, delta_column)
        return int(1 if 1.2 * straight_distance < path_distance else 0)

else:  # pragma: no cover - environment dependent
    _large_cur_flag_path_kernel = None


def numba_large_cur_flag_from_neighbor_counts(
    points: np.ndarray,
    neighbor_counts: np.ndarray,
) -> int:
    """Exact CPU-JIT companion for the ordered ``LargeCurFlag`` tail."""

    if _large_cur_flag_path_kernel is None:
        from .fragments import _large_cur_flag_from_neighbor_counts

        return int(_large_cur_flag_from_neighbor_counts(points, neighbor_counts))
    points_value = np.ascontiguousarray(np.asarray(points, dtype=np.int64))
    counts_value = np.ascontiguousarray(
        np.asarray(neighbor_counts, dtype=np.int64).reshape(-1)
    )
    return int(_large_cur_flag_path_kernel(points_value, counts_value))


def cuda_large_cur_flags(
    label_image: np.ndarray,
    fragment_coordinates: Sequence[np.ndarray],
    *,
    device: int = 0,
    threads_per_block: int = 256,
) -> np.ndarray:
    """Batch ``LargeCurFlag`` neighborhood counts on CUDA.

    The GPU computes only the independent 3x3 component-neighborhood counts.
    The ordered nearest-point path and the final strict threshold comparison
    use the optional no-fastmath Numba companion, which retains the MATLAB
    traversal, tie, ``sqrt`` and floating-point operation order.  The helper
    falls back to ``fragments._large_cur_flag_from_neighbor_counts`` when
    Numba is unavailable.
    """

    if int(threads_per_block) <= 0:
        raise ValueError("threads_per_block must be positive")
    labels = np.asarray(label_image)
    if labels.ndim != 2:
        raise ValueError(f"label_image must be a 2-D matrix, got shape {labels.shape}")
    labels_float = np.asarray(labels, dtype=np.float64)
    if not np.all(np.isfinite(labels_float)) or not np.all(
        labels_float == np.floor(labels_float)
    ):
        raise ValueError("label_image must contain finite integer labels")
    labels_host = np.ascontiguousarray(labels_float, dtype=np.int64)

    groups: list[np.ndarray] = []
    for coordinates in fragment_coordinates:
        points = np.asarray(coordinates)
        if points.size == 0:
            raise ValueError("LargeCurFlag requires at least one point")
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("each fragment coordinate array must have shape (N, 2)")
        if not np.all(np.isfinite(points)) or not np.all(points == np.floor(points)):
            raise ValueError("fragment coordinates must contain finite integer values")
        groups.append(points.astype(np.int64, copy=False))

    if not groups:
        return np.empty((0,), dtype=np.int64)

    concatenated_one_based = np.ascontiguousarray(np.concatenate(groups, axis=0))
    rows = concatenated_one_based[:, 0] - 1
    columns = concatenated_one_based[:, 1] - 1
    height, width = labels_host.shape
    if (
        np.any(rows < 0)
        or np.any(rows >= height)
        or np.any(columns < 0)
        or np.any(columns >= width)
    ):
        raise IndexError("fragment coordinate lies outside label_image")

    group_labels = np.empty(concatenated_one_based.shape[0], dtype=np.int64)
    offset = 0
    for points in groups:
        size = points.shape[0]
        point_rows = points[:, 0] - 1
        point_columns = points[:, 1] - 1
        observed_labels = labels_host[point_rows, point_columns]
        if np.any(observed_labels == 0) or np.any(observed_labels != observed_labels[0]):
            raise ValueError("all points in a fragment must have one nonzero label")
        group_labels[offset : offset + size] = observed_labels[0]
        offset += size

    coordinates_zero_based = np.ascontiguousarray(
        np.column_stack((rows, columns)), dtype=np.int32
    )
    group_labels = np.ascontiguousarray(group_labels, dtype=np.int64)
    counts_host = np.zeros(coordinates_zero_based.shape[0], dtype=np.int64)

    try:
        from numba import cuda
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("CUDA backend requires numba") from exc
    if not cuda.is_available():
        raise RuntimeError("Numba CUDA is not available")

    cuda.select_device(int(device))
    d_labels = cuda.to_device(labels_host)
    d_coordinates = cuda.to_device(coordinates_zero_based)
    d_group_labels = cuda.to_device(group_labels)
    d_counts = cuda.to_device(counts_host)
    kernel = _large_cur_neighbor_count_kernel()
    blocks = (
        coordinates_zero_based.shape[0] + int(threads_per_block) - 1
    ) // int(threads_per_block)
    kernel[blocks, int(threads_per_block)](
        d_labels,
        d_coordinates,
        d_group_labels,
        d_counts,
    )
    d_counts.copy_to_host(counts_host)

    flags = np.empty((len(groups),), dtype=np.int64)
    offset = 0
    for group_index, points in enumerate(groups):
        size = points.shape[0]
        flags[group_index] = numba_large_cur_flag_from_neighbor_counts(
            points,
            counts_host[offset : offset + size],
        )
        offset += size
    return flags


def numba_label_components(
    binary: np.ndarray,
    connectivity: int = 8,
) -> tuple[np.ndarray, int]:
    """JIT the exact reference scan/DFS for order-sensitive sorting.

    This is intentionally a CPU-side deterministic accelerator.  It is used
    only as an opt-in companion to the CUDA pipeline because a parallel GPU
    connected-component implementation could change MATLAB label numbering.
    """

    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    value = np.asarray(binary)
    if value.ndim != 2:
        raise ValueError(f"binary must be a 2-D matrix, got shape {value.shape}")
    if _label_components_kernel is None:
        from .fragments import label_components

        return label_components(value, connectivity=connectivity)
    if connectivity == 4:
        row_offsets = np.asarray((-1, 0, 0, 1), dtype=np.int64)
        column_offsets = np.asarray((0, -1, 1, 0), dtype=np.int64)
    else:
        row_offsets = np.asarray(
            (-1, -1, -1, 0, 0, 1, 1, 1), dtype=np.int64
        )
        column_offsets = np.asarray(
            (-1, 0, 1, -1, 1, -1, 0, 1), dtype=np.int64
        )
    return _label_components_kernel(value != 0, row_offsets, column_offsets)


__all__ = [
    "cuda_large_cur_flags",
    "cuda_neighbor_counts",
    "cuda_square_mask",
    "cuda_thin",
    "numba_large_cur_flag_from_neighbor_counts",
    "numba_label_components",
]
