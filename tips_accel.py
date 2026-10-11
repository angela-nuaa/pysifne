"""Serial JIT helpers for the order-sensitive TipReg primitives."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - environment dependent
    njit = None


if njit is not None:

    @njit(cache=True, fastmath=False)
    def _endpoint_mask_kernel(binary: np.ndarray) -> np.ndarray:
        height, width = binary.shape
        output = np.zeros((height, width), dtype=np.bool_)
        queue_rows = np.empty(9, dtype=np.int64)
        queue_columns = np.empty(9, dtype=np.int64)
        visited = np.zeros((3, 3), dtype=np.bool_)

        # MATLAB find/bwlookup order: columns outside the 3x3 neighborhood
        # are evaluated as zero-valued padding.  The tiny flood fill below is
        # exactly bwlabel(~nhood, 4), not an endpoint degree heuristic.
        for column in range(width):
            for row in range(height):
                if not binary[row, column]:
                    continue
                center_value = True
                components = 0
                for local_row in range(3):
                    for local_column in range(3):
                        visited[local_row, local_column] = False
                for local_row in range(3):
                    for local_column in range(3):
                        global_row = row + local_row - 1
                        global_column = column + local_column - 1
                        foreground = False
                        if (
                            0 <= global_row < height
                            and 0 <= global_column < width
                        ):
                            foreground = binary[global_row, global_column]
                        if foreground or visited[local_row, local_column]:
                            continue
                        components += 1
                        visited[local_row, local_column] = True
                        queue_rows[0] = local_row
                        queue_columns[0] = local_column
                        queue_size = 1
                        while queue_size > 0:
                            queue_size -= 1
                            current_row = queue_rows[queue_size]
                            current_column = queue_columns[queue_size]
                            for row_offset, column_offset in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                                neighbor_row = current_row + row_offset
                                neighbor_column = current_column + column_offset
                                if not (
                                    0 <= neighbor_row < 3
                                    and 0 <= neighbor_column < 3
                                ):
                                    continue
                                if visited[neighbor_row, neighbor_column]:
                                    continue
                                neighbor_global_row = row + neighbor_row - 1
                                neighbor_global_column = column + neighbor_column - 1
                                neighbor_foreground = False
                                if (
                                    0 <= neighbor_global_row < height
                                    and 0 <= neighbor_global_column < width
                                ):
                                    neighbor_foreground = binary[
                                        neighbor_global_row, neighbor_global_column
                                    ]
                                if neighbor_foreground:
                                    continue
                                visited[neighbor_row, neighbor_column] = True
                                queue_rows[queue_size] = neighbor_row
                                queue_columns[queue_size] = neighbor_column
                                queue_size += 1
                if center_value and components == 1:
                    output[row, column] = True
        return output


    @njit(cache=True, fastmath=False)
    def _local_centroids_kernel(
        selected_coordinates: np.ndarray,
        offsets: np.ndarray,
    ) -> np.ndarray:
        tip_count = offsets.shape[0] - 1
        output = np.empty((tip_count, 2), dtype=np.float64)
        total = selected_coordinates.shape[0]
        visited = np.zeros(total, dtype=np.bool_)
        stack = np.empty(total, dtype=np.int64)
        for tip_index in range(tip_count):
            start = int(offsets[tip_index])
            end = int(offsets[tip_index + 1])
            for index in range(start, end):
                visited[index] = False
            if start >= end:
                output[tip_index, 0] = np.nan
                output[tip_index, 1] = np.nan
                continue

            stack_size = 1
            stack[0] = start
            visited[start] = True
            while stack_size > 0:
                stack_size -= 1
                current = stack[stack_size]
                current_row = selected_coordinates[current, 0]
                current_column = selected_coordinates[current, 1]
                for candidate in range(start, end):
                    if visited[candidate]:
                        continue
                    if (
                        abs(selected_coordinates[candidate, 0] - current_row) <= 1
                        and abs(selected_coordinates[candidate, 1] - current_column) <= 1
                    ):
                        visited[candidate] = True
                        stack[stack_size] = candidate
                        stack_size += 1

            row_sum = 0.0
            column_sum = 0.0
            count = 0
            # selected_coordinates is already in MATLAB find order.  Scanning
            # it again, rather than summing the DFS stack, reproduces
            # regionprops' PixelList order for component 1.
            for index in range(start, end):
                if visited[index]:
                    row_sum += selected_coordinates[index, 0]
                    column_sum += selected_coordinates[index, 1]
                    count += 1
            output[tip_index, 0] = row_sum / float(count)
            output[tip_index, 1] = column_sum / float(count)
        return output

else:  # pragma: no cover - environment dependent.
    _endpoint_mask_kernel = None
    _local_centroids_kernel = None


def numba_detect_endpoints(binary: np.ndarray) -> np.ndarray:
    """Return the exact endpoint LUT mask, with MATLAB column-major order."""

    if _endpoint_mask_kernel is None:
        raise ImportError("numba is required for numba_detect_endpoints")
    value = np.asarray(binary, dtype=bool)
    if value.ndim != 2:
        raise ValueError("binary must be a 2-D array")
    return _endpoint_mask_kernel(np.ascontiguousarray(value))


def numba_local_centroids(
    selected_coordinates: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Compute TipReg's first 8-connected local component centroids."""

    if _local_centroids_kernel is None:
        raise ImportError("numba is required for numba_local_centroids")
    coordinates = np.asarray(selected_coordinates, dtype=np.int64)
    boundaries = np.asarray(offsets, dtype=np.int64).reshape(-1)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("selected_coordinates must have shape (N, 2)")
    if boundaries.size == 0 or boundaries[0] != 0 or boundaries[-1] != coordinates.shape[0]:
        raise ValueError("offsets must span selected_coordinates from zero to its size")
    if np.any(boundaries[1:] < boundaries[:-1]):
        raise ValueError("offsets must be non-decreasing")
    return _local_centroids_kernel(
        np.ascontiguousarray(coordinates),
        np.ascontiguousarray(boundaries),
    )


__all__ = ["numba_detect_endpoints", "numba_local_centroids"]
