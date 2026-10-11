"""Deterministic CUDA primitives for the SortFilament nearest-removal loop.

The MATLAB-compatible ordering is inherently sequential between successive
points.  This kernel keeps that state transition in one CUDA block, while
parallelising only the independent distance scan for the current point.  A
reduction compares ``(distance_squared, original_index)`` explicitly, so
equal-distance ties retain the first row of the source temporary array.
"""

from __future__ import annotations

import math

import numpy as np

from .sorting import numba_order_points


try:
    from numba import cuda
except ImportError:  # pragma: no cover - environment dependent
    cuda = None


if cuda is not None:

    @cuda.jit(fastmath=False)
    def _order_points_kernel(points, start, active, ordered):
        shared_scores = cuda.shared.array(1024, dtype=np.float64)
        shared_indices = cuda.shared.array(1024, dtype=np.int32)
        shared_valid = cuda.shared.array(1024, dtype=np.int32)
        shared_query = cuda.shared.array(2, dtype=np.float64)

        thread_index = cuda.threadIdx.x
        block_size = cuda.blockDim.x
        point_count = points.shape[0]

        if thread_index == 0:
            shared_query[0] = start[0]
            shared_query[1] = start[1]
        cuda.syncthreads()

        for output_index in range(point_count):
            local_valid = 0
            local_index = -1
            local_score = math.inf
            candidate_index = thread_index
            while candidate_index < point_count:
                if active[candidate_index]:
                    delta_row = points[candidate_index, 0] - shared_query[0]
                    delta_column = points[candidate_index, 1] - shared_query[1]
                    score = delta_row * delta_row + delta_column * delta_column
                    # The source starts with an Inf sentinel.  Treating NaN as
                    # Inf preserves: later finite values win, while an all-
                    # NaN/Inf scan retains the first active candidate.
                    if math.isnan(score):
                        score = math.inf
                    if local_valid == 0:
                        local_valid = 1
                        local_index = candidate_index
                        local_score = score
                    elif score < local_score:
                        local_index = candidate_index
                        local_score = score
                candidate_index += block_size

            shared_valid[thread_index] = local_valid
            shared_indices[thread_index] = local_index
            shared_scores[thread_index] = local_score
            cuda.syncthreads()

            stride = block_size // 2
            while stride > 0:
                if thread_index < stride:
                    right_index = thread_index + stride
                    right_valid = shared_valid[right_index]
                    left_valid = shared_valid[thread_index]
                    choose_right = False
                    if right_valid != 0:
                        if left_valid == 0:
                            choose_right = True
                        elif shared_scores[right_index] < shared_scores[thread_index]:
                            choose_right = True
                        elif (
                            shared_scores[right_index] == shared_scores[thread_index]
                            and shared_indices[right_index] < shared_indices[thread_index]
                        ):
                            choose_right = True
                    if choose_right:
                        shared_valid[thread_index] = right_valid
                        shared_indices[thread_index] = shared_indices[right_index]
                        shared_scores[thread_index] = shared_scores[right_index]
                cuda.syncthreads()
                stride //= 2

            if thread_index == 0:
                selected_index = shared_indices[0]
                ordered[output_index, 0] = points[selected_index, 0]
                ordered[output_index, 1] = points[selected_index, 1]
                shared_query[0] = points[selected_index, 0]
                shared_query[1] = points[selected_index, 1]
                active[selected_index] = False
            cuda.syncthreads()


    @cuda.jit(fastmath=False)
    def _order_filament_kernel(
        points,
        point_labels,
        component_labels,
        point_counts,
        component_counts,
        output,
        output_counts,
        starts,
        active,
    ):
        """Order every component of one composite in one CUDA block.

        ``SortFilament`` recreates a fresh ``temp`` and a fresh active mask
        for every label in ``l_unique``.  The kernel intentionally does the
        same instead of treating the input as one graph: a repeated label is
        therefore traversed repeatedly, and each component keeps the source
        compact-row order for ties.
        """

        shared_scores = cuda.shared.array(1024, dtype=np.float64)
        shared_indices = cuda.shared.array(1024, dtype=np.int32)
        shared_valid = cuda.shared.array(1024, dtype=np.int32)
        shared_query = cuda.shared.array(2, dtype=np.float64)

        thread_index = cuda.threadIdx.x
        block_size = cuda.blockDim.x
        filament_index = cuda.blockIdx.x
        point_count = point_counts[filament_index]
        component_count = component_counts[filament_index]

        if thread_index == 0:
            shared_query[0] = starts[filament_index, 0]
            shared_query[1] = starts[filament_index, 1]
            output_counts[filament_index] = 0
        cuda.syncthreads()

        output_index = 0
        for component_index in range(component_count):
            current_label = component_labels[filament_index, component_index]

            # This is the source's ``active = true(size(temp,1),1)``.
            # Resetting per label is required for duplicate labels.
            reset_index = thread_index
            while reset_index < point_count:
                active[filament_index, reset_index] = True
                reset_index += block_size
            cuda.syncthreads()

            while True:
                local_valid = 0
                local_index = -1
                local_score = math.inf
                candidate_index = thread_index
                while candidate_index < point_count:
                    if (
                        active[filament_index, candidate_index]
                        and point_labels[filament_index, candidate_index] == current_label
                    ):
                        delta_row = (
                            points[filament_index, candidate_index, 0]
                            - shared_query[0]
                        )
                        delta_column = (
                            points[filament_index, candidate_index, 1]
                            - shared_query[1]
                        )
                        score = delta_row * delta_row + delta_column * delta_column
                        # Match the MATLAB/Numba compact-row sentinel.  The
                        # first active candidate wins when every score is Inf
                        # or NaN; finite scores still replace it strictly.
                        if math.isnan(score):
                            score = math.inf
                        if local_valid == 0:
                            local_valid = 1
                            local_index = candidate_index
                            local_score = score
                        elif score < local_score:
                            local_index = candidate_index
                            local_score = score
                    candidate_index += block_size

                shared_valid[thread_index] = local_valid
                shared_indices[thread_index] = local_index
                shared_scores[thread_index] = local_score
                cuda.syncthreads()

                stride = block_size // 2
                while stride > 0:
                    if thread_index < stride:
                        right_index = thread_index + stride
                        right_valid = shared_valid[right_index]
                        left_valid = shared_valid[thread_index]
                        choose_right = False
                        if right_valid != 0:
                            if left_valid == 0:
                                choose_right = True
                            elif shared_scores[right_index] < shared_scores[thread_index]:
                                choose_right = True
                            elif (
                                shared_scores[right_index]
                                == shared_scores[thread_index]
                                and shared_indices[right_index]
                                < shared_indices[thread_index]
                            ):
                                choose_right = True
                        if choose_right:
                            shared_valid[thread_index] = right_valid
                            shared_indices[thread_index] = shared_indices[right_index]
                            shared_scores[thread_index] = shared_scores[right_index]
                    cuda.syncthreads()
                    stride //= 2

                if thread_index == 0:
                    if shared_valid[0] != 0:
                        selected_index = shared_indices[0]
                        output[filament_index, output_index, 0] = points[
                            filament_index, selected_index, 0
                        ]
                        output[filament_index, output_index, 1] = points[
                            filament_index, selected_index, 1
                        ]
                        shared_query[0] = points[filament_index, selected_index, 0]
                        shared_query[1] = points[filament_index, selected_index, 1]
                        active[filament_index, selected_index] = False
                        output_index += 1
                    # ``shared_valid[0]`` is also the synchronized loop
                    # termination flag: no candidate means this label's temp
                    # was empty/exhausted.
                cuda.syncthreads()
                if shared_valid[0] == 0:
                    break

        if thread_index == 0:
            output_counts[filament_index] = output_index


    @cuda.jit(fastmath=False)
    def _order_filaments_kernel(
        points,
        component_lengths,
        component_counts,
        output,
        output_counts,
        starts,
        active,
    ):
        """Batch the exact component loop across all composites.

        Host preparation packs each source ``temp`` consecutively.  A block
        owns one composite, and the block still performs the component loop
        and every nearest transition serially.  Only the independent scan of
        the current compact ``temp`` is parallel.
        """

        shared_scores = cuda.shared.array(1024, dtype=np.float64)
        shared_indices = cuda.shared.array(1024, dtype=np.int32)
        shared_valid = cuda.shared.array(1024, dtype=np.int32)
        shared_query = cuda.shared.array(2, dtype=np.float64)

        thread_index = cuda.threadIdx.x
        block_size = cuda.blockDim.x
        filament_index = cuda.blockIdx.x
        component_count = component_counts[filament_index]

        if thread_index == 0:
            shared_query[0] = starts[filament_index, 0]
            shared_query[1] = starts[filament_index, 1]
            output_counts[filament_index] = 0
        cuda.syncthreads()

        offset = 0
        output_index = 0
        for component_index in range(component_count):
            segment_length = component_lengths[filament_index, component_index]
            segment_end = offset + segment_length

            reset_index = offset + thread_index
            while reset_index < segment_end:
                active[filament_index, reset_index] = True
                reset_index += block_size
            cuda.syncthreads()

            while True:
                local_valid = 0
                local_index = -1
                local_score = math.inf
                candidate_index = offset + thread_index
                while candidate_index < segment_end:
                    if active[filament_index, candidate_index]:
                        delta_row = (
                            points[filament_index, candidate_index, 0]
                            - shared_query[0]
                        )
                        delta_column = (
                            points[filament_index, candidate_index, 1]
                            - shared_query[1]
                        )
                        score = delta_row * delta_row + delta_column * delta_column
                        if math.isnan(score):
                            score = math.inf
                        if local_valid == 0:
                            local_valid = 1
                            local_index = candidate_index
                            local_score = score
                        elif score < local_score:
                            local_index = candidate_index
                            local_score = score
                    candidate_index += block_size

                shared_valid[thread_index] = local_valid
                shared_indices[thread_index] = local_index
                shared_scores[thread_index] = local_score
                cuda.syncthreads()

                stride = block_size // 2
                while stride > 0:
                    if thread_index < stride:
                        right_index = thread_index + stride
                        right_valid = shared_valid[right_index]
                        left_valid = shared_valid[thread_index]
                        choose_right = False
                        if right_valid != 0:
                            if left_valid == 0:
                                choose_right = True
                            elif shared_scores[right_index] < shared_scores[thread_index]:
                                choose_right = True
                            elif (
                                shared_scores[right_index]
                                == shared_scores[thread_index]
                                and shared_indices[right_index]
                                < shared_indices[thread_index]
                            ):
                                choose_right = True
                        if choose_right:
                            shared_valid[thread_index] = right_valid
                            shared_indices[thread_index] = shared_indices[right_index]
                            shared_scores[thread_index] = shared_scores[right_index]
                    cuda.syncthreads()
                    stride //= 2

                if thread_index == 0:
                    if shared_valid[0] != 0:
                        selected_index = shared_indices[0]
                        output[filament_index, output_index, 0] = points[
                            filament_index, selected_index, 0
                        ]
                        output[filament_index, output_index, 1] = points[
                            filament_index, selected_index, 1
                        ]
                        shared_query[0] = points[filament_index, selected_index, 0]
                        shared_query[1] = points[filament_index, selected_index, 1]
                        active[filament_index, selected_index] = False
                        output_index += 1
                cuda.syncthreads()
                if shared_valid[0] == 0:
                    break

            offset = segment_end

        if thread_index == 0:
            output_counts[filament_index] = output_index

else:  # pragma: no cover - environment dependent
    _order_points_kernel = None
    _order_filament_kernel = None
    _order_filaments_kernel = None


def _next_power_of_two(value: int) -> int:
    result = 1
    while result < value and result < 1024:
        result *= 2
    return result


def cuda_order_points(
    points: np.ndarray,
    start: np.ndarray,
    *,
    device: int = 0,
) -> np.ndarray:
    """Return the exact SortFilament point order using one CUDA block.

    The CPU implementation remains the fallback when CUDA is unavailable or
    when Numba's simulator/device cannot execute the kernel.  No path-length
    arithmetic is performed on the device.
    """

    value = np.asarray(points, dtype=np.float64)
    query = np.asarray(start, dtype=np.float64).reshape(-1)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {value.shape}")
    if query.size != 2:
        raise ValueError(f"start must contain two coordinates, got {query.shape}")
    if value.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    if _order_points_kernel is None:
        return numba_order_points(value, query)
    if not cuda.is_available():
        return numba_order_points(value, query)

    points_host = np.ascontiguousarray(value, dtype=np.float64)
    start_host = np.ascontiguousarray(query, dtype=np.float64)
    active_host = np.ones(points_host.shape[0], dtype=np.bool_)
    ordered_host = np.empty_like(points_host)
    block_size = _next_power_of_two(points_host.shape[0])
    if points_host.shape[0] > 1024:
        block_size = 1024

    cuda.select_device(int(device))
    device_points = cuda.to_device(points_host)
    device_start = cuda.to_device(start_host)
    device_active = cuda.to_device(active_host)
    device_ordered = cuda.device_array_like(ordered_host)
    _order_points_kernel[1, block_size](
        device_points,
        device_start,
        device_active,
        device_ordered,
    )
    device_ordered.copy_to_host(ordered_host)
    return ordered_host


def cuda_order_points_adaptive(
    points: np.ndarray,
    start: np.ndarray,
    *,
    device: int = 0,
    minimum_points: int = 64,
) -> np.ndarray:
    """Use CUDA only when a filament is large enough to amortise transfers."""

    value = np.asarray(points, dtype=np.float64)
    if value.shape[0] < int(minimum_points):
        return numba_order_points(value, np.asarray(start, dtype=np.float64))
    return cuda_order_points(value, start, device=device)


def _order_filament_cpu(
    points: np.ndarray,
    point_labels: np.ndarray,
    component_labels: np.ndarray,
    start: np.ndarray,
) -> np.ndarray:
    """Exact fallback for the full-composite callback."""

    value = np.asarray(points, dtype=np.float64)
    labels = np.asarray(point_labels, dtype=np.float64).reshape(-1)
    components = np.asarray(component_labels, dtype=np.float64).reshape(-1)
    changing = np.asarray(start, dtype=np.float64).reshape(-1).copy()
    ordered: list[np.ndarray] = []
    for component_label in components:
        temp = value[labels == component_label].copy()
        if temp.shape[0] == 0:
            continue
        component_order = numba_order_points(temp, changing)
        for next_point in component_order:
            ordered.append(next_point.copy())
            changing = next_point.copy()
    return (
        np.vstack(ordered).astype(np.float64, copy=False)
        if ordered
        else np.empty((0, 2), dtype=np.float64)
    )


def cuda_order_filament(
    points: np.ndarray,
    point_labels: np.ndarray,
    component_labels: np.ndarray,
    start: np.ndarray,
    *,
    device: int = 0,
) -> np.ndarray:
    """Order a complete MATLAB ``SortFilament`` composite in one launch.

    The host-side callback has one composite, while the device keeps the
    original per-component reset and sequential nearest transition.  This
    removes one kernel launch and three device transfers per fragment without
    changing the source traversal order.
    """

    value = np.asarray(points, dtype=np.float64)
    labels = np.asarray(point_labels, dtype=np.float64).reshape(-1)
    components = np.asarray(component_labels, dtype=np.float64).reshape(-1)
    query = np.asarray(start, dtype=np.float64).reshape(-1)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {value.shape}")
    if labels.shape[0] != value.shape[0]:
        raise ValueError("point_labels must have one row per point")
    if query.size != 2:
        raise ValueError("start must contain two coordinates")
    if value.shape[0] == 0 or components.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    if (
        not np.all(np.isfinite(labels))
        or not np.all(np.isfinite(components))
        or not np.all(labels == np.floor(labels))
        or not np.all(components == np.floor(components))
        or np.any(np.abs(labels) > np.iinfo(np.int32).max)
        or np.any(np.abs(components) > np.iinfo(np.int32).max)
    ):
        return _order_filament_cpu(value, labels, components, query)
    if _order_filament_kernel is None or not cuda.is_available():
        return _order_filament_cpu(value, labels, components, query)

    points_host = np.ascontiguousarray(value[None, :, :], dtype=np.float64)
    labels_host = np.ascontiguousarray(labels[None, :].astype(np.int32), dtype=np.int32)
    components_host = np.ascontiguousarray(
        components[None, :].astype(np.int32), dtype=np.int32
    )
    point_counts_host = np.asarray([value.shape[0]], dtype=np.int32)
    component_counts_host = np.asarray([components.shape[0]], dtype=np.int32)
    starts_host = np.ascontiguousarray(query[None, :], dtype=np.float64)
    output_host = np.empty((1, value.shape[0] * max(1, components.shape[0]), 2), dtype=np.float64)
    output_counts_host = np.zeros(1, dtype=np.int32)
    active_host = np.zeros((1, value.shape[0]), dtype=np.bool_)
    block_size = _next_power_of_two(value.shape[0])
    if block_size > 1024:
        block_size = 1024

    cuda.select_device(int(device))
    device_points = cuda.to_device(points_host)
    device_labels = cuda.to_device(labels_host)
    device_components = cuda.to_device(components_host)
    device_point_counts = cuda.to_device(point_counts_host)
    device_component_counts = cuda.to_device(component_counts_host)
    device_output = cuda.device_array_like(output_host)
    device_output_counts = cuda.to_device(output_counts_host)
    device_starts = cuda.to_device(starts_host)
    device_active = cuda.to_device(active_host)
    _order_filament_kernel[1, block_size](
        device_points,
        device_labels,
        device_components,
        device_point_counts,
        device_component_counts,
        device_output,
        device_output_counts,
        device_starts,
        device_active,
    )
    device_output_counts.copy_to_host(output_counts_host)
    device_output.copy_to_host(output_host)
    return output_host[0, : int(output_counts_host[0]), :].copy()


def cuda_order_filaments(
    points_values,
    point_label_values,
    component_label_values,
    starts,
    *,
    device: int = 0,
):
    """Order every composite with one deterministic multi-block launch."""

    points_list = list(points_values)
    point_labels_list = list(point_label_values)
    component_labels_list = list(component_label_values)
    starts_list = list(starts)
    composite_count = len(points_list)
    if not (
        len(point_labels_list) == composite_count
        and len(component_labels_list) == composite_count
        and len(starts_list) == composite_count
    ):
        raise ValueError("all batch inputs must have the same composite count")
    if composite_count == 0:
        return []

    normalized_points: list[np.ndarray] = []
    normalized_point_labels: list[np.ndarray] = []
    normalized_component_labels: list[np.ndarray] = []
    normalized_starts: list[np.ndarray] = []
    packed_components: list[list[np.ndarray]] = []
    component_lengths_list: list[list[int]] = []
    max_packed = 0
    max_component_length = 0
    max_component_count = 0

    for points, point_labels, component_labels, start in zip(
        points_list, point_labels_list, component_labels_list, starts_list
    ):
        value = np.asarray(points, dtype=np.float64)
        labels = np.asarray(point_labels, dtype=np.float64).reshape(-1)
        components = np.asarray(component_labels, dtype=np.float64).reshape(-1)
        query = np.asarray(start, dtype=np.float64).reshape(-1)
        if value.ndim != 2 or value.shape[1] != 2:
            raise ValueError(f"points must have shape (N, 2), got {value.shape}")
        if labels.shape[0] != value.shape[0]:
            raise ValueError("point_labels must have one row per point")
        if query.size != 2:
            raise ValueError("each start must contain two coordinates")
        if (
            not np.all(np.isfinite(labels))
            or not np.all(np.isfinite(components))
            or not np.all(labels == np.floor(labels))
            or not np.all(components == np.floor(components))
            or np.any(np.abs(labels) > np.iinfo(np.int32).max)
            or np.any(np.abs(components) > np.iinfo(np.int32).max)
        ):
            return [
                _order_filament_cpu(p, l, c, s)
                for p, l, c, s in zip(
                    points_list,
                    point_labels_list,
                    component_labels_list,
                    starts_list,
                )
            ]
        chunks: list[np.ndarray] = []
        lengths: list[int] = []
        for component_label in components:
            temp = value[labels == component_label].copy()
            chunks.append(temp)
            lengths.append(int(temp.shape[0]))
            max_component_length = max(max_component_length, int(temp.shape[0]))
        packed_count = int(sum(lengths))
        max_packed = max(max_packed, packed_count)
        max_component_count = max(max_component_count, len(lengths))
        normalized_points.append(value)
        normalized_point_labels.append(labels)
        normalized_component_labels.append(components)
        normalized_starts.append(query)
        packed_components.append(chunks)
        component_lengths_list.append(lengths)

    if max_packed == 0 or _order_filaments_kernel is None or not cuda.is_available():
        return [
            _order_filament_cpu(p, l, c, s)
            for p, l, c, s in zip(
                normalized_points,
                normalized_point_labels,
                normalized_component_labels,
                normalized_starts,
            )
        ]

    points_host = np.zeros((composite_count, max_packed, 2), dtype=np.float64)
    component_lengths_host = np.zeros(
        (composite_count, max_component_count), dtype=np.int32
    )
    starts_host = np.zeros((composite_count, 2), dtype=np.float64)
    for composite_index, chunks in enumerate(packed_components):
        offset = 0
        starts_host[composite_index, :] = normalized_starts[composite_index]
        for component_index, chunk in enumerate(chunks):
            length = int(chunk.shape[0])
            component_lengths_host[composite_index, component_index] = length
            if length:
                points_host[composite_index, offset : offset + length, :] = chunk
            offset += length
    component_counts_host = np.asarray(
        [len(lengths) for lengths in component_lengths_list], dtype=np.int32
    )
    output_host = np.empty_like(points_host)
    output_counts_host = np.zeros(composite_count, dtype=np.int32)
    active_host = np.zeros((composite_count, max_packed), dtype=np.bool_)
    block_size = _next_power_of_two(max_component_length)
    if block_size > 1024:
        block_size = 1024

    cuda.select_device(int(device))
    device_points = cuda.to_device(points_host)
    device_lengths = cuda.to_device(component_lengths_host)
    device_counts = cuda.to_device(component_counts_host)
    device_output = cuda.device_array_like(output_host)
    device_output_counts = cuda.to_device(output_counts_host)
    device_starts = cuda.to_device(starts_host)
    device_active = cuda.to_device(active_host)
    _order_filaments_kernel[composite_count, block_size](
        device_points,
        device_lengths,
        device_counts,
        device_output,
        device_output_counts,
        device_starts,
        device_active,
    )
    device_output_counts.copy_to_host(output_counts_host)
    device_output.copy_to_host(output_host)
    result: list[np.ndarray] = []
    for composite_index, lengths in enumerate(component_lengths_list):
        expected = int(sum(lengths))
        actual = int(output_counts_host[composite_index])
        if actual != expected:
            raise RuntimeError(
                "CUDA batch SortFilament returned an unexpected point count: "
                f"composite {composite_index}, got {actual}, expected {expected}"
            )
        result.append(output_host[composite_index, :actual, :].copy())
    return result


__all__ = [
    "cuda_order_points",
    "cuda_order_points_adaptive",
    "cuda_order_filament",
    "cuda_order_filaments",
]
