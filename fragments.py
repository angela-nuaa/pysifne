"""Reference implementation of the SIFNE fragment-extraction stage.

The functions in this module follow the active code path in
``func_analysis_mt.m`` and ``IterGenFragment.m``.  Coordinates exposed by the
helper functions are MATLAB-style one-based ``(row, column)`` pairs; NumPy
array indexing is converted explicitly at each access.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Callable, Sequence

import numpy as np

from .lft_oft import lft_oft
from .segmentation import bwmorph_thin, imbinarize, mat2gray


EIGHT_NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _integer_parameter(value: Real, *, name: str) -> int:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric != math.floor(numeric):
        raise ValueError(f"{name} must be a finite integer, got {value!r}")
    return int(numeric)


def matlab_find(binary: np.ndarray) -> np.ndarray:
    """Return MATLAB ``find(binary)`` coordinates in column-major order."""

    value = np.asarray(binary)
    if value.ndim != 2:
        raise ValueError(f"binary must be a 2-D matrix, got shape {value.shape}")
    mask = value != 0
    height, _ = mask.shape
    linear_indices = np.flatnonzero(mask.ravel(order="F"))
    rows = linear_indices % height
    columns = linear_indices // height
    return np.column_stack((rows + 1, columns + 1)).astype(np.int64, copy=False)


def remove_border_margin(binary: np.ndarray, radius: Real) -> np.ndarray:
    """Reproduce the four margin assignments in ``func_analysis_mt.m``.

    In particular, ``(end-R):end`` is inclusive in MATLAB and therefore
    removes ``R+1`` rows/columns at the bottom/right, while ``1:R`` removes
    ``R`` rows/columns at the top/left.
    """

    value = np.asarray(binary)
    if value.ndim != 2:
        raise ValueError(f"binary must be a 2-D matrix, got shape {value.shape}")
    radius_int = _integer_parameter(radius, name="radius")
    if radius_int < 0:
        raise ValueError("radius must be non-negative")

    result = value.astype(bool, copy=True)
    if radius_int > 0:
        result[:radius_int, :] = False
        result[:, :radius_int] = False
    # MATLAB (end-radius):end, translated from one-based inclusive indexing.
    result[-(radius_int + 1) :, :] = False
    result[:, -(radius_int + 1) :] = False
    return result


def _validate_coordinate_matrix(coordinates: np.ndarray) -> np.ndarray:
    value = np.asarray(coordinates)
    if value.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(
            "coordinates must have shape (N, 2) in MATLAB one-based row/column form"
        )
    if not np.all(np.isfinite(value)) or not np.all(value == np.floor(value)):
        raise ValueError("coordinates must contain finite integer values")
    return value.astype(np.int64, copy=False)


def _neighbor_counts(
    binary: np.ndarray,
    coordinates: np.ndarray,
    *,
    neighbor_count_impl: Callable[..., np.ndarray] | None = None,
) -> np.ndarray:
    """Count the eight neighbors in the source's N1..N8 order."""

    value = np.asarray(binary).astype(bool, copy=False)
    coordinates = _validate_coordinate_matrix(coordinates)
    height, width = value.shape
    if neighbor_count_impl is not None:
        # Validate borders in the same MATLAB-compatible coordinate space
        # before handing the independent per-coordinate work to a backend.
        # This keeps the error behavior deterministic and leaves the output
        # order entirely controlled by ``coordinates``.
        for row_one, column_one in coordinates:
            row = int(row_one) - 1
            column = int(column_one) - 1
            if row <= 0 or row >= height - 1 or column <= 0 or column >= width - 1:
                raise IndexError(
                    "junction neighbor lookup requires an interior pixel, "
                    f"got MATLAB coordinate ({row_one}, {column_one})"
                )
        counts = np.asarray(neighbor_count_impl(value, coordinates))
        if counts.shape != (coordinates.shape[0],):
            raise ValueError(
                "neighbor_count_impl must return one count per coordinate, "
                f"got shape {counts.shape}"
            )
        if not np.all(np.isfinite(counts)) or not np.all(counts == np.floor(counts)):
            raise ValueError("neighbor_count_impl returned non-integer counts")
        return counts.astype(np.int64, copy=False)

    counts = np.zeros(coordinates.shape[0], dtype=np.int64)

    for index, (row_one, column_one) in enumerate(coordinates):
        row = int(row_one) - 1
        column = int(column_one) - 1
        if row <= 0 or row >= height - 1 or column <= 0 or column >= width - 1:
            # The MATLAB caller removes the image margins before evaluating
            # these sub2ind expressions.  A border point would be an invalid
            # MATLAB sub2ind call, so fail rather than silently pad it.
            raise IndexError(
                "junction neighbor lookup requires an interior pixel, "
                f"got MATLAB coordinate ({row_one}, {column_one})"
            )
        count = 0
        for row_offset, column_offset in EIGHT_NEIGHBOR_OFFSETS:
            count += int(value[row + row_offset, column + column_offset])
        counts[index] = count
    return counts


def detect_junction_pixels(
    binary: np.ndarray,
    *,
    neighbor_count_impl: Callable[..., np.ndarray] | None = None,
) -> np.ndarray:
    """Return skeleton pixels with at least three 8-neighbors."""

    coordinates = matlab_find(binary)
    counts = _neighbor_counts(
        binary, coordinates, neighbor_count_impl=neighbor_count_impl
    )
    return coordinates[counts >= 3].copy()


def detect_isolated_pixels(
    binary: np.ndarray,
    *,
    neighbor_count_impl: Callable[..., np.ndarray] | None = None,
) -> np.ndarray:
    """Return skeleton pixels with zero 8-neighbors."""

    coordinates = matlab_find(binary)
    counts = _neighbor_counts(
        binary, coordinates, neighbor_count_impl=neighbor_count_impl
    )
    return coordinates[counts == 0].copy()


def remove_junction_neighborhood(
    binary: np.ndarray,
    junctions: np.ndarray,
    radius: Real,
) -> np.ndarray:
    """Zero each inclusive square block centered at a junction pixel."""

    value = np.asarray(binary)
    if value.ndim != 2:
        raise ValueError(f"binary must be a 2-D matrix, got shape {value.shape}")
    coordinates = _validate_coordinate_matrix(junctions)
    radius_int = _integer_parameter(radius, name="junction radius")
    if radius_int < 0:
        raise ValueError("junction radius must be non-negative")

    result = value.astype(bool, copy=True)
    height, width = result.shape
    for row_one, column_one in coordinates:
        row = int(row_one) - 1
        column = int(column_one) - 1
        row_start = row - radius_int
        row_stop = row + radius_int + 1
        column_start = column - radius_int
        column_stop = column + radius_int + 1
        if (
            row_start < 0
            or row_stop > height
            or column_start < 0
            or column_stop > width
        ):
            raise IndexError(
                "junction removal block exceeds the MATLAB image boundary, "
                f"center=({row_one}, {column_one}), radius={radius_int}"
            )
        result[row_start:row_stop, column_start:column_stop] = False
    return result


def remove_isolated_pixels(
    binary: np.ndarray,
    *,
    neighbor_count_impl: Callable[..., np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the singleton list computed once by the MATLAB code."""

    isolated = detect_isolated_pixels(
        binary, neighbor_count_impl=neighbor_count_impl
    )
    result = np.asarray(binary).astype(bool, copy=True)
    for row_one, column_one in isolated:
        result[int(row_one) - 1, int(column_one) - 1] = False
    return result, isolated


@dataclass(frozen=True)
class JunctionRemovalResult:
    """Intermediate values produced by junction and singleton removal."""

    margin_removed: np.ndarray
    junctions: np.ndarray
    junction_removed: np.ndarray
    isolated_pixels: np.ndarray
    skeleton: np.ndarray
    labels: np.ndarray
    num_components: int
    all_points: np.ndarray


def remove_junctions_and_isolated(
    binary: np.ndarray,
    *,
    margin_radius: Real,
    junction_radius: Real,
    neighbor_count_impl: Callable[..., np.ndarray] | None = None,
    label_components_impl: Callable[..., tuple[np.ndarray, int]] | None = None,
) -> JunctionRemovalResult:
    """Run the active pre-fragment cleanup path and retain diagnostics."""

    margin_removed = remove_border_margin(binary, margin_radius)
    junctions = detect_junction_pixels(
        margin_removed, neighbor_count_impl=neighbor_count_impl
    )
    junction_removed = remove_junction_neighborhood(
        margin_removed, junctions, junction_radius
    )
    skeleton, isolated_pixels = remove_isolated_pixels(
        junction_removed, neighbor_count_impl=neighbor_count_impl
    )
    component_labeler = (
        label_components
        if label_components_impl is None
        else label_components_impl
    )
    labels, num_components = component_labeler(skeleton, connectivity=8)
    return JunctionRemovalResult(
        margin_removed=margin_removed,
        junctions=junctions,
        junction_removed=junction_removed,
        isolated_pixels=isolated_pixels,
        skeleton=skeleton,
        labels=labels,
        num_components=num_components,
        all_points=matlab_find(skeleton),
    )


def label_components(
    binary: np.ndarray,
    connectivity: int = 8,
) -> tuple[np.ndarray, int]:
    """Implement MATLAB ``bwlabel`` with deterministic column-major seeds."""

    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    value = np.asarray(binary)
    if value.ndim != 2:
        raise ValueError(f"binary must be a 2-D matrix, got shape {value.shape}")
    mask = value != 0
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.float64)
    if connectivity == 4:
        offsets = ((-1, 0), (0, -1), (0, 1), (1, 0))
    else:
        offsets = EIGHT_NEIGHBOR_OFFSETS

    next_label = 0
    # MATLAB's image-linear order is column-major: rows advance inside each
    # column, then columns advance from left to right.
    for column in range(width):
        for row in range(height):
            if not mask[row, column] or labels[row, column] != 0:
                continue
            next_label += 1
            labels[row, column] = next_label
            pending = [(row, column)]
            while pending:
                current_row, current_column = pending.pop()
                for row_offset, column_offset in offsets:
                    neighbor_row = current_row + row_offset
                    neighbor_column = current_column + column_offset
                    if (
                        0 <= neighbor_row < height
                        and 0 <= neighbor_column < width
                        and mask[neighbor_row, neighbor_column]
                        and labels[neighbor_row, neighbor_column] == 0
                    ):
                        labels[neighbor_row, neighbor_column] = next_label
                        pending.append((neighbor_row, neighbor_column))
    return labels, next_label


def remove_short_fragments(
    binary: np.ndarray,
    minimum_length: Real,
    *,
    connectivity: int = 8,
    label_components_impl: Callable[..., tuple[np.ndarray, int]] | None = None,
) -> np.ndarray:
    """Reproduce ``bwareaopen(binary, minimum_length, connectivity)``."""

    minimum = _integer_parameter(minimum_length, name="minimum fragment length")
    if minimum < 0:
        raise ValueError("minimum fragment length must be non-negative")
    component_labeler = (
        label_components
        if label_components_impl is None
        else label_components_impl
    )
    labels, num_components = component_labeler(binary, connectivity=connectivity)
    if num_components == 0:
        return np.zeros_like(np.asarray(binary), dtype=bool)

    sizes = np.bincount(labels.astype(np.int64).ravel(), minlength=num_components + 1)
    keep = sizes >= minimum
    keep[0] = False
    return keep[labels.astype(np.int64)]


def _large_cur_flag_from_neighbor_counts(
    points: np.ndarray,
    neighbor_counts: np.ndarray,
) -> int:
    """Finish ``LargeCurFlag`` after its independent neighborhood counts."""

    points = _validate_coordinate_matrix(points)
    if points.shape[0] == 0:
        raise ValueError("LargeCurFlag requires at least one point")
    counts = np.asarray(neighbor_counts, dtype=np.int64).reshape(-1)
    if counts.shape[0] != points.shape[0]:
        raise ValueError("neighbor_counts must contain one value per point")
    x = points[:, 0].astype(np.int64) - int(np.min(points[:, 0])) + 1
    y = points[:, 1].astype(np.int64) - int(np.min(points[:, 1])) + 1
    x = x + 1
    y = y + 1
    tip_indices = np.flatnonzero(counts == 2)
    if tip_indices.size != 2:
        return 1

    x_change = x.astype(np.float64).copy()
    y_change = y.astype(np.float64).copy()
    tip_change = np.array(
        [float(x[tip_indices[0]]), float(y[tip_indices[0]])], dtype=np.float64
    )
    distance = 0.0
    for _ in range(points.shape[0]):
        best_index = 0
        best_distance_squared = math.inf
        # Explicit strict comparison preserves the first nearest point on a
        # tie, matching the deterministic order supplied to dsearchn.
        for index in range(x_change.size):
            delta_x = tip_change[0] - x_change[index]
            delta_y = tip_change[1] - y_change[index]
            distance_squared = delta_x * delta_x + delta_y * delta_y
            if distance_squared < best_distance_squared:
                best_distance_squared = distance_squared
                best_index = index
        distance += math.sqrt(best_distance_squared)
        tip_change = np.array(
            [x_change[best_index], y_change[best_index]], dtype=np.float64
        )
        x_change = np.delete(x_change, best_index)
        y_change = np.delete(y_change, best_index)

    straight_distance = math.hypot(
        float(x[tip_indices[0]] - x[tip_indices[1]]),
        float(y[tip_indices[0]] - y[tip_indices[1]]),
    )
    return int(1 if 1.2 * straight_distance < distance else 0)


def large_cur_flag(coordinates: np.ndarray) -> int:
    """Reproduce ``LargeCurFlag(x,y)`` for MATLAB one-based coordinates."""

    points = _validate_coordinate_matrix(coordinates)
    if points.shape[0] == 0:
        raise ValueError("LargeCurFlag requires at least one point")

    x = points[:, 0].astype(np.int64) - int(np.min(points[:, 0])) + 1
    y = points[:, 1].astype(np.int64) - int(np.min(points[:, 1])) + 1
    temp = np.zeros((int(np.max(x)) + 2, int(np.max(y)) + 2), dtype=bool)
    x = x + 1
    y = y + 1
    temp[x - 1, y - 1] = True

    neighbor_offsets_with_center = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    neighbor_counts = np.zeros(points.shape[0], dtype=np.int64)
    for index, (row, column) in enumerate(zip(x, y, strict=True)):
        count = 0
        for row_offset, column_offset in neighbor_offsets_with_center:
            count += int(temp[row - 1 + row_offset, column - 1 + column_offset])
        neighbor_counts[index] = count

    return _large_cur_flag_from_neighbor_counts(points, neighbor_counts)


def _assign_points(matrix: np.ndarray, coordinates: np.ndarray, value: Real) -> None:
    for row_one, column_one in _validate_coordinate_matrix(coordinates):
        matrix[int(row_one) - 1, int(column_one) - 1] = value


def assemble_fragments(
    all_fragments: np.ndarray,
    new_fragment_coordinates: np.ndarray,
) -> np.ndarray:
    """Reproduce the final ``AllFragments(x,y)=1`` fragment assembly."""

    result = np.asarray(all_fragments).astype(bool, copy=True)
    _assign_points(result, new_fragment_coordinates, 1)
    return result


def _extract_new_fragments(
    origin_image: np.ndarray,
    all_fragments: np.ndarray,
    *,
    iteration_radius: int,
    radius: int,
    roi_mask: np.ndarray,
    n_angles: Real,
    junction_radius: int,
    minimum_length: int,
    lft_oft_impl: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    neighbor_count_impl: Callable[..., np.ndarray] | None = None,
    label_components_impl: Callable[..., tuple[np.ndarray, int]] | None = None,
    mask2_impl: Callable[[np.ndarray, int], np.ndarray] | None = None,
    thin_impl: Callable[[np.ndarray], np.ndarray] | None = None,
    large_cur_flags_impl: Callable[
        [np.ndarray, Sequence[np.ndarray]], np.ndarray
    ]
    | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run one literal iteration of ``IterGenFragment``."""

    component_labeler = (
        label_components
        if label_components_impl is None
        else label_components_impl
    )
    previous_labels, previous_num = component_labeler(all_fragments, connectivity=8)
    del previous_labels

    origin = np.asarray(origin_image)
    current = np.asarray(all_fragments).astype(bool, copy=True)
    height, width = current.shape
    new_image = np.zeros((height, width), dtype=np.float64)
    origin_height, origin_width = origin.shape
    new_image[radius : radius + origin_height, radius : radius + origin_width] = origin
    new_image = mat2gray(new_image)

    if mask2_impl is None:
        all_points = matlab_find(current)
        mask2 = np.zeros_like(current, dtype=bool)
        # MATLAB loops i then j and writes the same mask2.  Keep both loops
        # and the MATLAB coordinate conversion explicit in the reference path.
        for row_one, column_one in all_points:
            row = int(row_one) - 1
            column = int(column_one) - 1
            for row_offset in range(-iteration_radius, iteration_radius + 1):
                for column_offset in range(-iteration_radius, iteration_radius + 1):
                    mask2[row + row_offset, column + column_offset] = True
    else:
        mask2 = np.asarray(mask2_impl(current, iteration_radius), dtype=bool)
        if mask2.shape != current.shape:
            raise ValueError(
                "mask2_impl must return an array with the current image shape"
            )

    master_mask = np.asarray(roi_mask, dtype=np.float64) - mask2.astype(np.float64)
    # The default remains the verified CPU reference.  This callback is an
    # execution-backend seam only; all MATLAB-ordered mask construction,
    # thinning, labeling, and fragment state transitions stay on this path.
    compute_lft_oft = lft_oft if lft_oft_impl is None else lft_oft_impl
    oft_image, _, _ = compute_lft_oft(new_image, radius, n_angles, master_mask)
    oft_image = mat2gray(oft_image)
    compute_thin = bwmorph_thin if thin_impl is None else thin_impl
    raw_skeleton = compute_thin(imbinarize(oft_image))

    raw_skeleton = remove_border_margin(raw_skeleton, radius)
    junctions = detect_junction_pixels(
        raw_skeleton, neighbor_count_impl=neighbor_count_impl
    )
    raw_skeleton = remove_junction_neighborhood(
        raw_skeleton, junctions, junction_radius
    )
    raw_skeleton, isolated = remove_isolated_pixels(
        raw_skeleton, neighbor_count_impl=neighbor_count_impl
    )

    raw_skeleton = remove_short_fragments(
        raw_skeleton,
        minimum_length,
        connectivity=8,
        label_components_impl=label_components_impl,
    )
    raw_labels, raw_num = component_labeler(raw_skeleton, connectivity=8)
    # ``find(raw_labels == label)`` in the source is repeated once per
    # component.  One column-major scan plus stable label masks is exactly
    # the same coordinate stream and avoids rescanning the full image for
    # every fragment.
    raw_points = matlab_find(raw_labels != 0)
    raw_point_labels = raw_labels[
        raw_points[:, 0] - 1,
        raw_points[:, 1] - 1,
    ] if raw_points.shape[0] else np.empty((0,), dtype=np.float64)
    large_flags: list[int] = []
    if large_cur_flags_impl is None:
        for label in range(1, raw_num + 1):
            fragment_points = raw_points[raw_point_labels == float(label)]
            flag = large_cur_flag(fragment_points)
            large_flags.append(flag)
            if flag == 1:
                _assign_points(raw_labels, fragment_points, 0)
    else:
        fragment_points_by_label = [
            raw_points[raw_point_labels == float(label)]
            for label in range(1, raw_num + 1)
        ]
        accelerated_flags = np.asarray(
            large_cur_flags_impl(raw_labels, fragment_points_by_label),
            dtype=np.int64,
        ).reshape(-1)
        if accelerated_flags.shape[0] != raw_num:
            raise ValueError(
                "large_cur_flags_impl must return one flag per component, "
                f"got {accelerated_flags.shape[0]} for {raw_num}"
            )
        for label, fragment_points, flag_value in zip(
            range(1, raw_num + 1),
            fragment_points_by_label,
            accelerated_flags,
            strict=True,
        ):
            flag = int(flag_value)
            large_flags.append(flag)
            if flag == 1:
                _assign_points(raw_labels, fragment_points, 0)

    raw_labels[raw_labels != 0] = 1
    raw_labels, _ = component_labeler(raw_labels != 0, connectivity=8)
    new_points = matlab_find(raw_labels != 0)
    current = assemble_fragments(current, new_points)

    current = remove_border_margin(current, radius)
    junctions_after_merge = detect_junction_pixels(
        current, neighbor_count_impl=neighbor_count_impl
    )
    current = remove_junction_neighborhood(
        current, junctions_after_merge, junction_radius
    )
    current, isolated_after_merge = remove_isolated_pixels(
        current, neighbor_count_impl=neighbor_count_impl
    )
    current = remove_short_fragments(
        current,
        minimum_length,
        connectivity=8,
        label_components_impl=label_components_impl,
    )
    _, current_num = component_labeler(current, connectivity=8)

    diagnostics: dict[str, object] = {
        "previous_num": previous_num,
        "new_raw_skeleton": raw_skeleton,
        "new_junctions": junctions,
        "new_isolated_pixels": isolated,
        "new_labels_before_large_cur_flag": raw_labels,
        "large_cur_flags": np.asarray(large_flags, dtype=np.int64),
        "junctions_after_merge": junctions_after_merge,
        "isolated_after_merge": isolated_after_merge,
        "current_num": current_num,
    }
    return current, diagnostics


def iterative_fragment_extraction(
    origin_image: np.ndarray,
    all_fragments: np.ndarray,
    iteration: int,
    radius: Real,
    roi_mask: np.ndarray,
    n_angles: Real,
    iteration_remove_radius: Real,
    threshold: Real,
    junction_radius: Real,
    minimum_length: Real,
    *,
    lft_oft_impl: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    neighbor_count_impl: Callable[..., np.ndarray] | None = None,
    label_components_impl: Callable[..., tuple[np.ndarray, int]] | None = None,
    mask2_impl: Callable[[np.ndarray, int], np.ndarray] | None = None,
    thin_impl: Callable[[np.ndarray], np.ndarray] | None = None,
    large_cur_flags_impl: Callable[
        [np.ndarray, Sequence[np.ndarray]], np.ndarray
    ]
    | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Reproduce ``IterGenFragment`` using the verified Python ``lft_oft``."""

    del threshold  # MATLAB passes it, but the active code calls imbinarize(OFT_Img).
    iteration_count = _integer_parameter(iteration, name="iteration")
    radius_int = _integer_parameter(radius, name="radius")
    iteration_radius = _integer_parameter(
        iteration_remove_radius, name="iteration remove radius"
    )
    junction_radius_int = _integer_parameter(junction_radius, name="junction radius")
    minimum_length_int = _integer_parameter(
        minimum_length, name="minimum fragment length"
    )
    if iteration_count < 0 or iteration_radius < 0:
        raise ValueError("iteration and iteration remove radius must be non-negative")

    current = np.asarray(all_fragments).astype(bool, copy=True)
    diagnostics: list[dict[str, object]] = []
    for _ in range(iteration_count):
        current, iteration_diagnostics = _extract_new_fragments(
            origin_image,
            current,
            iteration_radius=iteration_radius,
            radius=radius_int,
            roi_mask=roi_mask,
            n_angles=n_angles,
            junction_radius=junction_radius_int,
            minimum_length=minimum_length_int,
            lft_oft_impl=lft_oft_impl,
            neighbor_count_impl=neighbor_count_impl,
            label_components_impl=label_components_impl,
            mask2_impl=mask2_impl,
            thin_impl=thin_impl,
            large_cur_flags_impl=large_cur_flags_impl,
        )
        diagnostics.append(iteration_diagnostics)
    return current, diagnostics


def extract_stage_04_fragments(
    origin_image: np.ndarray,
    raw_skeleton: np.ndarray,
    radius: Real,
    roi_mask: np.ndarray,
    n_angles: Real,
    threshold: Real,
    *,
    iteration: int = 1,
    junction_size: Real = 7,
    iteration_remove_radius: Real = 3,
    minimum_length: Real = 6,
    lft_oft_impl: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    neighbor_count_impl: Callable[..., np.ndarray] | None = None,
    label_components_impl: Callable[..., tuple[np.ndarray, int]] | None = None,
    mask2_impl: Callable[[np.ndarray, int], np.ndarray] | None = None,
    thin_impl: Callable[[np.ndarray], np.ndarray] | None = None,
    large_cur_flags_impl: Callable[
        [np.ndarray, Sequence[np.ndarray]], np.ndarray
    ]
    | None = None,
) -> dict[str, object]:
    """Run the active ``func_analysis_mt.m`` path through stage 04."""

    radius_int = _integer_parameter(radius, name="radius")
    junction_size_int = _integer_parameter(junction_size, name="junction size")
    if junction_size_int < 1:
        raise ValueError("junction size must be positive")
    junction_radius = (junction_size_int - 1) // 2
    minimum_length_int = _integer_parameter(
        minimum_length, name="minimum fragment length"
    )

    junction_result = remove_junctions_and_isolated(
        raw_skeleton,
        margin_radius=radius_int,
        junction_radius=junction_radius,
        neighbor_count_impl=neighbor_count_impl,
        label_components_impl=label_components_impl,
    )
    all_fragments_before_iteration = remove_short_fragments(
        junction_result.skeleton,
        minimum_length_int,
        connectivity=8,
        label_components_impl=label_components_impl,
    )
    component_labeler = (
        label_components
        if label_components_impl is None
        else label_components_impl
    )
    labels_before_iteration, num_before_iteration = component_labeler(
        all_fragments_before_iteration, connectivity=8
    )

    final_fragments, iteration_diagnostics = iterative_fragment_extraction(
        origin_image,
        all_fragments_before_iteration,
        iteration,
        radius_int,
        roi_mask,
        n_angles,
        iteration_remove_radius,
        threshold,
        junction_radius,
        minimum_length_int,
        lft_oft_impl=lft_oft_impl,
        neighbor_count_impl=neighbor_count_impl,
        label_components_impl=label_components_impl,
        mask2_impl=mask2_impl,
        thin_impl=thin_impl,
        large_cur_flags_impl=large_cur_flags_impl,
    )
    labels, num = component_labeler(final_fragments, connectivity=8)

    return {
        "R_Junc": np.asarray([[float(junction_radius)]], dtype=np.float64),
        "Size_Junc": np.asarray([[float(junction_size_int)]], dtype=np.float64),
        "MIN_FragmentLength": np.asarray([[float(minimum_length_int)]], dtype=np.float64),
        "margin_removed": junction_result.margin_removed,
        "junctions": junction_result.junctions.astype(np.float64),
        "junction_removed": junction_result.junction_removed,
        "isolated_pixels": junction_result.isolated_pixels.astype(np.float64),
        "AllFragments_after_junction": junction_result.skeleton,
        "L_after_junction": junction_result.labels,
        "num_after_junction": np.asarray(
            [[float(junction_result.num_components)]], dtype=np.float64
        ),
        "AllFragments_before_iteration": all_fragments_before_iteration,
        "L_before_iteration": labels_before_iteration,
        "num_before_iteration": np.asarray(
            [[float(num_before_iteration)]], dtype=np.float64
        ),
        "AllFragments": final_fragments,
        "L": labels,
        "num": np.asarray([[float(num)]], dtype=np.float64),
        "Allpts": matlab_find(final_fragments).astype(np.float64),
        "CrPts": junction_result.junctions.astype(np.float64),
        "RawCrPts": junction_result.junctions.astype(np.float64),
        "iteration_diagnostics": iteration_diagnostics,
    }


__all__ = [
    "JunctionRemovalResult",
    "assemble_fragments",
    "bwmorph_thin",
    "detect_isolated_pixels",
    "detect_junction_pixels",
    "extract_stage_04_fragments",
    "iterative_fragment_extraction",
    "label_components",
    "large_cur_flag",
    "matlab_find",
    "remove_border_margin",
    "remove_isolated_pixels",
    "remove_junction_neighborhood",
    "remove_junctions_and_isolated",
    "remove_short_fragments",
]
