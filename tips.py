"""MATLAB-equivalent endpoint and tip registration primitives.

This module follows the active ``func_analysis_mt.m`` -> ``TipReg.m`` path.
The public coordinate convention is zero-based ``(row, column)``.  The
MATLAB compatibility steps are kept explicit inside the functions: MATLAB
``find`` order is column-major, ``TipReg`` uses one-based local coordinates,
and ``regionprops(..., 'Centroid')`` returns ``[column, row]`` before the
source swaps it to ``[row, column]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Callable, Sequence

import numpy as np

from .fragments import label_components, matlab_find


FOUR_NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (0, -1),
    (0, 1),
    (1, 0),
)


# The frozen MATLAB R2023b references were generated on Windows.  For a
# handful of double-precision arguments, Windows MATLAB's ``atan`` rounds one
# ULP differently from both glibc ``atan`` and CUDA ``atan``.  PyTorch's CPU
# double implementation matches the frozen MATLAB values.  Keep this as a
# numeric compatibility backend only: the TipReg formula, branches, and
# operation order below remain the MATLAB source formula.
_TORCH_ATAN_BACKEND = None
_TORCH_ATAN_INITIALIZED = False
_TORCH_ATAN_CACHE: dict[float, float] = {}


def _matlab_compatible_atan(value: float) -> float:
    """Evaluate scalar double ``atan`` with the frozen MATLAB rounding.

    PySIFNE remains usable without PyTorch; in that case the platform
    ``math.atan`` implementation is the documented fallback.  The backend is
    loaded lazily so importing the reference implementation does not make
    PyTorch a hard dependency.
    """

    global _TORCH_ATAN_BACKEND, _TORCH_ATAN_INITIALIZED
    cache_key = float(value)
    cached = _TORCH_ATAN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not _TORCH_ATAN_INITIALIZED:
        _TORCH_ATAN_INITIALIZED = True
        try:
            import torch
        except ImportError:
            _TORCH_ATAN_BACKEND = False
        else:
            _TORCH_ATAN_BACKEND = torch

    if _TORCH_ATAN_BACKEND is False:
        result = math.atan(value)
        _TORCH_ATAN_CACHE[cache_key] = result
        return result

    torch = _TORCH_ATAN_BACKEND
    argument = torch.tensor([value], dtype=torch.float64, device="cpu")
    result = float(torch.atan(argument)[0].item())
    _TORCH_ATAN_CACHE[cache_key] = result
    return result


def _as_image(value: np.ndarray, *, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got shape {image.shape}")
    return image


def _as_point(point: Sequence[Real], *, name: str) -> np.ndarray:
    value = np.asarray(point)
    if value.size != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    value = value.reshape(2)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain finite coordinates")
    return value.astype(np.float64, copy=False)


def _as_integer_point_zero_based(
    point: Sequence[Real], *, name: str
) -> tuple[int, int]:
    value = _as_point(point, name=name)
    if not np.all(value == np.floor(value)):
        raise ValueError(f"{name} must contain integer coordinates")
    return int(value[0]), int(value[1])


def check_dir_between2pts(
    p1: Sequence[Real], p2: Sequence[Real]
) -> float:
    """Reproduce ``check_dir_between2pts.m`` exactly.

    ``p1`` and ``p2`` are image-matrix coordinates in ``(row, column)``
    order.  The one-based/zero-based origin does not affect the differences,
    so this function accepts either origin as long as both points use the same
    one.  The source intentionally uses ``atan`` plus a branch, rather than
    ``atan2``; that branch structure is retained verbatim.
    """

    first = _as_point(p1, name="p1")
    second = _as_point(p2, name="p2")
    row1, column1 = float(first[0]), float(first[1])
    row2, column2 = float(second[0]), float(second[1])

    if column2 > column1 and row2 == row1:
        return 0.0
    if column2 < column1 and row2 == row1:
        return -180.0
    if column2 == column1 and row2 > row1:
        return -90.0
    if column2 == column1 and row2 < row1:
        return 90.0

    numerator = row1 - row2
    denominator = column2 - column1
    if column2 < column1:
        # MATLAB's ``sign`` is zero only for a zero numerator.  That case is
        # already handled by the horizontal branches above.
        return (
            180.0 * _matlab_compatible_atan(numerator / denominator) / math.pi
            + float(np.sign(numerator)) * 180.0
        )

    # For identical points MATLAB evaluates 0/0 here and returns NaN.  Make
    # that behavior explicit instead of raising Python's ZeroDivisionError.
    if denominator == 0.0:
        return float("nan")
    return 180.0 * _matlab_compatible_atan(numerator / denominator) / math.pi


def _background_components_4(patch: np.ndarray) -> int:
    """Count 4-connected foreground components in a small boolean patch."""

    foreground = np.asarray(patch, dtype=bool)
    height, width = foreground.shape
    visited = np.zeros_like(foreground, dtype=bool)
    components = 0

    # The endpoint LUT is generated with bwlabel(..., 4).  Traversal order is
    # irrelevant for the count, but the explicit loop mirrors MATLAB find's
    # row-within-column scan closely and avoids library-specific semantics.
    for column in range(width):
        for row in range(height):
            if not foreground[row, column] or visited[row, column]:
                continue
            components += 1
            visited[row, column] = True
            pending = [(row, column)]
            while pending:
                current_row, current_column = pending.pop()
                for row_offset, column_offset in FOUR_NEIGHBOR_OFFSETS:
                    neighbor_row = current_row + row_offset
                    neighbor_column = current_column + column_offset
                    if (
                        0 <= neighbor_row < height
                        and 0 <= neighbor_column < width
                        and foreground[neighbor_row, neighbor_column]
                        and not visited[neighbor_row, neighbor_column]
                    ):
                        visited[neighbor_row, neighbor_column] = True
                        pending.append((neighbor_row, neighbor_column))
    return components


def _is_matlab_endpoint_neighborhood(neighborhood: np.ndarray) -> bool:
    """Evaluate one ``images.internal.lutendpoints`` neighborhood.

    MATLAB's LUT is generated by ``endpoint_fcn``:

    ``if nhood(2,2), [~, count] = bwlabel(~nhood, 4); count == 1``.

    This direct formulation also handles the zero-valued neighborhood outside
    an image boundary, which is how ``bwlookup`` supplies border pixels.
    """

    patch = np.asarray(neighborhood, dtype=bool)
    if patch.shape != (3, 3):
        raise ValueError(f"neighborhood must have shape (3, 3), got {patch.shape}")
    if not patch[1, 1]:
        return False
    return _background_components_4(~patch) == 1


def detect_endpoints(binary: np.ndarray) -> np.ndarray:
    """Return MATLAB ``bwmorph(binary, 'endpoints')`` pixels.

    The returned coordinates are zero-based ``(row, column)`` pairs in the
    same order as MATLAB ``find``.  The endpoint predicate is the exact LUT
    generator used by MATLAB R2023b, not an 8-neighbor degree heuristic.
    """

    image = _as_image(binary, name="binary") != 0
    height, width = image.shape
    padded = np.pad(image, 1, mode="constant", constant_values=False)
    endpoint_mask = np.zeros_like(image, dtype=bool)

    # ``matlab_find`` supplies the required column-major endpoint order.
    for row_one, column_one in matlab_find(image):
        row = int(row_one) - 1
        column = int(column_one) - 1
        neighborhood = padded[row : row + 3, column : column + 3]
        if _is_matlab_endpoint_neighborhood(neighborhood):
            endpoint_mask[row, column] = True

    endpoint_coordinates_one_based = matlab_find(endpoint_mask)
    return endpoint_coordinates_one_based - 1


@dataclass(frozen=True)
class TipLocalWindow:
    """Intermediate values from the local-coordinate part of ``TipReg``."""

    local_image: np.ndarray
    fragment_coordinates_one_based: np.ndarray
    selected_coordinates_one_based: np.ndarray
    tip_one_based: np.ndarray
    fragment_min_one_based: np.ndarray


def _select_endpoint_local_coordinates(
    fragment_coordinates_one_based: np.ndarray,
    tip_zero_based: Sequence[Real],
    reg_r: Real,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select the same local pixels as ``TipReg`` without materialising background.

    The active caller supplies a fragment's coordinates in MATLAB ``find``
    order.  Cropping is only an implementation detail for the subsequent
    connected-component calculation; the returned coordinate order and the
    distance expression are unchanged.
    """

    fragment_coordinates = np.asarray(
        fragment_coordinates_one_based, dtype=np.int64
    )
    if fragment_coordinates.ndim != 2 or fragment_coordinates.shape[1] != 2:
        raise ValueError(
            "fragment_coordinates_one_based must have shape (N, 2), got "
            f"{fragment_coordinates.shape}"
        )
    if fragment_coordinates.shape[0] == 0:
        raise ValueError("fragment_coordinates_one_based must not be empty")
    tip_row, tip_column = _as_integer_point_zero_based(
        tip_zero_based, name="tip_zero_based"
    )
    radius = float(reg_r)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError(f"reg_r must be a finite non-negative number, got {reg_r!r}")

    fragment_min_one_based = fragment_coordinates.min(axis=0)
    local_coordinates_one_based = (
        fragment_coordinates - fragment_min_one_based + 1
    )
    tip_one_based = np.asarray((tip_row + 1, tip_column + 1), dtype=np.float64)
    tip_local_one_based = tip_one_based - fragment_min_one_based + 1
    distances = np.sqrt(
        (local_coordinates_one_based[:, 0] - tip_local_one_based[0]) ** 2
        + (local_coordinates_one_based[:, 1] - tip_local_one_based[1]) ** 2
    )
    selected_coordinates_one_based = local_coordinates_one_based[
        distances <= radius
    ]
    return (
        fragment_min_one_based,
        selected_coordinates_one_based,
        tip_local_one_based,
        local_coordinates_one_based,
    )


def extract_endpoint_local_window(
    label_image: np.ndarray,
    label: Real,
    tip_zero_based: Sequence[Real],
    reg_r: Real,
    *,
    _fragment_coordinates_one_based: np.ndarray | None = None,
) -> TipLocalWindow:
    """Reproduce ``TipReg``'s local fragment image and circular selection."""

    image = _as_image(label_image, name="label_image")
    tip_row, tip_column = _as_integer_point_zero_based(
        tip_zero_based, name="tip_zero_based"
    )
    radius = float(reg_r)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError(f"reg_r must be a finite non-negative number, got {reg_r!r}")
    if not (0 <= tip_row < image.shape[0] and 0 <= tip_column < image.shape[1]):
        raise IndexError(f"tip_zero_based is outside label_image: {tip_zero_based!r}")

    # MATLAB [x,y] = find(LL == label): x=row, y=column, one-based and
    # column-major.  Keep this exact order through local image construction.
    if _fragment_coordinates_one_based is None:
        fragment_coordinates_one_based = matlab_find(image == label)
    else:
        fragment_coordinates_one_based = np.asarray(
            _fragment_coordinates_one_based, dtype=np.int64
        )
        if (
            fragment_coordinates_one_based.ndim != 2
            or fragment_coordinates_one_based.shape[1] != 2
        ):
            raise ValueError(
                "_fragment_coordinates_one_based must have shape (N, 2), got "
                f"{fragment_coordinates_one_based.shape}"
            )
    if fragment_coordinates_one_based.shape[0] == 0:
        raise ValueError(f"label {label!r} does not occur in label_image")

    (
        fragment_min_one_based,
        selected_coordinates_one_based,
        tip_local_one_based,
        local_coordinates_one_based,
    ) = _select_endpoint_local_coordinates(
        fragment_coordinates_one_based,
        tip_zero_based,
        reg_r,
    )

    local_height = int(local_coordinates_one_based[:, 0].max())
    local_width = int(local_coordinates_one_based[:, 1].max())
    local_image = np.zeros((local_height, local_width), dtype=bool)
    for row_one, column_one in selected_coordinates_one_based:
        local_image[int(row_one) - 1, int(column_one) - 1] = True

    return TipLocalWindow(
        local_image=local_image,
        fragment_coordinates_one_based=fragment_coordinates_one_based,
        selected_coordinates_one_based=selected_coordinates_one_based,
        tip_one_based=tip_local_one_based,
        fragment_min_one_based=fragment_min_one_based,
    )


def local_centroid(local_image: np.ndarray) -> np.ndarray:
    """Reproduce the first ``regionprops(temp, 'Centroid')`` result.

    MATLAB's default 2-D ``regionprops`` connectivity is 8.  ``TipReg`` uses
    ``center(1)`` when more than one local component exists, so this function
    deliberately keeps only the first component in MATLAB column-major order
    instead of averaging every foreground pixel in ``local_image``.
    """

    image = _as_image(local_image, name="local_image") != 0
    component_labels, component_count = label_components(image, connectivity=8)
    if component_count == 0:
        raise ValueError("local_image must contain at least one foreground pixel")
    coordinates_one_based = matlab_find(component_labels == 1)
    if coordinates_one_based.shape[0] == 0:
        raise ValueError("local_image must contain at least one foreground pixel")

    # regionprops returns [column, row].  TipReg immediately changes it to
    # [row, column].  Accumulate in find order to keep the floating operation
    # order deterministic and close to MATLAB's column-major pixel list.
    row_sum = 0.0
    column_sum = 0.0
    for row_one, column_one in coordinates_one_based:
        row_sum += float(row_one)
        column_sum += float(column_one)
    count = float(coordinates_one_based.shape[0])
    return np.asarray((row_sum / count, column_sum / count), dtype=np.float64)


def _local_centroid_from_selected_coordinates(
    selected_coordinates_one_based: np.ndarray,
) -> np.ndarray:
    """Compute TipReg's first local component on a tight equivalent crop.

    A translated crop has the same MATLAB column-major foreground order and
    the same 8-neighbor relation as the original zero-padded local image.
    The returned centroid is translated back to the original local one-based
    coordinates before the scalar sums are performed.
    """

    selected = np.asarray(selected_coordinates_one_based, dtype=np.int64)
    if selected.ndim != 2 or selected.shape[1] != 2:
        raise ValueError(
            "selected_coordinates_one_based must have shape (N, 2), got "
            f"{selected.shape}"
        )
    if selected.shape[0] == 0:
        raise ValueError("local_image must contain at least one foreground pixel")

    row_min = int(np.min(selected[:, 0]))
    row_max = int(np.max(selected[:, 0]))
    column_min = int(np.min(selected[:, 1]))
    column_max = int(np.max(selected[:, 1]))
    cropped = np.zeros(
        (row_max - row_min + 1, column_max - column_min + 1), dtype=bool
    )
    cropped[
        selected[:, 0] - row_min,
        selected[:, 1] - column_min,
    ] = True
    component_labels, component_count = label_components(cropped, connectivity=8)
    if component_count == 0:
        raise ValueError("local_image must contain at least one foreground pixel")
    component_coordinates_one_based = matlab_find(component_labels == 1)
    if component_coordinates_one_based.shape[0] == 0:
        raise ValueError("local_image must contain at least one foreground pixel")

    row_sum = 0.0
    column_sum = 0.0
    for row_one, column_one in component_coordinates_one_based:
        row_sum += float(int(row_one) + row_min - 1)
        column_sum += float(int(column_one) + column_min - 1)
    count = float(component_coordinates_one_based.shape[0])
    return np.asarray((row_sum / count, column_sum / count), dtype=np.float64)


def register_single_tip(
    label_image: np.ndarray,
    label: Real,
    tip_zero_based: Sequence[Real],
    reg_r: Real,
    *,
    _fragment_coordinates_one_based: np.ndarray | None = None,
    _sparse_local: bool = False,
    _skip_orientation: bool = False,
) -> np.ndarray:
    """Return ``[orientation, centroid_row0, centroid_col0]`` for one tip."""

    if _sparse_local:
        if _fragment_coordinates_one_based is None:
            raise ValueError("sparse local registration requires fragment coordinates")
        (
            fragment_min_one_based,
            selected_coordinates_one_based,
            tip_local_one_based,
            _local_coordinates_one_based,
        ) = _select_endpoint_local_coordinates(
            _fragment_coordinates_one_based,
            tip_zero_based,
            reg_r,
        )
        center_local_one_based = _local_centroid_from_selected_coordinates(
            selected_coordinates_one_based
        )
    else:
        local = extract_endpoint_local_window(
            label_image,
            label,
            tip_zero_based,
            reg_r,
            _fragment_coordinates_one_based=_fragment_coordinates_one_based,
        )
        fragment_min_one_based = local.fragment_min_one_based
        tip_local_one_based = local.tip_one_based
        center_local_one_based = local_centroid(local.local_image)
    orientation = (
        0.0
        if _skip_orientation
        else check_dir_between2pts(center_local_one_based, tip_local_one_based)
    )

    # TipReg uses ``center + [min(x) min(y)]``.  Do not simplify this to a
    # visually tempting ``min - 1`` expression: center is one-based local
    # coordinates, so this is the exact MATLAB global one-based result.
    center_global_one_based = center_local_one_based + fragment_min_one_based
    center_global_zero_based = center_global_one_based - 1.0
    return np.asarray(
        (orientation, center_global_zero_based[0], center_global_zero_based[1]),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class TipRegistrationResult:
    """Stage-05 outputs in the Python zero-based coordinate convention."""

    endpoint_mask: np.ndarray
    endpoint_coordinates: np.ndarray
    tip_labels: np.ndarray
    temp_info: np.ndarray
    all_tips: np.ndarray

    @property
    def all_tips_matlab(self) -> np.ndarray:
        """Return ``func_analysis_mt.m``'s one-based ``all_tips`` table."""

        result = self.all_tips.copy()
        if result.shape[0] != 0:
            result[:, 0:2] += 1.0
            result[:, 4:6] += 1.0
        return result

    @property
    def temp_info_matlab(self) -> np.ndarray:
        """Return MATLAB ``tempInfo`` (orientation, row1, column1)."""

        result = self.temp_info.copy()
        if result.shape[0] != 0:
            result[:, 1:3] += 1.0
        return result


def register_filament_tips(
    all_fragments: np.ndarray,
    label_image: np.ndarray,
    reg_r: Real,
    *,
    orientation_impl: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    endpoint_mask_impl: Callable[[np.ndarray], np.ndarray] | None = None,
    centroid_impl: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> TipRegistrationResult:
    """Reproduce the active ``func_analysis_mt.m`` tip-registration block."""

    fragments = _as_image(all_fragments, name="all_fragments") != 0
    labels = _as_image(label_image, name="label_image")
    if fragments.shape != labels.shape:
        raise ValueError(
            "all_fragments and label_image must have the same shape, got "
            f"{fragments.shape} and {labels.shape}"
        )

    if endpoint_mask_impl is None:
        endpoint_coordinates = detect_endpoints(fragments)
        endpoint_mask = np.zeros_like(fragments, dtype=bool)
        for row, column in endpoint_coordinates:
            endpoint_mask[int(row), int(column)] = True
    else:
        endpoint_mask = np.asarray(endpoint_mask_impl(fragments), dtype=bool)
        if endpoint_mask.shape != fragments.shape:
            raise ValueError(
                "endpoint_mask_impl must return an array with the fragment image shape"
            )
        # MATLAB find(endpoint_mask) defines the externally visible order.
        endpoint_coordinates = matlab_find(endpoint_mask) - 1

    tip_count = endpoint_coordinates.shape[0]
    tip_labels = np.empty(tip_count, dtype=np.float64)
    temp_info = np.empty((tip_count, 3), dtype=np.float64)
    fragment_mins = np.empty((tip_count, 2), dtype=np.float64)
    label_coordinates: dict[float, list[tuple[int, int]]] = {}
    for index, (row, column) in enumerate(endpoint_coordinates):
        label = float(labels[int(row), int(column)])
        tip_labels[index] = label

    # Every per-tip TipReg call previously rescanned the complete label image
    # with ``find(LL == label)``.  One column-major scan followed by stable
    # grouping is exactly the same coordinate stream for every label.
    requested_labels = {float(value) for value in tip_labels}
    for row_one, column_one in matlab_find(labels != 0):
        label = float(labels[int(row_one) - 1, int(column_one) - 1])
        if label in requested_labels:
            label_coordinates.setdefault(label, []).append(
                (int(row_one), int(column_one))
            )
    cached_coordinates = {
        label: np.asarray(rows, dtype=np.int64).reshape(-1, 2)
        for label, rows in label_coordinates.items()
    }
    centers_for_orientation: np.ndarray | None = None
    tip_locals_for_orientation: np.ndarray | None = None
    if centroid_impl is None:
        for index, (row, column) in enumerate(endpoint_coordinates):
            label = float(tip_labels[index])
            fragment_coordinates = cached_coordinates.get(label)
            if fragment_coordinates is None:
                fragment_mins[index, :] = 0.0
            else:
                fragment_mins[index, :] = np.min(fragment_coordinates, axis=0)
            temp_info[index, :] = register_single_tip(
                labels,
                label,
                (int(row), int(column)),
                reg_r,
                _fragment_coordinates_one_based=fragment_coordinates,
                _sparse_local=fragment_coordinates is not None,
                _skip_orientation=orientation_impl is not None,
            )
    else:
        # Keep the exact NumPy selection expression and MATLAB coordinate
        # convention, but batch only the subsequent first-component centroid
        # reduction.  The callback receives one concatenated stream plus
        # offsets, so no endpoint or fragment can change another's order.
        selected_parts: list[np.ndarray] = []
        tip_local_parts: list[np.ndarray] = []
        offsets = [0]
        all_fragment_coordinates_available = True
        for index, (row, column) in enumerate(endpoint_coordinates):
            label = float(tip_labels[index])
            fragment_coordinates = cached_coordinates.get(label)
            if fragment_coordinates is None:
                all_fragment_coordinates_available = False
                break
            fragment_mins[index, :] = np.min(fragment_coordinates, axis=0)
            _, selected_coordinates, tip_local, _ = _select_endpoint_local_coordinates(
                fragment_coordinates,
                (int(row), int(column)),
                reg_r,
            )
            selected_parts.append(selected_coordinates)
            tip_local_parts.append(tip_local.copy())
            offsets.append(offsets[-1] + selected_coordinates.shape[0])

        if not all_fragment_coordinates_available:
            # This is the same fallback used by the readable implementation
            # for an endpoint whose label is absent from the cached table.
            for index, (row, column) in enumerate(endpoint_coordinates):
                label = float(tip_labels[index])
                fragment_coordinates = cached_coordinates.get(label)
                if fragment_coordinates is None:
                    fragment_mins[index, :] = 0.0
                else:
                    fragment_mins[index, :] = np.min(fragment_coordinates, axis=0)
                temp_info[index, :] = register_single_tip(
                    labels,
                    label,
                    (int(row), int(column)),
                    reg_r,
                    _fragment_coordinates_one_based=fragment_coordinates,
                    _sparse_local=fragment_coordinates is not None,
                    _skip_orientation=orientation_impl is not None,
                )
        else:
            selected_coordinates_all = (
                np.vstack(selected_parts).astype(np.int64, copy=False)
                if selected_parts
                else np.empty((0, 2), dtype=np.int64)
            )
            centers = np.asarray(
                centroid_impl(
                    selected_coordinates_all,
                    np.asarray(offsets, dtype=np.int64),
                ),
                dtype=np.float64,
            )
            if centers.shape != (tip_count, 2):
                raise ValueError(
                    "centroid_impl must return one (row, column) center per endpoint"
                )
            centers_for_orientation = centers
            tip_locals_for_orientation = np.asarray(
                tip_local_parts,
                dtype=np.float64,
            )
            temp_info[:, 0] = 0.0
            temp_info[:, 1:3] = centers + fragment_mins - 1.0
            if orientation_impl is None:
                for index in range(tip_count):
                    _, _, tip_local, _ = _select_endpoint_local_coordinates(
                        cached_coordinates[float(tip_labels[index])],
                        (
                            int(endpoint_coordinates[index, 0]),
                            int(endpoint_coordinates[index, 1]),
                        ),
                        reg_r,
                    )
                    temp_info[index, 0] = check_dir_between2pts(
                        centers[index], tip_local
                    )

    if orientation_impl is not None and tip_count:
        # TipReg computes direction in local one-based coordinates.  Its
        # centroid is later translated with ``center + fragment_min`` while
        # the endpoint uses ``tip_local + fragment_min - 1``; retain that
        # intentional one-pixel offset instead of using global coordinates.
        if centers_for_orientation is not None and tip_locals_for_orientation is not None:
            # Use the exact scalar local values produced by the selection loop.
            # Reconstructing them from global centroids with a vectorized
            # subtraction changes the last bit for some curved fixtures.
            center_local = centers_for_orientation
            tip_local = tip_locals_for_orientation
        else:
            center_local = temp_info[:, 1:3] - fragment_mins + 1.0
            tip_local = endpoint_coordinates.astype(np.float64) - fragment_mins + 2.0
        orientations = np.asarray(
            orientation_impl(center_local, tip_local),
            dtype=np.float64,
        ).reshape(-1)
        if orientations.size != tip_count:
            raise ValueError(
                "orientation_impl must return one orientation per endpoint"
            )
        temp_info[:, 0] = orientations

    all_tips = np.empty((tip_count, 6), dtype=np.float64)
    if tip_count:
        all_tips[:, 0:2] = endpoint_coordinates.astype(np.float64)
        all_tips[:, 2] = tip_labels
        all_tips[:, 3:6] = temp_info
    return TipRegistrationResult(
        endpoint_mask=endpoint_mask,
        endpoint_coordinates=endpoint_coordinates,
        tip_labels=tip_labels,
        temp_info=temp_info,
        all_tips=all_tips,
    )
