"""Readable reference implementation of ``LFT_OFT_mex.c``.

The implementation intentionally follows the C-MEX loop order and MATLAB
column-major indexing.  It is a reference implementation, not an optimized
production kernel.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np


# The C source uses ``# define PI 3.141593`` rather than math.pi.
C_PI = 3.141593


def _as_column_major_flat(array: np.ndarray, *, name: str) -> tuple[np.ndarray, int, int]:
    """Return a float64 MATLAB-compatible flat view and its H/W dimensions."""

    value = np.asarray(array, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got shape {value.shape}")
    height, width = value.shape
    # C indexes MATLAB arrays as column-major ``i * H + j``.
    return np.asarray(value, dtype=np.float64, order="F").ravel(order="F"), height, width


def _c_rounded_offset(value: float) -> int:
    """Reproduce the C expression ``(int)floor(value + 0.5)``."""

    return int(math.floor(value + 0.5))


def _clamp(value: int, upper: int) -> int:
    """Reproduce the four explicit C boundary clamps for one coordinate."""

    if value < 0:
        return 0
    if value > upper - 1:
        return upper - 1
    return value


def _sample_index(i: int, j: int, q: float, angle: float, height: int, width: int) -> int:
    """Return the C-MEX column-major index for one line-transform sample."""

    x = i + _c_rounded_offset(q * math.cos(angle))
    y = j - _c_rounded_offset(q * math.sin(angle))
    x = _clamp(x, width)
    y = _clamp(y, height)
    return x * height + y


def _angle_values(start: float, stop: float, step: float) -> list[float]:
    """Generate angles with the C ``for(k; k < stop; k += step)`` rule."""

    values: list[float] = []
    angle = start
    while angle < stop:
        values.append(angle)
        angle += step
    return values


def lft_oft(
    image: np.ndarray,
    radius: Real,
    n_angles: Real,
    roi_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute SIFNE LFT and OFT exactly following ``LFT_OFT_mex.c``.

    Parameters
    ----------
    image:
        2-D image passed to the MEX function as MATLAB ``double``.
    radius:
        C-MEX ``R`` parameter.  The current SIFNE reference uses ``10``.
    n_angles:
        C-MEX ``Nangles`` parameter.  The current SIFNE reference uses ``20``.
    roi_mask:
        2-D mask.  A pixel is processed only when its value is exactly
        ``1.0``, matching ``ROIMask[...] == 1.0`` in the C source.

    Returns
    -------
    oft_image, lft_image, lft_orientations:
        Float64 matrices with MATLAB ``H x W`` shape and MATLAB-compatible
        coordinates.  The return order matches
        ``[OFT_Img, LFT_Img, LFT_Orientations]``.
    """

    if not isinstance(radius, Real) or not math.isfinite(float(radius)):
        raise ValueError("radius must be a finite real number")
    if not isinstance(n_angles, Real) or not math.isfinite(float(n_angles)):
        raise ValueError("n_angles must be a finite real number")

    radius_value = float(radius)
    n_angles_value = float(n_angles)
    if radius_value < 0:
        raise ValueError("radius must be non-negative")
    if n_angles_value <= 0:
        raise ValueError("n_angles must be positive")

    image_flat, height, width = _as_column_major_flat(image, name="image")
    roi_flat, roi_height, roi_width = _as_column_major_flat(roi_mask, name="roi_mask")
    if (roi_height, roi_width) != (height, width):
        raise ValueError(
            "roi_mask must have the same shape as image: "
            f"{(height, width)} != {(roi_height, roi_width)}"
        )

    # mxCreateNumericArray initializes all three MEX outputs to zero.
    oft_flat = np.zeros(image_flat.size, dtype=np.float64)
    lft_flat = np.zeros(image_flat.size, dtype=np.float64)
    orientation_flat = np.zeros(image_flat.size, dtype=np.float64)

    radius_index = int(radius_value)
    angle_interval = C_PI / n_angles_value
    angles = _angle_values(-C_PI / 2.0, C_PI / 2.0, angle_interval)

    # LFT.  Keep i as the column coordinate and j as the row coordinate,
    # exactly as in the C source.
    for i in range(radius_index, width - radius_index):
        for j in range(radius_index, height - radius_index):
            current_index = i * height + j
            if roi_flat[current_index] == 1.0:
                max_angle_intensity = 0.0
                max_intensity_angle = 0.0
                for angle in angles:
                    line_sum = 0.0
                    q = -radius_value
                    while q < radius_value + 1.0:
                        sample_index = _sample_index(i, j, q, angle, height, width)
                        line_sum = line_sum + image_flat[sample_index]
                        q += 1.0
                    # Strict comparison preserves the first C angle on ties.
                    if max_angle_intensity < line_sum:
                        max_angle_intensity = line_sum
                        max_intensity_angle = angle
                lft_flat[current_index] = max_angle_intensity / (radius_value + radius_value + 1.0)
                orientation_flat[current_index] = max_intensity_angle

    # OFT.  The C code reads the center-pixel LFT value once per q iteration;
    # retain that operation order instead of simplifying the expression.
    for i in range(radius_index, width - radius_index):
        for j in range(radius_index, height - radius_index):
            current_index = i * height + j
            if roi_flat[current_index] == 1.0:
                max_angle_intensity = 0.0
                for angle in angles:
                    line_sum = 0.0
                    q = -radius_value
                    # The C source uses ``q < R + 1`` here (21 samples for
                    # the reference R=10), just as the LFT loop does.
                    while q < radius_value + 1.0:
                        rho = lft_flat[current_index]
                        sample_index = _sample_index(i, j, q, angle, height, width)
                        theta = orientation_flat[sample_index]
                        line_sum = line_sum + rho * math.cos(2.0 * (theta - angle))
                        q += 1.0
                    if max_angle_intensity < line_sum:
                        max_angle_intensity = line_sum
                oft_flat[current_index] = max_angle_intensity

    # Return MATLAB-shaped matrices.  The flat arrays retain the C/MATLAB
    # storage convention, while the logical H x W values remain unchanged.
    return (
        oft_flat.reshape((height, width), order="F"),
        lft_flat.reshape((height, width), order="F"),
        orientation_flat.reshape((height, width), order="F"),
    )


__all__ = ["C_PI", "lft_oft"]
