"""Optional CUDA acceleration for the verified LFT/OFT reference kernel.

This module is intentionally separate from :mod:`pysifne.lft_oft`.  The
default Python reference path is untouched.  The CUDA path keeps the original
loop nesting and accumulation order; the host precomputes the exact angle and
rounded-offset tables used by the reference implementation, so the device
does not substitute a different angle sampler or filament detector.

The backend is opt-in and must pass the frozen differential fixtures before it
is used for a real-data benchmark.
"""

from __future__ import annotations

import math
from functools import lru_cache
from numbers import Real

import numpy as np

from .lft_oft import (
    C_PI,
    _angle_values,
    _as_column_major_flat,
    _c_rounded_offset,
)
from ._matlab_cos_tables import matlab_windows_factor_table


def cuda_available() -> bool:
    """Return whether Numba can access a CUDA device."""

    try:
        from numba import cuda

        return bool(cuda.is_available())
    except Exception:
        return False


def _host_tables(
    radius_value: float,
    n_angles_value: float,
    *,
    factor_mode: str = "matlab_windows",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    angles = np.asarray(
        _angle_values(-C_PI / 2.0, C_PI / 2.0, C_PI / n_angles_value),
        dtype=np.float64,
    )
    q_values: list[float] = []
    q = -radius_value
    while q < radius_value + 1.0:
        q_values.append(q)
        q += 1.0

    x_offsets = np.empty((angles.size, len(q_values)), dtype=np.int32)
    y_offsets = np.empty_like(x_offsets)
    for angle_index, angle in enumerate(angles):
        for q_index, q_value in enumerate(q_values):
            x_offsets[angle_index, q_index] = _c_rounded_offset(
                q_value * math.cos(float(angle))
            )
            y_offsets[angle_index, q_index] = -_c_rounded_offset(
                q_value * math.sin(float(angle))
            )

    # The extra row is the zero orientation used by unprocessed pixels and by
    # the C-MEX max initialization when every line sum is non-positive.  The
    # Windows/MATLAB table is frozen because Linux libm and the Windows CRT
    # can differ by one ULP for the same double argument.  ``platform`` is
    # retained as an explicit diagnostic mode, never as a silent fallback.
    if factor_mode == "matlab_windows":
        requested_n_angles = int(n_angles_value)
        if float(requested_n_angles) != n_angles_value:
            raise ValueError(
                "matlab_windows factor tables require integer Nangles"
            )
        factors = matlab_windows_factor_table(requested_n_angles)
        expected_shape = (angles.size + 1, angles.size)
        if factors.shape != expected_shape:
            raise RuntimeError(
                "MATLAB factor table angle count does not match the C angle loop: "
                f"{factors.shape} != {expected_shape}"
            )
    elif factor_mode == "platform":
        factors = np.empty((angles.size + 1, angles.size), dtype=np.float64)
        for theta_index, theta in enumerate(angles):
            for angle_index, angle in enumerate(angles):
                factors[theta_index, angle_index] = math.cos(
                    2.0 * (float(theta) - float(angle))
                )
        zero_theta_index = angles.size
        for angle_index, angle in enumerate(angles):
            factors[zero_theta_index, angle_index] = math.cos(-2.0 * float(angle))
    else:
        raise ValueError(
            f"unknown OFT factor_mode={factor_mode!r}; "
            "expected 'matlab_windows' or 'platform'"
        )
    return angles, x_offsets, y_offsets, factors


@lru_cache(maxsize=1)
def _cuda_kernels():
    from numba import cuda

    @cuda.jit(device=True, inline=False)
    def multiply_no_fma(left, right):
        return left * right

    @cuda.jit(device=True, inline=False)
    def add_rounded(left, right):
        return left + right

    @cuda.jit(fastmath=False)
    def lft_kernel(
        image_flat,
        roi_flat,
        lft_flat,
        orientation_flat,
        orientation_index,
        angles,
        x_offsets,
        y_offsets,
        height,
        width,
        radius_index,
        zero_theta_index,
    ):
        linear = cuda.grid(1)
        size = height * width
        if linear >= size:
            return
        i = linear // height
        j = linear - i * height
        if i < radius_index or i >= width - radius_index:
            return
        if j < radius_index or j >= height - radius_index:
            return
        if roi_flat[linear] != 1.0:
            return

        n_angles = angles.size
        n_q = x_offsets.shape[1]
        max_intensity = 0.0
        max_angle_index = zero_theta_index
        for angle_index in range(n_angles):
            line_sum = 0.0
            for q_index in range(n_q):
                x = i + x_offsets[angle_index, q_index]
                y = j + y_offsets[angle_index, q_index]
                if x < 0:
                    x = 0
                elif x >= width:
                    x = width - 1
                if y < 0:
                    y = 0
                elif y >= height:
                    y = height - 1
                line_sum = line_sum + image_flat[x * height + y]
            if max_intensity < line_sum:
                max_intensity = line_sum
                max_angle_index = angle_index
        lft_flat[linear] = max_intensity / (n_q * 1.0)
        if max_angle_index == zero_theta_index:
            orientation_flat[linear] = 0.0
        else:
            orientation_flat[linear] = angles[max_angle_index]
        orientation_index[linear] = max_angle_index

    @cuda.jit(fastmath=False)
    def oft_angle_kernel(
        lft_flat,
        roi_flat,
        oft_flat,
        orientation_index,
        factors,
        x_offsets,
        y_offsets,
        product_flat,
        height,
        width,
        radius_index,
        angle_index,
    ):
        """Accumulate one OFT angle in the original q order.

        The former CUDA path launched separate product and add kernels for
        every q sample, then launched a third kernel for the angle maximum.
        Every operation was per-pixel and serial in q.  The product temporary
        is retained as an explicit rounding point while the product/add/max
        work is fused into one launch per angle.  The accumulator is local to
        the pixel thread, preserving q order without a global line-sum access
        on every sample.
        """

        linear = cuda.grid(1)
        size = height * width
        if linear >= size:
            return
        i = linear // height
        j = linear - i * height
        if i < radius_index or i >= width - radius_index:
            return
        if j < radius_index or j >= height - radius_index:
            return
        product_flat[linear] = 0.0
        if roi_flat[linear] != 1.0:
            return

        line_sum = 0.0
        n_q = x_offsets.shape[1]
        for q_index in range(n_q):
            x = i + x_offsets[angle_index, q_index]
            y = j + y_offsets[angle_index, q_index]
            if x < 0:
                x = 0
            elif x >= width:
                x = width - 1
            if y < 0:
                y = 0
            elif y >= height:
                y = height - 1
            sample = x * height + y
            theta_index = orientation_index[sample]
            rho = lft_flat[linear]
            product = multiply_no_fma(rho, factors[theta_index, angle_index])
            product_flat[linear] = product
            line_sum = add_rounded(line_sum, product_flat[linear])

        if oft_flat[linear] < line_sum:
            oft_flat[linear] = line_sum

    return lft_kernel, oft_angle_kernel


def cuda_lft_oft(
    image: np.ndarray,
    radius: Real,
    n_angles: Real,
    roi_mask: np.ndarray,
    *,
    device: int = 0,
    threads_per_block: int = 256,
    factor_mode: str = "matlab_windows",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the verified LFT/OFT loop on CUDA without changing its semantics.

    ``factor_mode="matlab_windows"`` uses versioned factors generated by the
    Windows runtime that produced the frozen MATLAB fixtures.  This avoids a
    platform-dependent Linux ``cos`` result in the OFT accumulation.  The
    explicit ``factor_mode="platform"`` is available only for diagnostics.
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
    if int(threads_per_block) <= 0:
        raise ValueError("threads_per_block must be positive")

    try:
        from numba import cuda
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("CUDA backend requires numba") from exc
    if not cuda_available():
        raise RuntimeError("Numba CUDA is not available")

    image_flat, height, width = _as_column_major_flat(image, name="image")
    roi_flat, roi_height, roi_width = _as_column_major_flat(roi_mask, name="roi_mask")
    if (height, width) != (roi_height, roi_width):
        raise ValueError("roi_mask must have the same shape as image")
    angles, x_offsets, y_offsets, factors = _host_tables(
        radius_value, n_angles_value, factor_mode=factor_mode
    )
    radius_index = int(radius_value)
    zero_theta_index = int(angles.size)
    size = image_flat.size
    lft_flat = np.zeros(size, dtype=np.float64)
    oft_flat = np.zeros(size, dtype=np.float64)
    orientation_flat = np.zeros(size, dtype=np.float64)
    orientation_index = np.full(size, zero_theta_index, dtype=np.int32)

    cuda.select_device(int(device))
    d_image = cuda.to_device(np.ascontiguousarray(image_flat))
    d_roi = cuda.to_device(np.ascontiguousarray(roi_flat))
    d_lft = cuda.to_device(lft_flat)
    d_oft = cuda.to_device(oft_flat)
    d_product = cuda.to_device(np.zeros(size, dtype=np.float64))
    d_orientation = cuda.to_device(orientation_flat)
    d_orientation_index = cuda.to_device(orientation_index)
    d_angles = cuda.to_device(np.ascontiguousarray(angles))
    d_x_offsets = cuda.to_device(np.ascontiguousarray(x_offsets))
    d_y_offsets = cuda.to_device(np.ascontiguousarray(y_offsets))
    d_factors = cuda.to_device(np.ascontiguousarray(factors))
    lft_kernel, oft_angle_kernel = _cuda_kernels()
    blocks = (size + int(threads_per_block) - 1) // int(threads_per_block)
    lft_kernel[blocks, int(threads_per_block)](
        d_image,
        d_roi,
        d_lft,
        d_orientation,
        d_orientation_index,
        d_angles,
        d_x_offsets,
        d_y_offsets,
        height,
        width,
        radius_index,
        zero_theta_index,
    )
    for angle_index in range(angles.size):
        oft_angle_kernel[blocks, int(threads_per_block)](
            d_lft,
            d_roi,
            d_oft,
            d_orientation_index,
            d_factors,
            d_x_offsets,
            d_y_offsets,
            d_product,
            height,
            width,
            radius_index,
            angle_index,
        )
    d_lft.copy_to_host(lft_flat)
    d_oft.copy_to_host(oft_flat)
    d_orientation.copy_to_host(orientation_flat)
    return (
        oft_flat.reshape((height, width), order="F"),
        lft_flat.reshape((height, width), order="F"),
        orientation_flat.reshape((height, width), order="F"),
    )


__all__ = ["cuda_available", "cuda_lft_oft"]
