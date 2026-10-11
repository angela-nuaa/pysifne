"""MATLAB-equivalent SIFNE segmentation primitives.

This module follows the active segmentation path in ``func_analysis_mt.m``:

``OFT_Img -> mat2gray -> imbinarize(I) -> bwmorph(BW, 'thin', Inf)``.

The ``DefaultFactor`` threshold is reported for parity with the MATLAB
workspace, but it is not passed to ``imbinarize`` in the active MATLAB code.
"""

from __future__ import annotations

from numbers import Real
from typing import Callable

import numpy as np
from skimage.morphology import thin as _skimage_thin


def mat2gray(
    image: np.ndarray,
    limits: tuple[Real, Real] | None = None,
) -> np.ndarray:
    """Reproduce MATLAB ``mat2gray`` for the numeric image path used here.

    With no explicit limits MATLAB uses the image minimum and maximum.  For a
    constant image its source code leaves the image unchanged before clipping,
    so a positive constant maps to one, while zero and negative constants map
    to zero.
    """

    value = np.asarray(image)
    if not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_:
        raise TypeError(f"image must be numeric or logical, got {value.dtype}")
    value = value.astype(np.float64, copy=False)

    if limits is None:
        if value.size == 0:
            return value.copy()
        lower = float(np.min(value))
        upper = float(np.max(value))
    else:
        if len(limits) != 2:
            raise ValueError("limits must contain exactly [amin, amax]")
        lower = float(limits[0])
        upper = float(limits[1])

    if upper == lower:
        normalized = value.copy()
    else:
        # MATLAB mat2gray computes delta first, then calls imlincomb(delta,
        # A, -lower*delta, 'double').  Keeping that operation order matters
        # for the last bit of the stored stage-02 reference.
        delta = 1.0 / (upper - lower)
        normalized = delta * value + (-lower * delta)

    # MATLAB source: I = max(0,min(I,1));
    return np.maximum(0.0, np.minimum(normalized, 1.0))


def _matlab_im2uint8(image: np.ndarray) -> np.ndarray:
    """Reproduce the normalized floating-point branch of ``im2uint8``."""

    value = np.asarray(image)
    if not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_:
        raise TypeError(f"image must be numeric or logical, got {value.dtype}")

    if value.dtype == np.uint8:
        return value.copy()

    # graythresh first calls im2uint8 on the intensity image.  For the active
    # SIFNE path the input is mat2gray output in [0, 1].  floor(x + 0.5)
    # matches MATLAB's positive-value rounding, including half-integers.
    value = value.astype(np.float64, copy=False)
    value = np.clip(value, 0.0, 1.0)
    return np.floor(value * 255.0 + 0.5).astype(np.uint8)


def graythresh(image: np.ndarray) -> float:
    """Reproduce MATLAB ``graythresh``/``otsuthresh`` for a 2-D image.

    MATLAB converts the input to uint8, builds a 256-bin histogram, computes
    Otsu's between-class variance, averages all tied maximizing bins, and
    returns ``(index - 1) / 255``.  The implementation keeps that sequence.
    """

    uint8_image = _matlab_im2uint8(image)
    counts = np.bincount(uint8_image.ravel(), minlength=256).astype(np.float64)
    total = float(np.sum(counts))
    if total == 0.0:
        return 0.0

    probability = counts / total
    omega = np.cumsum(probability)
    mu = np.cumsum(probability * np.arange(1.0, 257.0))
    mu_total = mu[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        between_class_variance = (
            (mu_total * omega - mu) ** 2 / (omega * (1.0 - omega))
        )

    finite = np.isfinite(between_class_variance)
    if not np.any(finite):
        return 0.0

    maximum = float(np.max(between_class_variance[finite]))
    maximizing_bins = np.flatnonzero(between_class_variance == maximum)
    # The zero-based bin index is MATLAB's one-based idx minus one.
    return float(np.mean(maximizing_bins) / 255.0)


def imbinarize(
    image: np.ndarray,
    threshold: Real | np.ndarray | None = None,
) -> np.ndarray:
    """Reproduce the global ``imbinarize`` branch used by SIFNE.

    If no threshold is supplied, MATLAB computes Otsu's threshold internally.
    For floating-point images the active comparison is strictly ``I > T``.
    """

    value = np.asarray(image)
    if threshold is None:
        threshold_value: float | np.ndarray = graythresh(value)
    else:
        threshold_value = np.asarray(threshold, dtype=np.float64)

    if np.issubdtype(value.dtype, np.floating):
        return value > threshold_value
    if np.issubdtype(value.dtype, np.unsignedinteger):
        max_value = float(np.iinfo(value.dtype).max)
        return value > (threshold_value * max_value)
    if np.issubdtype(value.dtype, np.signedinteger):
        limits = np.iinfo(value.dtype)
        return value > (float(limits.min) + (float(limits.max) - limits.min) * threshold_value)
    if value.dtype == np.bool_:
        return value.copy()
    raise TypeError(f"image must be a supported numeric type, got {value.dtype}")


def bwmorph_thin(binary: np.ndarray) -> np.ndarray:
    """Reproduce ``bwmorph(binary, 'thin', Inf)`` for a 2-D binary image."""

    value = np.asarray(binary)
    if value.ndim != 2:
        raise ValueError(f"binary must be a 2-D matrix, got shape {value.shape}")
    # scikit-image's 8-neighbor iterative thinning has been checked against
    # the MATLAB R2023b bwmorph thin result on the frozen MATLAB fixture.
    return _skimage_thin(value.astype(bool, copy=False), max_num_iter=None)


def segment_oft(
    oft_image: np.ndarray,
    *,
    default_factor: Real = 1.42,
    thin_impl: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Run the active MATLAB SIFNE segmentation path.

    Returns ``(I, BW, RawSke, otsu_threshold, threshold_used)``.  The last
    value mirrors MATLAB's ``ThreshOpt1Edit``; it is intentionally not passed
    into ``imbinarize`` because the active MATLAB code calls ``imbinarize(I)``.
    """

    normalized = mat2gray(oft_image)
    otsu_threshold = graythresh(normalized)
    threshold_used = float(default_factor) * otsu_threshold
    if threshold_used >= 1.0:
        threshold_used = otsu_threshold
    binary = imbinarize(normalized)
    compute_thin = bwmorph_thin if thin_impl is None else thin_impl
    raw_skeleton = compute_thin(binary)
    return normalized, binary, raw_skeleton, otsu_threshold, threshold_used


__all__ = [
    "bwmorph_thin",
    "graythresh",
    "imbinarize",
    "mat2gray",
    "segment_oft",
]
