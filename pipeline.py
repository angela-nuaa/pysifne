"""MATLAB-free end-to-end execution of the verified PySIFNE stages."""

from __future__ import annotations

import hashlib
import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy.io import savemat

from .config import PY_SIFNE_PROFILE, PY_SIFNE_VERSION, SIFNEConfig
from .fragments import extract_stage_04_fragments
from .grouping import group_fragments
from .lft_oft import lft_oft
from .pairing import pair_filament_tips
from .result import FragmentStageResult, SIFNEResult
from .segmentation import segment_oft
from .sorting import (
    detect_tips_sparse_label_vectorized,
    numba_nearest_active_index,
    sort_filaments,
)
from .tips import register_filament_tips


def _matlab_round_positive(value: np.ndarray) -> np.ndarray:
    """Round non-negative image intensities as MATLAB's image conversion does."""

    return np.floor(value + 0.5)


def _to_uint8_intensity(value: np.ndarray) -> np.ndarray:
    """Implement the image-class conversion used before active ``imadjust``."""

    array = np.asarray(value)
    if array.dtype == np.uint8:
        return array.copy()
    if array.dtype == np.bool_:
        return np.where(array, 255, 0).astype(np.uint8)
    if array.dtype == np.uint16:
        normalized = array.astype(np.float64) / 65535.0
    elif array.dtype == np.int16:
        normalized = (array.astype(np.float64) + 32768.0) / 65535.0
    else:
        normalized = np.asarray(array, dtype=np.float64)
        normalized = np.clip(normalized, 0.0, 1.0)
    return _matlab_round_positive(normalized * 255.0).clip(0, 255).astype(np.uint8)


def _stretchlim_uint8(image: np.ndarray) -> tuple[float, float]:
    """Reproduce the default ``stretchlim`` histogram limits for uint8 data."""

    counts = np.bincount(np.asarray(image, dtype=np.uint8).ravel(), minlength=256)
    cdf = np.cumsum(counts, dtype=np.float64) / float(np.sum(counts))
    low_candidates = np.flatnonzero(cdf > 0.01)
    high_candidates = np.flatnonzero(cdf >= 0.99)
    if low_candidates.size == 0 or high_candidates.size == 0:
        return 0.0, 1.0
    low = int(low_candidates[0])
    high = int(high_candidates[0])
    # MATLAB returns [1;nbins] for a flat histogram bin, before converting
    # the indices to [0,1].
    if low == high:
        return 0.0, 1.0
    return low / 255.0, high / 255.0


def _imadjust_default_uint8(image: np.ndarray) -> np.ndarray:
    """Reproduce ``imadjust(im2uint8(image))`` for the SIFNE input path."""

    uint8_image = _to_uint8_intensity(image)
    low, high = _stretchlim_uint8(uint8_image)
    if high == low:
        return uint8_image.copy()
    normalized = uint8_image.astype(np.float64) / 255.0
    normalized = np.maximum(low, np.minimum(high, normalized))
    adjusted = (normalized - low) / (high - low)
    return _matlab_round_positive(adjusted * 255.0).clip(0, 255).astype(np.uint8)


def _rgb2gray_uint8(image: np.ndarray) -> np.ndarray:
    """Use MATLAB-compatible luminance weights for an RGB uint8 image."""

    value = np.asarray(image)
    if value.shape[-1] != 3:
        raise ValueError("a color input must have three channels")
    # These are the ITU-R BT.601 weights used by MATLAB rgb2gray for RGB data.
    gray = (
        0.298936021293775 * value[..., 0].astype(np.float64)
        + 0.587043074451258 * value[..., 1].astype(np.float64)
        + 0.114020904214967 * value[..., 2].astype(np.float64)
    )
    return _matlab_round_positive(gray).clip(0, 255).astype(np.uint8)


def _prepare_origin_image(value: np.ndarray) -> np.ndarray:
    """Match the ``LoadImg`` block through its final ``imadjust`` call."""

    image = np.squeeze(np.asarray(value))
    if image.ndim == 3:
        image = _rgb2gray_uint8(image)
    if image.ndim != 2:
        raise ValueError(f"input image must become a 2-D matrix, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float64, copy=False)
        minimum = float(np.min(image))
        maximum = float(np.max(image))
        if maximum != minimum:
            image = (image - minimum) / (maximum - minimum)
        else:
            image = np.zeros_like(image, dtype=np.float64)
    return _imadjust_default_uint8(image)


def _read_input(image_or_path: Any) -> tuple[np.ndarray, str | None, str]:
    if isinstance(image_or_path, (str, Path)):
        path = Path(image_or_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        image = tifffile.imread(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return np.asarray(image), str(path), digest

    image = np.asarray(image_or_path)
    digest = hashlib.sha256()
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(repr(tuple(image.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(image).tobytes())
    return image, None, digest.hexdigest()


def _prepare_roi_mask(config: SIFNEConfig, origin_shape: tuple[int, int], radius: int) -> np.ndarray:
    height, width = origin_shape
    padded_shape = (height + 2 * radius, width + 2 * radius)
    if config.roi_mask is None:
        return np.ones(padded_shape, dtype=np.float64)
    if isinstance(config.roi_mask, (str, Path)):
        mask = np.asarray(tifffile.imread(config.roi_mask))
    else:
        mask = np.asarray(config.roi_mask)
    mask = np.squeeze(mask)
    if mask.shape == padded_shape:
        return (mask != 0).astype(np.float64)
    if mask.shape != origin_shape:
        raise ValueError(
            f"roi_mask must have shape {origin_shape} or {padded_shape}, got {mask.shape}"
        )
    result = np.zeros(padded_shape, dtype=np.float64)
    result[radius : radius + height, radius : radius + width] = mask != 0
    return result


def _as_int_scalar(value: np.ndarray) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def analyze_mt(
    image_or_path: Any,
    *,
    config: SIFNEConfig | None = None,
    backend: str = "reference",
    device: int = 0,
    threads_per_block: int = 256,
    factor_mode: str = "matlab_windows",
    skip_unpaired_fragments: bool = False,
) -> SIFNEResult:
    """Run all verified SIFNE stages without MATLAB or external processes.

    ``backend="reference"`` is the unchanged readable Python implementation.
    ``backend="cuda"`` is an opt-in execution backend for the same LFT/OFT
    loop.  Its default ``factor_mode="matlab_windows"`` uses the frozen
    cosine factors validated against the MATLAB fixtures.  The optional
    ``skip_unpaired_fragments`` flag is an explicit audit-mode repair for
    source-workflow ungrouped fragments and is disabled by default.
    """

    cfg = SIFNEConfig() if config is None else config
    if cfg.mode != "faithful":
        raise NotImplementedError(
            "Only faithful mode is implemented; robust mode is reserved for a future API."
        )
    if backend == "reference":
        lft_oft_impl = lft_oft
        pair_distance_impl = None
        pair_direction_impl = None
        tip_orientation_impl = None
        sort_label_impl = None
        sort_nearest_impl = None
        sort_order_impl = None
        sort_filament_impl = None
        sort_filaments_impl = None
        sort_tip_impl = None
        sort_reuse_path_metrics = False
        neighbor_count_impl = None
        mask2_impl = None
        grouping_trace_impl = None
        sort_retain_intermediates = True
        grouping_retain_diagnostics = True
        pairing_pool_impl = None
        pairing_retain_local_results = True
        tip_endpoint_impl = None
        tip_centroid_impl = None
        thin_impl = None
        large_cur_flags_impl = None
    elif backend == "cuda":
        from .lft_oft_accel import cuda_lft_oft
        from .pairing_accel import (
            cpu_pair_directions_exact,
            cpu_pairwise_directions_exact,
            cuda_pairwise_distances,
            numba_pairing_pools,
        )
        from .fragments_accel import (
            cuda_large_cur_flags,
            cuda_neighbor_counts,
            cuda_square_mask,
            cuda_thin,
            numba_label_components,
        )
        from .sorting_accel import cuda_order_filaments
        from .grouping_accel import numba_trace_composite
        from .tips_accel import numba_detect_endpoints, numba_local_centroids

        lft_oft_impl = partial(
            cuda_lft_oft,
            device=int(device),
            threads_per_block=int(threads_per_block),
            factor_mode=factor_mode,
        )
        pair_distance_impl = partial(
            cuda_pairwise_distances,
            device=int(device),
            threads_per_block=int(threads_per_block),
        )
        pair_direction_impl = cpu_pairwise_directions_exact
        tip_orientation_impl = cpu_pair_directions_exact
        sort_label_impl = numba_label_components
        sort_nearest_impl = numba_nearest_active_index
        sort_order_impl = None
        sort_filament_impl = None
        sort_filaments_impl = partial(cuda_order_filaments, device=int(device))
        sort_tip_impl = detect_tips_sparse_label_vectorized
        sort_reuse_path_metrics = True
        neighbor_count_impl = partial(
            cuda_neighbor_counts,
            device=int(device),
            threads_per_block=int(threads_per_block),
        )
        mask2_impl = partial(
            cuda_square_mask,
            device=int(device),
            threads_per_block=int(threads_per_block),
        )
        grouping_trace_impl = numba_trace_composite
        sort_retain_intermediates = False
        grouping_retain_diagnostics = False
        pairing_pool_impl = numba_pairing_pools
        pairing_retain_local_results = False
        tip_endpoint_impl = numba_detect_endpoints
        tip_centroid_impl = numba_local_centroids
        thin_impl = partial(
            cuda_thin,
            device=int(device),
            threads_per_block=int(threads_per_block),
        )
        large_cur_flags_impl = partial(
            cuda_large_cur_flags,
            device=int(device),
            threads_per_block=int(threads_per_block),
        )
    else:
        raise ValueError(f"unknown LFT/OFT backend: {backend!r}")
    raw_input, input_path, input_sha256 = _read_input(image_or_path)
    origin_image = _prepare_origin_image(raw_input)
    radius = int(cfg.radius)
    height, width = origin_image.shape
    origin_image_margin = np.zeros((height + 2 * radius, width + 2 * radius), dtype=np.uint8)
    origin_image_margin[radius : radius + height, radius : radius + width] = origin_image
    roi_mask = _prepare_roi_mask(cfg, origin_image.shape, radius)

    oft_img, lft_img, lft_orientations = lft_oft_impl(
        origin_image_margin.astype(np.float64),
        radius,
        int(cfg.n_orientations),
        roi_mask,
    )
    normalized_oft, bw, raw_skeleton, otsu_threshold, threshold_used = segment_oft(
        oft_img,
        default_factor=cfg.default_factor,
        thin_impl=thin_impl,
    )
    fragment_data = extract_stage_04_fragments(
        origin_image,
        raw_skeleton,
        radius,
        roi_mask,
        int(cfg.n_orientations),
        threshold_used,
        iteration=int(cfg.iteration),
        junction_size=int(cfg.junction_size),
        iteration_remove_radius=int(cfg.iteration_remove_radius),
        minimum_length=int(cfg.minimum_fragment_length),
        lft_oft_impl=lft_oft_impl,
        neighbor_count_impl=neighbor_count_impl,
        label_components_impl=sort_label_impl,
        mask2_impl=mask2_impl,
        thin_impl=thin_impl,
        large_cur_flags_impl=large_cur_flags_impl,
    )
    fragment_result = FragmentStageResult(fragment_data)
    junctions_one_based = np.asarray(fragment_data["CrPts"], dtype=np.float64).reshape(-1, 2)
    junctions = junctions_one_based - 1.0 if junctions_one_based.size else junctions_one_based

    tip_result = register_filament_tips(
        fragment_result.all_fragments,
        fragment_result.labels,
        cfg.registration_radius_value,
        orientation_impl=tip_orientation_impl,
        endpoint_mask_impl=tip_endpoint_impl,
        centroid_impl=tip_centroid_impl,
    )
    distance_matrix = (
        None
        if pair_distance_impl is None
        else pair_distance_impl(tip_result.all_tips[:, 0:2])
    )
    direction_matrix = (
        None
        if pair_direction_impl is None
        else pair_direction_impl(tip_result.all_tips[:, 0:2])
    )
    pairing_result = pair_filament_tips(
        tip_result.all_tips,
        fragment_result.labels,
        cfg.fan_angle_value,
        cfg.fan_radius_value,
        cfg.min_info.reshape(-1),
        cfg.c1_weight,
        cfg.c3_weight,
        partner_slots=int(cfg.partner_slots),
        distance_matrix=distance_matrix,
        direction_matrix=direction_matrix,
        vectorized_local_search=direction_matrix is not None,
        pool_impl=pairing_pool_impl,
        retain_local_results=pairing_retain_local_results,
    )
    del distance_matrix
    del direction_matrix
    grouping_result = group_fragments(
        pairing_result.all_tips,
        fragment_result.labels,
        num_fragments=fragment_result.number,
        trace_impl=grouping_trace_impl,
        retain_diagnostics=grouping_retain_diagnostics,
        skip_unpaired_fragments=skip_unpaired_fragments,
    )
    if grouping_result.composite_count == 0:
        raise ValueError("faithful SIFNE grouping produced no composite filaments")
    full_length = int(cfg.full_length_factor) * int(grouping_result.all_filament.shape[0])
    sorting_result = sort_filaments(
        grouping_result.all_filament,
        grouping_result.all_connects,
        fragment_result.labels.shape,
        full_length,
        filament_lengths=[trace.filament_length for trace in grouping_result.traces],
        connection_lengths=[trace.connection_count for trace in grouping_result.traces],
        label_components_impl=sort_label_impl,
        tip_detector_impl=sort_tip_impl,
        nearest_index_impl=sort_nearest_impl,
        order_points_impl=sort_order_impl,
        order_filament_impl=sort_filament_impl,
        order_filaments_impl=sort_filaments_impl,
        reuse_path_metrics=sort_reuse_path_metrics,
        retain_intermediates=sort_retain_intermediates,
    )

    provenance = {
        "pysifne_version": PY_SIFNE_VERSION,
        "pysifne_profile": PY_SIFNE_PROFILE,
        "config": cfg.to_dict(),
        "input_sha256": input_sha256,
        "input_path": input_path,
        "mode": cfg.mode,
        "lft_oft_backend": backend,
        "lft_oft_device": int(device),
        "lft_oft_threads_per_block": int(threads_per_block),
        "lft_oft_factor_mode": factor_mode if backend == "cuda" else "reference",
        "source_reference": cfg.source_reference,
        "analysis_info_definition": "MATLAB pixel-space [orientation,length_px,end_to_end_px,centroid_row,centroid_col]",
        "skip_unpaired_fragments": bool(skip_unpaired_fragments),
        "unpaired_fragment_ids": list(grouping_result.unpaired_fragment_ids),
    }
    return SIFNEResult(
        input_image=np.asarray(raw_input),
        origin_image=origin_image,
        origin_image_margin=origin_image_margin,
        roi_mask=roi_mask,
        oft_img=oft_img,
        lft_img=lft_img,
        lft_orientations=lft_orientations,
        normalized_oft=normalized_oft,
        bw=bw,
        raw_skeleton=raw_skeleton,
        fragments=fragment_result,
        tips=tip_result,
        pairing=pairing_result,
        grouping=grouping_result,
        sorting=sorting_result,
        junctions=junctions,
        otsu_threshold=float(otsu_threshold),
        threshold_used=float(threshold_used),
        config=cfg,
        provenance=provenance,
    )


def export_matlab_compatibility(
    result: SIFNEResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Export the core MATLAB-consumer files using ``scipy.io.savemat``."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, tuple[str, np.ndarray]] = {
        "OFT_Img.mat": ("OFT_Img", result.oft_img),
        "LFT_Img.mat": ("LFT_Img", result.lft_img),
        "LFT_Orientations.mat": ("LFT_Orientations", result.lft_orientations),
        "RawSke.mat": ("RawSke", result.raw_skeleton),
        "AllFragments.mat": ("AllFragments", result.fragments.all_fragments),
        "R.mat": ("R", np.asarray([[float(result.config.radius)]], dtype=np.float64)),
        "NofOrientations_FT.mat": (
            "NofOrientations_FT",
            np.asarray([[float(result.config.n_orientations)]], dtype=np.float64),
        ),
        "ROI_Mask.mat": ("ROI_Mask", result.roi_mask),
        # The current core pipeline exposes the pre-grouping junction list.
        # It is the exact ``CrPts`` reference for the frozen core stages;
        # later overlap-collapse ``NewCrPts`` remains a separate dependency.
        "NewCrPts.mat": ("NewCrPts", result.junctions_matlab),
        "all_tips.mat": ("all_tips", result.all_tips_matlab),
        "new_partner_list.mat": ("new_partner_list", result.partner_list),
        "all_filament.mat": ("all_filament", result.all_filament_matlab),
        "all_connects.mat": ("all_connects", result.all_connects_matlab),
        "all_sorted_filament.mat": (
            "all_sorted_filament",
            result.all_sorted_filament_matlab,
        ),
        "AnalysisInfo.mat": ("AnalysisInfo", result.analysis_info),
    }
    written: dict[str, Path] = {}
    for filename, (variable, value) in arrays.items():
        path = destination / filename
        savemat(path, {variable: np.asarray(value)}, do_compression=True)
        written[filename] = path
    provenance_path = destination / "provenance.json"
    provenance_path.write_text(
        json.dumps(dict(result.provenance), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    written[provenance_path.name] = provenance_path
    return written


__all__ = ["analyze_mt", "export_matlab_compatibility"]
