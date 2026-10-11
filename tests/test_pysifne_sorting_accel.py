from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pytest

from pysifne.fragments import label_components
from pysifne.fragments_accel import numba_label_components
from pysifne.lft_oft_accel import cuda_available
from pysifne.sorting import (
    detect_tips_sparse_label_vectorized,
    numba_nearest_active_index,
    numba_order_points,
    sort_filaments_matlab,
)
from pysifne.sorting_accel import (
    cuda_order_filaments,
    cuda_order_points,
    cuda_order_points_adaptive,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_ROOT = (
    REPO_ROOT
    / "data"
    / "derived"
    / "reference-freezing"
    / "sifne-fixture-20260915-r4"
    / "npz"
)


def _load(name: str) -> dict[str, np.ndarray]:
    with np.load(STAGE_ROOT / name, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def test_numba_label_components_preserves_reference_scan_and_dfs() -> None:
    pytest.importorskip("numba")
    rng = np.random.default_rng(20260915)
    for connectivity in (4, 8):
        for shape in ((7, 9), (13, 11)):
            binary = rng.random(shape) > 0.68
            expected, expected_count = label_components(binary, connectivity)
            actual, actual_count = numba_label_components(binary, connectivity)
            assert np.array_equal(actual, expected)
            assert actual_count == expected_count

    raw = _load("stage_04_fragments.npz")
    expected, expected_count = label_components(raw["AllFragments"], 8)
    actual, actual_count = numba_label_components(raw["AllFragments"], 8)
    assert np.array_equal(actual, expected)
    assert actual_count == expected_count


def test_numba_labeler_preserves_stage08_sorting_exactly() -> None:
    pytest.importorskip("numba")
    stage04 = _load("stage_04_fragments.npz")
    stage07 = _load("stage_07_grouped.npz")
    stage08 = _load("stage_08_sorted.npz")
    actual = sort_filaments_matlab(
        stage07["all_filament"],
        stage07["all_connects"],
        stage04["L"].shape,
        int(stage08["Fullength"].reshape(-1)[0]),
        label_components_impl=numba_label_components,
        nearest_index_impl=numba_nearest_active_index,
        order_points_impl=numba_order_points,
    )
    assert np.array_equal(actual.all_sorted_filament_matlab, stage08["all_sorted_filament"])
    assert np.array_equal(actual.analysis_info, stage08["AnalysisInfo"])


def test_cuda_order_kernel_preserves_stage08_sorting_exactly() -> None:
    pytest.importorskip("numba")
    if not cuda_available():
        pytest.skip("CUDA device is not available")
    stage04 = _load("stage_04_fragments.npz")
    stage07 = _load("stage_07_grouped.npz")
    stage08 = _load("stage_08_sorted.npz")
    actual = sort_filaments_matlab(
        stage07["all_filament"],
        stage07["all_connects"],
        stage04["L"].shape,
        int(stage08["Fullength"].reshape(-1)[0]),
        label_components_impl=numba_label_components,
        nearest_index_impl=numba_nearest_active_index,
        order_points_impl=partial(cuda_order_points, device=0),
    )
    assert np.array_equal(actual.all_sorted_filament_matlab, stage08["all_sorted_filament"])
    assert np.array_equal(actual.analysis_info, stage08["AnalysisInfo"])


def test_cuda_adaptive_order_preserves_stage08_sorting_exactly() -> None:
    pytest.importorskip("numba")
    if not cuda_available():
        pytest.skip("CUDA device is not available")
    stage04 = _load("stage_04_fragments.npz")
    stage07 = _load("stage_07_grouped.npz")
    stage08 = _load("stage_08_sorted.npz")
    actual = sort_filaments_matlab(
        stage07["all_filament"],
        stage07["all_connects"],
        stage04["L"].shape,
        int(stage08["Fullength"].reshape(-1)[0]),
        label_components_impl=numba_label_components,
        nearest_index_impl=numba_nearest_active_index,
        order_points_impl=partial(cuda_order_points_adaptive, device=0),
    )
    assert np.array_equal(actual.all_sorted_filament_matlab, stage08["all_sorted_filament"])
    assert np.array_equal(actual.analysis_info, stage08["AnalysisInfo"])


def test_cuda_batch_order_preserves_stage08_sorting_exactly() -> None:
    pytest.importorskip("numba")
    if not cuda_available():
        pytest.skip("CUDA device is not available")
    stage04 = _load("stage_04_fragments.npz")
    stage07 = _load("stage_07_grouped.npz")
    stage08 = _load("stage_08_sorted.npz")
    actual = sort_filaments_matlab(
        stage07["all_filament"],
        stage07["all_connects"],
        stage04["L"].shape,
        int(stage08["Fullength"].reshape(-1)[0]),
        label_components_impl=numba_label_components,
        nearest_index_impl=numba_nearest_active_index,
        order_filaments_impl=partial(cuda_order_filaments, device=0),
    )
    assert np.array_equal(actual.all_sorted_filament_matlab, stage08["all_sorted_filament"])
    assert np.array_equal(actual.analysis_info, stage08["AnalysisInfo"])


def test_cuda_sort_companions_preserve_all_direct_sorting_outputs() -> None:
    """The optional CUDA-pipeline helpers must be exact on every frozen case."""

    sort_root = (
        REPO_ROOT
        / "data"
        / "derived"
        / "reference-freezing"
        / "sortfilament-20260915"
        / "npz"
    )
    case_names = (
        "single_horizontal",
        "single_vertical",
        "diagonal_pos",
        "diagonal_neg",
        "curved",
        "two_fragment_gap",
        "three_fragment_serial",
        "reverse_traversal",
        "equal_distance_tie",
        "irregular_path",
        "two_independent",
        "connection_direction",
    )
    for case_name in case_names:
        reference = _load(sort_root / f"{case_name}.npz")
        if not np.any(reference["supported"].reshape(-1).astype(bool)):
            continue
        kwargs = dict(order_filaments_impl=None)
        baseline = sort_filaments_matlab(
            reference["filament_records"],
            reference["connect_records"],
            tuple(reference["image_shape"].reshape(-1).astype(np.int64)),
            int(reference["Fullength"].reshape(-1)[0]),
            **kwargs,
        )
        optimized = sort_filaments_matlab(
            reference["filament_records"],
            reference["connect_records"],
            tuple(reference["image_shape"].reshape(-1).astype(np.int64)),
            int(reference["Fullength"].reshape(-1)[0]),
            tip_detector_impl=detect_tips_sparse_label_vectorized,
            reuse_path_metrics=True,
            **kwargs,
        )
        assert np.array_equal(
            optimized.all_sorted_filament_matlab,
            baseline.all_sorted_filament_matlab,
        ), case_name
        assert np.array_equal(optimized.analysis_info, baseline.analysis_info), case_name
        assert len(optimized.per_filament) == len(baseline.per_filament)
        for optimized_trace, baseline_trace in zip(
            optimized.per_filament, baseline.per_filament
        ):
            assert np.array_equal(
                optimized_trace.ordered_pixels,
                baseline_trace.ordered_pixels,
            ), case_name
            assert np.array_equal(
                optimized_trace.step_deltas,
                baseline_trace.step_deltas,
            ), case_name
            assert np.array_equal(
                optimized_trace.step_distances,
                baseline_trace.step_distances,
            ), case_name
            assert np.array_equal(
                optimized_trace.cumulative_distances,
                baseline_trace.cumulative_distances,
            ), case_name
