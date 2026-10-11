from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pysifne.sorting import (
    compute_centroid,
    compute_orientation,
    fill_line_matlab,
    sort_filaments_matlab,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_ROOT = (
    REPO_ROOT
    / "tests" / "fixtures"
    / "core"
    / "npz"
)
SORT_ROOT = (
    REPO_ROOT
    / "tests" / "fixtures"
    / "sorting"
    / "npz"
)

FILL_CASES = (
    "horizontal",
    "vertical",
    "positive_diagonal",
    "negative_diagonal",
    "steep_slope",
    "shallow_slope",
    "reverse_endpoints",
    "identical_endpoints",
)

SORT_CASES = (
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
    "single_pixel",
    "two_independent",
    "connection_direction",
)

MACHINE_ATOL = 16.0 * np.finfo(np.float64).eps


def _load(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise AssertionError(f"Missing MATLAB fixture: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _as_3d(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2:
        return array[:, :, None]
    return array


@pytest.mark.parametrize("case_name", FILL_CASES)
def test_filline_direct_fixture_is_exact(case_name: str) -> None:
    reference = _load(SORT_ROOT / f"fill_{case_name}.npz")
    actual = fill_line_matlab(
        reference["input_image"],
        float(reference["value"].reshape(-1)[0]),
        *reference["p1"].reshape(-1),
        *reference["p2"].reshape(-1),
    )
    assert np.array_equal(actual, reference["output_image"])


def _build_numeric_mask(
    reference: dict[str, np.ndarray], composite_index: int
) -> np.ndarray:
    shape = tuple(reference["image_shape"].reshape(-1).astype(np.int64))
    records = _as_3d(reference["filament_records"])
    lengths = reference["filament_lengths"].reshape(-1).astype(int)
    mask = np.zeros(shape, dtype=np.float64)
    for row, column in records[: lengths[composite_index], :, composite_index]:
        mask[int(row) - 1, int(column) - 1] = 1.0
    return mask


@pytest.mark.parametrize("case_name", SORT_CASES)
def test_sortfilament_direct_fixture_is_exact(case_name: str) -> None:
    reference = _load(SORT_ROOT / f"{case_name}.npz")
    supported = reference["supported"].reshape(-1).astype(bool)
    if not np.any(supported):
        # MATLAB's explicit tips(1)/tips(2) access does not support a
        # single-pixel object; the fixture records that behavior explicitly.
        assert case_name == "single_pixel"
        assert np.array_equal(reference["all_sorted_filament"], np.zeros((0, 2)))
        return

    result = sort_filaments_matlab(
        reference["filament_records"],
        reference["connect_records"],
        tuple(reference["image_shape"].reshape(-1).astype(np.int64)),
        int(reference["Fullength"].reshape(-1)[0]),
    )
    assert np.array_equal(result.all_sorted_filament_matlab, reference["all_sorted_filament"])
    assert result.analysis_info.shape == reference["AnalysisInfo"].shape
    np.testing.assert_allclose(
        result.analysis_info,
        reference["AnalysisInfo"],
        rtol=0.0,
        atol=MACHINE_ATOL,
    )

    records = _as_3d(reference["ordered_records"])
    expected_steps = _as_3d(reference["step_deltas"])
    expected_distances = np.asarray(reference["step_distances"])
    expected_cumulative = np.asarray(reference["cumulative_distances"])
    lengths = reference["ordered_lengths"].reshape(-1).astype(int)
    for index, trace in enumerate(result.per_filament):
        length = int(lengths[index])
        assert np.array_equal(
            trace.sorted_filament_matlab,
            records[:, :, index] if records.shape[2] > 1 else records[:, :, 0],
        )
        assert np.array_equal(
            trace.ordered_pixels + 1.0,
            records[:length, :, index] if records.shape[2] > 1 else records[:length, :, 0],
        )
        assert np.array_equal(
            trace.step_deltas,
            expected_steps[: max(length - 1, 0), :, index]
            if expected_steps.shape[2] > 1
            else expected_steps[: max(length - 1, 0), :, 0],
        )
        expected_step_column = (
            expected_distances[: max(length - 1, 0), index]
            if expected_distances.ndim == 2
            else expected_distances[: max(length - 1, 0), 0]
        )
        assert np.array_equal(trace.step_distances, expected_step_column)
        expected_cumulative_column = (
            expected_cumulative[:length, index]
            if expected_cumulative.ndim == 2
            else expected_cumulative[:length, 0]
        )
        assert np.array_equal(trace.cumulative_distances, expected_cumulative_column)

    # Equal-distance branch selection remains order-sensitive.
    if case_name == "equal_distance_tie":
        assert np.array_equal(
            result.per_filament[0].ordered_pixels + 1.0,
            reference["ordered_records"][:6, :, 0]
            if reference["ordered_records"].ndim == 3
            else reference["ordered_records"][:6, :],
        )


def test_orientation_and_centroid_are_independently_exact_or_machine_level() -> None:
    for case_name in SORT_CASES:
        reference = _load(SORT_ROOT / f"{case_name}.npz")
        supported = reference["supported"].reshape(-1).astype(bool)
        if not np.any(supported):
            continue
        for index, is_supported in enumerate(supported):
            if not is_supported:
                continue
            mask = _build_numeric_mask(reference, index)
            actual_orientation = compute_orientation(mask)
            actual_centroid = compute_centroid(mask)
            np.testing.assert_allclose(
                actual_orientation,
                reference["orientation"].reshape(-1)[index],
                rtol=0.0,
                atol=MACHINE_ATOL,
            )
            np.testing.assert_allclose(
                actual_centroid,
                reference["centroid_row_column"][index, :],
                rtol=0.0,
                atol=MACHINE_ATOL,
            )


def test_stage_08_sorted_matches_existing_matlab_fixture_exactly() -> None:
    with np.load(STAGE_ROOT / "stage_04_fragments.npz", allow_pickle=False) as data:
        stage04 = {key: data[key] for key in data.files}
    with np.load(STAGE_ROOT / "stage_07_grouped.npz", allow_pickle=False) as data:
        stage07 = {key: data[key] for key in data.files}
    with np.load(STAGE_ROOT / "stage_08_sorted.npz", allow_pickle=False) as data:
        stage08 = {key: data[key] for key in data.files}

    result = sort_filaments_matlab(
        stage07["all_filament"],
        stage07["all_connects"],
        stage04["L"].shape,
        int(stage08["Fullength"].reshape(-1)[0]),
    )
    assert np.array_equal(
        result.all_sorted_filament_matlab,
        stage08["all_sorted_filament"],
    )
    assert np.array_equal(result.analysis_info, stage08["AnalysisInfo"])

    trace = result.per_filament[0]
    assert np.array_equal(trace.ordered_pixels + 1.0, stage08["all_sorted_filament"])
    assert np.array_equal(trace.analysis_info, stage08["AnalysisInfo"][0, :])

