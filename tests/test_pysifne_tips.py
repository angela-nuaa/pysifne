from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pysifne.tips import (
    check_dir_between2pts,
    detect_endpoints,
    extract_endpoint_local_window,
    local_centroid,
    register_filament_tips,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_FIXTURE_ROOT = (
    REPO_ROOT
    / "data"
    / "derived"
    / "reference-freezing"
    / "sifne-fixture-20260915-r4"
    / "npz"
)
TIP_FIXTURE_ROOT = (
    REPO_ROOT
    / "data"
    / "derived"
    / "reference-freezing"
    / "tipreg-20260915"
    / "npz"
)

DIRECT_CASES = (
    "horizontal",
    "vertical",
    "plus45",
    "minus45",
    "curved",
    "near_boundary",
    "two_fragments",
    "symmetric_tie",
)


def _load(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise AssertionError(f"Missing MATLAB reference fixture: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def test_check_dir_between2pts_matches_matlab_fixture_exactly() -> None:
    reference = _load(TIP_FIXTURE_ROOT / "check_dir_between2pts.npz")
    actual = np.asarray(
        [
            check_dir_between2pts(p1, p2)
            for p1, p2 in zip(reference["P1"], reference["P2"])
        ],
        dtype=np.float64,
    )
    expected = reference["ExpectedOrient"].reshape(-1)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("case_name", DIRECT_CASES)
def test_direct_matlab_tipreg_fixture_is_exact(case_name: str) -> None:
    reference = _load(TIP_FIXTURE_ROOT / f"{case_name}.npz")
    actual = register_filament_tips(
        reference["AllFragments"],
        reference["L"],
        float(reference["RegR"][0, 0]),
    )

    # Endpoint mask/order and every column of TipReg's one-based output are
    # compared before any tolerance-based fallback is considered.
    assert np.array_equal(actual.endpoint_mask, reference["SkeTermi"])
    assert np.array_equal(actual.all_tips_matlab, reference["all_tips"])
    assert np.array_equal(actual.temp_info_matlab, reference["tempInfo"])


def test_stage_05_tips_matches_existing_golden_exactly() -> None:
    fragments = _load(STAGE_FIXTURE_ROOT / "stage_04_fragments.npz")
    reference = _load(STAGE_FIXTURE_ROOT / "stage_05_tips.npz")
    actual = register_filament_tips(
        fragments["AllFragments"],
        fragments["L"],
        float(reference["RegR"][0, 0]),
    )

    assert actual.all_tips.shape[0] == reference["all_tips"].shape[0]
    assert np.array_equal(actual.all_tips_matlab, reference["all_tips"])
    assert np.array_equal(actual.temp_info_matlab, reference["tempInfo"])

    # Keep the requested observables explicit: count, coordinate/order, and
    # registered orientation/centroid columns each have their own assertion.
    assert np.array_equal(actual.all_tips_matlab[:, 0:3], reference["all_tips"][:, 0:3])
    assert np.array_equal(actual.all_tips_matlab[:, 3], reference["all_tips"][:, 3])
    assert np.array_equal(actual.all_tips_matlab[:, 4:6], reference["all_tips"][:, 4:6])


def test_endpoint_lut_and_column_major_order_are_exact_on_direct_fixtures() -> None:
    for case_name in DIRECT_CASES:
        reference = _load(TIP_FIXTURE_ROOT / f"{case_name}.npz")
        # Construct the expected order from MATLAB's all_tips table instead
        # of assuming a NumPy row-major traversal order.
        expected_coordinates = reference["all_tips"][:, 0:2].astype(np.int64) - 1
        actual_coordinates = detect_endpoints(reference["AllFragments"])
        assert np.array_equal(actual_coordinates, expected_coordinates), case_name


def test_local_window_centroid_and_boundary_conversion_use_tipreg_coordinates() -> None:
    reference = _load(TIP_FIXTURE_ROOT / "near_boundary.npz")
    all_tips = reference["all_tips"]
    tip_row_zero = int(all_tips[0, 0]) - 1
    tip_column_zero = int(all_tips[0, 1]) - 1
    label = all_tips[0, 2]
    local = extract_endpoint_local_window(
        reference["L"], label, (tip_row_zero, tip_column_zero), float(reference["RegR"][0, 0])
    )
    center_local_one_based = local_centroid(local.local_image)
    expected_global_one_based = all_tips[0, 4:6]
    expected_local_one_based = expected_global_one_based - local.fragment_min_one_based

    assert np.array_equal(local.fragment_min_one_based, np.array([1, 3]))
    assert np.array_equal(local.tip_one_based, np.array([1.0, 1.0]))
    assert np.allclose(center_local_one_based, expected_local_one_based, rtol=0.0, atol=0.0)
    assert local.local_image.shape[0] >= 1
    assert local.local_image.shape[1] >= 1
