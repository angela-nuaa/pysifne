from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pysifne.pairing import (
    enforce_bidirectional_pairing,
    local_search_detailed,
    pair_filament_tips,
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
PAIRING_ROOT = (
    REPO_ROOT
    / "data"
    / "derived"
    / "reference-freezing"
    / "pairing-20260915"
    / "npz"
)

DIRECT_CASES = (
    "collinear_allowed",
    "distance_too_far",
    "direction_fail",
    "fan_angle_fail",
    "score_order",
    "equal_distance_tie",
    "inner_fragment_suppression",
    "no_candidates",
    "angle_wrap",
    "boundary_tip",
)


def _load(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise AssertionError(f"Missing MATLAB reference fixture: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _python_inputs(reference: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    tips = reference["all_tips_input"].copy()
    # MATLAB fixture stores image coordinates as one-based; the Python API is
    # explicitly zero-based for row/column coordinates.
    tips[:, 0:2] -= 1.0
    tips[:, 4:6] -= 1.0
    base = reference["base_loc"].reshape(2).copy() - 1.0
    return tips, base


def _to_matlab_tip_table(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    if result.ndim == 2 and result.shape[0] != 0:
        result[:, 0:2] += 1.0
        result[:, 4:6] += 1.0
    return result


def _assert_vector_exact(actual: np.ndarray, expected: np.ndarray) -> None:
    """Compare values/order exactly while allowing MATLAB row/column vectors."""

    actual_value = np.asarray(actual)
    expected_value = np.asarray(expected)
    assert actual_value.size == expected_value.size
    assert np.array_equal(actual_value.reshape(-1), expected_value.reshape(-1))


def _assert_matrix_exact(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.asarray(actual).shape == np.asarray(expected).shape
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("case_name", DIRECT_CASES)
def test_local_search_direct_fixture_is_exact(case_name: str) -> None:
    reference = _load(PAIRING_ROOT / f"{case_name}.npz")
    tips, base = _python_inputs(reference)
    result = local_search_detailed(
        reference["L_GlobalIndex"],
        reference["label_list"],
        float(reference["fanAngle"][0, 0]),
        tips,
        float(reference["r"][0, 0]),
        float(reference["base_ori"][0, 0]),
        base,
        reference["MIN_Info"].reshape(-1),
        float(reference["C1weight"][0, 0]),
        float(reference["C3weight"][0, 0]),
    )

    _assert_vector_exact(result.all_dist, reference["All_Dist"])
    _assert_vector_exact(result.distance_pool, reference["DistancePool"])
    _assert_vector_exact(result.fan_pool, reference["FanPool"])
    _assert_matrix_exact(
        _to_matlab_tip_table(result.candidate_tips), reference["CandidateTips"]
    )
    _assert_vector_exact(result.candidate_distances, reference["CandidateDistances"])
    _assert_vector_exact(result.candidate_dir, reference["CandidateDir"])
    _assert_vector_exact(result.angle_diff, reference["AngleDiff"])
    _assert_vector_exact(result.min3_list, reference["MIN3_List"])
    _assert_vector_exact(result.gap_orient, reference["GapOrient"])
    _assert_vector_exact(result.gap_orient_diff, reference["GapOrient_Diff"])
    _assert_matrix_exact(
        _to_matlab_tip_table(result.shortlist_candidate_tips),
        reference["ShortListCandidateTips"],
    )
    _assert_vector_exact(result.shortlist_pool, reference["ShortListPool"])
    _assert_vector_exact(result.index_global_pool, reference["local_search_return"])


def test_equal_distance_candidate_order_is_not_reduced_to_set_equality() -> None:
    reference = _load(PAIRING_ROOT / "equal_distance_tie.npz")
    tips, base = _python_inputs(reference)
    result = local_search_detailed(
        reference["L_GlobalIndex"],
        reference["label_list"],
        float(reference["fanAngle"][0, 0]),
        tips,
        float(reference["r"][0, 0]),
        float(reference["base_ori"][0, 0]),
        base,
        reference["MIN_Info"].reshape(-1),
        1.0,
        1.0,
    )
    expected_order = reference["local_search_return"].reshape(-1)
    assert np.array_equal(result.index_global_pool, expected_order)
    assert np.array_equal(result.index_global_pool[:2], np.array([2.0, 3.0]))


def test_stage_06_pairing_matches_existing_matlab_fixture_exactly() -> None:
    stage04 = _load(STAGE_ROOT / "stage_04_fragments.npz")
    stage05 = _load(STAGE_ROOT / "stage_05_tips.npz")
    reference = _load(STAGE_ROOT / "stage_06_pairing.npz")

    stage05_tips = stage05["all_tips"].copy()
    stage05_tips[:, 0:2] -= 1.0
    stage05_tips[:, 4:6] -= 1.0
    result = pair_filament_tips(
        stage05_tips,
        stage04["L"],
        float(reference["FanAngle"][0, 0]),
        float(reference["FanR"][0, 0]),
        reference["MIN_Info"].reshape(-1),
        float(reference["C1"][0, 0]),
        float(reference["C3"][0, 0]),
    )

    assert np.array_equal(result.l_global_index, reference["L_GlobalIndex"])
    assert np.array_equal(result.label_list, reference["label_list"])
    assert np.array_equal(result.new_partner_list, reference["new_partner_list"])
    assert np.array_equal(
        result.all_tips_pre_bidir_matlab, reference["all_tips_pre_bidir"]
    )
    assert np.array_equal(
        result.all_tips_post_bidir_matlab, reference["all_tips_post_bidir"]
    )
    assert np.array_equal(result.all_tips_matlab, reference["all_tips"])
    assert np.count_nonzero(result.new_partner_list) == np.count_nonzero(
        reference["new_partner_list"]
    )


def test_stage_06_each_local_search_row_matches_trimmed_partner_reference() -> None:
    stage04 = _load(STAGE_ROOT / "stage_04_fragments.npz")
    stage05 = _load(STAGE_ROOT / "stage_05_tips.npz")
    reference = _load(STAGE_ROOT / "stage_06_pairing.npz")
    stage05_tips = stage05["all_tips"].copy()
    stage05_tips[:, 0:2] -= 1.0
    stage05_tips[:, 4:6] -= 1.0
    result = pair_filament_tips(
        stage05_tips,
        stage04["L"],
        float(reference["FanAngle"][0, 0]),
        float(reference["FanR"][0, 0]),
        reference["MIN_Info"].reshape(-1),
        float(reference["C1"][0, 0]),
        float(reference["C3"][0, 0]),
    )

    expected = np.zeros((reference["new_partner_list"].shape[0], 100))
    expected[:, : reference["new_partner_list"].shape[1]] = reference[
        "new_partner_list"
    ]
    actual = np.vstack(
        [item.index_global_pool for item in result.local_search_results]
    )
    assert np.array_equal(actual, expected)


def test_bidirectional_cleanup_matches_one_way_matlab_fixture_exactly() -> None:
    reference = _load(PAIRING_ROOT / "bidir_one_way.npz")
    actual = enforce_bidirectional_pairing(reference["all_tips_bidir_input"])
    assert np.array_equal(actual, reference["all_tips_bidir_output"])
    # The one-way row-1 -> row-3 relationship is removed, while reciprocal
    # row-1 -> row-2 and row-1 -> row-4 relationships remain in order.
    assert np.array_equal(actual[0, 8:], np.array([2.0, 4.0, 0.0]))

