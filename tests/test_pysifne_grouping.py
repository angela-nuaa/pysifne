from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pysifne.grouping import build_fragment_pixel_table, group_fragments


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_ROOT = (
    REPO_ROOT
    / "tests" / "fixtures"
    / "core"
    / "npz"
)
GROUPING_ROOT = (
    REPO_ROOT
    / "tests" / "fixtures"
    / "grouping"
    / "npz"
)

DIRECT_CASES = (
    "single_fragment",
    "two_serial",
    "three_serial",
    "two_independent",
    "one_valid_partner",
    "reverse_traversal",
    "id_spatial_reverse",
    "cycle_like",
    "branch_like",
    "duplicate_partner",
    "empty_pairing",
    "empty_input",
)


def _load(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise AssertionError(f"Missing MATLAB grouping fixture: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _python_tips(reference: dict[str, np.ndarray]) -> np.ndarray:
    tips = reference["all_tips_input"].copy()
    if tips.shape[0]:
        tips[:, 0:2] -= 1.0
        tips[:, 4:6] -= 1.0
    return tips


def _record_3d(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        return np.empty((0, 2, 0), dtype=array.dtype)
    if array.ndim == 2:
        return array[:, :, None]
    assert array.ndim == 3
    return array


def _python_membership_padded(
    result, expected: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    expected_matrix = np.asarray(expected)
    if expected_matrix.size == 0:
        expected_matrix = np.empty((0, 0), dtype=np.float64)
    actual_matrix = np.zeros_like(expected_matrix)
    for index, membership in enumerate(result.fragment_memberships):
        actual_matrix[index, : membership.size] = membership
    actual_lengths = np.asarray(
        [membership.size for membership in result.fragment_memberships], dtype=np.float64
    )
    return actual_matrix, actual_lengths


def _python_connection_records(result, expected: np.ndarray) -> np.ndarray:
    expected_3d = _record_3d(expected)
    actual = np.zeros_like(expected_3d)
    for index, trace in enumerate(result.traces):
        rows = trace.connection_points.shape[0]
        actual[:rows, :, index] = trace.connection_points + 1.0
    if expected.ndim == 2 and actual.shape[2] == 1:
        return actual[:, :, 0]
    return actual


@pytest.mark.parametrize("case_name", DIRECT_CASES)
def test_grouping_direct_fixture_is_exact(case_name: str) -> None:
    reference = _load(GROUPING_ROOT / f"{case_name}.npz")
    result = group_fragments(
        _python_tips(reference),
        reference["L"],
        num_fragments=int(reference["num_fragments"].reshape(-1)[0]),
    )

    assert np.array_equal(result.all_filament_matlab, reference["all_filament"])
    assert np.array_equal(result.all_connects_matlab, reference["all_connects"])
    assert np.array_equal(
        result.all_connects_shortlist_matlab,
        reference["all_connects_shortlist"],
    )
    assert result.composite_count == int(reference["composite_count"].reshape(-1)[0])

    memberships, lengths = _python_membership_padded(
        result, reference["membership_records"]
    )
    assert np.array_equal(memberships, reference["membership_records"])
    composite_count = result.composite_count
    assert np.array_equal(
        lengths,
        reference["membership_lengths"].reshape(-1)[:composite_count],
    )
    assert np.array_equal(
        _python_connection_records(result, reference["connection_records"]),
        reference["connection_records"],
    )
    assert np.array_equal(
        np.asarray([trace.connection_count for trace in result.traces], dtype=np.float64),
        reference["connection_lengths"].reshape(-1)[:composite_count],
    )

    # These arrays are the loop's observable state, not unordered graph
    # summaries.  They catch updates made in the wrong source-order position.
    assert np.array_equal(result.lpts_initial_matlab, reference["Lpts_initial"])
    assert np.array_equal(result.lpts_final_matlab, reference["Lpts_final"])
    assert np.array_equal(result.n_of_lives_initial, reference["NofLives_initial"].reshape(-1))
    assert np.array_equal(result.n_of_lives_final, reference["NofLives_final"].reshape(-1))
    assert np.array_equal(result.label_list_final, reference["label_list_final"].reshape(-1))
    assert np.array_equal(
        result.new_partner_list_initial, reference["new_partner_list_initial"]
    )
    assert np.array_equal(result.new_partner_list_final, reference["new_partner_list_final"])
    if case_name == "one_valid_partner":
        assert np.count_nonzero(reference["raw_partner_candidates"]) == 3
        assert np.count_nonzero(reference["new_partner_list_input"]) == 2
    assert np.array_equal(
        result.iteration_n_of_lives_before, reference["state_noflives_before"]
    )
    assert np.array_equal(
        result.iteration_n_of_lives_after, reference["state_noflives_after"]
    )
    assert np.array_equal(
        result.iteration_label_list_after, reference["state_label_after"]
    )
    assert np.array_equal(
        result.iteration_lpts_labels_after,
        reference["state_lpts_label_after"],
    )


def test_stage_07_grouped_matches_existing_stage_fixture_exactly() -> None:
    stage04 = _load(STAGE_ROOT / "stage_04_fragments.npz")
    stage06 = _load(STAGE_ROOT / "stage_06_pairing.npz")
    reference = _load(STAGE_ROOT / "stage_07_grouped.npz")

    tips = stage06["all_tips"].copy()
    tips[:, 0:2] -= 1.0
    tips[:, 4:6] -= 1.0
    result = group_fragments(
        tips,
        stage04["L"],
        num_fragments=int(stage04["num"].reshape(-1)[0]),
    )

    assert result.composite_count == 1
    assert np.array_equal(result.all_filament_matlab, reference["all_filament"])
    assert np.array_equal(result.all_connects_matlab, reference["all_connects"])
    assert np.array_equal(
        result.all_connects_shortlist_matlab,
        reference["all_connects_shortlist"],
    )
    assert [membership.tolist() for membership in result.fragment_memberships] == [[1.0, 2.0]]


def test_fragment_pixel_table_keeps_matlab_find_column_major_order() -> None:
    labels = np.zeros((3, 4), dtype=np.float64)
    labels[2, 0] = 1
    labels[0, 1] = 2
    labels[1, 1] = 2
    labels[0, 3] = 1
    table = build_fragment_pixel_table(labels)
    assert np.array_equal(
        table,
        np.array(
            [
                [2.0, 0.0, 1.0],
                [0.0, 1.0, 2.0],
                [1.0, 1.0, 2.0],
                [0.0, 3.0, 1.0],
            ]
        ),
    )
