from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pysifne.fragments import iterative_fragment_extraction, label_components, large_cur_flag


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = (
    REPO_ROOT
    / "tests" / "fixtures"
    / "itergen-fragment"
    / "npz"
)

CASE_NAMES = (
    "straight_line",
    "junction_cross",
    "curved_filament",
    "short_fragment",
    "large_curvature",
)


def _component_signatures(labels: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], frozenset[int]]]:
    value = np.asarray(labels)
    height, _ = value.shape
    signatures: list[tuple[int, tuple[int, int, int, int], frozenset[int]]] = []
    for label in sorted(int(item) for item in np.unique(value) if item != 0):
        rows, columns = np.nonzero(value == label)
        linear = rows + columns * height
        bbox = (
            int(rows.min()),
            int(rows.max()),
            int(columns.min()),
            int(columns.max()),
        )
        signatures.append((int(rows.size), bbox, frozenset(int(item) for item in linear)))
    return sorted(signatures)


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_itergen_fragment_matches_independent_matlab_fixture(case_name: str) -> None:
    path = FIXTURE_ROOT / f"{case_name}.npz"
    if not path.is_file():
        raise AssertionError(f"Missing MATLAB IterGenFragment fixture: {path}")

    with np.load(path, allow_pickle=False) as data:
        actual, _ = iterative_fragment_extraction(
            data["OriginImg"],
            data["AllFragments"],
            int(data["iteration"][0, 0]),
            float(data["R"][0, 0]),
            data["ROI_Mask"],
            float(data["Nangles"][0, 0]),
            float(data["Iter_RemoveR"][0, 0]),
            float(data["Thresh"][0, 0]),
            float(data["R_Junc"][0, 0]),
            float(data["MIN_FragmentLength"][0, 0]),
        )
        expected = data["ExpectedAllFragments"]
        assert np.array_equal(actual, expected), case_name

        labels, num = label_components(actual, connectivity=8)
        assert num == int(data["ExpectedNum"][0, 0]), case_name
        expected_labels = data["ExpectedL"]
        if not np.array_equal(labels, expected_labels):
            assert _component_signatures(labels) == _component_signatures(expected_labels)


def test_large_cur_flag_matches_matlab_coordinate_fixtures() -> None:
    path = FIXTURE_ROOT / "large_cur_flag.npz"
    if not path.is_file():
        raise AssertionError(f"Missing MATLAB LargeCurFlag fixture: {path}")

    with np.load(path, allow_pickle=False) as data:
        cases = (
            data["LargeCurStraight"],
            data["LargeCurLShape"],
            data["LargeCurUShape"],
        )
        expected = data["LargeCurFlagExpected"].ravel()
        actual = np.asarray([large_cur_flag(case) for case in cases], dtype=np.float64)
        assert np.array_equal(actual, expected)
