from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pysifne.lft_oft import lft_oft


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = (
    REPO_ROOT
    / "tests" / "fixtures"
    / "lft-oft"
    / "npz"
)

CASE_NAMES = (
    "all_zero",
    "impulse",
    "horizontal_line",
    "vertical_line",
    "diagonal_45_line",
    "boundary_near",
    "non_all_one_roi",
    "random_r1_n5",
    "random_r2_n20",
    "random_r4_n13",
)


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_lft_oft_matches_matlab_differential_fixture(case_name: str) -> None:
    path = FIXTURE_ROOT / f"{case_name}.npz"
    if not path.is_file():
        raise AssertionError(f"Missing MATLAB differential fixture: {path}")

    with np.load(path, allow_pickle=False) as data:
        oft, lft, orientations = lft_oft(
            data["input_image"],
            float(data["R"][0, 0]),
            float(data["Nangles"][0, 0]),
            data["roi_mask"],
        )
        actual = {
            "OFT_Img": oft,
            "LFT_Img": lft,
            "LFT_Orientations": orientations,
        }
        for name, value in actual.items():
            expected = data[name]
            if not np.array_equal(value, expected):
                difference = np.argwhere(value != expected)
                first = difference[0].tolist() if difference.size else None
                max_abs = float(np.max(np.abs(value - expected)))
                raise AssertionError(
                    f"{case_name}/{name} differs: first_zero_based_row_col={first}, "
                    f"difference_pixels={difference.shape[0]}, max_abs={max_abs:.17g}"
                )
