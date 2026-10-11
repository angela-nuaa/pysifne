from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pysifne.lft_oft_accel import cuda_available, cuda_lft_oft


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "core"
DIFFERENTIAL_ROOT = (
    REPO_ROOT
    / "tests" / "fixtures"
    / "lft-oft"
    / "npz"
)

DIFFERENTIAL_CASES = (
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


def test_cuda_lft_oft_matches_stage_01_exactly() -> None:
    pytest.importorskip("numba")
    if not cuda_available():
        pytest.skip("CUDA device is not available")
    with np.load(FIXTURE_ROOT / "npz" / "stage_00_input.npz", allow_pickle=False) as inputs:
        with np.load(FIXTURE_ROOT / "npz" / "stage_01_oft.npz", allow_pickle=False) as golden:
            actual = cuda_lft_oft(
                inputs["OriginImg_Margin"],
                float(inputs["R"][0, 0]),
                float(inputs["NofOrientations_FT"][0, 0]),
                inputs["ROI_Mask"],
            )
            for value, name in zip(actual, ("OFT_Img", "LFT_Img", "LFT_Orientations")):
                assert np.array_equal(value, golden[name]), name


@pytest.mark.parametrize("case_name", DIFFERENTIAL_CASES)
def test_cuda_lft_oft_matches_matlab_differential_exactly(case_name: str) -> None:
    pytest.importorskip("numba")
    if not cuda_available():
        pytest.skip("CUDA device is not available")
    path = DIFFERENTIAL_ROOT / f"{case_name}.npz"
    with np.load(path, allow_pickle=False) as data:
        actual = cuda_lft_oft(
            data["input_image"],
            float(data["R"][0, 0]),
            float(data["Nangles"][0, 0]),
            data["roi_mask"],
        )
        for value, name in zip(actual, ("OFT_Img", "LFT_Img", "LFT_Orientations")):
            assert np.array_equal(value, data[name]), f"{case_name}/{name}"
