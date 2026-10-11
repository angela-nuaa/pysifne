from __future__ import annotations

from pathlib import Path

import numpy as np

from pysifne.lft_oft import C_PI, lft_oft


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "core"


def _load_golden() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    input_path = FIXTURE_ROOT / "npz" / "stage_00_input.npz"
    golden_path = FIXTURE_ROOT / "npz" / "stage_01_oft.npz"
    if not input_path.is_file() or not golden_path.is_file():
        raise AssertionError(
            "Missing SIFNE golden fixture. Expected stage_00_input.npz and "
            f"stage_01_oft.npz under {FIXTURE_ROOT}."
        )
    with np.load(input_path, allow_pickle=False) as input_data:
        inputs = {key: input_data[key] for key in input_data.files}
    with np.load(golden_path, allow_pickle=False) as golden_data:
        golden = {key: golden_data[key] for key in golden_data.files}
    return inputs, golden


def test_lft_oft_matches_stage_01_golden() -> None:
    inputs, golden = _load_golden()
    assert C_PI == 3.141593

    oft, lft, orientations = lft_oft(
        inputs["OriginImg_Margin"],
        float(inputs["R"][0, 0]),
        float(inputs["NofOrientations_FT"][0, 0]),
        inputs["ROI_Mask"],
    )

    assert oft.shape == golden["OFT_Img"].shape
    assert lft.shape == golden["LFT_Img"].shape
    assert orientations.shape == golden["LFT_Orientations"].shape
    assert oft.dtype == np.float64
    assert lft.dtype == np.float64
    assert orientations.dtype == np.float64

    # Tight absolute tolerances cover only platform/libm last-bit differences;
    # there is no relative tolerance and no algorithmic slack.
    np.testing.assert_allclose(oft, golden["OFT_Img"], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(lft, golden["LFT_Img"], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        orientations,
        golden["LFT_Orientations"],
        rtol=0.0,
        atol=1e-12,
    )
