from __future__ import annotations

from pathlib import Path

import numpy as np

from pysifne.segmentation import bwmorph_thin, graythresh, imbinarize, mat2gray


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = (
    REPO_ROOT
    / "data"
    / "derived"
    / "reference-freezing"
    / "sifne-fixture-20260915-r4"
    / "npz"
)


def _load_stage(name: str) -> dict[str, np.ndarray]:
    path = FIXTURE_ROOT / name
    if not path.is_file():
        raise AssertionError(f"Missing MATLAB reference fixture: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _skeleton_diagnostics(actual: np.ndarray, expected: np.ndarray) -> str:
    actual = actual.astype(bool, copy=False)
    expected = expected.astype(bool, copy=False)
    difference = np.argwhere(actual != expected)
    intersection = np.count_nonzero(actual & expected)
    union = np.count_nonzero(actual | expected)
    actual_count = np.count_nonzero(actual)
    expected_count = np.count_nonzero(expected)

    def degree(image: np.ndarray) -> np.ndarray:
        padded = np.pad(image.astype(np.uint8), 1, mode="constant")
        result = np.zeros_like(image, dtype=np.uint8)
        for row_offset in (-1, 0, 1):
            for col_offset in (-1, 0, 1):
                if row_offset == 0 and col_offset == 0:
                    continue
                result += padded[
                    1 + row_offset : 1 + row_offset + image.shape[0],
                    1 + col_offset : 1 + col_offset + image.shape[1],
                ]
        return result

    actual_degree = degree(actual)
    expected_degree = degree(expected)
    actual_endpoints = np.count_nonzero(actual & (actual_degree == 1))
    expected_endpoints = np.count_nonzero(expected & (expected_degree == 1))
    actual_junctions = np.count_nonzero(actual & (actual_degree >= 3))
    expected_junctions = np.count_nonzero(expected & (expected_degree >= 3))

    first = difference[0].tolist() if difference.size else None
    return (
        f"difference_pixels={difference.shape[0]}, first_zero_based_row_col={first}, "
        f"iou={intersection / union if union else 1.0:.17g}, "
        f"dice={2 * intersection / (actual_count + expected_count) if actual_count + expected_count else 1.0:.17g}, "
        f"endpoints_actual_expected={actual_endpoints}/{expected_endpoints}, "
        f"junctions_actual_expected={actual_junctions}/{expected_junctions}"
    )


def test_bw_matches_stage_02_matlab_exactly() -> None:
    oft = _load_stage("stage_01_oft.npz")
    reference = _load_stage("stage_02_bw.npz")

    normalized = mat2gray(oft["OFT_Img"])
    otsu_threshold = graythresh(normalized)
    threshold_used = 1.42 * otsu_threshold
    if threshold_used >= 1.0:
        threshold_used = otsu_threshold
    binary = imbinarize(normalized)

    assert np.array_equal(normalized, reference["I"])
    assert otsu_threshold == float(reference["OtsuThreshold"][0, 0])
    assert threshold_used == float(reference["ThresholdUsed"][0, 0])
    assert np.array_equal(binary, reference["BW"])


def test_rawske_matches_stage_03_matlab_exactly() -> None:
    bw = _load_stage("stage_02_bw.npz")["BW"]
    reference = _load_stage("stage_03_rawske.npz")["RawSke"]

    actual = bwmorph_thin(bw)
    if not np.array_equal(actual, reference):
        raise AssertionError(_skeleton_diagnostics(actual, reference))
