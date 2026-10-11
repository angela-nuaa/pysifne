from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pytest

from pysifne.fragments import (
    _large_cur_flag_from_neighbor_counts,
    _neighbor_counts,
    extract_stage_04_fragments,
    matlab_find,
    remove_border_margin,
)
from pysifne.fragments_accel import (
    cuda_large_cur_flags,
    cuda_neighbor_counts,
    cuda_square_mask,
    cuda_thin,
    numba_large_cur_flag_from_neighbor_counts,
)
from pysifne.lft_oft_accel import cuda_available, cuda_lft_oft
from pysifne.segmentation import bwmorph_thin


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = (
    REPO_ROOT
    / "tests" / "fixtures"
    / "core"
    / "npz"
)


def _load_stage(name: str) -> dict[str, np.ndarray]:
    with np.load(FIXTURE_ROOT / name, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _assert_diagnostics_exact(
    actual: list[dict[str, object]], expected: list[dict[str, object]]
) -> None:
    assert len(actual) == len(expected)
    for actual_iteration, expected_iteration in zip(actual, expected, strict=True):
        assert actual_iteration.keys() == expected_iteration.keys()
        for key in actual_iteration:
            actual_value = actual_iteration[key]
            expected_value = expected_iteration[key]
            if isinstance(actual_value, np.ndarray):
                assert isinstance(expected_value, np.ndarray)
                assert np.array_equal(actual_value, expected_value), key
            else:
                assert actual_value == expected_value, key


def test_cuda_itergen_lft_oft_callback_preserves_stage04_exactly() -> None:
    pytest.importorskip("numba")
    if not cuda_available():
        pytest.skip("CUDA device is not available")

    inputs = _load_stage("stage_00_input.npz")
    raw_skeleton = _load_stage("stage_03_rawske.npz")
    segmentation = _load_stage("stage_02_bw.npz")

    kwargs = dict(
        iteration=1,
        junction_size=7,
        iteration_remove_radius=3,
        minimum_length=6,
    )
    reference = extract_stage_04_fragments(
        inputs["OriginImg"],
        raw_skeleton["RawSke"],
        float(inputs["R"][0, 0]),
        inputs["ROI_Mask"],
        float(inputs["NofOrientations_FT"][0, 0]),
        float(segmentation["ThresholdUsed"][0, 0]),
        **kwargs,
    )
    cuda_impl = partial(
        cuda_lft_oft,
        device=0,
        threads_per_block=256,
        factor_mode="matlab_windows",
    )
    cuda_neighbor_impl = partial(
        cuda_neighbor_counts,
        device=0,
        threads_per_block=256,
    )
    cuda_mask2_impl = partial(
        cuda_square_mask,
        device=0,
        threads_per_block=256,
    )
    actual = extract_stage_04_fragments(
        inputs["OriginImg"],
        raw_skeleton["RawSke"],
        float(inputs["R"][0, 0]),
        inputs["ROI_Mask"],
        float(inputs["NofOrientations_FT"][0, 0]),
        float(segmentation["ThresholdUsed"][0, 0]),
        lft_oft_impl=cuda_impl,
        neighbor_count_impl=cuda_neighbor_impl,
        label_components_impl=None,
        mask2_impl=cuda_mask2_impl,
        **kwargs,
    )

    for key, expected in reference.items():
        if key == "iteration_diagnostics":
            _assert_diagnostics_exact(actual[key], expected)
        else:
            assert np.array_equal(actual[key], expected), key


def test_cuda_neighbor_counts_preserves_matlab_coordinate_order() -> None:
    pytest.importorskip("numba")
    if not cuda_available():
        pytest.skip("CUDA device is not available")
    raw_skeleton = _load_stage("stage_03_rawske.npz")["RawSke"]
    interior = remove_border_margin(raw_skeleton, 10)
    coordinates = matlab_find(interior)
    expected = _neighbor_counts(interior, coordinates)
    actual = cuda_neighbor_counts(interior, coordinates)
    assert np.array_equal(actual, expected)


def test_cuda_thin_matches_readable_guo_hall_on_direct_inputs() -> None:
    pytest.importorskip("numba")
    if not cuda_available():
        pytest.skip("CUDA device is not available")
    cases = {
        "zero": np.zeros((17, 19), dtype=bool),
        "one": np.ones((17, 19), dtype=bool),
        "impulse": np.pad(np.ones((1, 1), dtype=bool), 15),
        "cross": np.zeros((41, 37), dtype=bool),
        "diagonals": np.eye(47, dtype=bool),
    }
    cases["cross"][20, 4:33] = True
    cases["cross"][5:37, 18] = True
    cases["diagonals"] |= np.fliplr(cases["diagonals"])

    rng = np.random.default_rng(20260916)
    cases["random"] = rng.random((101, 97)) < 0.5
    with np.load(FIXTURE_ROOT / "stage_02_bw.npz", allow_pickle=False) as data:
        cases["stage_02_BW"] = data["BW"].astype(bool, copy=False)

    for name, value in cases.items():
        expected = bwmorph_thin(value)
        actual = cuda_thin(value, device=0, threads_per_block=256)
        assert np.array_equal(actual, expected), name


def test_cuda_thin_preserves_stage04_exactly() -> None:
    pytest.importorskip("numba")
    if not cuda_available():
        pytest.skip("CUDA device is not available")
    inputs = _load_stage("stage_00_input.npz")
    raw_skeleton = _load_stage("stage_03_rawske.npz")
    segmentation = _load_stage("stage_02_bw.npz")
    kwargs = dict(
        iteration=1,
        junction_size=7,
        iteration_remove_radius=3,
        minimum_length=6,
    )
    reference = extract_stage_04_fragments(
        inputs["OriginImg"],
        raw_skeleton["RawSke"],
        float(inputs["R"][0, 0]),
        inputs["ROI_Mask"],
        float(inputs["NofOrientations_FT"][0, 0]),
        float(segmentation["ThresholdUsed"][0, 0]),
        **kwargs,
    )
    accelerated = extract_stage_04_fragments(
        inputs["OriginImg"],
        raw_skeleton["RawSke"],
        float(inputs["R"][0, 0]),
        inputs["ROI_Mask"],
        float(inputs["NofOrientations_FT"][0, 0]),
        float(segmentation["ThresholdUsed"][0, 0]),
        lft_oft_impl=partial(
            cuda_lft_oft,
            device=0,
            threads_per_block=256,
            factor_mode="matlab_windows",
        ),
        neighbor_count_impl=partial(
            cuda_neighbor_counts,
            device=0,
            threads_per_block=256,
        ),
        label_components_impl=None,
        mask2_impl=partial(
            cuda_square_mask,
            device=0,
            threads_per_block=256,
        ),
        thin_impl=partial(cuda_thin, device=0, threads_per_block=256),
        large_cur_flags_impl=partial(
            cuda_large_cur_flags,
            device=0,
            threads_per_block=256,
        ),
        **kwargs,
    )
    for key, expected in reference.items():
        if key == "iteration_diagnostics":
            _assert_diagnostics_exact(accelerated[key], expected)
        else:
            assert np.array_equal(accelerated[key], expected), key


def test_cuda_large_cur_flags_matches_readable_reference_fixtures() -> None:
    pytest.importorskip("numba")
    if not cuda_available():
        pytest.skip("CUDA device is not available")
    with np.load(FIXTURE_ROOT.parent.parent / "itergen-fragment" / "npz" / "large_cur_flag.npz", allow_pickle=False) as data:
        cases = (
            data["LargeCurStraight"],
            data["LargeCurLShape"],
            data["LargeCurUShape"],
        )
        expected = data["LargeCurFlagExpected"].ravel().astype(np.int64)

    actual = []
    for points, expected_flag in zip(cases, expected, strict=True):
        points = np.asarray(points, dtype=np.int64)
        label_image = np.zeros(
            (int(points[:, 0].max()) + 2, int(points[:, 1].max()) + 2),
            dtype=np.float64,
        )
        label_image[points[:, 0] - 1, points[:, 1] - 1] = 1
        flags = cuda_large_cur_flags(label_image, [points])
        actual.append(int(flags[0]))
        assert int(flags[0]) == int(expected_flag)
    assert np.array_equal(np.asarray(actual, dtype=np.int64), expected)


def test_numba_large_cur_tail_is_exact_on_reference_inputs() -> None:
    pytest.importorskip("numba")
    with np.load(
        FIXTURE_ROOT.parent.parent
        / "itergen-fragment"
        / "npz"
        / "large_cur_flag.npz",
        allow_pickle=False,
    ) as data:
        cases = (
            data["LargeCurStraight"],
            data["LargeCurLShape"],
            data["LargeCurUShape"],
        )

    for points in cases:
        points = np.asarray(points, dtype=np.int64)
        shifted_rows = points[:, 0] - points[:, 0].min() + 2
        shifted_columns = points[:, 1] - points[:, 1].min() + 2
        label_image = np.zeros(
            (int(shifted_rows.max()) + 1, int(shifted_columns.max()) + 1),
            dtype=np.float64,
        )
        label_image[shifted_rows - 1, shifted_columns - 1] = 1
        counts = np.asarray(
            [
                np.sum(
                    label_image[
                        int(row) - 2 : int(row) + 1,
                        int(column) - 2 : int(column) + 1,
                    ]
                )
                for row, column in zip(shifted_rows, shifted_columns, strict=True)
            ],
            dtype=np.int64,
        )
        expected = _large_cur_flag_from_neighbor_counts(points, counts)
        actual = numba_large_cur_flag_from_neighbor_counts(points, counts)
        assert actual == expected
