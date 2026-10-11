from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pysifne.lft_oft_accel import cuda_available
from pysifne.pairing import generate_distance_candidates, pair_filament_tips
from pysifne.pairing_accel import (
    cpu_pairwise_directions_exact,
    cuda_pairwise_distances,
    numba_pairing_pools,
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
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _python_inputs(reference: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    tips = reference["all_tips_input"].copy()
    tips[:, 0:2] -= 1.0
    tips[:, 4:6] -= 1.0
    base = reference["base_loc"].reshape(2).copy() - 1.0
    return tips, base


@pytest.mark.parametrize("case_name", DIRECT_CASES)
def test_cuda_pairwise_distance_row_matches_matlab_distance_fixture(case_name: str) -> None:
    pytest.importorskip("numba")
    if not cuda_available():
        pytest.skip("CUDA device is not available")
    reference = _load(PAIRING_ROOT / f"{case_name}.npz")
    tips, base = _python_inputs(reference)
    matrix = cuda_pairwise_distances(tips[:, 0:2])
    base_index = int(reference["base_index"].reshape(-1)[0]) - 1
    actual, _ = generate_distance_candidates(
        tips,
        base,
        float(reference["FanR"][0, 0]),
        distance_values=matrix[base_index, :],
    )
    assert np.array_equal(actual, reference["All_Dist"].reshape(-1)), case_name


def test_cuda_distance_matrix_preserves_stage06_pairing_exactly() -> None:
    pytest.importorskip("numba")
    if not cuda_available():
        pytest.skip("CUDA device is not available")
    stage04 = _load(STAGE_ROOT / "stage_04_fragments.npz")
    stage05 = _load(STAGE_ROOT / "stage_05_tips.npz")
    reference = _load(STAGE_ROOT / "stage_06_pairing.npz")
    tips = stage05["all_tips"].copy()
    tips[:, 0:2] -= 1.0
    tips[:, 4:6] -= 1.0
    matrix = cuda_pairwise_distances(tips[:, 0:2])
    actual = pair_filament_tips(
        tips,
        stage04["L"],
        float(reference["FanAngle"][0, 0]),
        float(reference["FanR"][0, 0]),
        reference["MIN_Info"].reshape(-1),
        float(reference["C1"][0, 0]),
        float(reference["C3"][0, 0]),
        distance_matrix=matrix,
    )
    assert np.array_equal(actual.new_partner_list, reference["new_partner_list"])
    assert np.array_equal(actual.all_tips_pre_bidir_matlab, reference["all_tips_pre_bidir"])
    assert np.array_equal(actual.all_tips_post_bidir_matlab, reference["all_tips_post_bidir"])
    assert np.array_equal(actual.all_tips_matlab, reference["all_tips"])


def test_numba_batch_pairing_pool_matches_each_readable_local_search_row() -> None:
    pytest.importorskip("numba")
    if not cuda_available():
        pytest.skip("CUDA device is not available")
    stage04 = _load(STAGE_ROOT / "stage_04_fragments.npz")
    stage05 = _load(STAGE_ROOT / "stage_05_tips.npz")
    reference = _load(STAGE_ROOT / "stage_06_pairing.npz")
    tips = stage05["all_tips"].copy()
    tips[:, 0:2] -= 1.0
    tips[:, 4:6] -= 1.0
    distances = cuda_pairwise_distances(tips[:, 0:2])
    directions = cpu_pairwise_directions_exact(tips[:, 0:2])
    kwargs = dict(
        fan_angle=float(reference["FanAngle"][0, 0]),
        fan_r=float(reference["FanR"][0, 0]),
        min_info=reference["MIN_Info"].reshape(-1),
        c1_weight=float(reference["C1"][0, 0]),
        c3_weight=float(reference["C3"][0, 0]),
        distance_matrix=distances,
        direction_matrix=directions,
        vectorized_local_search=True,
    )
    readable = pair_filament_tips(tips, stage04["L"], **kwargs)
    accelerated = pair_filament_tips(
        tips,
        stage04["L"],
        **kwargs,
        pool_impl=numba_pairing_pools,
        retain_local_results=False,
    )
    expected_pools = np.vstack(
        [item.index_global_pool for item in readable.local_search_results]
    )
    assert np.array_equal(accelerated.new_partner_list_untrimmed, expected_pools)
    assert np.array_equal(accelerated.new_partner_list, readable.new_partner_list)
    assert np.array_equal(
        accelerated.all_tips_pre_bidir_matlab,
        readable.all_tips_pre_bidir_matlab,
    )
    assert np.array_equal(
        accelerated.all_tips_post_bidir_matlab,
        readable.all_tips_post_bidir_matlab,
    )
    assert np.array_equal(accelerated.all_tips_matlab, readable.all_tips_matlab)


@pytest.mark.parametrize("case_name", DIRECT_CASES)
def test_numba_batch_pairing_pool_matches_direct_fixture(case_name: str) -> None:
    pytest.importorskip("numba")
    if not cuda_available():
        pytest.skip("CUDA device is not available")
    reference = _load(PAIRING_ROOT / f"{case_name}.npz")
    tips, _ = _python_inputs(reference)
    distances = cuda_pairwise_distances(tips[:, 0:2])
    directions = cpu_pairwise_directions_exact(tips[:, 0:2])
    common = dict(
        fan_angle=float(reference["fanAngle"][0, 0]),
        fan_r=float(reference["r"][0, 0]),
        min_info=reference["MIN_Info"].reshape(-1),
        c1_weight=float(reference["C1weight"][0, 0]),
        c3_weight=float(reference["C3weight"][0, 0]),
        distance_matrix=distances,
        direction_matrix=directions,
        vectorized_local_search=True,
    )
    # The direct fixture's L_GlobalIndex has the exact image shape required by
    # the compatibility call; its contents do not affect the precomputed-matrix
    # path after tip IDs have been built by pair_filament_tips.
    label_image = reference["L_GlobalIndex"]
    readable = pair_filament_tips(tips, label_image, **common)
    accelerated = pair_filament_tips(
        tips,
        label_image,
        **common,
        pool_impl=numba_pairing_pools,
        retain_local_results=False,
    )
    expected = np.vstack(
        [item.index_global_pool for item in readable.local_search_results]
    )
    assert np.array_equal(accelerated.new_partner_list_untrimmed, expected)
    assert np.array_equal(accelerated.new_partner_list, readable.new_partner_list)
    assert np.array_equal(accelerated.all_tips_pre_bidir_matlab, readable.all_tips_pre_bidir_matlab)
    assert np.array_equal(accelerated.all_tips_post_bidir_matlab, readable.all_tips_post_bidir_matlab)
    assert np.array_equal(accelerated.all_tips_matlab, readable.all_tips_matlab)
