from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numba")

from pysifne.tips import register_filament_tips  # noqa: E402
from pysifne.tips_accel import (  # noqa: E402
    numba_detect_endpoints,
    numba_local_centroids,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TIP_ROOT = REPO_ROOT / "data" / "derived" / "reference-freezing" / "tipreg-20260915" / "npz"
STAGE_ROOT = REPO_ROOT / "data" / "derived" / "reference-freezing" / "sifne-fixture-20260915-r4" / "npz"


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _assert_same(readable, accelerated) -> None:
    assert np.array_equal(accelerated.endpoint_mask, readable.endpoint_mask)
    assert np.array_equal(accelerated.endpoint_coordinates, readable.endpoint_coordinates)
    assert np.array_equal(accelerated.tip_labels, readable.tip_labels)
    assert np.array_equal(accelerated.temp_info, readable.temp_info)
    assert np.array_equal(accelerated.all_tips, readable.all_tips)


def test_numba_tip_helpers_match_readable_direct_fixtures() -> None:
    for path in sorted(TIP_ROOT.glob("*.npz")):
        if path.stem == "check_dir_between2pts":
            continue
        reference = _load(path)
        readable = register_filament_tips(
            reference["AllFragments"],
            reference["L"],
            float(reference["RegR"].reshape(-1)[0]),
        )
        accelerated = register_filament_tips(
            reference["AllFragments"],
            reference["L"],
            float(reference["RegR"].reshape(-1)[0]),
            endpoint_mask_impl=numba_detect_endpoints,
            centroid_impl=numba_local_centroids,
        )
        _assert_same(readable, accelerated)


def test_numba_tip_helpers_match_stage05_exactly() -> None:
    fragments = _load(STAGE_ROOT / "stage_04_fragments.npz")
    reference = _load(STAGE_ROOT / "stage_05_tips.npz")
    accelerated = register_filament_tips(
        fragments["AllFragments"],
        fragments["L"],
        float(reference["RegR"].reshape(-1)[0]),
        endpoint_mask_impl=numba_detect_endpoints,
        centroid_impl=numba_local_centroids,
    )
    assert np.array_equal(accelerated.all_tips_matlab, reference["all_tips"])
    assert np.array_equal(accelerated.temp_info_matlab, reference["tempInfo"])
