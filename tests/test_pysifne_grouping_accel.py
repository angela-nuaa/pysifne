from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pysifne.grouping import group_fragments


numba = pytest.importorskip("numba")
from pysifne.grouping_accel import numba_trace_composite  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
GROUPING_ROOT = (
    REPO_ROOT
    / "data"
    / "derived"
    / "reference-freezing"
    / "grouping-20260915"
    / "npz"
)
STAGE_ROOT = (
    REPO_ROOT
    / "data"
    / "derived"
    / "reference-freezing"
    / "sifne-fixture-20260915-r4"
    / "npz"
)


def _tips_in_python(value: np.ndarray) -> np.ndarray:
    tips = np.asarray(value, dtype=np.float64).copy()
    if tips.shape[0]:
        tips[:, 0:2] -= 1.0
        tips[:, 4:6] -= 1.0
    return tips


def _assert_same(a, b) -> None:
    assert np.array_equal(a.all_filament, b.all_filament)
    assert np.array_equal(a.all_connects, b.all_connects)
    assert np.array_equal(a.all_connects_shortlist, b.all_connects_shortlist)
    assert np.array_equal(a.lpts_initial, b.lpts_initial)
    assert np.array_equal(a.lpts_final, b.lpts_final)
    assert np.array_equal(a.n_of_lives_initial, b.n_of_lives_initial)
    assert np.array_equal(a.n_of_lives_final, b.n_of_lives_final)
    assert np.array_equal(a.label_list_final, b.label_list_final)
    assert np.array_equal(a.new_partner_list_initial, b.new_partner_list_initial)
    assert np.array_equal(a.new_partner_list_final, b.new_partner_list_final)
    assert np.array_equal(a.iteration_n_of_lives_before, b.iteration_n_of_lives_before)
    assert np.array_equal(a.iteration_n_of_lives_after, b.iteration_n_of_lives_after)
    assert np.array_equal(a.iteration_label_list_after, b.iteration_label_list_after)
    assert np.array_equal(a.iteration_lpts_labels_after, b.iteration_lpts_labels_after)
    assert a.composite_count == b.composite_count
    assert [x.seed_fragment_id for x in a.traces] == [x.seed_fragment_id for x in b.traces]
    for trace_a, trace_b in zip(a.traces, b.traces):
        assert np.array_equal(trace_a.fragment_membership, trace_b.fragment_membership)
        assert np.array_equal(trace_a.filament_pixels, trace_b.filament_pixels)
        assert np.array_equal(trace_a.connection_points, trace_b.connection_points)
        assert np.array_equal(trace_a.n_of_lives_before, trace_b.n_of_lives_before)
        assert np.array_equal(trace_a.n_of_lives_after, trace_b.n_of_lives_after)
        assert np.array_equal(trace_a.label_list_after, trace_b.label_list_after)
        assert np.array_equal(trace_a.partner_list_after, trace_b.partner_list_after)
        assert np.array_equal(trace_a.lpts_labels_after, trace_b.lpts_labels_after)


def test_numba_grouping_direct_fixtures_match_readable_reference() -> None:
    paths = sorted(GROUPING_ROOT.glob("*.npz"))
    if not paths:
        pytest.skip("grouping direct fixtures are not present on this host")
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            reference = {key: data[key] for key in data.files}
        tips = _tips_in_python(reference["all_tips_input"])
        num_fragments = int(reference["num_fragments"].reshape(-1)[0])
        readable = group_fragments(
            tips.copy(),
            reference["L"],
            num_fragments=num_fragments,
        )
        accelerated = group_fragments(
            tips.copy(),
            reference["L"],
            num_fragments=num_fragments,
            trace_impl=numba_trace_composite,
        )
        _assert_same(readable, accelerated)


def test_numba_grouping_stage07_matches_readable_reference() -> None:
    stage04 = STAGE_ROOT / "stage_04_fragments.npz"
    stage06 = STAGE_ROOT / "stage_06_pairing.npz"
    if not stage04.is_file() or not stage06.is_file():
        pytest.skip("stage-04/stage-06 reference fixture is not present on this host")
    with np.load(stage04, allow_pickle=False) as data:
        fragments = {key: data[key] for key in data.files}
    with np.load(stage06, allow_pickle=False) as data:
        pairing = {key: data[key] for key in data.files}
    tips = _tips_in_python(pairing["all_tips"])
    num_fragments = int(fragments["num"].reshape(-1)[0])
    readable = group_fragments(
        tips.copy(),
        fragments["L"],
        num_fragments=num_fragments,
    )
    accelerated = group_fragments(
        tips.copy(),
        fragments["L"],
        num_fragments=num_fragments,
        trace_impl=numba_trace_composite,
    )
    _assert_same(readable, accelerated)
