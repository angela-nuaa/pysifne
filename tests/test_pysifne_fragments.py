from __future__ import annotations

from pathlib import Path

import numpy as np

from pysifne.fragments import (
    extract_stage_04_fragments,
    label_components,
    remove_junctions_and_isolated,
    remove_short_fragments,
)


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


def _component_signatures(labels: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], frozenset[int]]]:
    """Return label-independent component size, bbox, and F-order pixels."""

    value = np.asarray(labels)
    height, width = value.shape
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


def _assert_label_permutation_invariant(actual: np.ndarray, expected: np.ndarray) -> None:
    actual_signatures = _component_signatures(actual)
    expected_signatures = _component_signatures(expected)
    assert len(actual_signatures) == len(expected_signatures)
    assert actual_signatures == expected_signatures


def test_stage_04_fragments_matches_matlab_exactly() -> None:
    inputs = _load_stage("stage_00_input.npz")
    raw_skeleton = _load_stage("stage_03_rawske.npz")
    segmentation = _load_stage("stage_02_bw.npz")
    reference = _load_stage("stage_04_fragments.npz")

    actual = extract_stage_04_fragments(
        inputs["OriginImg"],
        raw_skeleton["RawSke"],
        float(inputs["R"][0, 0]),
        inputs["ROI_Mask"],
        float(inputs["NofOrientations_FT"][0, 0]),
        float(segmentation["ThresholdUsed"][0, 0]),
    )

    exact_keys = (
        "R_Junc",
        "Size_Junc",
        "MIN_FragmentLength",
        "AllFragments",
        "Allpts",
        "CrPts",
        "num",
        "RawCrPts",
        "AllFragments_before_iteration",
        "num_before_iteration",
    )
    for key in exact_keys:
        assert np.array_equal(actual[key], reference[key]), key

    for key in ("L", "L_before_iteration"):
        if not np.array_equal(actual[key], reference[key]):
            # Exact equality is the primary acceptance criterion.  This
            # fallback is intentionally limited to a pure label permutation.
            _assert_label_permutation_invariant(actual[key], reference[key])


def test_junction_intermediates_and_component_counts() -> None:
    inputs = _load_stage("stage_00_input.npz")
    raw_skeleton = _load_stage("stage_03_rawske.npz")
    segmentation = _load_stage("stage_02_bw.npz")
    reference = _load_stage("stage_04_fragments.npz")

    actual = extract_stage_04_fragments(
        inputs["OriginImg"],
        raw_skeleton["RawSke"],
        float(inputs["R"][0, 0]),
        inputs["ROI_Mask"],
        float(inputs["NofOrientations_FT"][0, 0]),
        float(segmentation["ThresholdUsed"][0, 0]),
    )

    # The frozen fixture has no crossing pixels; after the exact junction and
    # singleton pass, its pre-iteration skeleton is unchanged by bwareaopen.
    assert np.array_equal(actual["junctions"], reference["CrPts"])
    assert np.array_equal(actual["RawCrPts"], reference["RawCrPts"])
    assert np.array_equal(
        actual["AllFragments_after_junction"],
        reference["AllFragments_before_iteration"],
    )
    assert np.array_equal(actual["num_after_junction"], reference["num_before_iteration"])
    assert np.array_equal(actual["num_before_iteration"], reference["num_before_iteration"])

    assert int(actual["junctions"].shape[0]) == 0
    assert int(np.count_nonzero(actual["AllFragments_after_junction"])) == 75
    assert int(actual["num_before_iteration"][0, 0]) == 2


def test_junction_cleanup_matches_independent_matlab_fixture() -> None:
    path = (
        REPO_ROOT
        / "data"
        / "derived"
        / "reference-freezing"
        / "junction-cleanup-20260915"
        / "npz"
        / "junction_cleanup.npz"
    )
    if not path.is_file():
        raise AssertionError(f"Missing MATLAB junction fixture: {path}")

    with np.load(path, allow_pickle=False) as reference:
        actual = remove_junctions_and_isolated(
            reference["RawSke"],
            margin_radius=float(reference["R"][0, 0]),
            junction_radius=float(reference["R_Junc"][0, 0]),
        )
        assert np.array_equal(actual.margin_removed, reference["MarginRemoved"])
        assert np.array_equal(actual.junctions, reference["CrPts"])
        assert np.array_equal(actual.junction_removed, reference["JunctionRemoved"])
        assert np.array_equal(actual.isolated_pixels, reference["SinglePts"])
        assert np.array_equal(actual.skeleton, reference["AfterSingletonRemoval"])
        assert np.array_equal(actual.labels, reference["L_after_junction"])
        assert actual.num_components == int(reference["num_after_junction"][0, 0])

        after_short = remove_short_fragments(
            actual.skeleton,
            float(reference["MIN_FragmentLength"][0, 0]),
            connectivity=8,
        )
        labels_after_short, num_after_short = label_components(after_short, 8)
        assert np.array_equal(after_short, reference["AfterShortFilter"])
        assert np.array_equal(labels_after_short, reference["L_after_short_filter"])
        assert num_after_short == int(reference["num_after_short_filter"][0, 0])

        # Explicitly retain the requested intermediate observables.
        assert actual.junctions.shape[0] == 5
        assert actual.num_components == 5
        assert num_after_short == 4
