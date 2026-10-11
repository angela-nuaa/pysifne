"""End-to-end MATLAB-free PySIFNE regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from pysifne import SIFNEConfig, analyze_mt, export_matlab_compatibility


FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "tests" / "fixtures"
    / "core"
)
PIPELINE_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "tests" / "fixtures"
    / "pipeline"
)


def _load_stage(name: str) -> dict[str, np.ndarray]:
    with np.load(FIXTURE_ROOT / "npz" / name, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _load_fixture_stage(fixture_root: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(fixture_root / "npz" / name, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _assert_core_stages_exact(result, fixture_root: Path) -> None:
    stage00 = _load_fixture_stage(fixture_root, "stage_00_input.npz")
    stage01 = _load_fixture_stage(fixture_root, "stage_01_oft.npz")
    stage02 = _load_fixture_stage(fixture_root, "stage_02_bw.npz")
    stage03 = _load_fixture_stage(fixture_root, "stage_03_rawske.npz")
    stage04 = _load_fixture_stage(fixture_root, "stage_04_fragments.npz")
    stage05 = _load_fixture_stage(fixture_root, "stage_05_tips.npz")
    stage06 = _load_fixture_stage(fixture_root, "stage_06_pairing.npz")
    stage07 = _load_fixture_stage(fixture_root, "stage_07_grouped.npz")
    stage08 = _load_fixture_stage(fixture_root, "stage_08_sorted.npz")

    assert np.array_equal(result.input_image, stage00["input_img"])
    assert np.array_equal(result.origin_image, stage00["OriginImg"])
    assert np.array_equal(result.origin_image_margin, stage00["OriginImg_Margin"])
    assert np.array_equal(result.roi_mask, stage00["ROI_Mask"])
    assert np.array_equal(result.oft_img, stage01["OFT_Img"])
    assert np.array_equal(result.lft_img, stage01["LFT_Img"])
    assert np.array_equal(result.lft_orientations, stage01["LFT_Orientations"])
    assert np.array_equal(result.normalized_oft, stage02["I"])
    assert np.array_equal(result.bw, stage02["BW"])
    assert np.array_equal(result.raw_skeleton, stage03["RawSke"])
    for key in (
        "AllFragments",
        "Allpts",
        "CrPts",
        "L",
        "RawCrPts",
        "AllFragments_before_iteration",
        "L_before_iteration",
        "num_before_iteration",
        "num",
    ):
        assert np.array_equal(result.fragments[key], stage04[key]), key
    assert np.array_equal(result.tips.all_tips_matlab, stage05["all_tips"])
    assert np.array_equal(result.tips.temp_info_matlab, stage05["tempInfo"])
    assert np.array_equal(result.partner_list, stage06["new_partner_list"])
    assert np.array_equal(result.pairing.all_tips_matlab, stage06["all_tips"])
    assert np.array_equal(
        result.pairing.all_tips_pre_bidir_matlab,
        stage06["all_tips_pre_bidir"],
    )
    assert np.array_equal(
        result.pairing.all_tips_post_bidir_matlab,
        stage06["all_tips_post_bidir"],
    )
    assert np.array_equal(result.pairing.l_global_index, stage06["L_GlobalIndex"])
    assert np.array_equal(result.all_filament_matlab, stage07["all_filament"])
    assert np.array_equal(result.all_connects_matlab, stage07["all_connects"])
    assert np.array_equal(
        result.all_connects_shortlist_matlab,
        stage07["all_connects_shortlist"],
    )
    assert np.array_equal(
        result.all_sorted_filament_matlab,
        stage08["all_sorted_filament"],
    )
    assert np.array_equal(result.analysis_info, stage08["AnalysisInfo"])


def test_python_only_pipeline_matches_every_frozen_stage_from_tiff() -> None:
    result = analyze_mt(
        FIXTURE_ROOT / "input" / "fixture_input.tif",
        config=SIFNEConfig(source_reference="sifne-fixture-20260915-r4"),
    )
    stage01 = _load_stage("stage_01_oft.npz")
    stage02 = _load_stage("stage_02_bw.npz")
    stage03 = _load_stage("stage_03_rawske.npz")
    stage04 = _load_stage("stage_04_fragments.npz")
    stage05 = _load_stage("stage_05_tips.npz")
    stage06 = _load_stage("stage_06_pairing.npz")
    stage07 = _load_stage("stage_07_grouped.npz")
    stage08 = _load_stage("stage_08_sorted.npz")

    assert np.array_equal(result.oft_img, stage01["OFT_Img"])
    assert np.array_equal(result.lft_img, stage01["LFT_Img"])
    assert np.array_equal(result.lft_orientations, stage01["LFT_Orientations"])
    assert np.array_equal(result.normalized_oft, stage02["I"])
    assert np.array_equal(result.bw, stage02["BW"])
    assert np.array_equal(result.raw_skeleton, stage03["RawSke"])

    for key in (
        "AllFragments",
        "Allpts",
        "CrPts",
        "L",
        "RawCrPts",
        "AllFragments_before_iteration",
        "L_before_iteration",
        "num_before_iteration",
        "num",
    ):
        assert np.array_equal(result.fragments[key], stage04[key]), key

    assert np.array_equal(result.tips.all_tips_matlab, stage05["all_tips"])
    assert np.array_equal(result.tips.temp_info_matlab, stage05["tempInfo"])
    assert np.array_equal(result.partner_list, stage06["new_partner_list"])
    assert np.array_equal(result.pairing.all_tips_matlab, stage06["all_tips"])
    assert np.array_equal(result.pairing.all_tips_pre_bidir_matlab, stage06["all_tips_pre_bidir"])
    assert np.array_equal(result.pairing.all_tips_post_bidir_matlab, stage06["all_tips_post_bidir"])
    assert np.array_equal(result.pairing.l_global_index, stage06["L_GlobalIndex"])

    assert np.array_equal(result.all_filament_matlab, stage07["all_filament"])
    assert np.array_equal(result.all_connects_matlab, stage07["all_connects"])
    assert np.array_equal(
        result.all_connects_shortlist_matlab,
        stage07["all_connects_shortlist"],
    )

    assert np.array_equal(result.all_sorted_filament_matlab, stage08["all_sorted_filament"])
    assert np.array_equal(result.analysis_info, stage08["AnalysisInfo"])
    assert result.analysis_info.shape == (1, 5)


def test_pipeline_keeps_analysis_info_in_pixel_space_and_converts_only_on_request() -> None:
    result = analyze_mt(
        FIXTURE_ROOT / "input" / "fixture_input.tif",
        config=SIFNEConfig(input_pixel_size_um=0.0313),
    )
    assert np.array_equal(result.lengths_px, np.asarray([114.0]))
    assert np.array_equal(result.end_to_end_px, np.asarray([114.0]))
    np.testing.assert_array_equal(result.lengths_um(), np.asarray([114.0 * 0.0313]))
    np.testing.assert_array_equal(result.end_to_end_um(), np.asarray([114.0 * 0.0313]))
    np.testing.assert_allclose(result.lengths_um(0.02), np.asarray([2.28]), rtol=0.0, atol=1e-15)
    assert np.array_equal(result.analysis_info, np.asarray([[0.0, 114.0, 114.0, 58.0, 89.0]]))


def test_scipy_matlab_compatibility_export_has_consumer_variable_names(tmp_path: Path) -> None:
    result = analyze_mt(FIXTURE_ROOT / "input" / "fixture_input.tif")
    written = export_matlab_compatibility(result, tmp_path)
    expected = {
        "OFT_Img.mat": "OFT_Img",
        "LFT_Img.mat": "LFT_Img",
        "LFT_Orientations.mat": "LFT_Orientations",
        "RawSke.mat": "RawSke",
        "AllFragments.mat": "AllFragments",
        "R.mat": "R",
        "NofOrientations_FT.mat": "NofOrientations_FT",
        "ROI_Mask.mat": "ROI_Mask",
        "NewCrPts.mat": "NewCrPts",
        "all_tips.mat": "all_tips",
        "new_partner_list.mat": "new_partner_list",
        "all_filament.mat": "all_filament",
        "all_connects.mat": "all_connects",
        "all_sorted_filament.mat": "all_sorted_filament",
        "AnalysisInfo.mat": "AnalysisInfo",
    }
    assert set(expected).issubset(written)
    for filename, variable in expected.items():
        payload = loadmat(tmp_path / filename, squeeze_me=False, mat_dtype=True)
        assert variable in payload

    assert np.array_equal(
        loadmat(tmp_path / "AnalysisInfo.mat", mat_dtype=True)["AnalysisInfo"],
        result.analysis_info,
    )
    assert np.array_equal(
        loadmat(tmp_path / "all_sorted_filament.mat", mat_dtype=True)[
            "all_sorted_filament"
        ],
        result.all_sorted_filament_matlab,
    )
    assert np.array_equal(
        loadmat(tmp_path / "AllFragments.mat", mat_dtype=True)["AllFragments"],
        result.fragments.all_fragments,
    )
    assert (tmp_path / "provenance.json").is_file()


def test_pipeline_runtime_has_no_matlab_or_subprocess_dependency() -> None:
    source = (Path(__file__).resolve().parents[1] / "pipeline.py").read_text(
        encoding="utf-8"
    )
    assert "import subprocess" not in source
    assert "matlab.engine" not in source


@pytest.mark.parametrize(
    "case_name",
    [
        "single_filament",
        "two_fragment_gap",
        "multi_fragment_serial",
        "junction_image",
        "curved_largecur",
    ],
)
def test_five_independent_matlab_tiff_fixtures_match_python_pipeline(case_name: str) -> None:
    fixture_root = PIPELINE_FIXTURE_ROOT / case_name
    result = analyze_mt(fixture_root / "input" / "fixture_input.tif")
    _assert_core_stages_exact(result, fixture_root)
