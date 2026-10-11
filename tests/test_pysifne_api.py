from __future__ import annotations

from pathlib import Path

from pysifne import PY_SIFNE_PROFILE, PY_SIFNE_VERSION, SIFNEConfig
import pysifne.api as api


def test_compact_entry_selects_frozen_cuda_profile(monkeypatch) -> None:
    calls: dict[str, object] = {}
    sentinel = object()

    def fake_analyze_mt(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return sentinel

    config = SIFNEConfig()
    monkeypatch.setattr(api, "_cuda_is_available", lambda: True)
    monkeypatch.setattr(api, "analyze_mt", fake_analyze_mt)

    result = api.analyze(Path("fixture_input.tif"), config=config, device=1)

    assert result is sentinel
    assert calls["args"] == (Path("fixture_input.tif"),)
    assert calls["kwargs"] == {
        "config": config,
        "backend": "cuda",
        "device": 1,
        "threads_per_block": 256,
        "factor_mode": "matlab_windows",
    }


def test_compact_entry_falls_back_to_reference_without_cuda(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_analyze_mt(*args, **kwargs):
        calls["kwargs"] = kwargs
        return None

    monkeypatch.setattr(api, "_cuda_is_available", lambda: False)
    monkeypatch.setattr(api, "analyze_mt", fake_analyze_mt)

    api.analyze("fixture_input.tif")

    assert calls["kwargs"]["backend"] == "reference"
    assert calls["kwargs"]["factor_mode"] == "matlab_windows"


def test_frozen_profile_metadata_is_exposed() -> None:
    assert PY_SIFNE_VERSION == "0.2.0"
    assert PY_SIFNE_PROFILE == "faithful-matlab-r2023b-cuda-exact"
