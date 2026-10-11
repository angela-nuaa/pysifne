"""Small public entry points for the frozen PySIFNE profile."""

from __future__ import annotations

from typing import Any

from .config import SIFNEConfig
from .pipeline import analyze_mt
from .result import SIFNEResult


def _cuda_is_available() -> bool:
    """Return CUDA availability without making CUDA a hard import dependency."""

    try:
        from .lft_oft_accel import cuda_available
    except Exception:  # pragma: no cover - optional runtime dependency
        return False
    try:
        return bool(cuda_available())
    except Exception:  # pragma: no cover - driver/runtime dependent
        return False


def analyze(
    image_or_path: Any,
    *,
    config: SIFNEConfig | None = None,
    device: int = 0,
) -> SIFNEResult:
    """Run the frozen faithful PySIFNE profile.

    This is the compact entry point for normal use.  It selects the already
    validated CUDA path when a CUDA device is available and otherwise uses the
    readable reference path.  The CUDA path always uses the frozen
    ``matlab_windows`` factor table.  The lower-level :func:`analyze_mt`
    remains available when an explicit backend or kernel setting is required.

    Parameters that affect the MATLAB algorithm continue to live in
    :class:`SIFNEConfig`; no hidden physical-unit conversion is performed.
    """

    cfg = SIFNEConfig() if config is None else config
    backend = "cuda" if _cuda_is_available() else "reference"
    return analyze_mt(
        image_or_path,
        config=cfg,
        backend=backend,
        device=int(device),
        threads_per_block=256,
        factor_mode="matlab_windows",
    )


__all__ = ["analyze"]
