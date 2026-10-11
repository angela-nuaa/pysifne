"""Configuration for the faithful, MATLAB-free PySIFNE pipeline.

The values in this module mirror the active defaults in
``func_analysis_mt.m``.  In particular, ``internal_grouping_reference_pixel_size_um``
is the historical SIFNE value used to derive the search radius; it is not the
physical calibration of the input image.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np


# Frozen after MATLAB R2023b stage-by-stage validation and the exact CUDA
# LFT/OFT fusion validation completed on 2026-09-16.
PY_SIFNE_VERSION = "0.2.0"
PY_SIFNE_PROFILE = "faithful-matlab-r2023b-cuda-exact"


@dataclass(frozen=True)
class SIFNEConfig:
    """Parameters for the current faithful MATLAB-compatible execution path."""

    radius: int = 10
    n_orientations: int = 20
    default_factor: float = 1.42
    junction_size: int = 7
    minimum_fragment_length: int = 6
    iteration: int = 1
    iteration_remove_radius: int = 3
    internal_grouping_reference_pixel_size_um: float = 0.02
    max_curvature_rad_per_um: float = 1.0
    input_pixel_size_um: float | None = None
    output_pixel_size_um: float | None = None
    registration_radius: float | None = None
    fan_radius: float | None = None
    fan_angle_deg: float | None = None
    c1_weight: float = 1.0
    c3_weight: float = 1.0
    partner_slots: int = 100
    overlap: int = 1
    full_length_factor: int = 5
    roi_mask: np.ndarray | str | Path | None = None
    mode: str = "faithful"
    source_reference: str | None = None

    def __post_init__(self) -> None:
        integer_fields = (
            "radius",
            "n_orientations",
            "junction_size",
            "minimum_fragment_length",
            "iteration",
            "iteration_remove_radius",
            "partner_slots",
            "full_length_factor",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{name} must be an integer, got {value!r}")
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative, got {value!r}")
        if self.n_orientations == 0:
            raise ValueError("n_orientations must be positive")
        if self.junction_size == 0:
            raise ValueError("junction_size must be positive")
        if self.partner_slots == 0:
            raise ValueError("partner_slots must be positive")
        if self.full_length_factor == 0:
            raise ValueError("full_length_factor must be positive")
        if self.mode not in {"faithful", "robust"}:
            raise ValueError("mode must be 'faithful' or 'robust'")

        positive_fields = (
            "default_factor",
            "internal_grouping_reference_pixel_size_um",
            "max_curvature_rad_per_um",
            "c1_weight",
            "c3_weight",
        )
        for name in positive_fields:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("input_pixel_size_um", "output_pixel_size_um"):
            value = getattr(self, name)
            if value is not None and (
                not np.isfinite(float(value)) or float(value) <= 0
            ):
                raise ValueError(f"{name} must be finite and positive when supplied")
        if self.overlap not in (0, 1):
            raise ValueError("overlap must be 0 or 1")

    @property
    def registration_radius_value(self) -> float:
        """The ``RegR=2*R`` value used by the active ``TipReg`` call."""

        return float(2 * int(self.radius) if self.registration_radius is None else self.registration_radius)

    @property
    def fan_radius_value(self) -> float:
        """The ``FanR`` value used by ``local_search``."""

        if self.fan_radius is not None:
            return float(self.fan_radius)
        return 1.0 / (
            float(self.max_curvature_rad_per_um)
            * float(self.internal_grouping_reference_pixel_size_um)
        )

    @property
    def fan_angle_value(self) -> float:
        """The historical SIFNE fan-angle value, expressed in degrees."""

        if self.fan_angle_deg is not None:
            return float(self.fan_angle_deg)
        return 360.0 / 2.0 / np.pi

    @property
    def min_info(self) -> np.ndarray:
        """The MATLAB ``MIN_Info=[C1,FanR,C3]`` row vector."""

        fan_angle = self.fan_angle_value
        return np.asarray(
            [[fan_angle, self.fan_radius_value, fan_angle / 2.0]], dtype=np.float64
        )

    def physical_pixel_size_um(self, override: Real | None = None) -> float:
        """Resolve an explicit physical calibration for public µm helpers."""

        if override is not None:
            value = float(override)
        elif self.output_pixel_size_um is not None:
            value = float(self.output_pixel_size_um)
        elif self.input_pixel_size_um is not None:
            value = float(self.input_pixel_size_um)
        else:
            raise ValueError(
                "A physical pixel size is required; pass pixel_size_um or set "
                "input_pixel_size_um/output_pixel_size_um."
            )
        if not np.isfinite(value) or value <= 0:
            raise ValueError("pixel_size_um must be finite and positive")
        return value

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly configuration/provenance metadata."""

        result: dict[str, Any] = {
            "radius": int(self.radius),
            "n_orientations": int(self.n_orientations),
            "default_factor": float(self.default_factor),
            "junction_size": int(self.junction_size),
            "minimum_fragment_length": int(self.minimum_fragment_length),
            "iteration": int(self.iteration),
            "iteration_remove_radius": int(self.iteration_remove_radius),
            "internal_grouping_reference_pixel_size_um": float(
                self.internal_grouping_reference_pixel_size_um
            ),
            "max_curvature_rad_per_um": float(self.max_curvature_rad_per_um),
            "input_pixel_size_um": self.input_pixel_size_um,
            "output_pixel_size_um": self.output_pixel_size_um,
            "registration_radius": self.registration_radius_value,
            "fan_radius": self.fan_radius_value,
            "fan_angle_deg": self.fan_angle_value,
            "min_info": self.min_info.tolist(),
            "c1_weight": float(self.c1_weight),
            "c3_weight": float(self.c3_weight),
            "partner_slots": int(self.partner_slots),
            "overlap": int(self.overlap),
            "full_length_factor": int(self.full_length_factor),
            "mode": self.mode,
            "source_reference": self.source_reference,
        }
        if self.roi_mask is None:
            result["roi_mask"] = None
        elif isinstance(self.roi_mask, (str, Path)):
            result["roi_mask"] = str(self.roi_mask)
        else:
            mask = np.asarray(self.roi_mask)
            result["roi_mask"] = {
                "shape": list(mask.shape),
                "dtype": str(mask.dtype),
            }
        return result


__all__ = ["PY_SIFNE_PROFILE", "PY_SIFNE_VERSION", "SIFNEConfig"]
