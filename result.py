"""Result containers and physical-unit views for the PySIFNE pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .config import PY_SIFNE_VERSION, SIFNEConfig
from .grouping import GroupingResult
from .pairing import PairingResult
from .sorting import SortFilamentsResult
from .tips import TipRegistrationResult


@dataclass(frozen=True)
class FragmentStageResult:
    """Stage-04 arrays, retaining the names used by the MATLAB workspace."""

    data: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    @property
    def all_fragments(self) -> np.ndarray:
        return np.asarray(self.data["AllFragments"])

    @property
    def labels(self) -> np.ndarray:
        return np.asarray(self.data["L"])

    @property
    def number(self) -> int:
        return int(np.asarray(self.data["num"]).reshape(-1)[0])

    @property
    def all_points(self) -> np.ndarray:
        return np.asarray(self.data["Allpts"])


@dataclass(frozen=True)
class SIFNEResult:
    """All observable stages from one faithful Python-only analysis.

    Coordinates in Python-native arrays are zero-based ``(row, column)``.
    MATLAB-compatible one-based arrays are exposed by the ``*_matlab``
    properties.  Fragment/tip/partner IDs retain their MATLAB-style values
    because they are labels, not coordinates.
    """

    input_image: np.ndarray
    origin_image: np.ndarray
    origin_image_margin: np.ndarray
    roi_mask: np.ndarray
    oft_img: np.ndarray
    lft_img: np.ndarray
    lft_orientations: np.ndarray
    normalized_oft: np.ndarray
    bw: np.ndarray
    raw_skeleton: np.ndarray
    fragments: FragmentStageResult
    tips: TipRegistrationResult
    pairing: PairingResult
    grouping: GroupingResult
    sorting: SortFilamentsResult
    junctions: np.ndarray
    otsu_threshold: float
    threshold_used: float
    config: SIFNEConfig
    provenance: Mapping[str, Any]

    @property
    def partner_list(self) -> np.ndarray:
        return self.pairing.new_partner_list

    @property
    def all_tips(self) -> np.ndarray:
        return self.pairing.all_tips

    @property
    def all_tips_matlab(self) -> np.ndarray:
        return self.pairing.all_tips_matlab

    @property
    def junctions_matlab(self) -> np.ndarray:
        if self.junctions.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(self.junctions, dtype=np.float64) + 1.0

    @property
    def all_filament(self) -> np.ndarray:
        return self.grouping.all_filament

    @property
    def all_filament_matlab(self) -> np.ndarray:
        return self.grouping.all_filament_matlab

    @property
    def all_connects(self) -> np.ndarray:
        return self.grouping.all_connects

    @property
    def all_connects_matlab(self) -> np.ndarray:
        return self.grouping.all_connects_matlab

    @property
    def all_connects_shortlist(self) -> np.ndarray:
        return self.grouping.all_connects_shortlist

    @property
    def all_connects_shortlist_matlab(self) -> np.ndarray:
        return self.grouping.all_connects_shortlist_matlab

    @property
    def all_sorted_filament(self) -> np.ndarray:
        return self.sorting.all_sorted_filament

    @property
    def all_sorted_filament_matlab(self) -> np.ndarray:
        return self.sorting.all_sorted_filament_matlab

    @property
    def analysis_info(self) -> np.ndarray:
        """MATLAB's pixel-space ``[orientation,length,e2e,row,col]`` table."""

        return self.sorting.analysis_info

    @property
    def lengths_px(self) -> np.ndarray:
        return self.analysis_info[:, 1].copy()

    @property
    def end_to_end_px(self) -> np.ndarray:
        return self.analysis_info[:, 2].copy()

    def lengths_um(self, pixel_size_um: float | None = None) -> np.ndarray:
        return self.lengths_px * self.config.physical_pixel_size_um(pixel_size_um)

    def end_to_end_um(self, pixel_size_um: float | None = None) -> np.ndarray:
        return self.end_to_end_px * self.config.physical_pixel_size_um(pixel_size_um)

    def stage_arrays(self) -> dict[str, np.ndarray]:
        """Return the main stages in one dictionary for comparison/export."""

        return {
            "OFT_Img": self.oft_img,
            "LFT_Img": self.lft_img,
            "LFT_Orientations": self.lft_orientations,
            "I": self.normalized_oft,
            "BW": self.bw,
            "RawSke": self.raw_skeleton,
            "AllFragments": self.fragments.all_fragments,
            "L": self.fragments.labels,
            "all_tips": self.all_tips_matlab,
            "new_partner_list": self.partner_list,
            "all_filament": self.all_filament_matlab,
            "all_connects": self.all_connects_matlab,
            "all_sorted_filament": self.all_sorted_filament_matlab,
            "AnalysisInfo": self.analysis_info,
        }


__all__ = [
    "FragmentStageResult",
    "PY_SIFNE_VERSION",
    "SIFNEResult",
]
