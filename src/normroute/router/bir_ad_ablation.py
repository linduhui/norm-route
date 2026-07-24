"""Frozen, nested BIR-AD ablations for Stage 5 experiments.

These specifications define computations and Router feature inclusion only.
They do not contain, fabricate, or evaluate experiment outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


BIR_AD_ABLATION_PROTOCOL_VERSION = "stage5.bir_ad_ablation.v1"


@dataclass(frozen=True)
class BIRADAblationSpec:
    name: str
    description: str
    clarity_weights: tuple[float, float, float, float, float]
    use_structural_boundary_weighting: bool
    disagreement_penalty: float
    include_bai_features: bool
    include_consistency_features: bool
    include_representations: bool
    protocol_version: str = BIR_AD_ABLATION_PROTOCOL_VERSION

    def compute_kwargs(self) -> dict[str, Any]:
        return {
            "clarity_weights": self.clarity_weights,
            "use_structural_boundary_weighting": (
                self.use_structural_boundary_weighting
            ),
            "disagreement_penalty": self.disagreement_penalty,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "name": self.name,
            "description": self.description,
            "compute_kwargs": self.compute_kwargs(),
            "router_features": {
                "include_bai_features": self.include_bai_features,
                "include_consistency_features": (
                    self.include_consistency_features
                ),
                "include_representations": self.include_representations,
            },
        }


_ABLATIONS = (
    BIRADAblationSpec(
        name="sigma_l2",
        description="Feature-channel std and RMS L2 evidence with uniform patches.",
        clarity_weights=(0.5, 0.5, 0.0, 0.0, 0.0),
        use_structural_boundary_weighting=False,
        disagreement_penalty=0.0,
        include_bai_features=True,
        include_consistency_features=False,
        include_representations=False,
    ),
    BIRADAblationSpec(
        name="plus_sobel",
        description="Add aligned Sobel edge energy.",
        clarity_weights=(1 / 3, 1 / 3, 1 / 3, 0.0, 0.0),
        use_structural_boundary_weighting=False,
        disagreement_penalty=0.0,
        include_bai_features=True,
        include_consistency_features=False,
        include_representations=False,
    ),
    BIRADAblationSpec(
        name="plus_structural_boundary",
        description="Add feature-neighbour structural-boundary reweighting.",
        clarity_weights=(1 / 3, 1 / 3, 1 / 3, 0.0, 0.0),
        use_structural_boundary_weighting=True,
        disagreement_penalty=0.0,
        include_bai_features=True,
        include_consistency_features=False,
        include_representations=False,
    ),
    BIRADAblationSpec(
        name="plus_directional_evidence",
        description="Add structure-tensor coherence and two-sided contrast.",
        clarity_weights=(0.2, 0.2, 0.2, 0.2, 0.2),
        use_structural_boundary_weighting=True,
        disagreement_penalty=0.0,
        include_bai_features=True,
        include_consistency_features=False,
        include_representations=False,
    ),
    BIRADAblationSpec(
        name="plus_cross_modal_disagreement",
        description="Add explicit pixel-feature boundary disagreement penalty.",
        clarity_weights=(0.2, 0.2, 0.2, 0.2, 0.2),
        use_structural_boundary_weighting=True,
        disagreement_penalty=1.0,
        include_bai_features=True,
        include_consistency_features=False,
        include_representations=False,
    ),
    BIRADAblationSpec(
        name="plus_support_consistency",
        description="Add feature-matched patch-level normal-support consistency.",
        clarity_weights=(0.2, 0.2, 0.2, 0.2, 0.2),
        use_structural_boundary_weighting=True,
        disagreement_penalty=1.0,
        include_bai_features=True,
        include_consistency_features=True,
        include_representations=False,
    ),
    BIRADAblationSpec(
        name="representations_only",
        description="Clear and ambiguous representations without scalar BIR evidence.",
        clarity_weights=(0.2, 0.2, 0.2, 0.2, 0.2),
        use_structural_boundary_weighting=True,
        disagreement_penalty=1.0,
        include_bai_features=False,
        include_consistency_features=False,
        include_representations=True,
    ),
    BIRADAblationSpec(
        name="full",
        description="Complete BIR-AD scalar, consistency, and dual-representation bundle.",
        clarity_weights=(0.2, 0.2, 0.2, 0.2, 0.2),
        use_structural_boundary_weighting=True,
        disagreement_penalty=1.0,
        include_bai_features=True,
        include_consistency_features=True,
        include_representations=True,
    ),
)
BIR_AD_ABLATIONS = {spec.name: spec for spec in _ABLATIONS}


def get_bir_ad_ablation(name: str) -> BIRADAblationSpec:
    try:
        return BIR_AD_ABLATIONS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown BIR-AD ablation {name!r}; "
            f"expected one of {list(BIR_AD_ABLATIONS)}"
        ) from exc


def bir_ad_ablation_records() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in _ABLATIONS]


__all__ = [
    "BIR_AD_ABLATIONS",
    "BIR_AD_ABLATION_PROTOCOL_VERSION",
    "BIRADAblationSpec",
    "bir_ad_ablation_records",
    "get_bir_ad_ablation",
]
