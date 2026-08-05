"""Frozen FBDP-AD ablations for category-held-out Stage 5 experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


FBDP_AD_ABLATION_PROTOCOL_VERSION = "stage5.fbdp_ad_ablation.v1"


@dataclass(frozen=True)
class FBDPADAblationSpec:
    name: str
    description: str
    compute_overrides: Mapping[str, Any] = field(default_factory=dict)
    include_support_features: bool = True
    include_raw_residual_features: bool = True
    include_gated_residual_features: bool = True
    include_assignment_features: bool = True
    protocol_version: str = FBDP_AD_ABLATION_PROTOCOL_VERSION

    def compute_kwargs(self) -> dict[str, Any]:
        return dict(self.compute_overrides)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "name": self.name,
            "description": self.description,
            "compute_kwargs": self.compute_kwargs(),
            "router_features": {
                "include_support_features": self.include_support_features,
                "include_raw_residual_features": (
                    self.include_raw_residual_features
                ),
                "include_gated_residual_features": (
                    self.include_gated_residual_features
                ),
                "include_assignment_features": self.include_assignment_features,
            },
        }


_ABLATIONS = (
    FBDPADAblationSpec(
        name="single_bank",
        description="Union-bank nearest-prototype residual without decoupled mixing.",
        compute_overrides={"query_bank_mode": "single"},
    ),
    FBDPADAblationSpec(
        name="no_fbc",
        description="Remove FBC from the support-derived gate.",
        compute_overrides={"use_fbc_in_gate": False},
    ),
    FBDPADAblationSpec(
        name="no_objectness_gate",
        description="Force the FBDP query branch gate to one.",
        compute_overrides={"use_objectness_gate": False},
        include_gated_residual_features=False,
    ),
    FBDPADAblationSpec(
        name="border_only_background",
        description="Build background candidates only from border patches.",
        compute_overrides={"background_candidate_mode": "border_only"},
    ),
    FBDPADAblationSpec(
        name="low_objectness_only_background",
        description="Build background candidates only from low-objectness patches.",
        compute_overrides={"background_candidate_mode": "low_objectness_only"},
    ),
    FBDPADAblationSpec(
        name="no_cross_support_consistency",
        description="Remove cross-support consistency from foreground selection.",
        compute_overrides={"use_cross_support_consistency": False},
    ),
    FBDPADAblationSpec(
        name="pooling",
        description="Replace deterministic spherical k-means with prototype pooling.",
        compute_overrides={"prototype_method": "pooling"},
    ),
    FBDPADAblationSpec(
        name="same_position_consistency",
        description="Use strict same-position instead of local-window consistency.",
        compute_overrides={"consistency_mode": "same_position"},
    ),
    FBDPADAblationSpec(
        name="raw_residual_only",
        description="Expose ungated query residuals while withholding gated residuals.",
        include_gated_residual_features=False,
    ),
    FBDPADAblationSpec(
        name="full",
        description="Complete FBDP v2 support reliability and decoupled query bundle.",
    ),
)

FBDP_AD_ABLATIONS = {spec.name: spec for spec in _ABLATIONS}


def get_fbdp_ad_ablation(name: str) -> FBDPADAblationSpec:
    try:
        return FBDP_AD_ABLATIONS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown FBDP-AD ablation {name!r}; "
            f"expected one of {list(FBDP_AD_ABLATIONS)}"
        ) from exc


def fbdp_ad_ablation_records() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in _ABLATIONS]


_COMPUTE_DEFAULTS = {
    "query_bank_mode": "decoupled",
    "use_fbc_in_gate": True,
    "use_objectness_gate": True,
    "background_candidate_mode": "combined",
    "use_cross_support_consistency": True,
    "prototype_method": "kmeans",
    "consistency_mode": "local_window",
}


def validate_fbdp_ad_ablation_record(
    record: Mapping[str, Any],
    ablation: str | FBDPADAblationSpec,
) -> None:
    """Reject materialization from signatures built by another compute variant."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    spec = get_fbdp_ad_ablation(ablation) if isinstance(ablation, str) else ablation
    if not isinstance(spec, FBDPADAblationSpec):
        raise TypeError("ablation must be a name or FBDPADAblationSpec")
    expected = {**_COMPUTE_DEFAULTS, **spec.compute_kwargs()}
    mismatches = {
        field: {"expected": expected_value, "observed": record.get(field)}
        for field, expected_value in expected.items()
        if record.get(field) != expected_value
    }
    if mismatches:
        raise ValueError(
            f"FBDP signature compute settings disagree with {spec.name!r}: "
            f"{mismatches}"
        )


__all__ = [
    "FBDP_AD_ABLATIONS",
    "FBDP_AD_ABLATION_PROTOCOL_VERSION",
    "FBDPADAblationSpec",
    "fbdp_ad_ablation_records",
    "get_fbdp_ad_ablation",
    "validate_fbdp_ad_ablation_record",
]
