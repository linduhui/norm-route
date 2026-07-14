"""Stage 4 pre-route task protocol and leakage checks.

The policy boundary is the ``policy_features`` object. Top-level identity and
provenance fields are available to the runner for locating inputs, assigning
folds, and recording outputs, but they are not policy features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


STAGE4_TASK_PROTOCOL_VERSION = "stage4.pre_route.v1"

POLICY_FEATURE_ALLOWLIST = ("dataset", "category", "k_shot")
"""The complete set of fields a Stage 4 routing policy may inspect."""

PROVENANCE_FIELDS = ("seed", "support_set_id")
"""Runner-visible fields that must never be copied into policy features."""

CANDIDATE_EXPERTS = ("PatchCore", "WinCLIP", "AnomalyDINO")

FORBIDDEN_PRE_ROUTE_FIELDS = frozenset(
    {
        "label",
        "mask_path",
        "defect_type",
        "anomaly_type",
        "oracle_best_expert",
        "patchcore_score",
        "winclip_score",
        "anomalydino_score",
    }
)

PRE_ROUTE_TASK_FIELDS = frozenset(
    {
        "protocol_version",
        "task_id",
        "sample_id",
        "dataset",
        "category",
        "k_shot",
        "seed",
        "support_set_id",
        "candidate_experts",
        "policy_features",
    }
)


class Stage4ProtocolError(ValueError):
    """Raised when a Stage 4 task violates the frozen protocol."""


@dataclass(frozen=True)
class PreRouteTask:
    """A validated Stage 4 task before an expert has been selected or run."""

    task_id: str
    sample_id: str
    dataset: str
    category: str
    k_shot: int
    seed: int
    support_set_id: str
    candidate_experts: tuple[str, ...] = CANDIDATE_EXPERTS
    protocol_version: str = STAGE4_TASK_PROTOCOL_VERSION

    @property
    def policy_features(self) -> dict[str, str | int]:
        """Return the only mapping that a routing policy is allowed to consume."""

        return {
            "dataset": self.dataset,
            "category": self.category,
            "k_shot": self.k_shot,
        }

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol_version": self.protocol_version,
            "task_id": self.task_id,
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "category": self.category,
            "k_shot": self.k_shot,
            "seed": self.seed,
            "support_set_id": self.support_set_id,
            "candidate_experts": list(self.candidate_experts),
            "policy_features": self.policy_features,
        }
        validate_pre_route_task(payload)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PreRouteTask":
        """Create a typed task from an already serialized protocol mapping."""

        validate_pre_route_task(payload)
        return cls(
            task_id=payload["task_id"],
            sample_id=payload["sample_id"],
            dataset=payload["dataset"],
            category=payload["category"],
            k_shot=payload["k_shot"],
            seed=payload["seed"],
            support_set_id=payload["support_set_id"],
            candidate_experts=tuple(payload["candidate_experts"]),
            protocol_version=payload["protocol_version"],
        )


def validate_pre_route_task(payload: Mapping[str, Any], *, context: str = "pre-route task") -> None:
    """Validate one serialized task, including nested leakage and allowlist checks."""

    if not isinstance(payload, Mapping):
        raise Stage4ProtocolError(f"{context} must be a JSON object")

    validate_no_forbidden_fields(payload, context=context)
    fields = set(payload)
    missing = sorted(PRE_ROUTE_TASK_FIELDS - fields)
    extra = sorted(fields - PRE_ROUTE_TASK_FIELDS)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {missing}")
        if extra:
            details.append(f"unexpected fields: {extra}")
        raise Stage4ProtocolError(f"{context} has invalid schema ({'; '.join(details)})")

    if payload["protocol_version"] != STAGE4_TASK_PROTOCOL_VERSION:
        raise Stage4ProtocolError(
            f"{context} has protocol_version={payload['protocol_version']!r}; "
            f"expected {STAGE4_TASK_PROTOCOL_VERSION!r}"
        )

    for field in ("task_id", "sample_id", "dataset", "category", "support_set_id"):
        _require_nonempty_string(payload[field], field=field, context=context)
    _require_integer(payload["k_shot"], field="k_shot", context=context, minimum=1)
    _require_integer(payload["seed"], field="seed", context=context, minimum=0)

    candidates = payload["candidate_experts"]
    if not _is_sequence(candidates) or tuple(candidates) != CANDIDATE_EXPERTS:
        raise Stage4ProtocolError(
            f"{context} candidate_experts must be exactly {list(CANDIDATE_EXPERTS)!r}"
        )

    validate_policy_features(payload["policy_features"], payload=payload, context=context)


def validate_policy_features(
    features: Any,
    *,
    payload: Mapping[str, Any] | None = None,
    context: str = "policy features",
) -> None:
    """Enforce the exact policy feature allowlist and optional value consistency."""

    if not isinstance(features, Mapping):
        raise Stage4ProtocolError(f"{context} policy_features must be a JSON object")
    validate_no_forbidden_fields(features, context=f"{context} policy_features")

    feature_names = set(features)
    expected = set(POLICY_FEATURE_ALLOWLIST)
    missing = sorted(expected - feature_names)
    extra = sorted(feature_names - expected)
    if missing or extra:
        raise Stage4ProtocolError(
            f"{context} policy_features must contain exactly "
            f"{list(POLICY_FEATURE_ALLOWLIST)!r}; missing={missing}, extra={extra}"
        )

    if set(PROVENANCE_FIELDS).intersection(feature_names):
        raise Stage4ProtocolError(
            f"{context} policy_features contains provenance-only fields"
        )

    _require_nonempty_string(features["dataset"], field="dataset", context=context)
    _require_nonempty_string(features["category"], field="category", context=context)
    _require_integer(features["k_shot"], field="k_shot", context=context, minimum=1)

    if payload is not None:
        mismatched = [
            field
            for field in POLICY_FEATURE_ALLOWLIST
            if features[field] != payload[field]
        ]
        if mismatched:
            raise Stage4ProtocolError(
                f"{context} policy_features disagree with top-level fields: {mismatched}"
            )


def validate_no_forbidden_fields(value: Any, *, context: str) -> None:
    """Reject forbidden field names at any depth in a JSON-like value."""

    if isinstance(value, Mapping):
        forbidden = sorted(FORBIDDEN_PRE_ROUTE_FIELDS.intersection(value))
        if forbidden:
            raise Stage4ProtocolError(f"{context} contains forbidden fields: {forbidden}")
        for child in value.values():
            validate_no_forbidden_fields(child, context=context)
    elif _is_sequence(value):
        for child in value:
            validate_no_forbidden_fields(child, context=context)


def _require_nonempty_string(value: Any, *, field: str, context: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise Stage4ProtocolError(f"{context} field {field!r} must be a non-empty string")


def _require_integer(
    value: Any,
    *,
    field: str,
    context: str,
    minimum: int,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Stage4ProtocolError(
            f"{context} field {field!r} must be an integer >= {minimum}"
        )


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
