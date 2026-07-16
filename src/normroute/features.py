"""Leakage-safe feature construction for Stage 4 metadata routers.

The learned diagnostic baselines deliberately use a stricter boundary than
the general Stage 4 task protocol.  Only category, k-shot, the configured
tool budget, and predeclared static expert costs may reach an encoder.
Identity, provenance, query paths, and realized expert outcomes are rejected.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.normroute.agent.protocol import CANDIDATE_EXPERTS


METADATA_FEATURE_PROTOCOL_VERSION = "stage4.metadata_features.v1"
METADATA_FEATURE_ALLOWLIST = ("category", "k_shot", "budget", "static_cost")
"""The exhaustive raw-feature allowlist for learned metadata routers."""

FEATURE_DENYLIST = frozenset(
    {
        "dataset",
        "seed",
        "support_set_id",
        "query_path",
        "image_path",
        "sample_id",
        "task_id",
        "protocol_version",
        "candidate_experts",
        "policy_features",
        "expert_scores",
        "expert_score",
        "label",
        "mask",
        "mask_path",
        "defect_type",
        "anomaly_type",
        "ground_truth",
        "oracle_answer",
        "oracle_best_expert",
    }
)
"""Fields that must never be accepted as learned-router features."""

_MANIFEST_FIELDS = frozenset(
    {
        "protocol_version",
        "raw_feature_allowlist",
        "feature_denylist",
        "category_vocabulary",
        "candidate_experts",
        "encoded_feature_names",
        "numeric_scaling",
        "unknown_category_encoding",
    }
)


class MetadataFeatureError(ValueError):
    """Raised when metadata features cross the frozen leakage boundary."""


def build_metadata_feature_record(
    *,
    category: str,
    k_shot: int,
    budget: int,
    static_cost: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and validate one raw learned-router feature record."""

    record = {
        "category": category,
        "k_shot": k_shot,
        "budget": budget,
        "static_cost": dict(static_cost),
    }
    return validate_metadata_feature_record(record)


def validate_metadata_feature_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical record after enforcing the exact feature schema."""

    if not isinstance(record, Mapping):
        raise MetadataFeatureError("metadata feature record must be a mapping")
    _reject_denylisted_fields(record, context="metadata feature record")
    fields = set(record)
    expected = set(METADATA_FEATURE_ALLOWLIST)
    missing = sorted(expected - fields)
    extra = sorted(fields - expected)
    if missing or extra:
        raise MetadataFeatureError(
            "metadata features must contain exactly "
            f"{list(METADATA_FEATURE_ALLOWLIST)!r}; missing={missing}, extra={extra}"
        )

    category = _nonempty_string(record["category"], "category")
    k_shot = _positive_int(record["k_shot"], "k_shot")
    budget = _nonnegative_int(record["budget"], "budget")
    costs = canonical_static_costs(record["static_cost"])
    return {
        "category": category,
        "k_shot": k_shot,
        "budget": budget,
        "static_cost": costs,
    }


def canonical_static_costs(value: Any) -> dict[str, float]:
    """Validate a complete, canonical expert-to-static-cost mapping."""

    if not isinstance(value, Mapping):
        raise MetadataFeatureError("static_cost must be a mapping")
    _reject_denylisted_fields(value, context="static_cost")
    normalized: dict[str, float] = {}
    for raw_expert, raw_cost in value.items():
        expert = _canonical_expert(raw_expert)
        if expert in normalized:
            raise MetadataFeatureError(f"static_cost duplicates expert {expert!r}")
        normalized[expert] = _finite_nonnegative(raw_cost, f"static_cost.{raw_expert}")
    expected = set(CANDIDATE_EXPERTS)
    missing = sorted(expected - set(normalized))
    extra = sorted(set(normalized) - expected)
    if missing or extra:
        raise MetadataFeatureError(
            "static_cost must contain every candidate expert exactly once; "
            f"missing={missing}, extra={extra}"
        )
    return {expert: normalized[expert] for expert in CANDIDATE_EXPERTS}


def fit_feature_manifest(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fit a deterministic encoder manifest using training features only."""

    if not records:
        raise MetadataFeatureError("cannot fit a feature manifest without train records")
    canonical = [validate_metadata_feature_record(record) for record in records]
    categories = sorted({record["category"] for record in canonical})
    encoded_names = [f"category={category}" for category in categories]
    encoded_names.append("category=<UNKNOWN>")
    numeric_names = [
        "k_shot",
        "budget",
        *(f"static_cost.{expert}" for expert in CANDIDATE_EXPERTS),
    ]
    encoded_names.extend(numeric_names)

    numeric_values: dict[str, list[float]] = {name: [] for name in numeric_names}
    for record in canonical:
        numeric_values["k_shot"].append(float(record["k_shot"]))
        numeric_values["budget"].append(float(record["budget"]))
        for expert in CANDIDATE_EXPERTS:
            numeric_values[f"static_cost.{expert}"].append(
                float(record["static_cost"][expert])
            )
    scaling: dict[str, dict[str, float]] = {}
    for name in numeric_names:
        values = numeric_values[name]
        mean = math.fsum(values) / len(values)
        variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
        scale = math.sqrt(variance)
        if math.isclose(scale, 0.0, rel_tol=0.0, abs_tol=1e-15):
            scale = 1.0
        scaling[name] = {"mean": mean, "scale": scale}

    manifest = {
        "protocol_version": METADATA_FEATURE_PROTOCOL_VERSION,
        "raw_feature_allowlist": list(METADATA_FEATURE_ALLOWLIST),
        "feature_denylist": sorted(FEATURE_DENYLIST),
        "category_vocabulary": categories,
        "candidate_experts": list(CANDIDATE_EXPERTS),
        "encoded_feature_names": encoded_names,
        "numeric_scaling": scaling,
        "unknown_category_encoding": "dedicated_one_hot",
    }
    return validate_feature_manifest(manifest)


def encode_metadata_features(
    record: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[float]:
    """Encode one validated record using a frozen training-fold manifest."""

    canonical = validate_metadata_feature_record(record)
    frozen = validate_feature_manifest(manifest)
    categories = list(frozen["category_vocabulary"])
    category = canonical["category"]
    vector = [1.0 if category == known else 0.0 for known in categories]
    vector.append(0.0 if category in categories else 1.0)

    numeric = {
        "k_shot": float(canonical["k_shot"]),
        "budget": float(canonical["budget"]),
        **{
            f"static_cost.{expert}": float(canonical["static_cost"][expert])
            for expert in CANDIDATE_EXPERTS
        },
    }
    for name in (
        "k_shot",
        "budget",
        *(f"static_cost.{expert}" for expert in CANDIDATE_EXPERTS),
    ):
        parameters = frozen["numeric_scaling"][name]
        vector.append((numeric[name] - parameters["mean"]) / parameters["scale"])
    if len(vector) != len(frozen["encoded_feature_names"]):
        raise MetadataFeatureError("encoded feature vector disagrees with feature manifest")
    return vector


def validate_feature_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact, JSON-safe frozen encoder schema."""

    if not isinstance(value, Mapping):
        raise MetadataFeatureError("feature manifest must be a mapping")
    if set(value) != _MANIFEST_FIELDS:
        raise MetadataFeatureError(
            "feature manifest has invalid fields; "
            f"missing={sorted(_MANIFEST_FIELDS - set(value))}, "
            f"extra={sorted(set(value) - _MANIFEST_FIELDS)}"
        )
    if value["protocol_version"] != METADATA_FEATURE_PROTOCOL_VERSION:
        raise MetadataFeatureError("feature manifest protocol_version is invalid")
    if value["raw_feature_allowlist"] != list(METADATA_FEATURE_ALLOWLIST):
        raise MetadataFeatureError("feature manifest allowlist is not frozen")
    if value["feature_denylist"] != sorted(FEATURE_DENYLIST):
        raise MetadataFeatureError("feature manifest denylist is not frozen")
    if value["candidate_experts"] != list(CANDIDATE_EXPERTS):
        raise MetadataFeatureError("feature manifest candidate expert order is invalid")
    if value["unknown_category_encoding"] != "dedicated_one_hot":
        raise MetadataFeatureError("feature manifest unknown category encoding is invalid")

    categories_value = value["category_vocabulary"]
    if not isinstance(categories_value, list):
        raise MetadataFeatureError("category_vocabulary must be a list")
    categories = [_nonempty_string(item, "category_vocabulary item") for item in categories_value]
    if categories != sorted(set(categories)):
        raise MetadataFeatureError("category_vocabulary must be sorted and unique")

    expected_numeric_names = [
        "k_shot",
        "budget",
        *(f"static_cost.{expert}" for expert in CANDIDATE_EXPERTS),
    ]
    expected_encoded_names = [
        *(f"category={category}" for category in categories),
        "category=<UNKNOWN>",
        *expected_numeric_names,
    ]
    if value["encoded_feature_names"] != expected_encoded_names:
        raise MetadataFeatureError("encoded_feature_names do not match the frozen encoder")
    scaling_value = value["numeric_scaling"]
    if not isinstance(scaling_value, Mapping) or set(scaling_value) != set(expected_numeric_names):
        raise MetadataFeatureError("numeric_scaling has invalid feature names")
    scaling: dict[str, dict[str, float]] = {}
    for name in expected_numeric_names:
        parameters = scaling_value[name]
        if not isinstance(parameters, Mapping) or set(parameters) != {"mean", "scale"}:
            raise MetadataFeatureError(f"numeric_scaling.{name} is invalid")
        mean = _finite_number(parameters["mean"], f"numeric_scaling.{name}.mean")
        scale = _finite_number(parameters["scale"], f"numeric_scaling.{name}.scale")
        if scale <= 0:
            raise MetadataFeatureError(f"numeric_scaling.{name}.scale must be > 0")
        scaling[name] = {"mean": mean, "scale": scale}
    return {
        "protocol_version": METADATA_FEATURE_PROTOCOL_VERSION,
        "raw_feature_allowlist": list(METADATA_FEATURE_ALLOWLIST),
        "feature_denylist": sorted(FEATURE_DENYLIST),
        "category_vocabulary": categories,
        "candidate_experts": list(CANDIDATE_EXPERTS),
        "encoded_feature_names": expected_encoded_names,
        "numeric_scaling": scaling,
        "unknown_category_encoding": "dedicated_one_hot",
    }


def feature_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash a validated manifest using canonical JSON serialization."""

    canonical = validate_feature_manifest(manifest)
    serialized = json.dumps(
        canonical, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def write_feature_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically write ``feature_manifest.json``."""

    destination = Path(path)
    if destination.name != "feature_manifest.json":
        raise MetadataFeatureError("feature manifest filename must be feature_manifest.json")
    canonical = validate_feature_manifest(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(canonical, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_feature_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetadataFeatureError(f"could not load feature manifest {source}: {exc}") from exc
    return validate_feature_manifest(payload)


def _reject_denylisted_fields(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            if key in FEATURE_DENYLIST or key.endswith("_score") or key.endswith("_scores"):
                raise MetadataFeatureError(
                    f"{context} contains denylisted feature field {raw_key!r}"
                )
            _reject_denylisted_fields(child, context=context)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _reject_denylisted_fields(child, context=context)


def _canonical_expert(value: Any) -> str:
    normalized = "".join(character for character in str(value).lower() if character.isalnum())
    aliases = {
        "".join(character for character in expert.lower() if character.isalnum()): expert
        for expert in CANDIDATE_EXPERTS
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise MetadataFeatureError(f"unknown static-cost expert {value!r}") from exc


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MetadataFeatureError(f"{field} must be a non-empty string")
    return value.strip()


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MetadataFeatureError(f"{field} must be an integer >= 0")
    return value


def _positive_int(value: Any, field: str) -> int:
    parsed = _nonnegative_int(value, field)
    if parsed < 1:
        raise MetadataFeatureError(f"{field} must be an integer >= 1")
    return parsed


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise MetadataFeatureError(f"{field} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise MetadataFeatureError(f"{field} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise MetadataFeatureError(f"{field} must be a finite number")
    return parsed


def _finite_nonnegative(value: Any, field: str) -> float:
    parsed = _finite_number(value, field)
    if parsed < 0:
        raise MetadataFeatureError(f"{field} must be >= 0")
    return parsed


__all__ = [
    "FEATURE_DENYLIST",
    "METADATA_FEATURE_ALLOWLIST",
    "METADATA_FEATURE_PROTOCOL_VERSION",
    "MetadataFeatureError",
    "build_metadata_feature_record",
    "canonical_static_costs",
    "encode_metadata_features",
    "feature_manifest_sha256",
    "fit_feature_manifest",
    "load_feature_manifest",
    "validate_feature_manifest",
    "validate_metadata_feature_record",
    "write_feature_manifest",
]
