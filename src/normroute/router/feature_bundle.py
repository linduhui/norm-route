"""Leakage-safe Router feature assembly for Stage 5.

The Router consumes only model-ready numeric values from the normal-domain
signature and BIR-AD.  Identifiers, paths, seeds, fold membership, and support
set ids remain provenance and are never appended to the learned vector.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .bir_ad_ablation import BIRADAblationSpec, get_bir_ad_ablation


ROUTER_FEATURE_BUNDLE_PROTOCOL_VERSION = "stage5.router_feature_bundle.v1"

_PROVENANCE_FIELDS = (
    "task_id",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
    "encoder_fingerprint",
    "query_image_sha256",
)
_BIR_SCALAR_FIELDS = (
    ("bir_support_bai", "support_bai"),
    ("bir_support_bai_std", "support_bai_std"),
    ("bir_query_bai", "query_bai"),
    ("bir_query_support_boundary_shift", "query_support_boundary_shift"),
    ("bir_absolute_boundary_shift", "absolute_boundary_shift"),
    ("bir_support_bai_reliability", "support_bai_reliability"),
    ("bir_query_bai_reliability", "query_bai_reliability"),
    (
        "bir_support_pixel_feature_disagreement",
        "support_pixel_feature_disagreement",
    ),
    (
        "bir_query_pixel_feature_disagreement",
        "query_pixel_feature_disagreement",
    ),
)


class RouterFeatureBundleError(ValueError):
    """Raised when two signatures cannot form one Router input."""


@dataclass(frozen=True)
class RouterFeatureBundle:
    """One numeric Router input with non-learned provenance kept separately."""

    task_id: str
    dataset: str
    category: str
    k_shot: int
    seed: int
    support_set_id: str
    encoder_fingerprint: str
    query_image_sha256: str
    support_image_sha256s: tuple[str, ...]
    bir_normalization_sha256: str
    bir_alignment_fingerprint: str
    ablation_name: str
    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    protocol_version: str = ROUTER_FEATURE_BUNDLE_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if not self.feature_names or len(self.feature_names) != len(self.values):
            raise RouterFeatureBundleError(
                "Router feature names and values must be non-empty and aligned"
            )
        if len(set(self.feature_names)) != len(self.feature_names):
            raise RouterFeatureBundleError("Router feature names must be unique")
        if not all(math.isfinite(value) for value in self.values):
            raise RouterFeatureBundleError("Router features must be finite")

    def to_record(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "task_id": self.task_id,
            "dataset": self.dataset,
            "category": self.category,
            "k_shot": self.k_shot,
            "seed": self.seed,
            "support_set_id": self.support_set_id,
            "encoder_fingerprint": self.encoder_fingerprint,
            "query_image_sha256": self.query_image_sha256,
            "support_image_sha256s": list(self.support_image_sha256s),
            "bir_normalization_sha256": self.bir_normalization_sha256,
            "bir_alignment_fingerprint": self.bir_alignment_fingerprint,
            "ablation_name": self.ablation_name,
            "feature_names": list(self.feature_names),
            "values": list(self.values),
        }


def build_router_feature_bundle(
    normal_signature: Any,
    bir_ad_signature: Any,
    *,
    ablation: str | BIRADAblationSpec = "full",
) -> RouterFeatureBundle:
    """Join matching normal-domain and BIR-AD signatures without leakage."""

    normal = _record(normal_signature, "normal_signature")
    bir = _record(bir_ad_signature, "bir_ad_signature")
    _validate_matching_provenance(normal, bir)
    spec = (
        get_bir_ad_ablation(ablation)
        if isinstance(ablation, str)
        else ablation
    )
    if not isinstance(spec, BIRADAblationSpec):
        raise TypeError("ablation must be a name or BIRADAblationSpec")

    names: list[str] = []
    values: list[float] = []
    _extend_vector(names, values, "normal_niv", normal.get("niv"), length=6)
    _extend_vector(
        names,
        values,
        "normal_query_global_residual",
        normal.get("query_global_residual"),
    )
    _append_scalar(
        names,
        values,
        "normal_query_global_residual_l2",
        normal.get("query_global_residual_l2"),
    )
    for name in (
        "patch_nn_q50",
        "patch_nn_q90",
        "patch_nn_q95",
        "patch_nn_q99",
        "patch_nn_mean",
        "patch_nn_max",
    ):
        _append_scalar(names, values, f"normal_{name}", normal.get(name))

    if spec.include_bai_features:
        for feature_name, record_name in _BIR_SCALAR_FIELDS:
            _append_scalar(names, values, feature_name, bir.get(record_name))
    if spec.include_consistency_features:
        consistency_valid = bool(
            bir.get("support_boundary_consistency_valid", False)
        )
        consistency = bir.get("support_boundary_consistency")
        if consistency_valid and consistency is None:
            raise RouterFeatureBundleError(
                "valid support boundary consistency is missing"
            )
        _append_scalar(
            names,
            values,
            "bir_support_boundary_consistency",
            consistency if consistency_valid else 0.0,
        )
        _append_scalar(
            names,
            values,
            "bir_support_boundary_consistency_valid",
            1.0 if consistency_valid else 0.0,
        )
        _append_scalar(
            names,
            values,
            "bir_query_support_boundary_consistency",
            bir.get("query_support_boundary_consistency"),
        )
    if spec.include_representations:
        _extend_vector(
            names,
            values,
            "bir_clear_representation",
            bir.get("clear_representation"),
        )
        _extend_vector(
            names,
            values,
            "bir_ambiguous_representation",
            bir.get("ambiguous_representation"),
        )

    return RouterFeatureBundle(
        task_id=str(normal["task_id"]),
        dataset=str(normal["dataset"]),
        category=str(normal["category"]),
        k_shot=int(normal["k_shot"]),
        seed=int(normal["seed"]),
        support_set_id=str(normal["support_set_id"]),
        encoder_fingerprint=str(normal["encoder_fingerprint"]),
        query_image_sha256=str(normal["query_image_sha256"]),
        support_image_sha256s=tuple(
            sorted(_string_sequence(normal.get("support_image_sha256s")))
        ),
        bir_normalization_sha256=_nonempty_text(
            bir.get("normalization_sha256"), "normalization_sha256"
        ),
        bir_alignment_fingerprint=_nonempty_text(
            bir.get("alignment_fingerprint"), "alignment_fingerprint"
        ),
        ablation_name=spec.name,
        feature_names=tuple(names),
        values=tuple(values),
    )


def write_router_feature_bundles_jsonl(
    bundles: Sequence[RouterFeatureBundle],
    output_path: str | Path,
) -> Path:
    if isinstance(bundles, (str, bytes)) or not bundles:
        raise RouterFeatureBundleError("bundles must be non-empty")
    ordered = sorted(bundles, key=lambda item: item.task_id)
    if len({item.task_id for item in ordered}) != len(ordered):
        raise RouterFeatureBundleError("duplicate Router feature task_id")
    reference_names = ordered[0].feature_names
    reference_ablation = ordered[0].ablation_name
    reference_encoder = ordered[0].encoder_fingerprint
    reference_normalization = ordered[0].bir_normalization_sha256
    reference_alignment = ordered[0].bir_alignment_fingerprint
    for item in ordered[1:]:
        if item.feature_names != reference_names:
            raise RouterFeatureBundleError(
                "Router feature schema differs across tasks"
            )
        if item.ablation_name != reference_ablation:
            raise RouterFeatureBundleError(
                "Router feature artifact mixes BIR-AD ablations"
            )
        if item.encoder_fingerprint != reference_encoder:
            raise RouterFeatureBundleError(
                "Router feature artifact mixes encoder fingerprints"
            )
        if item.bir_normalization_sha256 != reference_normalization:
            raise RouterFeatureBundleError(
                "Router feature artifact mixes fold normalizations"
            )
        if item.bir_alignment_fingerprint != reference_alignment:
            raise RouterFeatureBundleError(
                "Router feature artifact mixes alignment transforms"
            )
    path = Path(output_path)
    text = "".join(
        json.dumps(
            item.to_record(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
        for item in ordered
    )
    _atomic_write_text(path, text)
    return path


def build_router_feature_bundles(
    normal_signatures: Sequence[Any],
    bir_ad_signatures: Sequence[Any],
    *,
    ablation: str | BIRADAblationSpec = "full",
) -> list[RouterFeatureBundle]:
    """Join two complete task collections and reject missing/duplicate rows."""

    normal_by_task = _records_by_task(normal_signatures, "normal signatures")
    bir_by_task = _records_by_task(bir_ad_signatures, "BIR-AD signatures")
    if set(normal_by_task) != set(bir_by_task):
        raise RouterFeatureBundleError(
            "normal and BIR-AD signature task sets disagree; "
            f"normal_only={sorted(set(normal_by_task) - set(bir_by_task))[:5]}, "
            f"bir_only={sorted(set(bir_by_task) - set(normal_by_task))[:5]}"
        )
    return [
        build_router_feature_bundle(
            normal_by_task[task_id],
            bir_by_task[task_id],
            ablation=ablation,
        )
        for task_id in sorted(normal_by_task)
    ]


def _record(value: Any, field: str) -> Mapping[str, Any]:
    if hasattr(value, "to_record"):
        value = value.to_record()
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a record or expose to_record()")
    return value


def _records_by_task(
    values: Sequence[Any], field: str
) -> dict[str, Mapping[str, Any]]:
    if isinstance(values, (str, bytes)) or not values:
        raise RouterFeatureBundleError(f"{field} must be non-empty")
    result: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(values):
        record = _record(value, f"{field}[{index}]")
        task_id = _nonempty_text(record.get("task_id"), "task_id")
        if task_id in result:
            raise RouterFeatureBundleError(
                f"{field} duplicates task_id={task_id!r}"
            )
        result[task_id] = record
    return result


def _validate_matching_provenance(
    normal: Mapping[str, Any], bir: Mapping[str, Any]
) -> None:
    for field in _PROVENANCE_FIELDS:
        if field not in normal or field not in bir:
            raise RouterFeatureBundleError(
                f"both signatures must contain provenance field {field!r}"
            )
        if normal[field] != bir[field]:
            raise RouterFeatureBundleError(
                f"signature provenance disagrees for {field}"
            )
    normal_supports = tuple(
        sorted(_string_sequence(normal.get("support_image_sha256s")))
    )
    bir_supports = tuple(
        sorted(_string_sequence(bir.get("support_image_sha256s")))
    )
    if normal_supports != bir_supports:
        raise RouterFeatureBundleError(
            "signature provenance disagrees for support image hashes"
        )


def _append_scalar(
    names: list[str], values: list[float], name: str, value: Any
) -> None:
    if isinstance(value, bool):
        numeric = float(value)
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise RouterFeatureBundleError(
                f"Router feature {name!r} must be numeric"
            ) from exc
    if not math.isfinite(numeric):
        raise RouterFeatureBundleError(
            f"Router feature {name!r} must be finite"
        )
    names.append(name)
    values.append(numeric)


def _extend_vector(
    names: list[str],
    values: list[float],
    prefix: str,
    vector: Any,
    *,
    length: int | None = None,
) -> None:
    if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
        raise RouterFeatureBundleError(
            f"Router feature vector {prefix!r} must be a sequence"
        )
    if not vector or (length is not None and len(vector) != length):
        expected = f" of length {length}" if length is not None else ""
        raise RouterFeatureBundleError(
            f"Router feature vector {prefix!r} must be non-empty{expected}"
        )
    for index, item in enumerate(vector):
        _append_scalar(names, values, f"{prefix}_{index:04d}", item)


def _string_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RouterFeatureBundleError(
            "support_image_sha256s must be a sequence"
        )
    result = tuple(str(item).strip() for item in value)
    if not result or any(not item for item in result):
        raise RouterFeatureBundleError(
            "support_image_sha256s must be non-empty strings"
        )
    return result


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RouterFeatureBundleError(f"{field} must be a non-empty string")
    return value.strip()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "ROUTER_FEATURE_BUNDLE_PROTOCOL_VERSION",
    "RouterFeatureBundle",
    "RouterFeatureBundleError",
    "build_router_feature_bundle",
    "build_router_feature_bundles",
    "write_router_feature_bundles_jsonl",
]
