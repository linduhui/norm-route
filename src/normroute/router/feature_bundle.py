"""Leakage-safe Router feature assembly for Stage 5.

The Router consumes only model-ready numeric values from the normal-domain
signature, BIR-AD, and FBDP-AD.  Identifiers, paths, seeds, fold membership,
and support set ids remain provenance and are never appended to the learned
vector.
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
from .fbdp_ad_ablation import FBDPADAblationSpec, get_fbdp_ad_ablation


ROUTER_FEATURE_BUNDLE_PROTOCOL_VERSION = "stage5.router_feature_bundle.v3"
ROUTER_FEATURE_BUNDLE_COMPATIBLE_PROTOCOL_VERSIONS = (
    "stage5.router_feature_bundle.v2",
    ROUTER_FEATURE_BUNDLE_PROTOCOL_VERSION,
)
ROUTER_FEATURE_VIEWS = (
    "normal_only",
    "normal_bir",
    "normal_fbdp",
    "normal_bir_fbdp",
)

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
_BIR_CORE_FIELDS = (
    ("bir_support_bai", "support_bai"),
    ("bir_support_bai_std", "support_bai_std"),
    ("bir_query_bai", "query_bai"),
    ("bir_query_support_boundary_shift", "query_support_boundary_shift"),
    ("bir_absolute_boundary_shift", "absolute_boundary_shift"),
)
_BIR_RELIABILITY_FIELDS = (
    ("bir_support_bai_reliability", "support_bai_reliability"),
    ("bir_query_bai_reliability", "query_bai_reliability"),
)
_BIR_DISAGREEMENT_FIELDS = (
    (
        "bir_support_pixel_feature_disagreement",
        "support_pixel_feature_disagreement",
    ),
    (
        "bir_query_pixel_feature_disagreement",
        "query_pixel_feature_disagreement",
    ),
)
_FBDP_SUPPORT_FIELDS = (
    ("fbdp_fbc", "foreground_background_confusion"),
    ("fbdp_objectness_gate", "objectness_gate"),
    ("fbdp_objectness_contrast", "objectness_contrast"),
    ("fbdp_support_reliability", "support_reliability"),
    ("fbdp_support_assignment_confidence", "support_assignment_confidence"),
    ("fbdp_prototype_compactness", "prototype_compactness"),
    ("fbdp_foreground_support_coverage", "foreground_support_coverage"),
    ("fbdp_foreground_candidate_ratio", "foreground_candidate_ratio"),
    ("fbdp_background_candidate_ratio", "background_candidate_ratio"),
)
_FBDP_RAW_RESIDUAL_FIELDS = (
    ("fbdp_residual_q50", "residual_q50"),
    ("fbdp_residual_q90", "residual_q90"),
    ("fbdp_residual_q95", "residual_q95"),
    ("fbdp_residual_q99", "residual_q99"),
    ("fbdp_residual_mean", "residual_mean"),
    ("fbdp_residual_max", "residual_max"),
)
_FBDP_GATED_RESIDUAL_FIELDS = (
    ("fbdp_gated_residual_q50", "gated_residual_q50"),
    ("fbdp_gated_residual_q90", "gated_residual_q90"),
    ("fbdp_gated_residual_q95", "gated_residual_q95"),
    ("fbdp_gated_residual_q99", "gated_residual_q99"),
    ("fbdp_gated_residual_mean", "gated_residual_mean"),
    ("fbdp_gated_residual_max", "gated_residual_max"),
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
    feature_view: str = "normal_bir"
    fbdp_ablation_name: str = "not_applicable"
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
        if self.feature_view not in ROUTER_FEATURE_VIEWS:
            raise RouterFeatureBundleError("invalid Router feature view")

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
            "feature_view": self.feature_view,
            "fbdp_ablation_name": self.fbdp_ablation_name,
            "feature_names": list(self.feature_names),
            "values": list(self.values),
        }


def build_router_feature_bundle(
    normal_signature: Any,
    bir_ad_signature: Any | None = None,
    *,
    fbdp_ad_signature: Any | None = None,
    ablation: str | BIRADAblationSpec = "full",
    fbdp_ablation: str | FBDPADAblationSpec = "full",
    feature_view: str | None = None,
) -> RouterFeatureBundle:
    """Join matching normal/BIR/FBDP signatures without leakage."""

    normal = _record(normal_signature, "normal_signature")
    resolved_view = _resolve_feature_view(
        feature_view,
        has_bir=bir_ad_signature is not None,
        has_fbdp=fbdp_ad_signature is not None,
    )
    include_bir = resolved_view in {"normal_bir", "normal_bir_fbdp"}
    include_fbdp = resolved_view in {"normal_fbdp", "normal_bir_fbdp"}
    bir: Mapping[str, Any] | None = None
    fbdp: Mapping[str, Any] | None = None
    spec: BIRADAblationSpec | None = None
    fbdp_spec: FBDPADAblationSpec | None = None
    if include_bir:
        if bir_ad_signature is None:
            raise RouterFeatureBundleError(
                f"feature_view={resolved_view!r} requires a BIR-AD signature"
            )
        bir = _record(bir_ad_signature, "bir_ad_signature")
        _validate_matching_provenance(normal, bir, "BIR-AD")
        spec = get_bir_ad_ablation(ablation) if isinstance(ablation, str) else ablation
        if not isinstance(spec, BIRADAblationSpec):
            raise TypeError("ablation must be a name or BIRADAblationSpec")
    if include_fbdp:
        if fbdp_ad_signature is None:
            raise RouterFeatureBundleError(
                f"feature_view={resolved_view!r} requires an FBDP-AD signature"
            )
        fbdp = _record(fbdp_ad_signature, "fbdp_ad_signature")
        _validate_matching_provenance(normal, fbdp, "FBDP-AD")
        fbdp_spec = (
            get_fbdp_ad_ablation(fbdp_ablation)
            if isinstance(fbdp_ablation, str)
            else fbdp_ablation
        )
        if not isinstance(fbdp_spec, FBDPADAblationSpec):
            raise TypeError(
                "fbdp_ablation must be a name or FBDPADAblationSpec"
            )

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

    if include_bir:
        assert bir is not None and spec is not None
        if spec.include_bai_core:
            for feature_name, record_name in _BIR_CORE_FIELDS:
                _append_scalar(names, values, feature_name, bir.get(record_name))
        if spec.include_reliability:
            for feature_name, record_name in _BIR_RELIABILITY_FIELDS:
                _append_scalar(names, values, feature_name, bir.get(record_name))
        if spec.include_disagreement:
            for feature_name, record_name in _BIR_DISAGREEMENT_FIELDS:
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

    if include_fbdp:
        assert fbdp is not None and fbdp_spec is not None
        if fbdp_spec.include_support_features:
            for feature_name, record_name in _FBDP_SUPPORT_FIELDS:
                _append_scalar(names, values, feature_name, fbdp.get(record_name))
            loo_valid = bool(
                fbdp.get("leave_one_out_reconstruction_valid", False)
            )
            loo_margin = fbdp.get("leave_one_out_reconstruction_margin")
            if loo_valid and loo_margin is None:
                raise RouterFeatureBundleError(
                    "valid FBDP leave-one-out margin is missing"
                )
            _append_scalar(
                names,
                values,
                "fbdp_leave_one_out_reconstruction_margin",
                loo_margin if loo_valid else 0.0,
            )
            for name, field in (
                ("fbdp_leave_one_out_reconstruction_valid", "leave_one_out_reconstruction_valid"),
                ("fbdp_support_consistency_valid", "support_consistency_valid"),
                ("fbdp_foreground_candidate_fallback", "foreground_candidate_fallback"),
            ):
                _append_scalar(names, values, name, 1.0 if bool(fbdp.get(field)) else 0.0)
        if fbdp_spec.include_raw_residual_features:
            for feature_name, record_name in _FBDP_RAW_RESIDUAL_FIELDS:
                _append_scalar(names, values, feature_name, fbdp.get(record_name))
        if fbdp_spec.include_gated_residual_features:
            for feature_name, record_name in _FBDP_GATED_RESIDUAL_FIELDS:
                _append_scalar(names, values, feature_name, fbdp.get(record_name))
        if fbdp_spec.include_assignment_features:
            _extend_vector(
                names,
                values,
                "fbdp_margin_quantile",
                fbdp.get("foreground_background_margin_quantiles"),
                length=4,
            )
            for name, field in (
                ("fbdp_margin_mean", "foreground_background_margin_mean"),
                ("fbdp_margin_min", "foreground_background_margin_min"),
                ("fbdp_margin_max", "foreground_background_margin_max"),
            ):
                _append_scalar(names, values, name, fbdp.get(field))
            _extend_vector(
                names,
                values,
                "fbdp_assignment_entropy_quantile",
                fbdp.get("assignment_entropy_quantiles"),
                length=4,
            )
            _append_scalar(
                names,
                values,
                "fbdp_assignment_entropy_mean",
                fbdp.get("assignment_entropy_mean"),
            )
            _append_scalar(
                names,
                values,
                "fbdp_assignment_entropy_max",
                fbdp.get("assignment_entropy_max"),
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
        bir_normalization_sha256=(
            _nonempty_text(bir.get("normalization_sha256"), "normalization_sha256")
            if bir is not None
            else "not_applicable"
        ),
        bir_alignment_fingerprint=(
            _nonempty_text(bir.get("alignment_fingerprint"), "alignment_fingerprint")
            if bir is not None
            else "not_applicable"
        ),
        ablation_name=(
            spec.name
            if spec is not None
            else (fbdp_spec.name if fbdp_spec is not None else "normal_only")
        ),
        feature_names=tuple(names),
        values=tuple(values),
        feature_view=resolved_view,
        fbdp_ablation_name=(
            fbdp_spec.name if fbdp_spec is not None else "not_applicable"
        ),
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
    reference_feature_view = ordered[0].feature_view
    reference_fbdp_ablation = ordered[0].fbdp_ablation_name
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
        if item.feature_view != reference_feature_view:
            raise RouterFeatureBundleError(
                "Router feature artifact mixes feature views"
            )
        if item.fbdp_ablation_name != reference_fbdp_ablation:
            raise RouterFeatureBundleError(
                "Router feature artifact mixes FBDP-AD ablations"
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
    bir_ad_signatures: Sequence[Any] | None = None,
    *,
    fbdp_ad_signatures: Sequence[Any] | None = None,
    ablation: str | BIRADAblationSpec = "full",
    fbdp_ablation: str | FBDPADAblationSpec = "full",
    feature_view: str | None = None,
) -> list[RouterFeatureBundle]:
    """Join complete task collections and reject missing/duplicate rows."""

    normal_by_task = _records_by_task(normal_signatures, "normal signatures")
    resolved_view = _resolve_feature_view(
        feature_view,
        has_bir=bir_ad_signatures is not None,
        has_fbdp=fbdp_ad_signatures is not None,
    )
    bir_by_task = (
        _records_by_task(bir_ad_signatures, "BIR-AD signatures")
        if bir_ad_signatures is not None
        else {}
    )
    fbdp_by_task = (
        _records_by_task(fbdp_ad_signatures, "FBDP-AD signatures")
        if fbdp_ad_signatures is not None
        else {}
    )
    if resolved_view in {"normal_bir", "normal_bir_fbdp"}:
        _validate_task_sets(normal_by_task, bir_by_task, "BIR-AD")
    if resolved_view in {"normal_fbdp", "normal_bir_fbdp"}:
        _validate_task_sets(normal_by_task, fbdp_by_task, "FBDP-AD")
    return [
        build_router_feature_bundle(
            normal_by_task[task_id],
            bir_by_task.get(task_id),
            fbdp_ad_signature=fbdp_by_task.get(task_id),
            ablation=ablation,
            fbdp_ablation=fbdp_ablation,
            feature_view=resolved_view,
        )
        for task_id in sorted(normal_by_task)
    ]


def _resolve_feature_view(
    value: str | None, *, has_bir: bool, has_fbdp: bool
) -> str:
    if value is None:
        if has_bir and has_fbdp:
            return "normal_bir_fbdp"
        if has_bir:
            return "normal_bir"
        if has_fbdp:
            return "normal_fbdp"
        return "normal_only"
    resolved = str(value).strip().lower()
    if resolved not in ROUTER_FEATURE_VIEWS:
        raise RouterFeatureBundleError(
            f"feature_view must be one of {ROUTER_FEATURE_VIEWS}"
        )
    return resolved


def _validate_task_sets(
    normal: Mapping[str, Any], other: Mapping[str, Any], name: str
) -> None:
    if set(normal) != set(other):
        raise RouterFeatureBundleError(
            f"normal and {name} signature task sets disagree; "
            f"normal_only={sorted(set(normal) - set(other))[:5]}, "
            f"{name.lower()}_only={sorted(set(other) - set(normal))[:5]}"
        )


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
    normal: Mapping[str, Any], other: Mapping[str, Any], name: str
) -> None:
    for field in _PROVENANCE_FIELDS:
        if field not in normal or field not in other:
            raise RouterFeatureBundleError(
                f"both signatures must contain provenance field {field!r}"
            )
        if normal[field] != other[field]:
            raise RouterFeatureBundleError(
                f"signature provenance disagrees for {field}"
            )
    normal_supports = tuple(
        sorted(_string_sequence(normal.get("support_image_sha256s")))
    )
    other_supports = tuple(
        sorted(_string_sequence(other.get("support_image_sha256s")))
    )
    if normal_supports != other_supports:
        raise RouterFeatureBundleError(
            f"normal/{name} provenance disagrees for support image hashes"
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
    "ROUTER_FEATURE_BUNDLE_COMPATIBLE_PROTOCOL_VERSIONS",
    "ROUTER_FEATURE_BUNDLE_PROTOCOL_VERSION",
    "ROUTER_FEATURE_VIEWS",
    "RouterFeatureBundle",
    "RouterFeatureBundleError",
    "build_router_feature_bundle",
    "build_router_feature_bundles",
    "write_router_feature_bundles_jsonl",
]
