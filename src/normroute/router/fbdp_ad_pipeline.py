"""Leakage-safe batch signatures for FBDP-AD Stage 5 routing.

The pipeline resolves every support set through the same audited task/support
contract as BIR-AD, reuses the content-addressed frozen feature cache, builds
one support context per unique support-image set, and emits one deterministic
signature per task.  Evaluator-side outcomes are not accepted by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .bir_ad_pipeline import (
    _resolve_tasks_and_supports,
    read_supports_csv,
    read_tasks_jsonl,
)
from .fbdp_ad import (
    FBDP_AD_PROTOCOL_VERSION,
    FBDP_AD_RESIDUAL_QUANTILES,
    FBDPADResult,
    compose_fbdp_ad_query,
    prepare_fbdp_ad_support_context,
)
from .feature_cache import FeatureCache


FBDP_AD_SIGNATURE_PROTOCOL_VERSION = "stage5.fbdp_ad_signature.v2"
FBDP_AD_SIGNATURES_NAME = "fbdp_ad_signatures.jsonl"
FBDP_AD_FAILURES_NAME = "fbdp_ad_failures.json"

FBDP_AD_SIGNATURE_COLUMNS = (
    "protocol_version",
    "fbdp_ad_protocol_version",
    "task_id",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
    "encoder_fingerprint",
    "query_image_sha256",
    "support_image_sha256s",
    "patch_grid_shape",
    "support_count",
    "query_patch_count",
    "support_patch_count",
    "foreground_candidate_count",
    "background_candidate_count",
    "foreground_candidate_ratio",
    "background_candidate_ratio",
    "foreground_prototype_count",
    "background_prototype_count",
    "foreground_background_confusion",
    "objectness_gate",
    "objectness_contrast",
    "support_reliability",
    "support_assignment_confidence",
    "prototype_compactness",
    "foreground_support_coverage",
    "leave_one_out_reconstruction_margin",
    "leave_one_out_reconstruction_valid",
    "support_consistency_valid",
    "foreground_candidate_fallback",
    "prototype_method",
    "consistency_mode",
    "consistency_window_radius",
    "background_candidate_mode",
    "query_bank_mode",
    "k1_gate_policy",
    "k1_gate_scale",
    "fallback_gate_cap",
    "use_cross_support_consistency",
    "use_fbc_in_gate",
    "use_objectness_gate",
    "temperature",
    "residual_quantile_levels",
    "residual_quantiles",
    "gated_residual_quantiles",
    "residual_q50",
    "residual_q90",
    "residual_q95",
    "residual_q99",
    "gated_residual_q50",
    "gated_residual_q90",
    "gated_residual_q95",
    "gated_residual_q99",
    "residual_mean",
    "residual_max",
    "gated_residual_mean",
    "gated_residual_max",
    "foreground_background_margin_quantiles",
    "foreground_background_margin_mean",
    "foreground_background_margin_min",
    "foreground_background_margin_max",
    "assignment_entropy_quantiles",
    "assignment_entropy_mean",
    "assignment_entropy_max",
)


class FBDPADPipelineError(RuntimeError):
    """Base failure for FBDP batch construction."""


class FBDPADArtifactError(FBDPADPipelineError):
    """Raised when a signature artifact is malformed or incomplete."""


@dataclass(frozen=True)
class FBDPADSignature:
    task_id: str
    dataset: str
    category: str
    k_shot: int
    seed: int
    support_set_id: str
    encoder_fingerprint: str
    query_image_sha256: str
    support_image_sha256s: tuple[str, ...]
    result: FBDPADResult
    protocol_version: str = FBDP_AD_SIGNATURE_PROTOCOL_VERSION

    def to_record(self) -> dict[str, Any]:
        np = _numpy()
        result = self.result
        context = result.support_context
        levels = tuple(float(value) for value in context.residual_quantile_levels)
        if levels != FBDP_AD_RESIDUAL_QUANTILES:
            raise FBDPADArtifactError(
                "batch FBDP signatures require frozen residual quantiles "
                f"{FBDP_AD_RESIDUAL_QUANTILES}"
            )
        raw = _float_list(result.residual_quantiles, np)
        gated = _float_list(result.gated_residual_quantiles, np)
        margin_quantiles = _float_list(
            result.foreground_background_margin_quantiles, np
        )
        entropy_quantiles = _float_list(
            result.foreground_assignment_entropy_quantiles, np
        )
        record = {
            "protocol_version": self.protocol_version,
            "fbdp_ad_protocol_version": result.protocol_version,
            "task_id": self.task_id,
            "dataset": self.dataset,
            "category": self.category,
            "k_shot": self.k_shot,
            "seed": self.seed,
            "support_set_id": self.support_set_id,
            "encoder_fingerprint": self.encoder_fingerprint,
            "query_image_sha256": self.query_image_sha256,
            "support_image_sha256s": list(self.support_image_sha256s),
            "patch_grid_shape": list(result.patch_grid_shape),
            "support_count": context.support_count,
            "query_patch_count": result.patch_count,
            "support_patch_count": (
                context.support_count * math.prod(context.patch_grid_shape)
            ),
            "foreground_candidate_count": context.foreground_candidate_count,
            "background_candidate_count": context.background_candidate_count,
            "foreground_candidate_ratio": context.foreground_candidate_ratio,
            "background_candidate_ratio": context.background_candidate_ratio,
            "foreground_prototype_count": context.foreground_prototype_count,
            "background_prototype_count": context.background_prototype_count,
            "foreground_background_confusion": context.fbc,
            "objectness_gate": context.objectness_gate,
            "objectness_contrast": context.objectness_contrast,
            "support_reliability": context.support_reliability,
            "support_assignment_confidence": (
                context.support_assignment_confidence
            ),
            "prototype_compactness": context.prototype_compactness,
            "foreground_support_coverage": context.foreground_support_coverage,
            "leave_one_out_reconstruction_margin": (
                context.leave_one_out_reconstruction_margin
            ),
            "leave_one_out_reconstruction_valid": (
                context.leave_one_out_reconstruction_valid
            ),
            "support_consistency_valid": context.support_consistency_valid,
            "foreground_candidate_fallback": (
                context.foreground_candidate_fallback
            ),
            "prototype_method": context.prototype_method,
            "consistency_mode": context.consistency_mode,
            "consistency_window_radius": context.consistency_window_radius,
            "background_candidate_mode": context.background_candidate_mode,
            "query_bank_mode": context.query_bank_mode,
            "k1_gate_policy": context.k1_gate_policy,
            "k1_gate_scale": context.k1_gate_scale,
            "fallback_gate_cap": context.fallback_gate_cap,
            "use_cross_support_consistency": (
                context.use_cross_support_consistency
            ),
            "use_fbc_in_gate": context.use_fbc_in_gate,
            "use_objectness_gate": context.use_objectness_gate,
            "temperature": context.temperature,
            "residual_quantile_levels": list(levels),
            "residual_quantiles": raw,
            "gated_residual_quantiles": gated,
            "residual_q50": raw[0],
            "residual_q90": raw[1],
            "residual_q95": raw[2],
            "residual_q99": raw[3],
            "gated_residual_q50": gated[0],
            "gated_residual_q90": gated[1],
            "gated_residual_q95": gated[2],
            "gated_residual_q99": gated[3],
            "residual_mean": float(result.residuals.mean(dtype=np.float64)),
            "residual_max": float(result.residuals.max()),
            "gated_residual_mean": float(
                result.gated_residuals.mean(dtype=np.float64)
            ),
            "gated_residual_max": float(result.gated_residuals.max()),
            "foreground_background_margin_quantiles": margin_quantiles,
            "foreground_background_margin_mean": float(
                result.foreground_background_margin.mean(dtype=np.float64)
            ),
            "foreground_background_margin_min": float(
                result.foreground_background_margin.min()
            ),
            "foreground_background_margin_max": float(
                result.foreground_background_margin.max()
            ),
            "assignment_entropy_quantiles": entropy_quantiles,
            "assignment_entropy_mean": float(
                result.foreground_assignment_entropy.mean(dtype=np.float64)
            ),
            "assignment_entropy_max": float(
                result.foreground_assignment_entropy.max()
            ),
        }
        if tuple(record) != FBDP_AD_SIGNATURE_COLUMNS:
            raise FBDPADArtifactError("FBDP signature schema drifted")
        _assert_finite_record(record)
        return record


class FBDPADTaskEncoder:
    """Encode complete tasks with support-context and feature-cache reuse."""

    def __init__(
        self,
        feature_cache: FeatureCache,
        feature_encoder: Any,
        *,
        batch_size: int = 32,
        patch_grid_shape: tuple[int, int] | None = None,
        fbdp_ad_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(feature_cache, FeatureCache):
            raise TypeError("feature_cache must be FeatureCache")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.feature_cache = feature_cache
        self.feature_encoder = feature_encoder
        self.batch_size = batch_size
        self.patch_grid_shape = patch_grid_shape
        self.fbdp_ad_kwargs = dict(fbdp_ad_kwargs or {})
        forbidden = {"patch_grid_shape", "residual_quantiles"}.intersection(
            self.fbdp_ad_kwargs
        )
        if forbidden:
            raise ValueError(
                f"fbdp_ad_kwargs contains encoder-controlled fields: {sorted(forbidden)}"
            )
        self.last_run_statistics: dict[str, Any] = {}

    def encode_tasks(
        self,
        tasks: Sequence[Mapping[str, Any]],
        support_sets: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]],
    ) -> list[FBDPADSignature]:
        try:
            normalized_tasks, task_support_paths = _resolve_tasks_and_supports(
                tasks, support_sets
            )
        except Exception as exc:
            if isinstance(exc, FBDPADPipelineError):
                raise
            raise FBDPADPipelineError(str(exc)) from exc
        unique_paths = sorted(
            {
                Path(task["query_path"])
                for task in normalized_tasks
            }.union(
                path
                for paths in task_support_paths.values()
                for path in paths
            ),
            key=str,
        )
        try:
            cached = self.feature_cache.get_or_encode(
                unique_paths,
                self.feature_encoder,
                batch_size=self.batch_size,
            )
        except Exception as exc:
            raise FBDPADPipelineError(f"feature-cache encoding failed: {exc}") from exc
        cached_by_path = dict(zip(unique_paths, cached))
        task_inputs = []
        for task in normalized_tasks:
            query_features = cached_by_path[Path(task["query_path"])]
            support_features = [
                cached_by_path[path]
                for path in task_support_paths[task["task_id"]]
            ]
            support_features.sort(key=lambda item: item.image_sha256)
            support_hashes = tuple(item.image_sha256 for item in support_features)
            if len(set(support_hashes)) != len(support_hashes):
                raise FBDPADPipelineError(
                    f"task {task['task_id']!r} contains byte-identical supports"
                )
            task_inputs.append(
                (support_hashes, task, query_features, support_features)
            )
        task_inputs.sort(key=lambda item: (item[0], item[1]["task_id"]))

        signatures: list[FBDPADSignature] = []
        active_hashes: tuple[str, ...] | None = None
        active_context = None
        context_build_count = 0
        context_cache_hits = 0
        for support_hashes, task, query_features, support_features in task_inputs:
            if support_hashes != active_hashes:
                try:
                    active_context = prepare_fbdp_ad_support_context(
                        [item.patch_features for item in support_features],
                        patch_grid_shape=self.patch_grid_shape,
                        residual_quantiles=FBDP_AD_RESIDUAL_QUANTILES,
                        **self.fbdp_ad_kwargs,
                    )
                except Exception as exc:
                    raise FBDPADPipelineError(
                        f"task {task['task_id']!r} support context failed: {exc}"
                    ) from exc
                active_hashes = support_hashes
                context_build_count += 1
            else:
                context_cache_hits += 1
            if active_context is None:
                raise FBDPADPipelineError("support context was not initialized")
            try:
                result = compose_fbdp_ad_query(
                    query_features.patch_features, active_context
                )
            except Exception as exc:
                raise FBDPADPipelineError(
                    f"task {task['task_id']!r} query composition failed: {exc}"
                ) from exc
            signatures.append(
                FBDPADSignature(
                    task_id=task["task_id"],
                    dataset=task["dataset"],
                    category=task["category"],
                    k_shot=task["k_shot"],
                    seed=task["seed"],
                    support_set_id=task["support_set_id"],
                    encoder_fingerprint=self.feature_cache.encoder_fingerprint,
                    query_image_sha256=query_features.image_sha256,
                    support_image_sha256s=support_hashes,
                    result=result,
                )
            )
        signatures.sort(key=lambda item: item.task_id)
        self.last_run_statistics = {
            "task_count": len(signatures),
            "unique_image_count": len(unique_paths),
            "unique_support_context_count": context_build_count,
            "support_context_cache_hits": context_cache_hits,
        }
        return signatures


def write_fbdp_ad_signatures_jsonl(
    signatures: Sequence[FBDPADSignature], output_path: str | Path
) -> Path:
    if isinstance(signatures, (str, bytes)) or not signatures:
        raise FBDPADArtifactError("signatures must be non-empty")
    ordered = sorted(signatures, key=lambda item: item.task_id)
    if len({item.task_id for item in ordered}) != len(ordered):
        raise FBDPADArtifactError("duplicate FBDP signature task_id")
    reference_encoder = ordered[0].encoder_fingerprint
    for item in ordered:
        if item.encoder_fingerprint != reference_encoder:
            raise FBDPADArtifactError("FBDP signatures mix encoder fingerprints")
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
    path = Path(output_path)
    _atomic_write_text(path, text)
    return path


def build_fbdp_ad_signatures(
    tasks: Sequence[Mapping[str, Any]],
    support_sets: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]],
    *,
    encoder: FBDPADTaskEncoder,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    signatures_path = destination / FBDP_AD_SIGNATURES_NAME
    failures_path = destination / FBDP_AD_FAILURES_NAME
    try:
        signatures = encoder.encode_tasks(tasks, support_sets)
        write_fbdp_ad_signatures_jsonl(signatures, signatures_path)
        _atomic_write_json(failures_path, [])
        return signatures_path, failures_path
    except Exception as exc:
        task_ids = [
            str(task.get("task_id", f"task-index-{index}"))
            if isinstance(task, Mapping)
            else f"task-index-{index}"
            for index, task in enumerate(tasks)
        ]
        failures = [
            {"task_id": task_id, "code": type(exc).__name__, "message": str(exc)}
            for task_id in task_ids
        ] or [{"task_id": None, "code": type(exc).__name__, "message": str(exc)}]
        _atomic_write_text(signatures_path, "")
        _atomic_write_json(failures_path, failures)
        raise


def _assert_finite_record(record: Mapping[str, Any]) -> None:
    def visit(value: Any, path: str) -> None:
        if value is None or isinstance(value, (str, bool)):
            return
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                raise FBDPADArtifactError(f"non-finite signature value at {path}")
            return
        if isinstance(value, Sequence):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        raise FBDPADArtifactError(f"unsupported signature value at {path}")

    for key, value in record.items():
        visit(value, key)


def _float_list(value: Any, np: Any) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not bool(np.all(np.isfinite(array))):
        raise FBDPADArtifactError("signature vector must be finite and one-dimensional")
    return [float(item) for item in array.tolist()]


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )


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


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise FBDPADArtifactError("FBDP signatures require NumPy") from exc
    return np


__all__ = [
    "FBDP_AD_FAILURES_NAME",
    "FBDP_AD_SIGNATURE_COLUMNS",
    "FBDP_AD_SIGNATURE_PROTOCOL_VERSION",
    "FBDP_AD_SIGNATURES_NAME",
    "FBDPADArtifactError",
    "FBDPADPipelineError",
    "FBDPADSignature",
    "FBDPADTaskEncoder",
    "build_fbdp_ad_signatures",
    "read_supports_csv",
    "read_tasks_jsonl",
    "write_fbdp_ad_signatures_jsonl",
]
