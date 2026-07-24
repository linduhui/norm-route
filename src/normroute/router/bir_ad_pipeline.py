"""Batch, fold-normalized BIR-AD signature production for Stage 5.

This module closes the engineering boundary between the verified feature cache,
strictly aligned pixel views, normal support sets, and Router-ready task
signatures.  It consumes no evaluator-only target, label, mask, defect type,
expert outcome, or Oracle field.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .bir_ad import (
    BIR_AD_PROTOCOL_VERSION,
    BIRADInputError,
    BIRADNormalizationStats,
    BIRADTaskResult,
    PatchAlignedImage,
    compose_bir_ad_task,
    compute_bir_ad,
    fit_bir_ad_normalization,
)
from .feature_cache import CachedImageFeatures, FeatureCache
from .normal_domain import (
    NormalDomainInputError,
    _group_support_sets,
    _normalize_tasks,
    _validate_task_supports,
)


BIR_AD_SIGNATURE_PROTOCOL_VERSION = "stage5.bir_ad_signature.v1"
BIR_AD_FOLD_NORMALIZATION_PROTOCOL_VERSION = (
    "stage5.bir_ad_fold_normalization.v1"
)
BIR_AD_SIGNATURES_NAME = "bir_ad_signatures.jsonl"
BIR_AD_FAILURES_NAME = "bir_ad_failures.json"

BIR_AD_SIGNATURE_COLUMNS = (
    "protocol_version",
    "bir_ad_protocol_version",
    "task_id",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
    "encoder_fingerprint",
    "normalization_sha256",
    "alignment_fingerprint",
    "query_image_sha256",
    "support_image_sha256s",
    "support_bai",
    "support_bai_variance",
    "support_bai_std",
    "query_bai",
    "query_support_boundary_shift",
    "absolute_boundary_shift",
    "bai_vector",
    "support_bai_reliability",
    "query_bai_reliability",
    "support_boundary_consistency",
    "support_boundary_consistency_valid",
    "query_support_boundary_consistency",
    "support_pixel_feature_disagreement",
    "query_pixel_feature_disagreement",
    "clear_representation",
    "ambiguous_representation",
    "query_patch_count",
    "support_patch_count",
)


class BIRADPipelineError(RuntimeError):
    """Base error for batch BIR-AD production."""


class BIRADArtifactError(BIRADPipelineError):
    """Raised when a BIR-AD artifact is malformed or cannot be written."""


@dataclass(frozen=True)
class BIRADSignature:
    """One leakage-safe task-level BIR-AD signature."""

    task_id: str
    dataset: str
    category: str
    k_shot: int
    seed: int
    support_set_id: str
    encoder_fingerprint: str
    normalization_sha256: str
    alignment_fingerprint: str
    query_image_sha256: str
    support_image_sha256s: tuple[str, ...]
    result: BIRADTaskResult
    protocol_version: str = BIR_AD_SIGNATURE_PROTOCOL_VERSION

    def to_record(self) -> dict[str, Any]:
        result = self.result
        support_patch_count = sum(item.patch_count for item in result.supports)
        record = {
            "protocol_version": self.protocol_version,
            "bir_ad_protocol_version": result.protocol_version,
            "task_id": self.task_id,
            "dataset": self.dataset,
            "category": self.category,
            "k_shot": self.k_shot,
            "seed": self.seed,
            "support_set_id": self.support_set_id,
            "encoder_fingerprint": self.encoder_fingerprint,
            "normalization_sha256": self.normalization_sha256,
            "alignment_fingerprint": self.alignment_fingerprint,
            "query_image_sha256": self.query_image_sha256,
            "support_image_sha256s": list(self.support_image_sha256s),
            "support_bai": result.support_bai,
            "support_bai_variance": result.support_bai_variance,
            "support_bai_std": result.support_bai_std,
            "query_bai": result.query_bai,
            "query_support_boundary_shift": (
                result.query_support_boundary_shift
            ),
            "absolute_boundary_shift": result.absolute_boundary_shift,
            "bai_vector": list(result.bai_vector),
            "support_bai_reliability": result.support_bai_reliability,
            "query_bai_reliability": result.query_bai_reliability,
            "support_boundary_consistency": result.support_boundary_consistency,
            "support_boundary_consistency_valid": (
                result.support_boundary_consistency_valid
            ),
            "query_support_boundary_consistency": (
                result.query_support_boundary_consistency
            ),
            "support_pixel_feature_disagreement": (
                result.support_pixel_feature_disagreement
            ),
            "query_pixel_feature_disagreement": (
                result.query_pixel_feature_disagreement
            ),
            "clear_representation": _float_list(result.clear_representation),
            "ambiguous_representation": _float_list(
                result.ambiguous_representation
            ),
            "query_patch_count": result.query.patch_count,
            "support_patch_count": support_patch_count,
        }
        if tuple(record) != BIR_AD_SIGNATURE_COLUMNS:
            raise BIRADArtifactError("BIR-AD signature record schema drifted")
        return record


@dataclass(frozen=True)
class BIRADFoldNormalizationArtifact:
    """Fold-scoped normalization plus auditable train-only provenance."""

    fold: str
    statistics: BIRADNormalizationStats
    encoder_fingerprint: str
    alignment_fingerprint: str
    train_categories: tuple[str, ...]
    train_support_set_ids: tuple[str, ...]
    training_image_sha256s: tuple[str, ...]
    protocol_version: str = BIR_AD_FOLD_NORMALIZATION_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.fold, str) or not self.fold.strip():
            raise BIRADArtifactError("fold must be a non-empty string")
        if (
            not isinstance(self.statistics, BIRADNormalizationStats)
            or not self.statistics.is_fitted
            or self.statistics.source != "training_categories"
        ):
            raise BIRADArtifactError(
                "fold normalization must contain fitted training-category statistics"
            )
        for field, value in (
            ("encoder_fingerprint", self.encoder_fingerprint),
            ("alignment_fingerprint", self.alignment_fingerprint),
        ):
            if not isinstance(value, str) or not value.strip():
                raise BIRADArtifactError(f"{field} must be a non-empty string")
        for field, values in (
            ("train_categories", self.train_categories),
            ("train_support_set_ids", self.train_support_set_ids),
            ("training_image_sha256s", self.training_image_sha256s),
        ):
            if (
                not isinstance(values, tuple)
                or not values
                or values != tuple(sorted(set(values)))
                or any(not isinstance(value, str) or not value for value in values)
            ):
                raise BIRADArtifactError(
                    f"{field} must be a non-empty sorted unique tuple"
                )
        if not all(_is_lower_sha256(value) for value in self.training_image_sha256s):
            raise BIRADArtifactError(
                "training_image_sha256s must contain lowercase SHA-256 values"
            )
        if self.protocol_version != BIR_AD_FOLD_NORMALIZATION_PROTOCOL_VERSION:
            raise BIRADArtifactError(
                "invalid BIR-AD fold-normalization protocol"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "fold": self.fold,
            "split": "train",
            "encoder_fingerprint": self.encoder_fingerprint,
            "alignment_fingerprint": self.alignment_fingerprint,
            "train_categories": list(self.train_categories),
            "train_support_set_ids": list(self.train_support_set_ids),
            "training_image_sha256s": list(self.training_image_sha256s),
            "statistics": self.statistics.to_dict(),
        }


class BIRADTaskEncoder:
    """Compose verified cache entries and aligned pixels into task signatures."""

    def __init__(
        self,
        feature_cache: FeatureCache,
        feature_encoder: Any,
        *,
        normalization_stats: BIRADNormalizationStats | None = None,
        normalization_artifact: BIRADFoldNormalizationArtifact | None = None,
        pixel_aligner: Any | None = None,
        batch_size: int = 32,
        patch_grid_shape: tuple[int, int] | None = None,
        require_fitted_normalization: bool = True,
        bir_ad_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if normalization_artifact is not None:
            if not isinstance(
                normalization_artifact, BIRADFoldNormalizationArtifact
            ):
                raise TypeError(
                    "normalization_artifact must be "
                    "BIRADFoldNormalizationArtifact"
                )
            if (
                normalization_artifact.encoder_fingerprint
                != feature_cache.encoder_fingerprint
            ):
                raise BIRADPipelineError(
                    "fold normalization encoder fingerprint disagrees with "
                    "the feature cache"
                )
            if (
                normalization_stats is not None
                and normalization_stats != normalization_artifact.statistics
            ):
                raise BIRADPipelineError(
                    "normalization_stats disagrees with normalization_artifact"
                )
            normalization_stats = normalization_artifact.statistics
        if not isinstance(normalization_stats, BIRADNormalizationStats):
            raise TypeError(
                "normalization_stats or normalization_artifact is required"
            )
        if require_fitted_normalization and not normalization_stats.is_fitted:
            raise BIRADInputError(
                "batch BIR-AD requires frozen fitted normalization statistics"
            )
        self.feature_cache = feature_cache
        self.feature_encoder = feature_encoder
        self.pixel_aligner = pixel_aligner or feature_encoder
        self.normalization_stats = normalization_stats
        self.normalization_artifact = normalization_artifact
        self.batch_size = batch_size
        self.patch_grid_shape = patch_grid_shape
        self.require_fitted_normalization = require_fitted_normalization
        self.bir_ad_kwargs = dict(bir_ad_kwargs or {})
        forbidden = {
            "normalization_stats",
            "patch_grid_shape",
            "require_fitted_normalization",
            "require_strict_alignment",
        }.intersection(self.bir_ad_kwargs)
        if forbidden:
            raise ValueError(
                f"bir_ad_kwargs contains encoder-controlled fields: {sorted(forbidden)}"
            )

    def encode_tasks(
        self,
        tasks: Sequence[Mapping[str, Any]],
        support_sets: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]],
    ) -> list[BIRADSignature]:
        normalized_tasks, task_support_paths = _resolve_tasks_and_supports(
            tasks, support_sets
        )
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
        cached = self.feature_cache.get_or_encode(
            unique_paths,
            self.feature_encoder,
            batch_size=self.batch_size,
        )
        cached_by_path = dict(zip(unique_paths, cached))
        aligned_by_path = self._align_unique_paths(unique_paths, cached_by_path)
        image_result_by_hash = self._compute_unique_image_results(
            unique_paths, cached_by_path, aligned_by_path
        )
        normalization_hash = _canonical_sha256(
            (
                self.normalization_artifact.to_dict()
                if self.normalization_artifact is not None
                else self.normalization_stats.to_dict()
            )
        )

        signatures = []
        for task in normalized_tasks:
            query_path = Path(task["query_path"])
            query_features = cached_by_path[query_path]
            support_features = [
                cached_by_path[path]
                for path in task_support_paths[task["task_id"]]
            ]
            support_features.sort(key=lambda item: item.image_sha256)
            support_hashes = tuple(item.image_sha256 for item in support_features)
            if len(set(support_hashes)) != len(support_hashes):
                raise BIRADPipelineError(
                    f"task {task['task_id']!r} contains byte-identical support "
                    "images; cross-support consistency requires distinct images"
                )
            query_result = image_result_by_hash[query_features.image_sha256]
            support_results = [
                image_result_by_hash[item.image_sha256]
                for item in support_features
            ]
            task_result = compose_bir_ad_task(
                query_result,
                support_results,
                consistency_temperature=float(
                    self.bir_ad_kwargs.get("consistency_temperature", 1.0)
                ),
                consistency_chunk_size=int(
                    self.bir_ad_kwargs.get("consistency_chunk_size", 1024)
                ),
            )
            fingerprints = {
                item.alignment_fingerprint
                for item in (task_result.query, *task_result.supports)
            }
            if None in fingerprints or len(fingerprints) != 1:
                raise BIRADPipelineError(
                    f"task {task['task_id']!r} mixes spatial transforms"
                )
            if (
                self.normalization_artifact is not None
                and next(iter(fingerprints))
                != self.normalization_artifact.alignment_fingerprint
            ):
                raise BIRADPipelineError(
                    f"task {task['task_id']!r} alignment fingerprint "
                    "disagrees with fold normalization"
                )
            signatures.append(
                BIRADSignature(
                    task_id=task["task_id"],
                    dataset=task["dataset"],
                    category=task["category"],
                    k_shot=task["k_shot"],
                    seed=task["seed"],
                    support_set_id=task["support_set_id"],
                    encoder_fingerprint=self.feature_cache.encoder_fingerprint,
                    normalization_sha256=normalization_hash,
                    alignment_fingerprint=next(iter(fingerprints)),
                    query_image_sha256=query_features.image_sha256,
                    support_image_sha256s=support_hashes,
                    result=task_result,
                )
            )
        signatures.sort(key=lambda item: item.task_id)
        return signatures

    def _align_unique_paths(
        self,
        paths: Sequence[Path],
        cached_by_path: Mapping[Path, CachedImageFeatures],
    ) -> dict[Path, PatchAlignedImage]:
        grouped: dict[tuple[int, int], list[Path]] = {}
        for path in paths:
            patch_count = int(cached_by_path[path].patch_features.shape[0])
            grid_shape = self.patch_grid_shape or _square_grid(patch_count)
            if grid_shape[0] * grid_shape[1] != patch_count:
                raise BIRADPipelineError(
                    f"patch grid {grid_shape} disagrees with {patch_count} tokens"
                )
            grouped.setdefault(grid_shape, []).append(path)
        aligned_by_path: dict[Path, PatchAlignedImage] = {}
        align_method = getattr(self.pixel_aligner, "align_images_for_patches", None)
        if align_method is None:
            raise BIRADPipelineError(
                "pixel_aligner must implement align_images_for_patches()"
            )
        for grid_shape, group_paths in sorted(grouped.items()):
            aligned = align_method(group_paths, patch_grid_shape=grid_shape)
            if isinstance(aligned, (str, bytes)) or len(aligned) != len(group_paths):
                raise BIRADPipelineError(
                    "pixel aligner did not return one result per image"
                )
            for path, item in zip(group_paths, aligned):
                if not isinstance(item, PatchAlignedImage):
                    raise BIRADPipelineError(
                        "pixel aligner must return PatchAlignedImage values"
                    )
                if item.source_image_sha256 != cached_by_path[path].image_sha256:
                    raise BIRADPipelineError(
                        f"aligned pixels/cache hash disagree for {path}"
                    )
                aligned_by_path[path] = item
        return aligned_by_path

    def _compute_unique_image_results(
        self,
        paths: Sequence[Path],
        cached_by_path: Mapping[Path, CachedImageFeatures],
        aligned_by_path: Mapping[Path, PatchAlignedImage],
    ) -> dict[str, Any]:
        result_by_hash = {}
        compute_kwargs = {
            key: value
            for key, value in self.bir_ad_kwargs.items()
            if key not in {"consistency_temperature", "consistency_chunk_size"}
        }
        for path in paths:
            cached = cached_by_path[path]
            if cached.image_sha256 in result_by_hash:
                continue
            result_by_hash[cached.image_sha256] = compute_bir_ad(
                cached.patch_features,
                aligned_by_path[path],
                normalization_stats=self.normalization_stats,
                require_fitted_normalization=self.require_fitted_normalization,
                require_strict_alignment=True,
                **compute_kwargs,
            )
        return result_by_hash


def fit_fold_bir_ad_normalization(
    tasks: Sequence[Mapping[str, Any]],
    support_sets: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]],
    fold_manifest_rows: Sequence[Mapping[str, Any]],
    *,
    fold: str,
    feature_cache: FeatureCache,
    feature_encoder: Any,
    pixel_aligner: Any | None = None,
    patch_grid_shape: tuple[int, int] | None = None,
    batch_size: int = 32,
    output_path: str | Path | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
) -> BIRADFoldNormalizationArtifact:
    """Fit one fold using unique official train/good support images only."""

    normalized_tasks, task_support_paths = _resolve_tasks_and_supports(
        tasks, support_sets
    )
    requested_fold = _text(fold, "fold")
    split_by_task: dict[str, str] = {}
    for index, row in enumerate(fold_manifest_rows):
        if not isinstance(row, Mapping):
            raise BIRADPipelineError(f"fold manifest row {index} is not a mapping")
        if str(row.get("fold", "")).strip() != requested_fold:
            continue
        task_id = _text(row.get("task_id"), "task_id")
        split = _text(row.get("split"), "split").casefold()
        if split not in {"train", "val", "test"}:
            raise BIRADPipelineError(f"invalid fold split {split!r}")
        if task_id in split_by_task:
            raise BIRADPipelineError(
                f"fold {requested_fold!r} duplicates task_id={task_id!r}"
            )
        split_by_task[task_id] = split
    train_tasks = [
        task for task in normalized_tasks
        if split_by_task.get(task["task_id"]) == "train"
    ]
    if not train_tasks:
        raise BIRADPipelineError(
            f"fold {requested_fold!r} has no train tasks"
        )
    missing = sorted(
        task["task_id"]
        for task in normalized_tasks
        if task["task_id"] not in split_by_task
    )
    if missing:
        raise BIRADPipelineError(
            f"fold {requested_fold!r} is missing task ids: {missing[:5]}"
        )

    train_support_ids = sorted(
        {task["support_set_id"] for task in train_tasks}
    )
    representative_paths = sorted(
        {
            path
            for task in train_tasks
            for path in task_support_paths[task["task_id"]]
        },
        key=str,
    )
    cached = feature_cache.get_or_encode(
        representative_paths,
        feature_encoder,
        batch_size=batch_size,
    )
    # Byte-identical images appearing in several support sets contribute once.
    path_by_hash: dict[str, Path] = {}
    cached_by_hash: dict[str, CachedImageFeatures] = {}
    for path, item in zip(representative_paths, cached):
        path_by_hash.setdefault(item.image_sha256, path)
        cached_by_hash.setdefault(item.image_sha256, item)
    unique_hashes = sorted(cached_by_hash)
    unique_paths = [path_by_hash[image_hash] for image_hash in unique_hashes]
    aligner = pixel_aligner or feature_encoder
    aligned_by_hash: dict[str, PatchAlignedImage] = {}
    grouped: dict[tuple[int, int], list[str]] = {}
    for image_hash in unique_hashes:
        patch_count = int(cached_by_hash[image_hash].patch_features.shape[0])
        grid = patch_grid_shape or _square_grid(patch_count)
        if grid[0] * grid[1] != patch_count:
            raise BIRADPipelineError("normalization patch grid/token count mismatch")
        grouped.setdefault(grid, []).append(image_hash)
    for grid, hashes in sorted(grouped.items()):
        paths = [path_by_hash[value] for value in hashes]
        aligned = aligner.align_images_for_patches(paths, patch_grid_shape=grid)
        if len(aligned) != len(hashes):
            raise BIRADPipelineError("normalization pixel alignment count mismatch")
        for image_hash, item in zip(hashes, aligned):
            if (
                not isinstance(item, PatchAlignedImage)
                or item.source_image_sha256 != image_hash
            ):
                raise BIRADPipelineError(
                    "normalization aligned pixels/cache identity mismatch"
                )
            aligned_by_hash[image_hash] = item
    fingerprints = {
        item.transform_fingerprint for item in aligned_by_hash.values()
    }
    if len(fingerprints) != 1:
        raise BIRADPipelineError("fold normalization mixes spatial transforms")
    statistics = fit_bir_ad_normalization(
        [cached_by_hash[value].patch_features for value in unique_hashes],
        [aligned_by_hash[value] for value in unique_hashes],
        split="train",
        patch_grid_shape=patch_grid_shape,
        **dict(fit_kwargs or {}),
    )
    artifact = BIRADFoldNormalizationArtifact(
        fold=requested_fold,
        statistics=statistics,
        encoder_fingerprint=feature_cache.encoder_fingerprint,
        alignment_fingerprint=next(iter(fingerprints)),
        train_categories=tuple(
            sorted({task["category"] for task in train_tasks})
        ),
        train_support_set_ids=tuple(train_support_ids),
        training_image_sha256s=tuple(unique_hashes),
    )
    if output_path is not None:
        write_fold_normalization_artifact(artifact, output_path)
    return artifact


def write_fold_normalization_artifact(
    artifact: BIRADFoldNormalizationArtifact,
    output_path: str | Path,
) -> Path:
    if not isinstance(artifact, BIRADFoldNormalizationArtifact):
        raise TypeError("artifact must be BIRADFoldNormalizationArtifact")
    path = Path(output_path)
    _atomic_write_json(path, artifact.to_dict())
    return path


def load_fold_normalization_artifact(
    path: str | Path,
) -> BIRADFoldNormalizationArtifact:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BIRADArtifactError(
            f"could not read BIR-AD fold normalization {source}: {exc}"
        ) from exc
    if payload.get("protocol_version") != BIR_AD_FOLD_NORMALIZATION_PROTOCOL_VERSION:
        raise BIRADArtifactError("invalid BIR-AD fold-normalization protocol")
    if payload.get("split") != "train":
        raise BIRADArtifactError("BIR-AD fold normalization must record train split")
    return BIRADFoldNormalizationArtifact(
        fold=_text(payload.get("fold"), "fold"),
        statistics=BIRADNormalizationStats.from_mapping(payload.get("statistics")),
        encoder_fingerprint=_text(
            payload.get("encoder_fingerprint"), "encoder_fingerprint"
        ),
        alignment_fingerprint=_text(
            payload.get("alignment_fingerprint"), "alignment_fingerprint"
        ),
        train_categories=_string_tuple(
            payload.get("train_categories"), "train_categories"
        ),
        train_support_set_ids=_string_tuple(
            payload.get("train_support_set_ids"), "train_support_set_ids"
        ),
        training_image_sha256s=_string_tuple(
            payload.get("training_image_sha256s"), "training_image_sha256s"
        ),
    )


def write_bir_ad_signatures_jsonl(
    signatures: Sequence[BIRADSignature],
    output_path: str | Path,
) -> Path:
    if isinstance(signatures, (str, bytes)) or not signatures:
        raise BIRADArtifactError("signatures must be non-empty")
    ordered = sorted(signatures, key=lambda item: item.task_id)
    if len({item.task_id for item in ordered}) != len(ordered):
        raise BIRADArtifactError("duplicate BIR-AD signature task_id")
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


def build_bir_ad_signatures(
    tasks: Sequence[Mapping[str, Any]],
    support_sets: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]],
    *,
    encoder: BIRADTaskEncoder,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Build complete signatures or save the explicit fatal failure."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    failures_path = destination / BIR_AD_FAILURES_NAME
    try:
        signatures = encoder.encode_tasks(tasks, support_sets)
        signature_path = write_bir_ad_signatures_jsonl(
            signatures, destination / BIR_AD_SIGNATURES_NAME
        )
        _atomic_write_json(failures_path, [])
        return signature_path, failures_path
    except Exception as exc:
        _atomic_write_json(
            failures_path,
            [{"code": type(exc).__name__, "message": str(exc)}],
        )
        raise


def read_tasks_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows = []
    try:
        with source.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise BIRADArtifactError(
                        f"{source}:{line_number} is blank"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise BIRADArtifactError(
                        f"{source}:{line_number} is not an object"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        if isinstance(exc, BIRADArtifactError):
            raise
        raise BIRADArtifactError(f"could not read tasks {source}: {exc}") from exc
    if not rows:
        raise BIRADArtifactError(f"task file is empty: {source}")
    return rows


def read_supports_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    try:
        with source.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise BIRADArtifactError(
            f"could not read supports {source}: {exc}"
        ) from exc
    if not rows:
        raise BIRADArtifactError(f"support file is empty: {source}")
    return rows


def read_fold_manifest_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    try:
        with source.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise BIRADArtifactError(
            f"could not read fold manifest {source}: {exc}"
        ) from exc
    if not rows:
        raise BIRADArtifactError(f"fold manifest is empty: {source}")
    return rows


def _resolve_tasks_and_supports(
    tasks: Sequence[Mapping[str, Any]],
    support_sets: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, tuple[Path, ...]]]:
    try:
        _assert_official_train_good_supports(support_sets)
        normalized_tasks = _normalize_tasks(tasks)
        grouped_supports = _group_support_sets(support_sets)
        paths = {}
        for task in normalized_tasks:
            support_id = task["support_set_id"]
            if support_id not in grouped_supports:
                raise NormalDomainInputError(
                    f"support_set_id {support_id!r} has no support rows"
                )
            paths[task["task_id"]] = _validate_task_supports(
                task, grouped_supports[support_id]
            )
        return normalized_tasks, paths
    except NormalDomainInputError as exc:
        raise BIRADPipelineError(str(exc)) from exc


def _assert_official_train_good_supports(
    support_sets: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]],
) -> None:
    """Require explicit split metadata or a canonical train/good path."""

    if isinstance(support_sets, Mapping):
        values = [
            value
            for group in support_sets.values()
            for value in group
        ]
    else:
        values = list(support_sets)
    for index, value in enumerate(values):
        if isinstance(value, Mapping):
            split = str(value.get("split", "")).strip().casefold()
            split = split.replace("\\", "/")
            path_value = value.get("image_path")
        else:
            split = ""
            path_value = value
        if split in {"train", "train/good", "good"}:
            continue
        normalized_path = str(path_value or "").replace("\\", "/").casefold()
        canonical_path = f"/{normalized_path.strip('/')}/"
        if "/train/good/" in canonical_path:
            continue
        raise NormalDomainInputError(
            f"support {index} has no verifiable official train/good provenance"
        )


def _square_grid(patch_count: int) -> tuple[int, int]:
    side = math.isqrt(patch_count)
    if side * side != patch_count:
        raise BIRADPipelineError(
            f"explicit patch_grid_shape is required for {patch_count} tokens"
        )
    return side, side


def _float_list(value: Any) -> list[float]:
    try:
        import numpy as np
    except ImportError as exc:
        raise BIRADArtifactError("BIR-AD signatures require NumPy") from exc
    return [float(item) for item in np.asarray(value, dtype=np.float32).tolist()]


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_lower_sha256(value: str) -> bool:
    return (
        len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _text(value: Any, field: str) -> str:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise BIRADArtifactError(f"{field} must be a non-empty string")
    return str(value).strip()


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BIRADArtifactError(f"{field} must be a sequence")
    result = tuple(_text(item, field) for item in value)
    if not result or len(set(result)) != len(result):
        raise BIRADArtifactError(f"{field} must be non-empty and unique")
    return result


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
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


__all__ = [
    "BIR_AD_FAILURES_NAME",
    "BIR_AD_FOLD_NORMALIZATION_PROTOCOL_VERSION",
    "BIR_AD_SIGNATURE_COLUMNS",
    "BIR_AD_SIGNATURE_PROTOCOL_VERSION",
    "BIR_AD_SIGNATURES_NAME",
    "BIRADArtifactError",
    "BIRADFoldNormalizationArtifact",
    "BIRADPipelineError",
    "BIRADSignature",
    "BIRADTaskEncoder",
    "build_bir_ad_signatures",
    "fit_fold_bir_ad_normalization",
    "load_fold_normalization_artifact",
    "read_fold_manifest_csv",
    "read_supports_csv",
    "read_tasks_jsonl",
    "write_bir_ad_signatures_jsonl",
    "write_fold_normalization_artifact",
]
