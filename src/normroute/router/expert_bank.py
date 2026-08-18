"""Auditable, train-category-only Expert Capability Profile Bank (ECPB)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any

from .teacher import TEACHER_PROTOCOL_VERSION, TeacherArtifact


EXPERT_BANK_PROTOCOL_VERSION = "stage5.expert_capability_bank.v3"
EXPERT_BANK_COMPATIBLE_PROTOCOL_VERSIONS = frozenset(
    {"stage5.expert_capability_bank.v2", EXPERT_BANK_PROTOCOL_VERSION}
)
CAPABILITY_BANK_NAME = "capability_bank.json"
CAPABILITY_FEATURE_NAMES = (
    "overall_skill",
    "boundary_skill",
    "fgbg_skill",
    "lowshot_skill",
    "texture_skill",
    "latency_p50_log1p",
    "latency_p95_log1p",
    "failure_rate",
)
_BOUNDARY_FEATURES = (
    "bir_query_bai",
    "bir_query_support_boundary_shift",
    "bir_absolute_boundary_shift",
)
_FGBG_FEATURES = (
    "fbdp_fbc",
    "fbdp_foreground_background_confusion",
)
_TEXTURE_FEATURES = (
    "normal_niv_0002",
    "normal_niv_2",
    "normal_niv_structure",
    "normal_niv_texture",
    "normal_query_texture_complexity",
)
_STAGE2_RUNTIME_COLUMNS = (
    "expert",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
    "num_predictions",
    "num_success",
    "num_failed",
    "average_runtime_ms",
)


class ExpertBankError(RuntimeError):
    """Base error for capability-bank construction."""


class ExpertBankInputError(ExpertBankError, ValueError):
    """Raised when capability statistics cannot be computed safely."""


class ExpertBankIsolationError(ExpertBankError, ValueError):
    """Raised when non-training outcomes enter capability aggregation."""


@dataclass(frozen=True)
class Stage2RuntimeSummaryRecord:
    """One aggregate runtime row, parsed only after the category gate."""

    expert: str
    dataset: str
    category: str
    k_shot: int
    seed: int
    support_set_id: str
    num_predictions: int
    num_success: int
    num_failed: int
    average_runtime_ms: float

    @property
    def key(self) -> tuple[str, str, str, str, int, int]:
        return (
            self.expert,
            self.dataset,
            self.category,
            self.support_set_id,
            self.k_shot,
            self.seed,
        )


@dataclass(frozen=True)
class SkillCurveBin:
    quantile_low: float
    quantile_high: float
    signal_low: float
    signal_high: float
    skill: float
    sample_count: int
    effective_weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantile_low": self.quantile_low,
            "quantile_high": self.quantile_high,
            "signal_low": self.signal_low,
            "signal_high": self.signal_high,
            "skill": self.skill,
            "sample_count": self.sample_count,
            "effective_weight": self.effective_weight,
        }


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float = 0.95

    def to_dict(self) -> dict[str, float]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ExpertCapabilityProfile:
    """Category-balanced quality, cost, reliability, and difficulty curves."""

    boundary_skill: float
    fgbg_skill: float
    lowshot_skill: float
    texture_skill: float | None
    latency: float | None
    latency_p50: float | None
    latency_p95: float | None
    failure_rate: float
    overall_skill: float
    sample_count: int
    effective_sample_weight: float
    boundary_sample_count: int
    fgbg_sample_count: int
    lowshot_sample_count: int
    texture_sample_count: int
    k_shot_skill: Mapping[str, float]
    failure_counts: Mapping[str, int]
    boundary_curve: tuple[SkillCurveBin, ...]
    fgbg_curve: tuple[SkillCurveBin, ...]
    texture_curve: tuple[SkillCurveBin, ...]
    confidence_intervals: Mapping[str, ConfidenceInterval]
    capability_vector: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_skill": self.boundary_skill,
            "fgbg_skill": self.fgbg_skill,
            "lowshot_skill": self.lowshot_skill,
            "texture_skill": self.texture_skill,
            "latency": self.latency,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "failure_rate": self.failure_rate,
            "overall_skill": self.overall_skill,
            "sample_count": self.sample_count,
            "effective_sample_weight": self.effective_sample_weight,
            "boundary_sample_count": self.boundary_sample_count,
            "fgbg_sample_count": self.fgbg_sample_count,
            "lowshot_sample_count": self.lowshot_sample_count,
            "texture_sample_count": self.texture_sample_count,
            "k_shot_skill": dict(sorted(self.k_shot_skill.items(), key=lambda item: int(item[0]))),
            "failure_counts": dict(sorted(self.failure_counts.items())),
            "boundary_curve": [item.to_dict() for item in self.boundary_curve],
            "fgbg_curve": [item.to_dict() for item in self.fgbg_curve],
            "texture_curve": [item.to_dict() for item in self.texture_curve],
            "confidence_intervals": {
                name: interval.to_dict()
                for name, interval in sorted(self.confidence_intervals.items())
            },
            "capability_vector": list(self.capability_vector),
        }


@dataclass(frozen=True)
class ExpertCapabilityBank:
    """Fold-specific ECPB containing aggregates, never per-query outcomes."""

    fold: str
    train_categories: tuple[str, ...]
    lowshot_k: int
    boundary_threshold: float
    fgbg_threshold: float
    texture_threshold: float | None
    profiles: Mapping[str, ExpertCapabilityProfile]
    bootstrap_replicates: int
    bootstrap_seed: int
    runtime_provenance: Mapping[str, Any]
    protocol_version: str = EXPERT_BANK_PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "fold": self.fold,
            "construction": "train_category_balanced_query_weighted_bootstrap",
            "train_categories": list(self.train_categories),
            "lowshot_k": self.lowshot_k,
            "repeat_weighting": "teacher_equal_category_equal_query_inverse_variant_frequency",
            "difficulty_thresholds": {
                "boundary": self.boundary_threshold,
                "fgbg": self.fgbg_threshold,
                "texture": self.texture_threshold,
            },
            "bootstrap": {
                "unit": "category",
                "replicates": self.bootstrap_replicates,
                "seed": self.bootstrap_seed,
                "confidence": 0.95,
            },
            "runtime_provenance": dict(self.runtime_provenance),
            "capability_feature_names": list(CAPABILITY_FEATURE_NAMES),
            "profiles": {
                expert: self.profiles[expert].to_dict()
                for expert in sorted(self.profiles)
            },
        }


def build_expert_capability_bank(
    teacher: TeacherArtifact | Sequence[Mapping[str, Any]],
    *,
    fold: str | None = None,
    train_categories: Sequence[str] | None = None,
    feature_records: Iterable[Mapping[str, Any]] | None = None,
    boundary_signals: Mapping[str, float] | None = None,
    fgbg_signals: Mapping[str, float] | None = None,
    texture_signals: Mapping[str, float] | None = None,
    runtime_summary_records: Sequence[Stage2RuntimeSummaryRecord] | None = None,
    bootstrap_replicates: int = 200,
    bootstrap_seed: int = 0,
) -> ExpertCapabilityBank:
    """Build category-balanced capability statistics from train rows only."""

    rows, resolved_fold, categories = _resolve_teacher_scope(
        teacher, fold=fold, train_categories=train_categories
    )
    selected = _validate_and_select_rows(rows, resolved_fold, categories)
    runtime_provenance = _validate_runtime_provenance(
        selected,
        categories=categories,
        runtime_summary_records=runtime_summary_records,
    )
    task_ids = tuple(sorted({str(row["task_id"]) for row in selected}))
    extracted_boundary: dict[str, float] = {}
    extracted_fgbg: dict[str, float] = {}
    extracted_texture: dict[str, float] = {}
    if feature_records is not None:
        extracted_boundary, extracted_fgbg, extracted_texture = (
            extract_capability_signals(feature_records, allowed_task_ids=set(task_ids))
        )
    boundary = dict(boundary_signals or extracted_boundary)
    fgbg = dict(fgbg_signals or extracted_fgbg)
    texture = dict(texture_signals or extracted_texture)
    _validate_signal_coverage("boundary", boundary, task_ids)
    _validate_signal_coverage("fgbg", fgbg, task_ids)
    if texture:
        _validate_signal_coverage("texture", texture, task_ids)

    task_rows = _one_row_per_task(selected)
    task_weights = {
        task_id: _positive(row.get("sample_weight"), "sample_weight")
        for task_id, row in task_rows.items()
    }
    boundary_edges = _difficulty_edges(boundary, task_ids, task_weights)
    fgbg_edges = _difficulty_edges(fgbg, task_ids, task_weights)
    texture_edges = (
        _difficulty_edges(texture, task_ids, task_weights) if texture else None
    )
    lowshot_k = min(_positive_int(row.get("k_shot"), "k_shot") for row in selected)
    replicates = _nonnegative_int(bootstrap_replicates, "bootstrap_replicates")
    seed = int(bootstrap_seed)

    by_expert: dict[str, list[Mapping[str, Any]]] = {}
    for row in selected:
        by_expert.setdefault(_text(row.get("expert_name"), "expert_name"), []).append(row)
    if len(by_expert) < 2:
        raise ExpertBankInputError("capability bank requires at least two experts")
    expected_tasks = set(task_ids)
    profiles: dict[str, ExpertCapabilityProfile] = {}
    for expert, expert_rows in sorted(by_expert.items()):
        if {str(row["task_id"]) for row in expert_rows} != expected_tasks or len(expert_rows) != len(expected_tasks):
            raise ExpertBankInputError(
                f"expert {expert!r} does not have exactly one row per train task"
            )
        boundary_rows = _high_difficulty_rows(expert_rows, boundary, boundary_edges[1])
        fgbg_rows = _high_difficulty_rows(expert_rows, fgbg, fgbg_edges[1])
        texture_rows = (
            _high_difficulty_rows(expert_rows, texture, texture_edges[1])
            if texture_edges is not None
            else []
        )
        lowshot_rows = [
            row for row in expert_rows
            if _positive_int(row.get("k_shot"), "k_shot") == lowshot_k
        ]
        successful_runtime_rows = [
            row for row in expert_rows
            if not bool(row.get("failed")) and row.get("runtime_ms") is not None
        ]
        latency_p50 = _weighted_runtime_quantile(successful_runtime_rows, 0.5)
        latency_p95 = _weighted_runtime_quantile(successful_runtime_rows, 0.95)
        if latency_p50 is None or latency_p95 is None:
            raise ExpertBankInputError(
                f"expert {expert!r} has no successful train runtime observations"
            )
        overall = _weighted_skill(expert_rows)
        boundary_skill = _weighted_skill(boundary_rows)
        fgbg_skill = _weighted_skill(fgbg_rows)
        lowshot_skill = _weighted_skill(lowshot_rows)
        texture_skill = _weighted_skill(texture_rows) if texture_rows else None
        failure_rate = _weighted_failure_rate(expert_rows)
        intervals = _bootstrap_intervals(
            expert_rows,
            boundary_rows=boundary_rows,
            fgbg_rows=fgbg_rows,
            lowshot_rows=lowshot_rows,
            texture_rows=texture_rows,
            replicates=replicates,
            seed=seed,
        )
        vector = (
            overall,
            boundary_skill,
            fgbg_skill,
            lowshot_skill,
            texture_skill if texture_skill is not None else overall,
            math.log1p(latency_p50) if latency_p50 is not None else 0.0,
            math.log1p(latency_p95) if latency_p95 is not None else 0.0,
            failure_rate,
        )
        profiles[expert] = ExpertCapabilityProfile(
            boundary_skill=boundary_skill,
            fgbg_skill=fgbg_skill,
            lowshot_skill=lowshot_skill,
            texture_skill=texture_skill,
            latency=latency_p50,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            failure_rate=failure_rate,
            overall_skill=overall,
            sample_count=len(expert_rows),
            effective_sample_weight=sum(_row_weight(row) for row in expert_rows),
            boundary_sample_count=len(boundary_rows),
            fgbg_sample_count=len(fgbg_rows),
            lowshot_sample_count=len(lowshot_rows),
            texture_sample_count=len(texture_rows),
            k_shot_skill={
                str(k): _weighted_skill([
                    row for row in expert_rows
                    if _positive_int(row.get("k_shot"), "k_shot") == k
                ])
                for k in sorted({_positive_int(row.get("k_shot"), "k_shot") for row in expert_rows})
            },
            failure_counts=dict(Counter(
                str(row.get("status") or "unknown")
                for row in expert_rows if bool(row.get("failed"))
            )),
            boundary_curve=_skill_curve(expert_rows, boundary, boundary_edges),
            fgbg_curve=_skill_curve(expert_rows, fgbg, fgbg_edges),
            texture_curve=(
                _skill_curve(expert_rows, texture, texture_edges)
                if texture_edges is not None else ()
            ),
            confidence_intervals=intervals,
            capability_vector=vector,
        )
    return ExpertCapabilityBank(
        fold=resolved_fold,
        train_categories=tuple(sorted(categories)),
        lowshot_k=lowshot_k,
        boundary_threshold=boundary_edges[1],
        fgbg_threshold=fgbg_edges[1],
        texture_threshold=texture_edges[1] if texture_edges is not None else None,
        profiles=profiles,
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
        runtime_provenance=runtime_provenance,
    )


build_capability_bank = build_expert_capability_bank


def extract_capability_signals(
    feature_records: Iterable[Mapping[str, Any]],
    *,
    allowed_task_ids: set[str] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Extract BIR/FBDP/texture signals, gating by task id before parsing."""

    boundary: dict[str, float] = {}
    fgbg: dict[str, float] = {}
    texture: dict[str, float] = {}
    for index, row in enumerate(feature_records):
        raw_task_id = row.get("task_id")
        if allowed_task_ids is not None and str(raw_task_id or "") not in allowed_task_ids:
            continue
        task_id = _text(raw_task_id, f"feature record {index} task_id")
        names = tuple(str(value) for value in row.get("feature_names", ()))
        values = row.get("values", ())
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ExpertBankInputError(f"feature record {index} values are invalid")
        if len(names) != len(values) or len(set(names)) != len(names):
            raise ExpertBankInputError(f"feature record {index} schema is invalid")
        by_name = {name: _finite(value, f"feature {name}") for name, value in zip(names, values)}
        boundary_values = [abs(by_name[name]) for name in _BOUNDARY_FEATURES if name in by_name]
        fgbg_values = [by_name[name] for name in _FGBG_FEATURES if name in by_name]
        texture_values = [abs(by_name[name]) for name in _TEXTURE_FEATURES if name in by_name]
        if not boundary_values or not fgbg_values:
            raise ExpertBankInputError(
                f"feature record {index} lacks BIR boundary or FBDP fgbg signals"
            )
        if task_id in boundary:
            raise ExpertBankInputError(f"duplicate feature task_id {task_id!r}")
        boundary[task_id] = sum(boundary_values) / len(boundary_values)
        fgbg[task_id] = sum(fgbg_values) / len(fgbg_values)
        if texture_values:
            texture[task_id] = sum(texture_values) / len(texture_values)
    if texture and set(texture) != set(boundary):
        raise ExpertBankInputError("texture signal coverage is partial")
    return boundary, fgbg, texture


def extract_difficulty_signals(
    feature_records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    boundary, fgbg, _ = extract_capability_signals(feature_records)
    return boundary, fgbg


def iter_feature_records_jsonl(path: str | Path) -> Iterable[Mapping[str, Any]]:
    """Stream a feature bundle so multi-gigabyte JSONL is never materialized."""

    source = Path(path)
    if not source.is_file():
        raise ExpertBankInputError(f"Router feature JSONL does not exist: {source}")
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExpertBankInputError(
                    f"{source}:{line_number} is invalid JSON"
                ) from exc
            if not isinstance(value, Mapping):
                raise ExpertBankInputError(f"{source}:{line_number} must be an object")
            yield value


def read_stage2_runtime_summary(
    path: str | Path,
    *,
    allowed_categories: set[str],
) -> tuple[Stage2RuntimeSummaryRecord, ...]:
    """Read train-category runtime aggregates without parsing held-out rows.

    The category check deliberately precedes expert, count, and runtime parsing,
    so malformed validation/test values cannot affect an ECPB artifact.
    """

    source = Path(path)
    if not source.is_file() or source.suffix.casefold() != ".csv":
        raise ExpertBankInputError(
            f"Stage 2 runtime summary must be an existing CSV: {source}"
        )
    if not allowed_categories:
        raise ExpertBankIsolationError(
            "runtime summary requires non-empty train categories"
        )
    records: dict[
        tuple[str, str, str, str, int, int], Stage2RuntimeSummaryRecord
    ] = {}
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing_columns = [
            name for name in _STAGE2_RUNTIME_COLUMNS if name not in fieldnames
        ]
        if missing_columns:
            raise ExpertBankInputError(
                f"{source} is missing Stage 2 runtime columns: {missing_columns}"
            )
        for line_number, raw in enumerate(reader, start=2):
            category = str(raw.get("category") or "").strip()
            if category not in allowed_categories:
                continue
            clean = {
                str(key): str(value if value is not None else "").strip()
                for key, value in raw.items()
            }
            missing_values = [
                name for name in _STAGE2_RUNTIME_COLUMNS if not clean.get(name)
            ]
            if missing_values:
                raise ExpertBankInputError(
                    f"{source}:{line_number} is missing runtime values: {missing_values}"
                )
            record = Stage2RuntimeSummaryRecord(
                expert=_text(clean["expert"], "expert").casefold(),
                dataset=_text(clean["dataset"], "dataset"),
                category=category,
                k_shot=_positive_int(clean["k_shot"], "k_shot"),
                seed=_nonnegative_int(clean["seed"], "seed"),
                support_set_id=_text(clean["support_set_id"], "support_set_id"),
                num_predictions=_positive_int(
                    clean["num_predictions"], "num_predictions"
                ),
                num_success=_nonnegative_int(clean["num_success"], "num_success"),
                num_failed=_nonnegative_int(clean["num_failed"], "num_failed"),
                average_runtime_ms=_nonnegative(
                    clean["average_runtime_ms"], "average_runtime_ms"
                ),
            )
            if record.num_success + record.num_failed != record.num_predictions:
                raise ExpertBankInputError(
                    f"{source}:{line_number} has inconsistent success/failure counts"
                )
            if record.key in records:
                raise ExpertBankInputError(
                    f"{source}:{line_number} duplicates runtime condition {record.key}"
                )
            records[record.key] = record
    observed_categories = {record.category for record in records.values()}
    if observed_categories != allowed_categories:
        raise ExpertBankIsolationError(
            "Stage 2 runtime summary does not exactly cover current-fold train categories"
        )
    return tuple(records[key] for key in sorted(records))


def write_capability_bank(bank: ExpertCapabilityBank, output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix.casefold() != ".json" or path.name != CAPABILITY_BANK_NAME:
        raise ExpertBankInputError(f"capability bank output must be named {CAPABILITY_BANK_NAME}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(bank.to_dict(), indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload.encode("utf-8"))
        os.replace(temporary, path)
    except Exception as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise ExpertBankError(f"could not write capability bank {path}: {exc}") from exc
    return path


def read_capability_bank(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExpertBankInputError(f"could not read capability bank {source}") from exc
    if (
        not isinstance(value, dict)
        or value.get("protocol_version") not in EXPERT_BANK_COMPATIBLE_PROTOCOL_VERSIONS
    ):
        raise ExpertBankInputError("capability bank protocol is incompatible")
    if not value.get("train_categories") or not isinstance(value.get("profiles"), dict):
        raise ExpertBankInputError("capability bank is incomplete")
    return value


def _resolve_teacher_scope(
    teacher: TeacherArtifact | Sequence[Mapping[str, Any]],
    *,
    fold: str | None,
    train_categories: Sequence[str] | None,
) -> tuple[list[Mapping[str, Any]], str, tuple[str, ...]]:
    if isinstance(teacher, TeacherArtifact):
        return list(teacher.rows), fold or teacher.fold, tuple(train_categories or teacher.train_categories)
    if fold is None or train_categories is None:
        raise ExpertBankIsolationError(
            "raw teacher rows require explicit fold and train_categories"
        )
    return list(teacher), fold, tuple(str(value) for value in train_categories)


def _validate_and_select_rows(
    rows: Sequence[Mapping[str, Any]], fold: str, categories: Sequence[str]
) -> list[Mapping[str, Any]]:
    if not fold or not categories or len(set(categories)) != len(categories):
        raise ExpertBankInputError("fold and unique train_categories are required")
    train_set = set(categories)
    selected = [row for row in rows if str(row.get("category", "")) in train_set]
    if not selected:
        raise ExpertBankInputError("no train-category teacher rows are available")
    for row in selected:
        if row.get("protocol_version") != TEACHER_PROTOCOL_VERSION:
            raise ExpertBankInputError("teacher protocol is incompatible")
        if row.get("evaluator_only") is not True:
            raise ExpertBankIsolationError("capability source must be evaluator-only")
        if str(row.get("fold", "")) != fold or row.get("split") != "train":
            raise ExpertBankIsolationError("capability statistics may use only current-fold train rows")
        if row.get("runtime_ms") is None:
            raise ExpertBankInputError(
                "capability latency requires runtime_ms on every train teacher row"
            )
        _nonnegative(row.get("runtime_ms"), "runtime_ms")
    return selected


def _runtime_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, int, int]:
    return (
        _text(row.get("expert_name"), "expert_name").casefold(),
        _text(row.get("dataset"), "dataset"),
        _text(row.get("category"), "category"),
        _text(row.get("support_set_id"), "support_set_id"),
        _positive_int(row.get("k_shot"), "k_shot"),
        _nonnegative_int(row.get("seed"), "seed"),
    )


def _validate_runtime_provenance(
    rows: Sequence[Mapping[str, Any]],
    *,
    categories: Sequence[str],
    runtime_summary_records: Sequence[Stage2RuntimeSummaryRecord] | None,
) -> dict[str, Any]:
    groups: dict[
        tuple[str, str, str, str, int, int], list[Mapping[str, Any]]
    ] = {}
    for row in rows:
        groups.setdefault(_runtime_key(row), []).append(row)
    base = {
        "source": "evaluator_only_routing_matrix.runtime_ms",
        "category_scope": "train_only",
        "train_categories": list(sorted(categories)),
        "teacher_runtime_row_count": len(rows),
        "teacher_runtime_coverage": 1.0,
        "latency_statistics": "category_and_query_balanced_weighted_p50_p95",
    }
    if runtime_summary_records is None:
        return {**base, "stage2_summary_cross_check": False}
    summary = {record.key: record for record in runtime_summary_records}
    if len(summary) != len(runtime_summary_records):
        raise ExpertBankInputError(
            "Stage 2 runtime summary contains duplicate conditions"
        )
    if set(summary) != set(groups):
        missing = sorted(set(groups) - set(summary))
        extra = sorted(set(summary) - set(groups))
        raise ExpertBankInputError(
            "Stage 2 runtime summary condition coverage disagrees with train teacher rows; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    total_predictions = 0
    total_failed = 0
    for key, condition_rows in groups.items():
        record = summary[key]
        failed = sum(bool(row.get("failed")) for row in condition_rows)
        mean_runtime = sum(
            float(row["runtime_ms"]) for row in condition_rows
        ) / len(condition_rows)
        if record.num_predictions != len(condition_rows):
            raise ExpertBankInputError(
                f"Stage 2 runtime count disagrees for condition {key}"
            )
        if (
            record.num_failed != failed
            or record.num_success != len(condition_rows) - failed
        ):
            raise ExpertBankInputError(
                f"Stage 2 failure counts disagree for condition {key}"
            )
        if not math.isclose(
            record.average_runtime_ms,
            mean_runtime,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ExpertBankInputError(
                f"Stage 2 average_runtime_ms disagrees for condition {key}: "
                f"summary={record.average_runtime_ms}, teacher={mean_runtime}"
            )
        total_predictions += record.num_predictions
        total_failed += record.num_failed
    return {
        **base,
        "stage2_summary_cross_check": True,
        "stage2_summary_record_count": len(runtime_summary_records),
        "stage2_summary_prediction_count": total_predictions,
        "stage2_summary_failure_count": total_failed,
        "condition_fields": [
            "expert",
            "dataset",
            "category",
            "support_set_id",
            "k_shot",
            "seed",
        ],
    }


def _one_row_per_task(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task_id = _text(row.get("task_id"), "task_id")
        current = result.setdefault(task_id, row)
        for field in ("category", "k_shot", "sample_weight", "query_group_id"):
            if current.get(field) != row.get(field):
                raise ExpertBankInputError(f"task metadata changed across experts: {task_id}")
    return result


def _difficulty_edges(
    signals: Mapping[str, float], task_ids: Sequence[str], weights: Mapping[str, float]
) -> tuple[float, float]:
    values = [_finite(signals[task_id], "difficulty signal") for task_id in task_ids]
    task_weights = [weights[task_id] for task_id in task_ids]
    return (
        _weighted_quantile(values, task_weights, 1.0 / 3.0),
        _weighted_quantile(values, task_weights, 2.0 / 3.0),
    )


def _skill_curve(
    rows: Sequence[Mapping[str, Any]],
    signals: Mapping[str, float],
    edges: tuple[float, float],
) -> tuple[SkillCurveBin, ...]:
    low, high = edges
    partitions = (
        [row for row in rows if signals[str(row["task_id"])] <= low],
        [row for row in rows if low < signals[str(row["task_id"])] < high],
        [row for row in rows if signals[str(row["task_id"])] >= high],
    )
    quantiles = ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0))
    result = []
    for index, subset in enumerate(partitions):
        if not subset:
            continue
        values = [signals[str(row["task_id"])] for row in subset]
        result.append(SkillCurveBin(
            quantile_low=quantiles[index][0],
            quantile_high=quantiles[index][1],
            signal_low=min(values),
            signal_high=max(values),
            skill=_weighted_skill(subset),
            sample_count=len(subset),
            effective_weight=sum(_row_weight(row) for row in subset),
        ))
    return tuple(result)


def _high_difficulty_rows(
    rows: Sequence[Mapping[str, Any]], signals: Mapping[str, float], threshold: float
) -> list[Mapping[str, Any]]:
    return [row for row in rows if signals[str(row["task_id"])] >= threshold]


def _weighted_skill(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        raise ExpertBankInputError("capability slice is empty")
    weights = [_row_weight(row) for row in rows]
    return sum(
        weight * _probability(
            row.get("correctness_probability", row.get("teacher_utility")),
            "correctness_probability",
        )
        for row, weight in zip(rows, weights)
    ) / sum(weights)


def _weighted_failure_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    weights = [_row_weight(row) for row in rows]
    return sum(weight * bool(row.get("failed")) for row, weight in zip(rows, weights)) / sum(weights)


def _weighted_runtime_quantile(rows: Sequence[Mapping[str, Any]], quantile: float) -> float | None:
    if not rows:
        return None
    return _weighted_quantile(
        [_nonnegative(row.get("runtime_ms"), "runtime_ms") for row in rows],
        [_row_weight(row) for row in rows],
        quantile,
    )


def _bootstrap_intervals(
    rows: Sequence[Mapping[str, Any]],
    *,
    boundary_rows: Sequence[Mapping[str, Any]],
    fgbg_rows: Sequence[Mapping[str, Any]],
    lowshot_rows: Sequence[Mapping[str, Any]],
    texture_rows: Sequence[Mapping[str, Any]],
    replicates: int,
    seed: int,
) -> dict[str, ConfidenceInterval]:
    slices = {
        "overall_skill": list(rows),
        "boundary_skill": list(boundary_rows),
        "fgbg_skill": list(fgbg_rows),
        "lowshot_skill": list(lowshot_rows),
    }
    if texture_rows:
        slices["texture_skill"] = list(texture_rows)
    estimates = {name: _weighted_skill(subset) for name, subset in slices.items()}
    estimates["failure_rate"] = _weighted_failure_rate(rows)
    successful_runtime_rows = [
        row
        for row in rows
        if not bool(row.get("failed")) and row.get("runtime_ms") is not None
    ]
    latency_p50 = _weighted_runtime_quantile(successful_runtime_rows, 0.5)
    latency_p95 = _weighted_runtime_quantile(successful_runtime_rows, 0.95)
    if latency_p50 is not None and latency_p95 is not None:
        estimates["latency_p50"] = latency_p50
        estimates["latency_p95"] = latency_p95
    if replicates == 0:
        return {
            name: ConfidenceInterval(value, value, value)
            for name, value in estimates.items()
        }
    categories = tuple(sorted({_text(row.get("category"), "category") for row in rows}))
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {name: [] for name in estimates}
    for _ in range(replicates):
        multiplicity = Counter(rng.choice(categories) for _ in categories)
        for name, subset in slices.items():
            expanded = [
                row for row in subset
                for _ in range(multiplicity.get(str(row["category"]), 0))
            ]
            if expanded:
                samples[name].append(_weighted_skill(expanded))
        expanded_all = [
            row for row in rows
            for _ in range(multiplicity.get(str(row["category"]), 0))
        ]
        samples["failure_rate"].append(_weighted_failure_rate(expanded_all))
        expanded_runtime = [
            row
            for row in expanded_all
            if not bool(row.get("failed")) and row.get("runtime_ms") is not None
        ]
        for name, quantile in (("latency_p50", 0.5), ("latency_p95", 0.95)):
            if name in samples:
                value = _weighted_runtime_quantile(expanded_runtime, quantile)
                samples[name].append(
                    estimates[name] if value is None else value
                )
    return {
        name: ConfidenceInterval(
            estimate=value,
            lower=_unweighted_quantile(samples[name], 0.025),
            upper=_unweighted_quantile(samples[name], 0.975),
        )
        for name, value in estimates.items()
    }


def _validate_signal_coverage(name: str, values: Mapping[str, float], task_ids: Sequence[str]) -> None:
    missing = [task_id for task_id in task_ids if task_id not in values]
    if missing:
        raise ExpertBankInputError(f"missing {name} signals for tasks: {missing[:5]}")
    for task_id in task_ids:
        _finite(values[task_id], f"{name} signal for {task_id}")


def _weighted_quantile(values: Sequence[float], weights: Sequence[float], quantile: float) -> float:
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    if not ordered:
        raise ExpertBankInputError("weighted quantile requires values")
    threshold = quantile * sum(weight for _, weight in ordered)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(ordered[-1][0])


def _unweighted_quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ExpertBankInputError("bootstrap produced no samples")
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _row_weight(row: Mapping[str, Any]) -> float:
    return _positive(row.get("sample_weight"), "sample_weight")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExpertBankInputError(f"{field} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ExpertBankInputError(f"{field} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ExpertBankInputError(f"{field} must be a positive integer") from exc
    if parsed <= 0 or (isinstance(value, float) and not value.is_integer()):
        raise ExpertBankInputError(f"{field} must be a positive integer")
    return parsed


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ExpertBankInputError(f"{field} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ExpertBankInputError(
            f"{field} must be a non-negative integer"
        ) from exc
    if parsed < 0 or (isinstance(value, float) and not value.is_integer()):
        raise ExpertBankInputError(f"{field} must be a non-negative integer")
    return parsed


def _finite(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ExpertBankInputError(f"{field} must be finite") from exc
    if not math.isfinite(parsed):
        raise ExpertBankInputError(f"{field} must be finite")
    return parsed


def _positive(value: Any, field: str) -> float:
    parsed = _finite(value, field)
    if parsed <= 0.0:
        raise ExpertBankInputError(f"{field} must be positive")
    return parsed


def _nonnegative(value: Any, field: str) -> float:
    parsed = _finite(value, field)
    if parsed < 0.0:
        raise ExpertBankInputError(f"{field} must be non-negative")
    return parsed


def _probability(value: Any, field: str) -> float:
    parsed = _finite(value, field)
    if not 0.0 <= parsed <= 1.0:
        raise ExpertBankInputError(f"{field} must be in [0,1]")
    return parsed


__all__ = [
    "CAPABILITY_BANK_NAME",
    "CAPABILITY_FEATURE_NAMES",
    "EXPERT_BANK_PROTOCOL_VERSION",
    "EXPERT_BANK_COMPATIBLE_PROTOCOL_VERSIONS",
    "ConfidenceInterval",
    "ExpertBankError",
    "ExpertBankInputError",
    "ExpertBankIsolationError",
    "ExpertCapabilityBank",
    "ExpertCapabilityProfile",
    "SkillCurveBin",
    "Stage2RuntimeSummaryRecord",
    "build_capability_bank",
    "build_expert_capability_bank",
    "extract_capability_signals",
    "extract_difficulty_signals",
    "iter_feature_records_jsonl",
    "read_capability_bank",
    "read_stage2_runtime_summary",
    "write_capability_bank",
]
