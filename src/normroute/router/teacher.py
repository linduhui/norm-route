"""Fold-isolated Stage 5 teacher targets.

The teacher is an evaluator-side component.  It accepts only a Stage 3
``routing_matrix_*`` stored below an ``evaluator_only`` directory, fits score
calibration on the current fold's training categories, uses validation rows
only to select the soft-target temperature, and never parses test outcomes.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence


TEACHER_PROTOCOL_VERSION = "stage5.teacher.v2"
TEACHER_PARQUET_NAME = "teacher.parquet"
TEACHER_COLUMNS = (
    "protocol_version",
    "evaluator_only",
    "fold",
    "split",
    "task_id",
    "image_id",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
    "query_group_id",
    "sample_weight",
    "expert_name",
    "label",
    "status",
    "failed",
    "failure_message",
    "raw_expert_score",
    "runtime_ms",
    "calibrated_anomaly_probability",
    "correctness_probability",
    "teacher_risk",
    "normalized_cost",
    "failure_penalty_value",
    "teacher_objective",
    "teacher_utility",
    "soft_utility_probability",
    "hard_oracle_expert",
    "hard_oracle",
    "calibration_score_mean",
    "calibration_score_scale",
    "calibration_slope",
    "calibration_intercept",
    "calibration_scope",
    "calibration_fit_categories",
    "temperature",
)
_SPLITS = ("train", "val", "test")
_KEY_COLUMNS = (
    "image_id",
    "dataset",
    "category",
    "support_set_id",
    "k_shot",
    "seed",
)


class TeacherError(RuntimeError):
    """Base error for Stage 5 teacher construction."""


class TeacherIsolationError(TeacherError, ValueError):
    """Raised when an evaluator-only or fold boundary would be crossed."""


class TeacherInputError(TeacherError, ValueError):
    """Raised when teacher inputs are incomplete or malformed."""


class TeacherDependencyError(TeacherError, ImportError):
    """Raised when optional Parquet dependencies are unavailable."""


@dataclass(frozen=True)
class ExpertScoreCalibration:
    """Train-only one-dimensional Platt calibration parameters."""

    expert_name: str
    score_mean: float
    score_scale: float
    slope: float
    intercept: float
    sample_count: int
    positive_count: int
    failure_count: int
    effective_sample_weight: float
    fit_categories: tuple[str, ...]

    def predict(self, score: float) -> float:
        standardized = (float(score) - self.score_mean) / self.score_scale
        return _sigmoid(self.slope * standardized + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expert_name": self.expert_name,
            "score_mean": self.score_mean,
            "score_scale": self.score_scale,
            "slope": self.slope,
            "intercept": self.intercept,
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "failure_count": self.failure_count,
            "effective_sample_weight": self.effective_sample_weight,
            "fit_categories": list(self.fit_categories),
        }


@dataclass(frozen=True)
class TeacherArtifact:
    """In-memory fold-scoped teacher artifact."""

    fold: str
    train_categories: tuple[str, ...]
    validation_categories: tuple[str, ...]
    experts: tuple[str, ...]
    calibrations: tuple[ExpertScoreCalibration, ...]
    temperature: float
    validation_cross_entropy: float
    rows: tuple[dict[str, Any], ...]
    temperature_frontier: tuple[dict[str, float], ...] = ()
    calibration_strategy: str = "leave_one_train_category_out"
    repeat_weighting: str = "equal_category_equal_query_inverse_variant_frequency"
    proper_loss: str = "negative_log_correctness_probability"
    cost_weight: float = 0.05
    failure_penalty: float = 2.0
    protocol_version: str = TEACHER_PROTOCOL_VERSION

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "evaluator_only": True,
            "fold": self.fold,
            "train_categories": list(self.train_categories),
            "validation_categories": list(self.validation_categories),
            "experts": list(self.experts),
            "temperature": self.temperature,
            "validation_cross_entropy": self.validation_cross_entropy,
            "temperature_frontier": [dict(item) for item in self.temperature_frontier],
            "calibration_strategy": self.calibration_strategy,
            "repeat_weighting": self.repeat_weighting,
            "proper_loss": self.proper_loss,
            "cost_weight": self.cost_weight,
            "failure_penalty": self.failure_penalty,
            "calibrations": [item.to_dict() for item in self.calibrations],
        }


@dataclass(frozen=True)
class _ExpertOutcome:
    status: str
    score: float | None
    runtime_ms: float | None
    failure_message: str

    @property
    def failed(self) -> bool:
        return self.status != "ok"


@dataclass(frozen=True)
class _TaskOutcome:
    manifest: Mapping[str, str]
    label: int
    experts: Mapping[str, _ExpertOutcome]


def build_teacher(
    *,
    routing_matrix_path: str | Path,
    fold_manifest_path: str | Path,
    fold: str,
    temperatures: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 1.0),
    cost_weight: float = 0.05,
    failure_penalty: float = 2.0,
    calibration_strategy: str = "leave_one_train_category_out",
    repeat_weighting: str = "equal_category_equal_query_inverse_variant_frequency",
) -> TeacherArtifact:
    """Build train-only soft teacher targets for one category-held-out fold.

    Calibration parameters and serialized teacher utilities are computed from
    training categories only.  Validation outcomes select ``temperature``;
    test-category rows are filtered before their outcome fields are parsed.
    """

    source = ensure_evaluator_only_routing_matrix(routing_matrix_path)
    manifest_rows = _read_fold_manifest(fold_manifest_path, fold)
    by_split = {
        split: [row for row in manifest_rows if row["split"] == split]
        for split in _SPLITS
    }
    split_categories = {
        split: tuple(sorted({row["category"] for row in rows}))
        for split, rows in by_split.items()
    }
    allowed_rows = by_split["train"] + by_split["val"]
    outcomes = _read_allowed_outcomes(
        source,
        allowed_rows,
        allowed_categories=set(split_categories["train"] + split_categories["val"]),
    )
    train_ids = tuple(row["task_id"] for row in by_split["train"])
    validation_ids = tuple(row["task_id"] for row in by_split["val"])
    experts = _consistent_experts(outcomes, (*train_ids, *validation_ids))
    if calibration_strategy not in {
        "leave_one_train_category_out",
        "full_train",
    }:
        raise TeacherInputError("unsupported calibration_strategy")
    if repeat_weighting not in {
        "equal_category_equal_query_inverse_variant_frequency",
        "uniform_task",
    }:
        raise TeacherInputError("unsupported repeat_weighting")
    if (
        calibration_strategy == "leave_one_train_category_out"
        and len(split_categories["train"]) < 2
    ):
        raise TeacherInputError(
            "category-cross-fitted teacher requires at least two train categories"
        )
    resolved_cost_weight = _nonnegative_hyperparameter(cost_weight, "cost_weight")
    resolved_failure_penalty = _nonnegative_hyperparameter(
        failure_penalty, "failure_penalty"
    )
    weight_builder = (
        _balanced_task_weights
        if repeat_weighting
        == "equal_category_equal_query_inverse_variant_frequency"
        else _uniform_task_weights
    )
    train_weights = weight_builder(train_ids, outcomes)
    validation_weights = weight_builder(validation_ids, outcomes)
    calibrations = _fit_expert_calibrations(
        train_ids, outcomes, experts, train_weights
    )
    calibration_by_expert = {
        item.expert_name: item for item in calibrations
    }
    candidates = _temperatures(temperatures)
    runtime_scale = _fit_runtime_scale(train_ids, outcomes, train_weights)
    selected_temperature, validation_loss, temperature_frontier = _select_temperature(
        validation_ids,
        outcomes,
        experts,
        calibration_by_expert,
        candidates,
        validation_weights,
        runtime_scale=runtime_scale,
        cost_weight=resolved_cost_weight,
        failure_penalty=resolved_failure_penalty,
    )
    crossfit_by_category: dict[str, dict[str, ExpertScoreCalibration]] = {}
    for held_out_category in split_categories["train"]:
        if calibration_strategy == "leave_one_train_category_out":
            fit_ids = tuple(
                task_id
                for task_id in train_ids
                if outcomes[task_id].manifest["category"] != held_out_category
            )
            fitted = _fit_expert_calibrations(
                fit_ids, outcomes, experts, train_weights
            )
            crossfit_by_category[held_out_category] = {
                item.expert_name: item for item in fitted
            }
        else:
            crossfit_by_category[held_out_category] = dict(calibration_by_expert)
    rows = _teacher_rows(
        train_ids,
        outcomes,
        experts,
        crossfit_by_category,
        train_weights,
        fold=fold,
        temperature=selected_temperature,
        runtime_scale=runtime_scale,
        cost_weight=resolved_cost_weight,
        failure_penalty=resolved_failure_penalty,
        calibration_scope=calibration_strategy,
    )
    return TeacherArtifact(
        fold=fold,
        train_categories=split_categories["train"],
        validation_categories=split_categories["val"],
        experts=experts,
        calibrations=calibrations,
        temperature=selected_temperature,
        validation_cross_entropy=validation_loss,
        rows=tuple(rows),
        temperature_frontier=temperature_frontier,
        cost_weight=resolved_cost_weight,
        failure_penalty=resolved_failure_penalty,
        calibration_strategy=calibration_strategy,
        repeat_weighting=repeat_weighting,
    )


def build_and_write_teacher(
    *,
    routing_matrix_path: str | Path,
    fold_manifest_path: str | Path,
    fold: str,
    output_path: str | Path,
    temperatures: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 1.0),
    cost_weight: float = 0.05,
    failure_penalty: float = 2.0,
    calibration_strategy: str = "leave_one_train_category_out",
    repeat_weighting: str = "equal_category_equal_query_inverse_variant_frequency",
) -> TeacherArtifact:
    """Build and atomically write one evaluator-only teacher Parquet."""

    artifact = build_teacher(
        routing_matrix_path=routing_matrix_path,
        fold_manifest_path=fold_manifest_path,
        fold=fold,
        temperatures=temperatures,
        cost_weight=cost_weight,
        failure_penalty=failure_penalty,
        calibration_strategy=calibration_strategy,
        repeat_weighting=repeat_weighting,
    )
    write_teacher_parquet(artifact, output_path)
    return artifact


def ensure_evaluator_only_routing_matrix(path: str | Path) -> Path:
    """Reject every teacher source except an evaluator-only routing matrix."""

    source = Path(path)
    parts = {part.casefold() for part in source.parts}
    if "evaluator_only" not in parts or not source.stem.casefold().startswith(
        "routing_matrix"
    ):
        raise TeacherIsolationError(
            "teacher may read only a routing_matrix* artifact under an "
            "evaluator_only directory"
        )
    if source.suffix.casefold() not in {".csv", ".parquet"}:
        raise TeacherIsolationError(
            "teacher routing matrix must be CSV or Parquet"
        )
    if not source.is_file():
        raise TeacherInputError(f"routing matrix does not exist: {source}")
    return source


def ensure_evaluator_only_teacher_output(path: str | Path) -> Path:
    destination = Path(path)
    if "evaluator_only" not in {part.casefold() for part in destination.parts}:
        raise TeacherIsolationError(
            "teacher targets must be written under an evaluator_only directory"
        )
    if destination.suffix.casefold() != ".parquet":
        raise TeacherIsolationError("teacher target output must be Parquet")
    return destination


def write_teacher_parquet(
    artifact: TeacherArtifact,
    output_path: str | Path,
) -> Path:
    """Atomically write a typed evaluator-only teacher target table."""

    destination = ensure_evaluator_only_teacher_output(output_path)
    if not artifact.rows:
        raise TeacherInputError("cannot write an empty teacher artifact")
    pa, pq = _pyarrow()
    records = [dict(row) for row in artifact.rows]
    for row in records:
        if tuple(row) != TEACHER_COLUMNS:
            raise TeacherInputError("teacher row does not match the frozen schema")
        if row["split"] != "train" or row["category"] not in artifact.train_categories:
            raise TeacherIsolationError("teacher Parquet may contain train categories only")
        if row["protocol_version"] != TEACHER_PROTOCOL_VERSION:
            raise TeacherInputError("teacher row has an incompatible protocol")
    schema = _teacher_schema(pa).with_metadata(
        {
            b"normroute.teacher.metadata": json.dumps(
                artifact.metadata(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        }
    )
    try:
        table = pa.Table.from_pylist(records, schema=schema)
    except Exception as exc:
        raise TeacherInputError(f"could not construct teacher table: {exc}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            version="2.6",
        )
        os.replace(temporary, destination)
    except Exception as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise TeacherError(f"could not write teacher Parquet {destination}: {exc}") from exc
    return destination


def read_teacher_parquet(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Read teacher rows while preserving the evaluator-only path boundary."""

    source = ensure_evaluator_only_teacher_output(path)
    if not source.is_file():
        raise TeacherInputError(f"teacher Parquet does not exist: {source}")
    _, pq = _pyarrow()
    try:
        table = pq.read_table(source)
    except Exception as exc:
        raise TeacherInputError(f"could not read teacher Parquet {source}: {exc}") from exc
    if tuple(table.column_names) != TEACHER_COLUMNS:
        raise TeacherInputError("teacher Parquet does not match the frozen schema")
    rows = tuple(dict(row) for row in table.to_pylist())
    if not rows or any(
        row.get("protocol_version") != TEACHER_PROTOCOL_VERSION
        or row.get("evaluator_only") is not True
        or row.get("split") != "train"
        for row in rows
    ):
        raise TeacherIsolationError("teacher Parquet is empty or not train/evaluator-only")
    return rows


def _read_fold_manifest(path: str | Path, fold: str) -> list[dict[str, str]]:
    source = Path(path)
    if not source.is_file():
        raise TeacherInputError(f"fold manifest does not exist: {source}")
    required = (
        "fold",
        "split",
        "task_id",
        "dataset",
        "category",
        "k_shot",
        "seed",
        "support_set_id",
    )
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise TeacherInputError(f"{source} is missing columns: {missing}")
        for line_number, raw in enumerate(reader, start=2):
            clean = {key: (value or "").strip() for key, value in raw.items()}
            if clean["fold"] != fold:
                continue
            if clean["split"] not in _SPLITS:
                raise TeacherInputError(f"{source}:{line_number} has invalid split")
            identity = next(
                (
                    clean.get(name, "")
                    for name in ("sample_id", "image_id", "query_id")
                    if clean.get(name, "")
                ),
                "",
            )
            missing_values = [name for name in required[1:] if not clean.get(name)]
            if not identity:
                missing_values.append("sample_id/image_id/query_id")
            if missing_values:
                raise TeacherInputError(
                    f"{source}:{line_number} is missing values: {missing_values}"
                )
            if clean["task_id"] in seen:
                raise TeacherInputError(f"{source} has duplicate task_id {clean['task_id']!r}")
            seen.add(clean["task_id"])
            clean["image_id"] = identity
            rows.append(clean)
    if not rows or {row["split"] for row in rows} != set(_SPLITS):
        raise TeacherInputError(f"{source} does not contain all splits for {fold}")
    categories = {
        split: {row["category"] for row in rows if row["split"] == split}
        for split in _SPLITS
    }
    if any(
        categories[left].intersection(categories[right])
        for index, left in enumerate(_SPLITS)
        for right in _SPLITS[index + 1 :]
    ):
        raise TeacherIsolationError("fold categories are not pairwise disjoint")
    return rows


def _read_allowed_outcomes(
    source: Path,
    manifest_rows: Sequence[Mapping[str, str]],
    *,
    allowed_categories: set[str],
) -> dict[str, _TaskOutcome]:
    lookup: dict[tuple[str, ...], Mapping[str, str]] = {}
    for row in manifest_rows:
        key = _manifest_key(row)
        if key in lookup:
            raise TeacherInputError("fold manifest has duplicate routing signature")
        lookup[key] = row
    staged: dict[str, dict[str, Any]] = {}
    for line_number, raw in _routing_rows(
        source, allowed_categories=allowed_categories
    ):
        # This category gate intentionally precedes label, score, runtime, status,
        # and expert parsing.  Test-category outcomes remain completely unused.
        category = str(raw.get("category") or "").strip()
        if category not in allowed_categories:
            continue
        clean = {
            str(key): str(value if value is not None else "").strip()
            for key, value in raw.items()
        }
        expert_column = "expert_name" if "expert_name" in clean else "expert"
        score_column = "final_score" if "final_score" in clean else "image_score"
        missing = [name for name in _KEY_COLUMNS if not clean.get(name)]
        if not clean.get(expert_column):
            missing.append("expert_name/expert")
        if not clean.get("label"):
            missing.append("label")
        if missing:
            raise TeacherInputError(f"{source}:{line_number} is missing values: {missing}")
        key = _matrix_key(clean)
        manifest = lookup.get(key)
        if manifest is None:
            raise TeacherIsolationError(
                f"{source}:{line_number} has an allowed-category row outside the fold manifest"
            )
        task_id = manifest["task_id"]
        label = _binary_label(clean["label"], source, line_number)
        status = (clean.get("status") or "ok").casefold()
        failed = status != "ok"
        score = None
        if not failed:
            if not clean.get(score_column):
                raise TeacherInputError(
                    f"{source}:{line_number} successful output has no expert score"
                )
            score = _finite_float(clean[score_column], source, line_number, score_column)
        runtime = None
        if clean.get("runtime_ms"):
            runtime = _finite_float(clean["runtime_ms"], source, line_number, "runtime_ms")
            if runtime < 0.0:
                raise TeacherInputError(f"{source}:{line_number} has negative runtime_ms")
        expert = clean[expert_column].casefold()
        current = staged.setdefault(
            task_id,
            {"manifest": manifest, "label": label, "experts": {}},
        )
        if current["label"] != label:
            raise TeacherInputError(f"{source}:{line_number} has conflicting task labels")
        if expert in current["experts"]:
            raise TeacherInputError(f"{source}:{line_number} duplicates an expert/task row")
        current["experts"][expert] = _ExpertOutcome(
            status=status,
            score=score,
            runtime_ms=runtime,
            failure_message=clean.get("error_message", ""),
        )
    expected = {row["task_id"] for row in manifest_rows}
    if set(staged) != expected:
        missing = sorted(expected - set(staged))
        extra = sorted(set(staged) - expected)
        raise TeacherInputError(
            f"routing matrix coverage disagrees with train/val manifest; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    return {
        task_id: _TaskOutcome(
            manifest=value["manifest"],
            label=int(value["label"]),
            experts=dict(value["experts"]),
        )
        for task_id, value in staged.items()
    }


def _routing_rows(
    source: Path,
    *,
    allowed_categories: set[str] | None = None,
) -> Iterable[tuple[int, Mapping[str, Any]]]:
    if source.suffix.casefold() == ".csv":
        with source.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            if "category" not in fields:
                raise TeacherInputError(f"{source} is missing category")
            for line_number, row in enumerate(reader, start=2):
                yield line_number, row
        return
    _, pq = _pyarrow()
    try:
        table = pq.read_table(
            source,
            filters=(
                [("category", "in", sorted(allowed_categories))]
                if allowed_categories is not None
                else None
            ),
        )
    except Exception as exc:
        raise TeacherInputError(f"could not read routing matrix {source}: {exc}") from exc
    if "category" not in table.column_names:
        raise TeacherInputError(f"{source} is missing category")
    for line_number, row in enumerate(table.to_pylist(), start=2):
        yield line_number, row


def _consistent_experts(
    outcomes: Mapping[str, _TaskOutcome], task_ids: Sequence[str]
) -> tuple[str, ...]:
    sets = {tuple(sorted(outcomes[task_id].experts)) for task_id in task_ids}
    if len(sets) != 1:
        raise TeacherInputError("routing matrix has inconsistent expert coverage")
    experts = next(iter(sets))
    if len(experts) < 2:
        raise TeacherInputError("teacher requires at least two experts")
    for task_id in task_ids:
        if all(item.failed for item in outcomes[task_id].experts.values()):
            raise TeacherInputError(f"all experts failed for task_id={task_id!r}")
    return experts


def _fit_expert_calibrations(
    task_ids: Sequence[str],
    outcomes: Mapping[str, _TaskOutcome],
    experts: Sequence[str],
    task_weights: Mapping[str, float],
) -> tuple[ExpertScoreCalibration, ...]:
    calibrations = []
    for expert in experts:
        successful = [
            (
                outcomes[task_id].label,
                outcomes[task_id].experts[expert].score,
                task_weights[task_id],
            )
            for task_id in task_ids
            if not outcomes[task_id].experts[expert].failed
        ]
        labels = [label for label, _, _ in successful]
        scores = [float(score) for _, score, _ in successful if score is not None]
        weights = [float(weight) for _, _, weight in successful]
        if len(scores) != len(successful) or set(labels) != {0, 1}:
            raise TeacherInputError(
                f"expert {expert!r} calibration requires successful train rows "
                "from both labels"
            )
        total_weight = sum(weights)
        if total_weight <= 0.0:
            raise TeacherInputError(f"expert {expert!r} has zero calibration weight")
        mean = sum(weight * score for score, weight in zip(scores, weights)) / total_weight
        variance = sum(
            weight * (value - mean) ** 2
            for value, weight in zip(scores, weights)
        ) / total_weight
        scale = math.sqrt(variance) if variance > 1e-12 else 1.0
        standardized = [(value - mean) / scale for value in scores]
        slope, intercept = _fit_logistic(standardized, labels, weights)
        fit_categories = tuple(
            sorted({outcomes[task_id].manifest["category"] for task_id in task_ids})
        )
        calibrations.append(
            ExpertScoreCalibration(
                expert_name=expert,
                score_mean=mean,
                score_scale=scale,
                slope=slope,
                intercept=intercept,
                sample_count=len(scores),
                positive_count=sum(labels),
                failure_count=sum(
                    outcomes[task_id].experts[expert].failed for task_id in task_ids
                ),
                effective_sample_weight=total_weight,
                fit_categories=fit_categories,
            )
        )
    return tuple(calibrations)


def _fit_logistic(
    values: Sequence[float],
    labels: Sequence[int],
    weights: Sequence[float],
) -> tuple[float, float]:
    total_weight = sum(weights)
    positive_rate = (
        sum(weight * label for label, weight in zip(labels, weights)) + 0.5
    ) / (total_weight + 1.0)
    slope = 0.0
    intercept = math.log(positive_rate / (1.0 - positive_rate))
    l2 = 1e-3
    for _ in range(64):
        grad_slope = l2 * slope
        grad_intercept = 0.0
        h_ss = l2
        h_si = 0.0
        h_ii = 1e-9
        for value, label, weight in zip(values, labels, weights):
            probability = _sigmoid(slope * value + intercept)
            residual = weight * (probability - label)
            curvature = weight * max(probability * (1.0 - probability), 1e-9)
            grad_slope += residual * value
            grad_intercept += residual
            h_ss += curvature * value * value
            h_si += curvature * value
            h_ii += curvature
        determinant = h_ss * h_ii - h_si * h_si
        if determinant <= 1e-14:
            break
        delta_slope = (h_ii * grad_slope - h_si * grad_intercept) / determinant
        delta_intercept = (-h_si * grad_slope + h_ss * grad_intercept) / determinant
        slope -= delta_slope
        intercept -= delta_intercept
        if max(abs(delta_slope), abs(delta_intercept)) < 1e-10:
            break
    if not all(math.isfinite(value) for value in (slope, intercept)):
        raise TeacherInputError("expert score calibration did not converge")
    return slope, intercept


def _select_temperature(
    task_ids: Sequence[str],
    outcomes: Mapping[str, _TaskOutcome],
    experts: Sequence[str],
    calibrations: Mapping[str, ExpertScoreCalibration],
    candidates: Sequence[float],
    task_weights: Mapping[str, float],
    *,
    runtime_scale: tuple[float, float],
    cost_weight: float,
    failure_penalty: float,
) -> tuple[float, float, tuple[dict[str, float], ...]]:
    ranked: list[tuple[float, float, float, float]] = []
    for temperature in candidates:
        weighted_loss = 0.0
        weighted_entropy = 0.0
        weighted_expected_utility = 0.0
        weight_sum = 0.0
        for task_id in task_ids:
            components = _objective_components(
                outcomes[task_id],
                experts,
                calibrations,
                runtime_scale=runtime_scale,
                cost_weight=cost_weight,
                failure_penalty=failure_penalty,
            )
            target = _normalize_nonnegative(
                [item[1] for item in components], "validation correctness utilities"
            )
            probabilities = _softmax(
                [-item[5] for item in components], temperature
            )
            weight = task_weights[task_id]
            weighted_loss += weight * -sum(
                target_value * math.log(max(probability, 1e-12))
                for target_value, probability in zip(target, probabilities)
            )
            weighted_entropy += weight * -sum(
                probability * math.log(max(probability, 1e-12))
                for probability in probabilities
            )
            weighted_expected_utility += weight * sum(
                probability * item[1]
                for probability, item in zip(probabilities, components)
            )
            weight_sum += weight
        ranked.append(
            (
                weighted_loss / weight_sum,
                temperature,
                weighted_entropy / weight_sum,
                weighted_expected_utility / weight_sum,
            )
        )
    loss, temperature, _, _ = min(ranked, key=lambda item: (item[0], item[1]))
    frontier = tuple(
        {
            "temperature": float(item[1]),
            "validation_soft_cross_entropy": float(item[0]),
            "validation_distribution_entropy": float(item[2]),
            "validation_expected_correctness": float(item[3]),
        }
        for item in sorted(ranked, key=lambda item: item[1])
    )
    return temperature, loss, frontier


def _teacher_rows(
    task_ids: Sequence[str],
    outcomes: Mapping[str, _TaskOutcome],
    experts: Sequence[str],
    calibrations_by_held_out_category: Mapping[
        str, Mapping[str, ExpertScoreCalibration]
    ],
    task_weights: Mapping[str, float],
    *,
    fold: str,
    temperature: float,
    runtime_scale: tuple[float, float],
    cost_weight: float,
    failure_penalty: float,
    calibration_scope: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        task = outcomes[task_id]
        held_out_category = task.manifest["category"]
        calibrations = calibrations_by_held_out_category[held_out_category]
        components = _objective_components(
            task,
            experts,
            calibrations,
            runtime_scale=runtime_scale,
            cost_weight=cost_weight,
            failure_penalty=failure_penalty,
        )
        distribution = _softmax([-item[5] for item in components], temperature)
        hard_index = min(
            range(len(experts)), key=lambda index: (components[index][5], index)
        )
        hard_expert = experts[hard_index]
        manifest = task.manifest
        for index, expert in enumerate(experts):
            result = task.experts[expert]
            calibration = calibrations[expert]
            (
                anomaly_probability,
                correctness_probability,
                risk,
                normalized_cost,
                failure_value,
                objective,
            ) = components[index]
            rows.append(
                {
                    "protocol_version": TEACHER_PROTOCOL_VERSION,
                    "evaluator_only": True,
                    "fold": fold,
                    "split": "train",
                    "task_id": task_id,
                    "image_id": manifest["image_id"],
                    "dataset": manifest["dataset"],
                    "category": manifest["category"],
                    "k_shot": int(manifest["k_shot"]),
                    "seed": int(manifest["seed"]),
                    "support_set_id": manifest["support_set_id"],
                    "query_group_id": _query_group_id(manifest),
                    "sample_weight": task_weights[task_id],
                    "expert_name": expert,
                    "label": task.label,
                    "status": result.status,
                    "failed": result.failed,
                    "failure_message": result.failure_message,
                    "raw_expert_score": result.score,
                    "runtime_ms": result.runtime_ms,
                    "calibrated_anomaly_probability": anomaly_probability,
                    "correctness_probability": correctness_probability,
                    "teacher_risk": risk,
                    "normalized_cost": normalized_cost,
                    "failure_penalty_value": failure_value,
                    "teacher_objective": objective,
                    "teacher_utility": math.exp(-min(objective, 700.0)),
                    "soft_utility_probability": distribution[index],
                    "hard_oracle_expert": hard_expert,
                    "hard_oracle": expert == hard_expert,
                    "calibration_score_mean": calibration.score_mean,
                    "calibration_score_scale": calibration.score_scale,
                    "calibration_slope": calibration.slope,
                    "calibration_intercept": calibration.intercept,
                    "calibration_scope": calibration_scope,
                    "calibration_fit_categories": json.dumps(
                        list(calibration.fit_categories), separators=(",", ":")
                    ),
                    "temperature": temperature,
                }
            )
    return rows


def _objective_components(
    task: _TaskOutcome,
    experts: Sequence[str],
    calibrations: Mapping[str, ExpertScoreCalibration],
    *,
    runtime_scale: tuple[float, float],
    cost_weight: float,
    failure_penalty: float,
) -> list[tuple[float | None, float, float, float, float, float]]:
    components: list[tuple[float | None, float, float, float, float, float]] = []
    for expert in experts:
        result = task.experts[expert]
        if result.failed or result.score is None:
            correctness = 0.0
            risk = -math.log(1e-12)
            normalized_cost = 1.0
            penalty_value = failure_penalty
            objective = risk + cost_weight * normalized_cost + penalty_value
            components.append(
                (None, correctness, risk, normalized_cost, penalty_value, objective)
            )
            continue
        probability = calibrations[expert].predict(result.score)
        correctness = probability if task.label == 1 else 1.0 - probability
        risk = -math.log(max(correctness, 1e-12))
        normalized_cost = _normalized_runtime(result.runtime_ms, runtime_scale)
        objective = risk + cost_weight * normalized_cost
        components.append(
            (probability, correctness, risk, normalized_cost, 0.0, objective)
        )
    return components


def _balanced_task_weights(
    task_ids: Sequence[str],
    outcomes: Mapping[str, _TaskOutcome],
) -> dict[str, float]:
    """Give every category and opaque query equal aggregate influence."""

    if not task_ids:
        raise TeacherInputError("cannot weight an empty split")
    category_groups: dict[str, dict[str, list[str]]] = {}
    for task_id in task_ids:
        manifest = outcomes[task_id].manifest
        category = manifest["category"]
        group_id = _query_group_id(manifest)
        category_groups.setdefault(category, {}).setdefault(group_id, []).append(task_id)
    raw: dict[str, float] = {}
    for groups in category_groups.values():
        group_mass = 1.0 / len(groups)
        for variants in groups.values():
            variant_weight = group_mass / len(variants)
            for task_id in variants:
                raw[task_id] = variant_weight
    normalization = len(task_ids) / sum(raw.values())
    return {task_id: raw[task_id] * normalization for task_id in task_ids}


def _uniform_task_weights(
    task_ids: Sequence[str],
    outcomes: Mapping[str, _TaskOutcome],
) -> dict[str, float]:
    del outcomes
    if not task_ids:
        raise TeacherInputError("cannot weight an empty split")
    return {task_id: 1.0 for task_id in task_ids}


def _query_group_id(manifest: Mapping[str, str]) -> str:
    return "|".join(
        (manifest["dataset"], manifest["category"], manifest["image_id"])
    )


def _fit_runtime_scale(
    task_ids: Sequence[str],
    outcomes: Mapping[str, _TaskOutcome],
    task_weights: Mapping[str, float],
) -> tuple[float, float]:
    values: list[float] = []
    weights: list[float] = []
    for task_id in task_ids:
        for result in outcomes[task_id].experts.values():
            if not result.failed and result.runtime_ms is not None:
                values.append(math.log1p(result.runtime_ms))
                weights.append(task_weights[task_id])
    if not values:
        return 0.0, 1.0
    lower = _weighted_quantile(values, weights, 0.1)
    upper = _weighted_quantile(values, weights, 0.9)
    if upper - lower <= 1e-12:
        upper = lower + 1.0
    return lower, upper


def _normalized_runtime(
    runtime_ms: float | None, runtime_scale: tuple[float, float]
) -> float:
    if runtime_ms is None:
        return 1.0
    lower, upper = runtime_scale
    value = (math.log1p(runtime_ms) - lower) / (upper - lower)
    return min(max(value, 0.0), 1.0)


def _weighted_quantile(
    values: Sequence[float], weights: Sequence[float], quantile: float
) -> float:
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    total = sum(weight for _, weight in ordered)
    threshold = quantile * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(ordered[-1][0])


def _normalize_nonnegative(values: Sequence[float], field: str) -> list[float]:
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise TeacherInputError(f"{field} must be finite and non-negative")
    total = sum(values)
    if total <= 1e-12:
        raise TeacherInputError(f"{field} have zero total mass")
    return [value / total for value in values]


def _nonnegative_hyperparameter(value: float, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TeacherInputError(f"{field} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise TeacherInputError(f"{field} must be finite and non-negative")
    return result


def _temperatures(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TeacherInputError("temperatures must be a sequence")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise TeacherInputError("temperatures must be numeric") from exc
    if (
        not result
        or any(not math.isfinite(value) or value <= 0.0 for value in result)
        or tuple(sorted(set(result))) != result
    ):
        raise TeacherInputError(
            "temperatures must be non-empty, sorted, unique, finite, and positive"
        )
    return result


def _manifest_key(row: Mapping[str, str]) -> tuple[str, ...]:
    return (
        row["image_id"],
        row["dataset"],
        row["category"],
        row["support_set_id"],
        str(int(row["k_shot"])),
        str(int(row["seed"])),
    )


def _matrix_key(row: Mapping[str, str]) -> tuple[str, ...]:
    try:
        return (
            row["image_id"],
            row["dataset"],
            row["category"],
            row["support_set_id"],
            str(int(row["k_shot"])),
            str(int(row["seed"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TeacherInputError("routing matrix has an invalid task signature") from exc


def _binary_label(value: str, path: Path, line_number: int) -> int:
    if value in {"0", "0.0"}:
        return 0
    if value in {"1", "1.0"}:
        return 1
    raise TeacherInputError(f"{path}:{line_number} has invalid binary label {value!r}")


def _finite_float(value: str, path: Path, line_number: int, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TeacherInputError(
            f"{path}:{line_number} has invalid numeric {field}"
        ) from exc
    if not math.isfinite(result):
        raise TeacherInputError(f"{path}:{line_number} has non-finite {field}")
    return result


def _softmax(utilities: Sequence[float], temperature: float) -> list[float]:
    logits = [value / temperature for value in utilities]
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + exp)
    exp = math.exp(max(value, -700.0))
    return exp / (1.0 + exp)


def _teacher_schema(pa: Any) -> Any:
    return pa.schema(
        [
            pa.field("protocol_version", pa.string(), nullable=False),
            pa.field("evaluator_only", pa.bool_(), nullable=False),
            pa.field("fold", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("task_id", pa.string(), nullable=False),
            pa.field("image_id", pa.string(), nullable=False),
            pa.field("dataset", pa.string(), nullable=False),
            pa.field("category", pa.string(), nullable=False),
            pa.field("k_shot", pa.int32(), nullable=False),
            pa.field("seed", pa.int64(), nullable=False),
            pa.field("support_set_id", pa.string(), nullable=False),
            pa.field("query_group_id", pa.string(), nullable=False),
            pa.field("sample_weight", pa.float64(), nullable=False),
            pa.field("expert_name", pa.string(), nullable=False),
            pa.field("label", pa.int8(), nullable=False),
            pa.field("status", pa.string(), nullable=False),
            pa.field("failed", pa.bool_(), nullable=False),
            pa.field("failure_message", pa.string(), nullable=False),
            pa.field("raw_expert_score", pa.float64(), nullable=True),
            pa.field("runtime_ms", pa.float64(), nullable=True),
            pa.field(
                "calibrated_anomaly_probability", pa.float64(), nullable=True
            ),
            pa.field("correctness_probability", pa.float64(), nullable=False),
            pa.field("teacher_risk", pa.float64(), nullable=False),
            pa.field("normalized_cost", pa.float64(), nullable=False),
            pa.field("failure_penalty_value", pa.float64(), nullable=False),
            pa.field("teacher_objective", pa.float64(), nullable=False),
            pa.field("teacher_utility", pa.float64(), nullable=False),
            pa.field("soft_utility_probability", pa.float64(), nullable=False),
            pa.field("hard_oracle_expert", pa.string(), nullable=False),
            pa.field("hard_oracle", pa.bool_(), nullable=False),
            pa.field("calibration_score_mean", pa.float64(), nullable=False),
            pa.field("calibration_score_scale", pa.float64(), nullable=False),
            pa.field("calibration_slope", pa.float64(), nullable=False),
            pa.field("calibration_intercept", pa.float64(), nullable=False),
            pa.field("calibration_scope", pa.string(), nullable=False),
            pa.field("calibration_fit_categories", pa.string(), nullable=False),
            pa.field("temperature", pa.float64(), nullable=False),
        ]
    )


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise TeacherDependencyError(
            "teacher Parquet requires pyarrow from the 'stage5' optional dependencies"
        ) from exc
    return pa, pq


__all__ = [
    "ExpertScoreCalibration",
    "TEACHER_COLUMNS",
    "TEACHER_PARQUET_NAME",
    "TEACHER_PROTOCOL_VERSION",
    "TeacherArtifact",
    "TeacherDependencyError",
    "TeacherError",
    "TeacherInputError",
    "TeacherIsolationError",
    "build_and_write_teacher",
    "build_teacher",
    "ensure_evaluator_only_routing_matrix",
    "ensure_evaluator_only_teacher_output",
    "read_teacher_parquet",
    "write_teacher_parquet",
]
