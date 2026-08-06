"""Leakage-separated training and inference datasets for Stage 5 routing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

from .teacher import TEACHER_PROTOCOL_VERSION, TeacherArtifact


STAGE5_DATASET_PROTOCOL_VERSION = "stage5.router_dataset.v2"
_FORBIDDEN_NAMES = frozenset(
    {
        "label",
        "labels",
        "mask",
        "masks",
        "mask_path",
        "defect_type",
        "anomaly_type",
        "ground_truth",
        "ground_truth_label",
        "target_label",
        "expert_score",
        "expert_scores",
        "expert_utility",
        "expert_utilities",
        "expert_outcome",
        "expert_outcomes",
        "patchcore_score",
        "winclip_score",
        "anomalydino_score",
        "teacher",
        "teacher_target",
        "teacher_utility",
        "oracle",
        "oracle_expert",
        "hard_oracle",
    }
)


class Stage5DatasetError(ValueError):
    """Raised when a Router dataset violates schema or isolation rules."""


@dataclass(frozen=True)
class RouterInferenceExample:
    task_id: str
    features: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "features": self.features}


@dataclass(frozen=True)
class RouterTrainingExample:
    task_id: str
    features: tuple[float, ...]
    target_distribution: tuple[float, ...]
    sample_weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "features": self.features,
            "target_distribution": self.target_distribution,
            "sample_weight": self.sample_weight,
        }


class RouterInferenceDataset(Sequence[dict[str, Any]]):
    """Model-facing dataset containing only task id and numeric features."""

    def __init__(self, feature_records: Sequence[Mapping[str, Any]]) -> None:
        normalized = _normalize_features(feature_records)
        self.feature_names = normalized[0]["feature_names"]
        self._examples = tuple(
            RouterInferenceExample(row["task_id"], row["features"])
            for row in normalized
        )
        self.group_ids = tuple(row["group_id"] for row in normalized)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._examples[index].to_dict()


class RouterTrainingDataset(Sequence[dict[str, Any]]):
    """Training dataset exposing soft targets but no raw evaluator outcomes."""

    def __init__(
        self,
        feature_records: Sequence[Mapping[str, Any]],
        teacher: TeacherArtifact | Sequence[Mapping[str, Any]],
    ) -> None:
        normalized = _normalize_features(feature_records)
        rows = list(teacher.rows) if isinstance(teacher, TeacherArtifact) else list(teacher)
        targets, weights, experts = _normalize_teacher_targets(rows)
        feature_ids = {row["task_id"] for row in normalized}
        if feature_ids != set(targets):
            raise Stage5DatasetError(
                "feature and teacher task coverage must match exactly; no row may be skipped"
            )
        self.feature_names = normalized[0]["feature_names"]
        self.experts = experts
        self._examples = tuple(
            RouterTrainingExample(
                row["task_id"],
                row["features"],
                targets[row["task_id"]],
                weights[row["task_id"]],
            )
            for row in normalized
        )
        self.group_ids = tuple(row["group_id"] for row in normalized)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._examples[index].to_dict()


InferenceDataset = RouterInferenceDataset
TrainingDataset = RouterTrainingDataset


def build_inference_dataset(
    feature_records: Sequence[Mapping[str, Any]],
) -> RouterInferenceDataset:
    return RouterInferenceDataset(feature_records)


def build_training_dataset(
    feature_records: Sequence[Mapping[str, Any]],
    teacher: TeacherArtifact | Sequence[Mapping[str, Any]],
) -> RouterTrainingDataset:
    return RouterTrainingDataset(feature_records, teacher)


def _normalize_features(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(records, (str, bytes)) or not records:
        raise Stage5DatasetError("feature_records must be a non-empty sequence")
    normalized: list[dict[str, Any]] = []
    schema: tuple[str, ...] | None = None
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise Stage5DatasetError(f"feature record {index} is not a mapping")
        findings = _forbidden_paths(record)
        if findings:
            raise Stage5DatasetError(
                f"feature record {index} contains forbidden inference fields: {findings}"
            )
        task_id = _text(record.get("task_id"), f"feature record {index} task_id")
        if task_id in seen:
            raise Stage5DatasetError(f"duplicate feature task_id {task_id!r}")
        seen.add(task_id)
        names = tuple(str(value) for value in record.get("feature_names", ()))
        values = record.get("values", record.get("features", ()))
        if not names or len(set(names)) != len(names):
            raise Stage5DatasetError(f"feature record {index} names are invalid")
        if schema is None:
            schema = names
        elif names != schema:
            raise Stage5DatasetError("feature schema changed within dataset")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise Stage5DatasetError(f"feature record {index} values are invalid")
        try:
            vector = tuple(float(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise Stage5DatasetError(f"feature record {index} values are not numeric") from exc
        if len(vector) != len(names) or not all(math.isfinite(value) for value in vector):
            raise Stage5DatasetError(f"feature record {index} values are misaligned/non-finite")
        normalized.append(
            {
                "task_id": task_id,
                "feature_names": names,
                "features": vector,
                "group_id": _query_group_id(record, task_id),
            }
        )
    return normalized


def _normalize_teacher_targets(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, tuple[float, ...]], dict[str, float], tuple[str, ...]]:
    if not rows:
        raise Stage5DatasetError("teacher rows must be non-empty")
    staged: dict[str, dict[str, float]] = {}
    sample_weights: dict[str, float] = {}
    expected_experts: tuple[str, ...] | None = None
    for index, row in enumerate(rows):
        if row.get("protocol_version") != TEACHER_PROTOCOL_VERSION:
            raise Stage5DatasetError(f"teacher row {index} has an incompatible protocol")
        if row.get("evaluator_only") is not True or row.get("split") != "train":
            raise Stage5DatasetError("training targets must be evaluator-only train rows")
        task_id = _text(row.get("task_id"), f"teacher row {index} task_id")
        expert = _text(row.get("expert_name"), f"teacher row {index} expert_name")
        probability = _probability(
            row.get("soft_utility_probability"),
            f"teacher row {index} soft_utility_probability",
        )
        if expert in staged.setdefault(task_id, {}):
            raise Stage5DatasetError(f"duplicate teacher task/expert row at {index}")
        staged[task_id][expert] = probability
        sample_weight = _positive_finite(
            row.get("sample_weight"), f"teacher row {index} sample_weight"
        )
        previous_weight = sample_weights.setdefault(task_id, sample_weight)
        if not math.isclose(previous_weight, sample_weight, rel_tol=1e-12, abs_tol=1e-12):
            raise Stage5DatasetError("teacher sample weight changed across experts")
    targets: dict[str, tuple[float, ...]] = {}
    for task_id, values in staged.items():
        experts = tuple(sorted(values))
        if expected_experts is None:
            expected_experts = experts
        elif experts != expected_experts:
            raise Stage5DatasetError("teacher expert coverage is inconsistent")
        total = sum(values.values())
        if not math.isclose(total, 1.0, rel_tol=1e-7, abs_tol=1e-7):
            raise Stage5DatasetError(f"teacher distribution for {task_id!r} does not sum to one")
        targets[task_id] = tuple(values[expert] / total for expert in experts)
    if expected_experts is None or len(expected_experts) < 2:
        raise Stage5DatasetError("teacher targets require at least two experts")
    return targets, sample_weights, expected_experts


def _query_group_id(record: Mapping[str, Any], task_id: str) -> tuple[str, str, str]:
    identity = next(
        (
            str(record[name]).strip()
            for name in (
                "query_image_sha256",
                "sample_id",
                "image_id",
                "query_id",
            )
            if record.get(name) is not None and str(record[name]).strip()
        ),
        "",
    )
    if not identity and "|" in task_id:
        identity = task_id.split("|", 1)[0]
    if not identity:
        raise Stage5DatasetError(
            "each feature record needs an opaque query identity for grouped sampling"
        )
    return (
        str(record.get("dataset", "")),
        str(record.get("category", "")),
        identity,
    )


def _forbidden_paths(value: Any, context: str = "root") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            normalized = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
            tokens = set(normalized.split("_"))
            if (
                normalized in _FORBIDDEN_NAMES
                or "label" in tokens
                or "mask" in tokens
                or "oracle" in tokens
                or "teacher" in tokens
                or (
                    "expert" in tokens
                    and tokens.intersection(
                        {"score", "scores", "utility", "utilities", "outcome", "outcomes"}
                    )
                )
                or (
                    "score" in tokens
                    and tokens.intersection({"patchcore", "winclip", "anomalydino"})
                )
            ):
                findings.append(f"{context}.{name}")
            findings.extend(_forbidden_paths(child, f"{context}.{name}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            # Feature names are data values but still form the model schema.
            if isinstance(child, str):
                normalized = re.sub(r"[^a-z0-9]+", "_", child.casefold()).strip("_")
                tokens = set(normalized.split("_"))
                if tokens.intersection({"label", "mask", "oracle", "teacher"}) or (
                    "expert" in tokens
                    and tokens.intersection(
                        {"score", "scores", "utility", "utilities", "outcome", "outcomes"}
                    )
                ) or (
                    "score" in tokens
                    and tokens.intersection({"patchcore", "winclip", "anomalydino"})
                ):
                    findings.append(f"{context}[{index}]")
            else:
                findings.extend(_forbidden_paths(child, f"{context}[{index}]"))
    return findings


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage5DatasetError(f"{field} must be a non-empty string")
    return value.strip()


def _probability(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise Stage5DatasetError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise Stage5DatasetError(f"{field} must be finite and in [0,1]")
    return parsed


def _positive_finite(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise Stage5DatasetError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise Stage5DatasetError(f"{field} must be finite and positive")
    return parsed


__all__ = [
    "InferenceDataset",
    "RouterInferenceDataset",
    "RouterInferenceExample",
    "RouterTrainingDataset",
    "RouterTrainingExample",
    "STAGE5_DATASET_PROTOCOL_VERSION",
    "Stage5DatasetError",
    "TrainingDataset",
    "build_inference_dataset",
    "build_training_dataset",
]
