"""Leakage-safe, fold-calibrated cost-aware Stage 4 routing.

Training rows provide expert quality and historical runtime estimates.  The
training fold is aggregated at global, category, and category+k-shot levels.
The validation fold is used only to select ``lambda`` from a frozen grid; test
rows are never parsed, hashed, or persisted.  Replay loads the resulting final
artifact and treats all runtime values as estimates from immutable Stage 2
runs, never as wall-clock time of the replay itself.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from src.normroute.agent.policy import Policy, TrainRecord
from src.normroute.agent.protocol import AgentTask, CANDIDATE_EXPERTS, RouteDecision


COST_AWARE_POLICY_NAME = "cost_aware"
COST_AWARE_PROTOCOL_VERSION = "stage4.policy.cost_aware.v1"
ESTIMATED_RUNTIME = "estimated_runtime"
DEFAULT_LAMBDA_GRID = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
SUPPORTED_METRICS = ("image_auroc", "image_ap")
RUNTIME_COLUMN_CANDIDATES = (
    "average_runtime_ms",
    "runtime_ms",
    "mean_runtime_ms",
)
FLOAT_TOLERANCE = 1e-12
FRONTIER_COLUMNS = (
    "fold",
    "metric",
    "lambda",
    "max_runtime_ms",
    "mean_validation_quality",
    "mean_estimated_runtime_ms",
    "normalized_quality",
    "normalized_runtime",
    "utility",
    "num_validation_runs",
    "feasible",
    "selected",
    "runtime_source",
)

_MANIFEST_COLUMNS = (
    "fold",
    "split",
    "task_id",
    "sample_id",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
)
_QUALITY_IDENTITY_COLUMNS = (
    "expert",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
)
_FORBIDDEN_KEYS = frozenset(
    {
        "label",
        "labels",
        "mask",
        "mask_path",
        "defect_type",
        "anomaly_type",
        "ground_truth",
        "oracle_best_expert",
        "test_label",
        "test_labels",
        "test_quality",
        "test_runtime",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "protocol_version",
        "policy_name",
        "fold",
        "train_seeds",
        "validation_seeds",
        "metric",
        "lambda_grid",
        "selected_lambda",
        "max_runtime_ms",
        "normalization",
        "selection_rule",
        "validation_selection_rule",
        "runtime_source",
        "statistics",
        "num_training_runs",
        "num_training_rows",
        "num_validation_runs",
        "num_validation_rows",
        "training_rows_sha256",
        "validation_rows_sha256",
        "training_manifest_runs_sha256",
        "validation_manifest_runs_sha256",
        "git_commit",
    }
)
_STATISTIC_FIELDS = frozenset(
    {
        "dataset",
        "category",
        "k_shot",
        "expert",
        "average_quality",
        "average_runtime_ms",
        "num_training_runs",
    }
)

RunKey = tuple[str, str, int, int, str]
ContextKey = tuple[str, str, int]


class CostAwarePolicyError(ValueError):
    """Raised when cost-aware calibration or replay would violate its contract."""


@dataclass(frozen=True)
class QualityRuntimeRow:
    expert: str
    dataset: str
    category: str
    k_shot: int
    seed: int
    support_set_id: str
    quality: float
    runtime_ms: float | None

    @property
    def run_key(self) -> RunKey:
        return (
            self.dataset,
            self.category,
            self.k_shot,
            self.seed,
            self.support_set_id,
        )

    def hash_record(self) -> dict[str, Any]:
        payload = {
            "expert": self.expert,
            "dataset": self.dataset,
            "category": self.category,
            "k_shot": self.k_shot,
            "seed": self.seed,
            "support_set_id": self.support_set_id,
            "quality": self.quality,
        }
        if self.runtime_ms is not None:
            payload["runtime_ms"] = self.runtime_ms
        return payload


@dataclass(frozen=True)
class CostStatistic:
    dataset: str
    category: str
    k_shot: int
    expert: str
    average_quality: float
    average_runtime_ms: float
    num_training_runs: int

    @property
    def context(self) -> ContextKey:
        return self.dataset, self.category, self.k_shot

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "category": self.category,
            "k_shot": self.k_shot,
            "expert": self.expert,
            "average_quality": self.average_quality,
            "average_runtime_ms": self.average_runtime_ms,
            "num_training_runs": self.num_training_runs,
        }


@dataclass(frozen=True)
class CostAwareFold:
    fold: str
    run_splits: dict[RunKey, str]

    @property
    def training_run_keys(self) -> set[RunKey]:
        return {key for key, split in self.run_splits.items() if split == "train"}

    @property
    def validation_run_keys(self) -> set[RunKey]:
        return {key for key, split in self.run_splits.items() if split == "val"}

    @property
    def train_seeds(self) -> tuple[int, ...]:
        return tuple(sorted({key[3] for key in self.training_run_keys}))

    @property
    def validation_seeds(self) -> tuple[int, ...]:
        return tuple(sorted({key[3] for key in self.validation_run_keys}))


def normalized_quality(value: float, values: Iterable[float]) -> float:
    """Min-max normalize quality within one routing context."""

    return _min_max(value, values, constant_value=1.0)


def normalized_runtime(value: float, values: Iterable[float]) -> float:
    """Min-max normalize runtime within one routing context."""

    return _min_max(value, values, constant_value=0.0)


def cost_aware_utility(
    quality: float,
    runtime_ms: float,
    *,
    quality_values: Iterable[float],
    runtime_values: Iterable[float],
    lambda_value: float,
) -> float:
    """Return ``normalized_quality - lambda * normalized_runtime`` exactly."""

    penalty = _finite_nonnegative(lambda_value, "lambda_value")
    return normalized_quality(quality, quality_values) - penalty * normalized_runtime(
        runtime_ms, runtime_values
    )


class CostAwarePolicy(Policy):
    """Route from a final train-calibrated, validation-selected policy artifact."""

    name = COST_AWARE_POLICY_NAME

    def __init__(self, artifact: Mapping[str, Any] | str | Path) -> None:
        payload = load_cost_aware_artifact(artifact)
        self._artifact = payload
        self._global = _index_statistics(payload["statistics"]["global"])
        self._category = _index_statistics(payload["statistics"]["category"])
        self._category_shot = _index_statistics(
            payload["statistics"]["category_shot"]
        )

    def fit(self, train_records: Sequence[TrainRecord]) -> None:
        # Calibration is evaluator-side.  Replay may never refit or alter the
        # final artifact, especially when the requested split is test.
        del train_records

    def select(self, task: AgentTask) -> RouteDecision:
        statistics, level = self._statistics_for(task.policy_features)
        selected, utility, norm_quality, norm_runtime = _select_statistic(
            statistics,
            lambda_value=float(self._artifact["selected_lambda"]),
            max_runtime_ms=self._artifact["max_runtime_ms"],
            candidates=task.candidate_experts,
            context=f"task_id={task.task_id!r}",
        )
        budget = self._artifact["max_runtime_ms"]
        budget_reason = (
            "no runtime ceiling"
            if budget is None
            else f"max_runtime_ms={_format_number(float(budget))}"
        )
        return RouteDecision(
            task_id=task.task_id,
            dataset=task.dataset,
            category=task.category,
            k_shot=task.k_shot,
            seed=task.seed,
            support_set_id=task.support_set_id,
            policy_name=self.name,
            selected_expert=selected.expert,
            decision_reason=(
                f"Cost-aware {level} rule selected {selected.expert}: "
                f"utility={_format_number(utility)} = normalized_quality="
                f"{_format_number(norm_quality)} - lambda="
                f"{_format_number(float(self._artifact['selected_lambda']))} * "
                f"normalized_runtime={_format_number(norm_runtime)}; {budget_reason}; "
                f"runtime_source={ESTIMATED_RUNTIME}."
            ),
            estimated_cost_ms=selected.average_runtime_ms,
            tool_calls=1,
        )

    def configuration(self) -> dict[str, Any]:
        return {
            "fold": self._artifact["fold"],
            "train_seeds": list(self._artifact["train_seeds"]),
            "validation_seeds": list(self._artifact["validation_seeds"]),
            "metric": self._artifact["metric"],
            "lambda_grid": list(self._artifact["lambda_grid"]),
            "selected_lambda": self._artifact["selected_lambda"],
            "max_runtime_ms": self._artifact["max_runtime_ms"],
            "runtime_source": ESTIMATED_RUNTIME,
            "artifact_protocol_version": COST_AWARE_PROTOCOL_VERSION,
        }

    def save(self, path: str | Path) -> Path:
        destination = _artifact_path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(destination, self._artifact)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "CostAwarePolicy":
        return cls(path)

    def _statistics_for(
        self, features: Mapping[str, Any]
    ) -> tuple[dict[str, CostStatistic], str]:
        exact = (
            str(features["dataset"]),
            str(features["category"]),
            int(features["k_shot"]),
        )
        if exact in self._category_shot:
            return self._category_shot[exact], "category+k-shot"
        category = (exact[0], exact[1], 0)
        if category in self._category:
            return self._category[category], "category fallback"
        return self._global[("*", "*", 0)], "global fallback"


def aggregate_training_statistics(
    rows: Sequence[QualityRuntimeRow],
) -> dict[str, list[dict[str, Any]]]:
    """Average train quality/runtime by expert, category, and k-shot.

    Dataset is retained in every key so equal category names from different
    datasets cannot be merged accidentally.  Category and global aggregates
    are included only as deterministic fallbacks for unseen k-shot contexts.
    """

    if not rows:
        raise CostAwarePolicyError("Cannot aggregate an empty training fold")
    for row in rows:
        if row.runtime_ms is None:
            raise CostAwarePolicyError(
                "Every training row requires runtime_ms for cost-aware routing"
            )
    levels: dict[str, dict[ContextKey, list[QualityRuntimeRow]]] = {
        "global": {},
        "category": {},
        "category_shot": {},
    }
    for row in rows:
        levels["global"].setdefault(("*", "*", 0), []).append(row)
        levels["category"].setdefault((row.dataset, row.category, 0), []).append(row)
        levels["category_shot"].setdefault(
            (row.dataset, row.category, row.k_shot), []
        ).append(row)

    result: dict[str, list[dict[str, Any]]] = {}
    for level, groups in levels.items():
        serialized: list[dict[str, Any]] = []
        for context, group_rows in sorted(groups.items()):
            by_expert = _complete_expert_groups(group_rows, context=f"{level}={context!r}")
            for expert in CANDIDATE_EXPERTS:
                expert_rows = by_expert[expert]
                serialized.append(
                    CostStatistic(
                        dataset=context[0],
                        category=context[1],
                        k_shot=context[2],
                        expert=expert,
                        average_quality=_mean(row.quality for row in expert_rows),
                        average_runtime_ms=_mean(
                            float(row.runtime_ms)
                            for row in expert_rows
                            if row.runtime_ms is not None
                        ),
                        num_training_runs=len(expert_rows),
                    ).to_dict()
                )
        result[level] = serialized
    return result


def calibrate_cost_aware_policy(
    *,
    expert_quality_by_run: str | Path,
    fold_manifest: str | Path,
    fold: str,
    output_path: str | Path,
    metric: str = "image_auroc",
    lambda_grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
    max_runtime_ms: float | None = None,
    runtime_column: str | None = None,
) -> tuple[Path, Path]:
    """Calibrate one fold and write its final artifact and cost-quality frontier."""

    outputs = calibrate_cost_aware_artifacts(
        expert_quality_by_run=expert_quality_by_run,
        fold_manifest=fold_manifest,
        output_root=Path(output_path).parent,
        metric=metric,
        lambda_grid=lambda_grid,
        max_runtime_ms=max_runtime_ms,
        runtime_column=runtime_column,
        folds=(fold,),
        explicit_artifact_path=output_path,
    )
    return outputs[0]


def calibrate_cost_aware_artifacts(
    *,
    expert_quality_by_run: str | Path,
    fold_manifest: str | Path,
    output_root: str | Path,
    metric: str = "image_auroc",
    lambda_grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
    max_runtime_ms: float | None = None,
    runtime_column: str | None = None,
    folds: Sequence[str] | None = None,
    explicit_artifact_path: str | Path | None = None,
) -> list[tuple[Path, Path]]:
    """Calibrate requested folds without ever reading test metric/runtime cells."""

    if metric not in SUPPORTED_METRICS:
        raise CostAwarePolicyError(
            f"metric must be one of {list(SUPPORTED_METRICS)!r}"
        )
    grid = _validate_lambda_grid(lambda_grid)
    runtime_limit = (
        None
        if max_runtime_ms is None
        else _finite_nonnegative(max_runtime_ms, "max_runtime_ms")
    )
    manifests = read_cost_aware_folds(fold_manifest, folds=folds)
    rows = read_train_validation_quality(
        expert_quality_by_run,
        manifest_folds=manifests,
        metric=metric,
        runtime_column=runtime_column,
    )
    if explicit_artifact_path is not None and len(manifests) != 1:
        raise CostAwarePolicyError(
            "An explicit artifact path can be used only when calibrating one fold"
        )

    root = Path(output_root)
    outputs: list[tuple[Path, Path]] = []
    for fold_name, manifest in manifests.items():
        train_rows = rows[fold_name]["train"]
        validation_rows = rows[fold_name]["val"]
        statistics = aggregate_training_statistics(train_rows)
        frontier = build_cost_quality_frontier(
            fold=fold_name,
            metric=metric,
            statistics=statistics,
            validation_rows=validation_rows,
            lambda_grid=grid,
            max_runtime_ms=runtime_limit,
        )
        selected_lambda = _selected_frontier_lambda(frontier)
        artifact = build_cost_aware_artifact(
            manifest=manifest,
            metric=metric,
            lambda_grid=grid,
            selected_lambda=selected_lambda,
            max_runtime_ms=runtime_limit,
            statistics=statistics,
            training_rows=train_rows,
            validation_rows=validation_rows,
        )
        if explicit_artifact_path is not None:
            artifact_path = _artifact_path(explicit_artifact_path)
        else:
            artifact_path = (
                root / COST_AWARE_POLICY_NAME / fold_name / "policy_artifact.json"
            )
        _ensure_safe_output(artifact_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(artifact_path, artifact)
        frontier_path = artifact_path.parent / "cost_quality_frontier.csv"
        write_cost_quality_frontier(frontier, frontier_path)
        outputs.append((artifact_path, frontier_path))
    return outputs


def build_cost_quality_frontier(
    *,
    fold: str,
    metric: str,
    statistics: Mapping[str, Sequence[Mapping[str, Any]]],
    validation_rows: Sequence[QualityRuntimeRow],
    lambda_grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
    max_runtime_ms: float | None = None,
) -> list[dict[str, Any]]:
    """Evaluate train-derived routes on validation for every lambda candidate."""

    if not validation_rows:
        raise CostAwarePolicyError(f"Fold {fold!r} has no validation rows")
    indexed = {
        level: _index_statistics(statistics[level])
        for level in ("global", "category", "category_shot")
    }
    by_run: dict[RunKey, dict[str, QualityRuntimeRow]] = {}
    for row in validation_rows:
        experts = by_run.setdefault(row.run_key, {})
        if row.expert in experts:
            raise CostAwarePolicyError(
                f"Duplicate validation row for run={row.run_key!r}, expert={row.expert}"
            )
        experts[row.expert] = row
    for run_key, experts in by_run.items():
        missing = sorted(set(CANDIDATE_EXPERTS) - set(experts))
        if missing:
            raise CostAwarePolicyError(
                f"Validation run {run_key!r} is missing experts: {missing}"
            )

    raw: list[dict[str, Any]] = []
    for lambda_value in _validate_lambda_grid(lambda_grid):
        qualities: list[float] = []
        runtimes: list[float] = []
        feasible = True
        for run_key, experts in sorted(by_run.items()):
            context = (run_key[0], run_key[1], run_key[2])
            context_statistics = indexed["category_shot"].get(context)
            if context_statistics is None:
                context_statistics = indexed["category"].get(
                    (context[0], context[1], 0)
                )
            if context_statistics is None:
                context_statistics = indexed["global"][("*", "*", 0)]
            try:
                selected, _, _, _ = _select_statistic(
                    context_statistics,
                    lambda_value=lambda_value,
                    max_runtime_ms=max_runtime_ms,
                    candidates=CANDIDATE_EXPERTS,
                    context=f"validation run={run_key!r}",
                )
            except CostAwarePolicyError:
                feasible = False
                break
            qualities.append(experts[selected.expert].quality)
            # Runtime is deliberately the train-fold estimate embedded in the
            # policy statistics, not validation/test replay wall-clock time.
            runtimes.append(selected.average_runtime_ms)
        raw.append(
            {
                "fold": fold,
                "metric": metric,
                "lambda": lambda_value,
                "max_runtime_ms": max_runtime_ms,
                "mean_validation_quality": _mean(qualities) if feasible else None,
                "mean_estimated_runtime_ms": _mean(runtimes) if feasible else None,
                "num_validation_runs": len(by_run),
                "feasible": feasible,
                "runtime_source": ESTIMATED_RUNTIME,
            }
        )

    feasible_rows = [row for row in raw if row["feasible"]]
    if not feasible_rows:
        raise CostAwarePolicyError(
            f"No lambda in the validation grid is feasible for max_runtime_ms={max_runtime_ms!r}"
        )
    quality_values = [float(row["mean_validation_quality"]) for row in feasible_rows]
    runtime_values = [float(row["mean_estimated_runtime_ms"]) for row in feasible_rows]
    for row in raw:
        if row["feasible"]:
            row["normalized_quality"] = normalized_quality(
                float(row["mean_validation_quality"]), quality_values
            )
            row["normalized_runtime"] = normalized_runtime(
                float(row["mean_estimated_runtime_ms"]), runtime_values
            )
            row["utility"] = row["normalized_quality"] - float(
                row["lambda"]
            ) * row["normalized_runtime"]
        else:
            row["normalized_quality"] = None
            row["normalized_runtime"] = None
            row["utility"] = None
        row["selected"] = False
    selected_lambda = _selected_frontier_lambda(raw)
    for row in raw:
        row["selected"] = bool(
            row["feasible"]
            and math.isclose(
                float(row["lambda"]),
                selected_lambda,
                rel_tol=FLOAT_TOLERANCE,
                abs_tol=FLOAT_TOLERANCE,
            )
        )
    return raw


def build_cost_aware_artifact(
    *,
    manifest: CostAwareFold,
    metric: str,
    lambda_grid: Sequence[float],
    selected_lambda: float,
    max_runtime_ms: float | None,
    statistics: Mapping[str, Sequence[Mapping[str, Any]]],
    training_rows: Sequence[QualityRuntimeRow],
    validation_rows: Sequence[QualityRuntimeRow],
) -> dict[str, Any]:
    """Build the final artifact; validation contributes only lambda and hashes."""

    canonical_train = _sorted_hash_records(training_rows)
    canonical_val = _sorted_hash_records(validation_rows)
    artifact = {
        "protocol_version": COST_AWARE_PROTOCOL_VERSION,
        "policy_name": COST_AWARE_POLICY_NAME,
        "fold": manifest.fold,
        "train_seeds": list(manifest.train_seeds),
        "validation_seeds": list(manifest.validation_seeds),
        "metric": metric,
        "lambda_grid": list(_validate_lambda_grid(lambda_grid)),
        "selected_lambda": selected_lambda,
        "max_runtime_ms": max_runtime_ms,
        "normalization": {
            "method": "min_max_within_routing_context",
            "quality_constant_value": 1.0,
            "runtime_constant_value": 0.0,
        },
        "selection_rule": (
            "maximize normalized_quality - lambda * normalized_runtime; "
            "then quality, runtime, candidate order"
        ),
        "validation_selection_rule": (
            "maximize validation frontier utility; then validation quality, "
            "estimated runtime, lambda"
        ),
        "runtime_source": ESTIMATED_RUNTIME,
        "statistics": {
            level: [dict(row) for row in statistics[level]]
            for level in ("global", "category", "category_shot")
        },
        "num_training_runs": len(manifest.training_run_keys),
        "num_training_rows": len(training_rows),
        "num_validation_runs": len(manifest.validation_run_keys),
        "num_validation_rows": len(validation_rows),
        "training_rows_sha256": _canonical_sha256(canonical_train),
        "validation_rows_sha256": _canonical_sha256(canonical_val),
        "training_manifest_runs_sha256": _canonical_sha256(
            [_run_key_record(key) for key in sorted(manifest.training_run_keys)]
        ),
        "validation_manifest_runs_sha256": _canonical_sha256(
            [_run_key_record(key) for key in sorted(manifest.validation_run_keys)]
        ),
        "git_commit": _git_commit(),
    }
    return load_cost_aware_artifact(artifact)


def load_cost_aware_artifact(
    source: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Strictly validate a final cost-aware artifact before test replay."""

    if isinstance(source, Mapping):
        payload: Any = dict(source)
        context = "cost-aware policy artifact"
    else:
        path = _artifact_path(source)
        context = str(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CostAwarePolicyError(f"Could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CostAwarePolicyError(f"{context} must be a JSON object")
    _reject_forbidden_keys(payload, context=context)
    missing = sorted(_ARTIFACT_FIELDS - set(payload))
    extra = sorted(set(payload) - _ARTIFACT_FIELDS)
    if missing or extra:
        raise CostAwarePolicyError(
            f"{context} has invalid fields; missing={missing}, extra={extra}"
        )
    if payload["protocol_version"] != COST_AWARE_PROTOCOL_VERSION:
        raise CostAwarePolicyError(
            f"{context} protocol_version must be {COST_AWARE_PROTOCOL_VERSION!r}"
        )
    if payload["policy_name"] != COST_AWARE_POLICY_NAME:
        raise CostAwarePolicyError(
            f"{context} policy_name must be {COST_AWARE_POLICY_NAME!r}"
        )
    _nonempty_string(payload["fold"], "fold", context)
    _nonempty_string(payload["git_commit"], "git_commit", context)
    if payload["metric"] not in SUPPORTED_METRICS:
        raise CostAwarePolicyError(f"{context} has invalid metric")
    grid = _validate_lambda_grid(payload["lambda_grid"])
    selected_lambda = _finite_nonnegative(payload["selected_lambda"], "selected_lambda")
    if not any(
        math.isclose(
            selected_lambda,
            candidate,
            rel_tol=FLOAT_TOLERANCE,
            abs_tol=FLOAT_TOLERANCE,
        )
        for candidate in grid
    ):
        raise CostAwarePolicyError(f"{context} selected_lambda is absent from lambda_grid")
    if payload["max_runtime_ms"] is not None:
        _finite_nonnegative(payload["max_runtime_ms"], "max_runtime_ms")
    if payload["runtime_source"] != ESTIMATED_RUNTIME:
        raise CostAwarePolicyError(
            f"{context} runtime_source must be {ESTIMATED_RUNTIME!r}"
        )
    expected_normalization = {
        "method": "min_max_within_routing_context",
        "quality_constant_value": 1.0,
        "runtime_constant_value": 0.0,
    }
    if payload["normalization"] != expected_normalization:
        raise CostAwarePolicyError(f"{context} has invalid normalization contract")
    for field in ("selection_rule", "validation_selection_rule"):
        _nonempty_string(payload[field], field, context)
    _validate_seed_list(payload["train_seeds"], "train_seeds", context)
    _validate_seed_list(payload["validation_seeds"], "validation_seeds", context)
    if set(payload["train_seeds"]).intersection(payload["validation_seeds"]):
        raise CostAwarePolicyError(f"{context} train/validation seeds overlap")
    for field in (
        "num_training_runs",
        "num_training_rows",
        "num_validation_runs",
        "num_validation_rows",
    ):
        _positive_int(payload[field], field, context)
    for field in (
        "training_rows_sha256",
        "validation_rows_sha256",
        "training_manifest_runs_sha256",
        "validation_manifest_runs_sha256",
    ):
        value = _nonempty_string(payload[field], field, context)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise CostAwarePolicyError(f"{context} field {field!r} must be SHA-256 hex")

    statistics = payload["statistics"]
    expected_levels = {"global", "category", "category_shot"}
    if not isinstance(statistics, dict) or set(statistics) != expected_levels:
        raise CostAwarePolicyError(
            f"{context} statistics must contain exactly {sorted(expected_levels)!r}"
        )
    indexed: dict[str, dict[ContextKey, dict[str, CostStatistic]]] = {}
    for level in sorted(expected_levels):
        indexed[level] = _index_statistics(statistics[level], context=f"{context}.{level}")
        for key, expert_stats in indexed[level].items():
            if set(expert_stats) != set(CANDIDATE_EXPERTS):
                raise CostAwarePolicyError(
                    f"{context}.{level} context {key!r} must contain every expert"
                )
    if set(indexed["global"]) != {("*", "*", 0)}:
        raise CostAwarePolicyError(f"{context} global statistics key is invalid")
    if not indexed["category"] or not indexed["category_shot"]:
        raise CostAwarePolicyError(f"{context} requires category and category+k-shot statistics")
    return payload


def read_cost_aware_folds(
    path: str | Path, *, folds: Sequence[str] | None = None
) -> dict[str, CostAwareFold]:
    """Read run-level train/validation/test assignments from the frozen manifest."""

    manifest_path = Path(path)
    requested = _normalize_folds(folds)
    requested_set = set(requested or ())
    run_splits: dict[str, dict[RunKey, str]] = {}
    task_ids: dict[str, set[str]] = {}
    try:
        handle = manifest_path.open("r", newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise CostAwarePolicyError(f"Could not read fold manifest {manifest_path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in _MANIFEST_COLUMNS if column not in fieldnames]
        if missing:
            raise CostAwarePolicyError(
                f"{manifest_path} is missing manifest columns: {missing}"
            )
        for line_number, raw in enumerate(reader, start=2):
            fold = str(raw.get("fold") or "").strip()
            if requested_set and fold not in requested_set:
                continue
            clean = {key: str(value or "").strip() for key, value in raw.items()}
            empty = [column for column in _MANIFEST_COLUMNS if not clean.get(column)]
            if empty:
                raise CostAwarePolicyError(
                    f"{manifest_path}:{line_number} has empty values: {empty}"
                )
            split = clean["split"]
            if split not in {"train", "val", "test"}:
                raise CostAwarePolicyError(
                    f"{manifest_path}:{line_number} has invalid split={split!r}"
                )
            key = _run_key(clean, manifest_path, line_number)
            previous = run_splits.setdefault(fold, {}).get(key)
            if previous is not None and previous != split:
                raise CostAwarePolicyError(
                    f"{manifest_path} assigns run {key!r} to {previous!r} and {split!r}"
                )
            run_splits[fold][key] = split
            if clean["task_id"] in task_ids.setdefault(fold, set()):
                raise CostAwarePolicyError(
                    f"{manifest_path} duplicates task_id={clean['task_id']!r} in {fold}"
                )
            task_ids[fold].add(clean["task_id"])
    if requested_set - set(run_splits):
        raise CostAwarePolicyError(
            f"Manifest has no rows for folds: {sorted(requested_set - set(run_splits))}"
        )
    if not run_splits:
        raise CostAwarePolicyError(f"{manifest_path} has no usable folds")
    result: dict[str, CostAwareFold] = {}
    for fold in sorted(run_splits, key=_natural_fold_key):
        manifest = CostAwareFold(fold=fold, run_splits=run_splits[fold])
        if not manifest.training_run_keys or not manifest.validation_run_keys:
            raise CostAwarePolicyError(f"{fold} requires non-empty train and validation runs")
        result[fold] = manifest
    return result


def read_train_validation_quality(
    path: str | Path,
    *,
    manifest_folds: Mapping[str, CostAwareFold],
    metric: str = "image_auroc",
    runtime_column: str | None = None,
) -> dict[str, dict[str, list[QualityRuntimeRow]]]:
    """Read train and validation cells, deliberately never parsing test outcomes."""

    quality_path = Path(path)
    result = {
        fold: {"train": [], "val": []} for fold in manifest_folds
    }
    seen = {
        fold: {"train": set(), "val": set()} for fold in manifest_folds
    }
    try:
        handle = quality_path.open("r", newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise CostAwarePolicyError(f"Could not read quality file {quality_path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = (*_QUALITY_IDENTITY_COLUMNS, metric)
        missing = [column for column in required if column not in fieldnames]
        if missing:
            raise CostAwarePolicyError(f"{quality_path} is missing columns: {missing}")
        runtime_name = _resolve_runtime_column(
            fieldnames, requested=runtime_column, path=quality_path
        )
        if runtime_name is None:
            raise CostAwarePolicyError(
                f"{quality_path} requires a runtime column for cost-aware routing"
            )
        for line_number, raw in enumerate(reader, start=2):
            identity = {
                column: str(raw.get(column) or "").strip()
                for column in _QUALITY_IDENTITY_COLUMNS
            }
            empty = [column for column, value in identity.items() if not value]
            if empty:
                raise CostAwarePolicyError(
                    f"{quality_path}:{line_number} has empty identity values: {empty}"
                )
            run_key = _run_key(identity, quality_path, line_number)
            expert = _canonical_expert(identity["expert"], quality_path, line_number)
            matched = False
            for fold, manifest in manifest_folds.items():
                split = manifest.run_splits.get(run_key)
                if split is None:
                    continue
                matched = True
                if split == "test":
                    # Critical isolation boundary.  Do not parse, validate, or
                    # hash test quality/runtime cells under any circumstances.
                    continue
                row_identity = (run_key, expert)
                if row_identity in seen[fold][split]:
                    raise CostAwarePolicyError(
                        f"Duplicate {split} row for fold={fold}, run={run_key!r}, expert={expert}"
                    )
                quality = _finite_number(
                    raw.get(metric), metric, quality_path, line_number
                )
                if not 0.0 <= quality <= 1.0:
                    raise CostAwarePolicyError(
                        f"{quality_path}:{line_number} {split} {metric} must be in [0, 1]"
                    )
                runtime: float | None = None
                if split == "train":
                    runtime = _finite_number(
                        raw.get(runtime_name), runtime_name, quality_path, line_number,
                        nonnegative=True,
                    )
                result[fold][split].append(
                    QualityRuntimeRow(
                        expert=expert,
                        dataset=run_key[0],
                        category=run_key[1],
                        k_shot=run_key[2],
                        seed=run_key[3],
                        support_set_id=run_key[4],
                        quality=quality,
                        runtime_ms=runtime,
                    )
                )
                seen[fold][split].add(row_identity)
            if not matched:
                raise CostAwarePolicyError(
                    f"{quality_path}:{line_number} run {run_key!r} is absent from requested folds"
                )
    for fold, manifest in manifest_folds.items():
        for split, run_keys in (
            ("train", manifest.training_run_keys),
            ("val", manifest.validation_run_keys),
        ):
            expected = {
                (run_key, expert)
                for run_key in run_keys
                for expert in CANDIDATE_EXPERTS
            }
            actual = seen[fold][split]
            if expected != actual:
                raise CostAwarePolicyError(
                    f"{split} quality coverage mismatch for {fold}; "
                    f"missing={sorted(expected - actual)[:10]}, extra={sorted(actual - expected)[:10]}"
                )
    return result


def write_cost_quality_frontier(
    rows: Sequence[Mapping[str, Any]], path: str | Path
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(FRONTIER_COLUMNS), extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        column: _csv_value(row.get(column))
                        for column in FRONTIER_COLUMNS
                    }
                )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _select_statistic(
    statistics: Mapping[str, CostStatistic],
    *,
    lambda_value: float,
    max_runtime_ms: float | None,
    candidates: Sequence[str],
    context: str,
) -> tuple[CostStatistic, float, float, float]:
    ordered = [expert for expert in CANDIDATE_EXPERTS if expert in candidates]
    missing = [expert for expert in ordered if expert not in statistics]
    if missing:
        raise CostAwarePolicyError(f"{context} is missing expert statistics: {missing}")
    all_stats = [statistics[expert] for expert in ordered]
    quality_values = [stat.average_quality for stat in all_stats]
    runtime_values = [stat.average_runtime_ms for stat in all_stats]
    feasible = [
        stat
        for stat in all_stats
        if max_runtime_ms is None
        or stat.average_runtime_ms <= float(max_runtime_ms) + FLOAT_TOLERANCE
    ]
    if not feasible:
        raise CostAwarePolicyError(
            f"{context} has no expert satisfying max_runtime_ms={max_runtime_ms!r}"
        )
    scored: list[tuple[CostStatistic, float, float, float]] = []
    for stat in feasible:
        nq = normalized_quality(stat.average_quality, quality_values)
        nr = normalized_runtime(stat.average_runtime_ms, runtime_values)
        scored.append((stat, nq - float(lambda_value) * nr, nq, nr))
    scored.sort(
        key=lambda item: (
            -item[1],
            -item[2],
            item[0].average_runtime_ms,
            CANDIDATE_EXPERTS.index(item[0].expert),
        )
    )
    return scored[0]


def _selected_frontier_lambda(rows: Sequence[Mapping[str, Any]]) -> float:
    feasible = [row for row in rows if row.get("feasible")]
    if not feasible:
        raise CostAwarePolicyError("Cost-quality frontier has no feasible lambda")
    selected = min(
        feasible,
        key=lambda row: (
            -float(row["utility"]),
            -float(row["mean_validation_quality"]),
            float(row["mean_estimated_runtime_ms"]),
            float(row["lambda"]),
        ),
    )
    return float(selected["lambda"])


def _index_statistics(
    rows: Any, *, context: str = "statistics"
) -> dict[ContextKey, dict[str, CostStatistic]]:
    if not isinstance(rows, list):
        raise CostAwarePolicyError(f"{context} must be a list")
    result: dict[ContextKey, dict[str, CostStatistic]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != _STATISTIC_FIELDS:
            raise CostAwarePolicyError(
                f"{context}[{index}] must contain exactly {sorted(_STATISTIC_FIELDS)!r}"
            )
        dataset = _nonempty_string(raw["dataset"], "dataset", context)
        category = _nonempty_string(raw["category"], "category", context)
        k_shot = _nonnegative_int(raw["k_shot"], "k_shot", context)
        expert = _canonical_expert_value(raw["expert"], context)
        if raw["expert"] != expert:
            raise CostAwarePolicyError(f"{context} must use canonical expert names")
        stat = CostStatistic(
            dataset=dataset,
            category=category,
            k_shot=k_shot,
            expert=expert,
            average_quality=_finite_unit(raw["average_quality"], "average_quality"),
            average_runtime_ms=_finite_nonnegative(
                raw["average_runtime_ms"], "average_runtime_ms"
            ),
            num_training_runs=_positive_int(
                raw["num_training_runs"], "num_training_runs", context
            ),
        )
        experts = result.setdefault(stat.context, {})
        if expert in experts:
            raise CostAwarePolicyError(
                f"{context} duplicates expert={expert!r} for {stat.context!r}"
            )
        experts[expert] = stat
    return result


def _complete_expert_groups(
    rows: Sequence[QualityRuntimeRow], *, context: str
) -> dict[str, list[QualityRuntimeRow]]:
    result = {expert: [] for expert in CANDIDATE_EXPERTS}
    for row in rows:
        result[row.expert].append(row)
    missing = [expert for expert, expert_rows in result.items() if not expert_rows]
    if missing:
        raise CostAwarePolicyError(f"{context} is missing experts: {missing}")
    return result


def _min_max(value: float, values: Iterable[float], *, constant_value: float) -> float:
    parsed = [_finite_number_value(item, "normalization value") for item in values]
    target = _finite_number_value(value, "normalization target")
    if not parsed:
        raise CostAwarePolicyError("Cannot normalize against an empty sequence")
    lower, upper = min(parsed), max(parsed)
    if math.isclose(lower, upper, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE):
        return constant_value
    return (target - lower) / (upper - lower)


def _validate_lambda_grid(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise CostAwarePolicyError("lambda_grid must be a non-empty sequence")
    parsed = tuple(_finite_nonnegative(value, "lambda_grid value") for value in values)
    if len(parsed) != len(set(parsed)) or tuple(sorted(parsed)) != parsed:
        raise CostAwarePolicyError("lambda_grid must be sorted with unique values")
    return parsed


def _resolve_runtime_column(
    fieldnames: Sequence[str], *, requested: str | None, path: Path
) -> str | None:
    if requested is not None:
        name = requested.strip()
        if not name or name not in fieldnames:
            raise CostAwarePolicyError(
                f"{path} does not contain requested runtime column {requested!r}"
            )
        return name
    return next((name for name in RUNTIME_COLUMN_CANDIDATES if name in fieldnames), None)


def _run_key(row: Mapping[str, Any], path: Path, line_number: int) -> RunKey:
    dataset = str(row.get("dataset") or "").strip()
    category = str(row.get("category") or "").strip()
    support = str(row.get("support_set_id") or "").strip()
    if not dataset or not category or not support:
        raise CostAwarePolicyError(f"{path}:{line_number} has empty run identity")
    return (
        dataset,
        category,
        _positive_csv_int(row.get("k_shot"), "k_shot", path, line_number),
        _nonnegative_csv_int(row.get("seed"), "seed", path, line_number),
        support,
    )


def _run_key_record(key: RunKey) -> dict[str, Any]:
    return {
        "dataset": key[0],
        "category": key[1],
        "k_shot": key[2],
        "seed": key[3],
        "support_set_id": key[4],
    }


def _sorted_hash_records(rows: Sequence[QualityRuntimeRow]) -> list[dict[str, Any]]:
    return [
        row.hash_record()
        for row in sorted(
            rows,
            key=lambda item: (*item.run_key, CANDIDATE_EXPERTS.index(item.expert)),
        )
    ]


def _canonical_expert(value: str, path: Path, line_number: int) -> str:
    try:
        return _canonical_expert_value(value, str(path))
    except CostAwarePolicyError as exc:
        raise CostAwarePolicyError(f"{path}:{line_number} has unknown expert={value!r}") from exc


def _canonical_expert_value(value: Any, context: str) -> str:
    normalized = "".join(character for character in str(value).lower() if character.isalnum())
    mapping = {
        "".join(character for character in expert.lower() if character.isalnum()): expert
        for expert in CANDIDATE_EXPERTS
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise CostAwarePolicyError(f"{context} has unknown expert={value!r}") from exc


def _finite_number(
    value: Any,
    field: str,
    path: Path,
    line_number: int,
    nonnegative: bool = False,
) -> float:
    try:
        parsed = float(str(value or "").strip())
    except ValueError as exc:
        raise CostAwarePolicyError(
            f"{path}:{line_number} has invalid {field}={value!r}"
        ) from exc
    if not math.isfinite(parsed) or (nonnegative and parsed < 0):
        raise CostAwarePolicyError(f"{path}:{line_number} has invalid {field}={value!r}")
    return parsed


def _finite_number_value(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise CostAwarePolicyError(f"{field} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CostAwarePolicyError(f"{field} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise CostAwarePolicyError(f"{field} must be a finite number")
    return parsed


def _finite_nonnegative(value: Any, field: str) -> float:
    parsed = _finite_number_value(value, field)
    if parsed < 0:
        raise CostAwarePolicyError(f"{field} must be >= 0")
    return parsed


def _finite_unit(value: Any, field: str) -> float:
    parsed = _finite_number_value(value, field)
    if not 0.0 <= parsed <= 1.0:
        raise CostAwarePolicyError(f"{field} must be in [0, 1]")
    return parsed


def _nonnegative_int(value: Any, field: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CostAwarePolicyError(f"{context} field {field!r} must be an integer >= 0")
    return value


def _positive_int(value: Any, field: str, context: str) -> int:
    parsed = _nonnegative_int(value, field, context)
    if parsed < 1:
        raise CostAwarePolicyError(f"{context} field {field!r} must be an integer >= 1")
    return parsed


def _nonnegative_csv_int(value: Any, field: str, path: Path, line_number: int) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise CostAwarePolicyError(
            f"{path}:{line_number} {field} must be an integer"
        ) from exc
    if parsed < 0:
        raise CostAwarePolicyError(f"{path}:{line_number} {field} must be >= 0")
    return parsed


def _positive_csv_int(value: Any, field: str, path: Path, line_number: int) -> int:
    parsed = _nonnegative_csv_int(value, field, path, line_number)
    if parsed < 1:
        raise CostAwarePolicyError(f"{path}:{line_number} {field} must be >= 1")
    return parsed


def _nonempty_string(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CostAwarePolicyError(f"{context} field {field!r} must be a non-empty string")
    return value.strip()


def _validate_seed_list(value: Any, field: str, context: str) -> None:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in value)
        or value != sorted(set(value))
    ):
        raise CostAwarePolicyError(
            f"{context} {field} must be a sorted unique non-empty integer list"
        )


def _normalize_folds(folds: Sequence[str] | None) -> tuple[str, ...] | None:
    if folds is None:
        return None
    cleaned = tuple(str(fold).strip() for fold in folds)
    if not cleaned or any(not fold for fold in cleaned) or len(cleaned) != len(set(cleaned)):
        raise CostAwarePolicyError("folds must contain unique non-empty names")
    return cleaned


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise CostAwarePolicyError("Cannot average an empty sequence")
    return math.fsum(materialized) / len(materialized)


def _canonical_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(records), sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reject_forbidden_keys(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(
            key for key in value if str(key).strip().lower() in _FORBIDDEN_KEYS
        )
        if forbidden:
            raise CostAwarePolicyError(
                f"{context} contains forbidden evaluator/test fields: {forbidden}"
            )
        for child in value.values():
            _reject_forbidden_keys(child, context=context)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child, context=context)


def _artifact_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        preferred = candidate / "policy_artifact.json"
        fallback = candidate / "policy.json"
        return fallback if fallback.is_file() and not preferred.is_file() else preferred
    return candidate


def _ensure_safe_output(path: Path) -> None:
    lowered = {part.lower() for part in path.parts}
    if "evaluator_only" in lowered or "oracle" in lowered:
        raise CostAwarePolicyError(
            "Cost-aware policy artifacts must not be written under evaluator_only/oracle"
        )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return _format_number(value)
    return value


def _format_number(value: float) -> str:
    return f"{value:.12g}"


def _natural_fold_key(value: str) -> tuple[str, int, str]:
    prefix = value.rstrip("0123456789")
    suffix = value[len(prefix) :]
    return prefix, int(suffix) if suffix else -1, value


def _git_commit() -> str:
    project_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


# Compatibility aliases for callers that use a shorter mathematical name.
utility = cost_aware_utility
normalize_quality = normalized_quality
normalize_runtime = normalized_runtime


__all__ = [
    "COST_AWARE_POLICY_NAME",
    "COST_AWARE_PROTOCOL_VERSION",
    "DEFAULT_LAMBDA_GRID",
    "ESTIMATED_RUNTIME",
    "FRONTIER_COLUMNS",
    "CostAwareFold",
    "CostAwarePolicy",
    "CostAwarePolicyError",
    "CostStatistic",
    "QualityRuntimeRow",
    "aggregate_training_statistics",
    "build_cost_aware_artifact",
    "build_cost_quality_frontier",
    "calibrate_cost_aware_artifacts",
    "calibrate_cost_aware_policy",
    "cost_aware_utility",
    "load_cost_aware_artifact",
    "normalize_quality",
    "normalize_runtime",
    "normalized_quality",
    "normalized_runtime",
    "read_cost_aware_folds",
    "read_train_validation_quality",
    "utility",
    "write_cost_quality_frontier",
]
