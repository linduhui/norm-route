"""Lightweight learned Stage 4 metadata-routing diagnostic baselines.

Two deliberately small classifiers are provided: a decision tree and a
multinomial logistic model.  They are evaluator-side trained from run-level
best-expert labels, use validation only for hyperparameter selection, and are
loaded as frozen JSON artifacts for replay.  They are diagnostic baselines,
not the final NORM-Route method.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

from src.normroute.agent.policy import Policy, TrainRecord
from src.normroute.agent.protocol import AgentTask, CANDIDATE_EXPERTS, RouteDecision
from src.normroute.features import (
    MetadataFeatureError,
    build_metadata_feature_record,
    canonical_static_costs,
    encode_metadata_features,
    feature_manifest_sha256,
    fit_feature_manifest,
    validate_feature_manifest,
    write_feature_manifest,
)


DECISION_TREE_METADATA = "decision_tree_metadata"
MULTINOMIAL_LOGISTIC_METADATA = "multinomial_logistic_metadata"
LEARNED_METADATA_POLICY_NAMES = (
    DECISION_TREE_METADATA,
    MULTINOMIAL_LOGISTIC_METADATA,
)
LEARNED_METADATA_PROTOCOL_VERSION = "stage4.policy.learned_metadata.v1"
LEARNED_METADATA_STATE_VERSION = "stage4.policy.learned_metadata_state.v1"
MODEL_ARTIFACT_FILENAME = "model_artifact.json"
FEATURE_MANIFEST_FILENAME = "feature_manifest.json"
TRAIN_METADATA_FILENAME = "train_metadata.json"
VALIDATION_PREDICTIONS_FILENAME = "validation_predictions.csv"
FAILURES_FILENAME = "failures.json"
DIAGNOSTIC_BASELINE_ROLE = "diagnostic_baseline_not_final_norm_route"

DEFAULT_TREE_MAX_DEPTH_GRID: tuple[int | None, ...] = (1, 2, 3, 4, 6, None)
DEFAULT_LOGISTIC_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_STATIC_COSTS = {expert: 1.0 for expert in CANDIDATE_EXPERTS}
DEFAULT_STATIC_COST_UNIT = "abstract_unit"
SUPPORTED_METRICS = ("image_auroc", "image_ap")
MANIFEST_COLUMNS = (
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
QUALITY_IDENTITY_COLUMNS = (
    "expert",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
)
VALIDATION_PREDICTION_COLUMNS = (
    "fold",
    "split",
    "run_identity_sha256",
    "category",
    "k_shot",
    "best_expert_label",
    "predicted_expert",
    "selected_probability",
    "margin",
    "correct",
)
_MODEL_ARTIFACT_FIELDS = frozenset(
    {
        "protocol_version",
        "policy_name",
        "baseline_role",
        "diagnostic_baseline",
        "frozen",
        "fold",
        "train_seeds",
        "validation_seeds",
        "metric",
        "label_source",
        "selected_hyperparameter",
        "candidate_experts",
        "budget",
        "static_cost",
        "static_cost_unit",
        "feature_manifest_sha256",
        "train_metadata_sha256",
        "model",
    }
)

RunKey = tuple[str, str, int, int, str]


class LearnedMetadataPolicyError(ValueError):
    """Raised when safe training, serialization, or frozen replay fails."""


@dataclass(frozen=True)
class LearnedFold:
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


@dataclass(frozen=True)
class ExpertOutcome:
    expert: str
    run_key: RunKey
    metric_value: float


@dataclass(frozen=True)
class LabeledRun:
    run_key: RunKey
    category: str
    k_shot: int
    best_expert: str

    def feature_record(
        self, *, budget: int, static_cost: Mapping[str, float]
    ) -> dict[str, Any]:
        return build_metadata_feature_record(
            category=self.category,
            k_shot=self.k_shot,
            budget=budget,
            static_cost=static_cost,
        )


class _FrozenLearnedMetadataPolicy(Policy):
    """Shared inference-only implementation for learned metadata policies."""

    expected_model_kind: str

    def __init__(self, *, artifact: str | Path) -> None:
        model_artifact, feature_manifest, train_metadata = _load_frozen_bundle(artifact)
        if model_artifact["policy_name"] != self.name:
            raise LearnedMetadataPolicyError(
                f"artifact policy_name={model_artifact['policy_name']!r} does not match {self.name!r}"
            )
        if model_artifact["model"].get("kind") != self.expected_model_kind:
            raise LearnedMetadataPolicyError(
                f"artifact model kind does not match {self.expected_model_kind!r}"
            )
        self._artifact = model_artifact
        self._feature_manifest = feature_manifest
        self._train_metadata = train_metadata
        self._frozen = True

    def fit(self, train_records: Sequence[TrainRecord]) -> None:
        # Replay always calls Policy.fit.  A learned test policy must remain the
        # byte-equivalent frozen model selected before test, so no record is read.
        del train_records
        if not self._frozen:
            raise LearnedMetadataPolicyError("learned metadata model is not frozen")

    def select(self, task: AgentTask) -> RouteDecision:
        if not self._frozen:
            raise LearnedMetadataPolicyError("test routing requires a frozen model")
        record = build_metadata_feature_record(
            category=task.category,
            k_shot=task.k_shot,
            budget=self._artifact["budget"],
            static_cost=self._artifact["static_cost"],
        )
        vector = encode_metadata_features(record, self._feature_manifest)
        probabilities = _predict_probabilities(self._artifact["model"], vector)
        selected_index = max(
            range(len(CANDIDATE_EXPERTS)), key=lambda index: (probabilities[index], -index)
        )
        selected = CANDIDATE_EXPERTS[selected_index]
        ordered = sorted(probabilities, reverse=True)
        selected_probability = probabilities[selected_index]
        margin = ordered[0] - ordered[1]
        static_cost = float(self._artifact["static_cost"][selected])
        unit = self._artifact["static_cost_unit"]
        estimated_cost_ms = static_cost if unit == "ms" else 0.0
        hyperparameter_name, hyperparameter_value = next(
            iter(self._artifact["selected_hyperparameter"].items())
        )
        reason = (
            f"Diagnostic baseline ({self.name}) loaded a frozen {self.expected_model_kind} "
            f"from fold {self._artifact['fold']}; {hyperparameter_name}="
            f"{hyperparameter_value!r}; selected_probability={selected_probability:.6f}; "
            f"margin={margin:.6f}; features=category,k_shot,budget,static_cost only; "
            f"selected_static_cost={static_cost:g} {unit}. This is not final NORM-Route."
        )
        return RouteDecision(
            task_id=task.task_id,
            dataset=task.dataset,
            category=task.category,
            k_shot=task.k_shot,
            seed=task.seed,
            support_set_id=task.support_set_id,
            policy_name=self.name,
            selected_expert=selected,
            decision_reason=reason,
            estimated_cost_ms=estimated_cost_ms,
            tool_calls=1,
            selected_probability=selected_probability,
            margin=margin,
        )

    def configuration(self) -> dict[str, Any]:
        return {
            "fold": self._artifact["fold"],
            "train_seeds": list(self._artifact["train_seeds"]),
            "validation_seeds": list(self._artifact["validation_seeds"]),
            "budget": self._artifact["budget"],
            "static_cost_unit": self._artifact["static_cost_unit"],
            "diagnostic_baseline": True,
            "baseline_role": DIAGNOSTIC_BASELINE_ROLE,
            "frozen_model": True,
            "feature_manifest_sha256": self._artifact["feature_manifest_sha256"],
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.is_dir():
            destination = destination / "policy.json"
        state = {
            "state_version": LEARNED_METADATA_STATE_VERSION,
            "policy_name": self.name,
            "model_artifact": self._artifact,
            "feature_manifest": self._feature_manifest,
            "train_metadata": self._train_metadata,
        }
        _write_json_atomic(destination, state)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "_FrozenLearnedMetadataPolicy":
        return cls(artifact=path)


class DecisionTreeMetadataPolicy(_FrozenLearnedMetadataPolicy):
    """Frozen decision-tree metadata diagnostic baseline."""

    name = DECISION_TREE_METADATA
    expected_model_kind = "decision_tree"


class MultinomialLogisticMetadataPolicy(_FrozenLearnedMetadataPolicy):
    """Frozen multinomial-logistic metadata diagnostic baseline."""

    name = MULTINOMIAL_LOGISTIC_METADATA
    expected_model_kind = "multinomial_logistic"


def calibrate_learned_metadata_policy(
    *,
    expert_quality_by_run: str | Path,
    fold_manifest: str | Path,
    fold: str,
    output_path: str | Path,
    policy_name: str,
    metric: str = "image_auroc",
    budget: int = 1,
    static_costs: Mapping[str, Any] | str | Path | None = None,
    static_cost_unit: str = DEFAULT_STATIC_COST_UNIT,
    tree_max_depth_grid: Sequence[int | None] = DEFAULT_TREE_MAX_DEPTH_GRID,
    logistic_c_grid: Sequence[float] = DEFAULT_LOGISTIC_C_GRID,
    training_seed: int = 0,
) -> Path:
    """Train one fold, select its hyperparameter on val, and freeze JSON artifacts."""

    if policy_name not in LEARNED_METADATA_POLICY_NAMES:
        raise LearnedMetadataPolicyError(
            f"policy_name must be one of {list(LEARNED_METADATA_POLICY_NAMES)!r}"
        )
    if metric not in SUPPORTED_METRICS:
        raise LearnedMetadataPolicyError(
            f"metric must be one of {list(SUPPORTED_METRICS)!r}"
        )
    parsed_budget = _nonnegative_int(budget, "budget")
    if parsed_budget < 1:
        raise LearnedMetadataPolicyError("budget must be >= 1 for one-expert routing")
    parsed_training_seed = _nonnegative_int(training_seed, "training_seed")
    unit = _nonempty_string(static_cost_unit, "static_cost_unit")
    costs, cost_source = _resolve_static_costs(static_costs)
    folds = read_learned_folds(fold_manifest, folds=(fold,))
    outcomes = read_train_validation_outcomes(
        expert_quality_by_run,
        manifest_folds=folds,
        metric=metric,
    )[fold]
    train_runs = _build_run_labels(outcomes["train"], split="train", fold=fold)
    validation_runs = _build_run_labels(outcomes["val"], split="val", fold=fold)
    feature_records = [
        run.feature_record(budget=parsed_budget, static_cost=costs) for run in train_runs
    ]
    feature_manifest = fit_feature_manifest(feature_records)
    train_x = [encode_metadata_features(record, feature_manifest) for record in feature_records]
    train_y = [CANDIDATE_EXPERTS.index(run.best_expert) for run in train_runs]
    validation_records = [
        run.feature_record(budget=parsed_budget, static_cost=costs)
        for run in validation_runs
    ]
    validation_x = [
        encode_metadata_features(record, feature_manifest) for record in validation_records
    ]
    validation_y = [CANDIDATE_EXPERTS.index(run.best_expert) for run in validation_runs]

    if policy_name == DECISION_TREE_METADATA:
        depth_grid = _validate_depth_grid(tree_max_depth_grid)
        selected_value, model, frontier = _select_tree_depth(
            train_x, train_y, validation_x, validation_y, depth_grid
        )
        selected_hyperparameter = {"max_depth": selected_value}
    else:
        c_grid = _validate_c_grid(logistic_c_grid)
        selected_value, model, frontier = _select_logistic_c(
            train_x, train_y, validation_x, validation_y, c_grid
        )
        selected_hyperparameter = {"C": selected_value}

    validation_predictions = _validation_prediction_rows(
        fold=fold,
        runs=validation_runs,
        vectors=validation_x,
        model=model,
    )
    manifest = folds[fold]
    train_label_records = [_labeled_run_hash_record(run) for run in train_runs]
    validation_label_records = [_labeled_run_hash_record(run) for run in validation_runs]
    train_metadata = {
        "protocol_version": LEARNED_METADATA_PROTOCOL_VERSION,
        "policy_name": policy_name,
        "baseline_role": DIAGNOSTIC_BASELINE_ROLE,
        "diagnostic_baseline": True,
        "final_norm_route": False,
        "fold": fold,
        "metric": metric,
        "label_source": "run_level_best_expert_from_split_outcomes",
        "label_tie_break": "candidate_expert_order",
        "feature_source": "category,k_shot,configured_budget,predeclared_static_cost_only",
        "config": {
            "policy_name": policy_name,
            "metric": metric,
            "budget": parsed_budget,
            "static_cost": costs,
            "static_cost_unit": unit,
            "tree_max_depth_grid": list(_validate_depth_grid(tree_max_depth_grid)),
            "logistic_c_grid": list(_validate_c_grid(logistic_c_grid)),
        },
        "seed": parsed_training_seed,
        "training_seed": parsed_training_seed,
        "train_seeds": list(manifest.train_seeds),
        "validation_seeds": list(manifest.validation_seeds),
        "num_training_runs": len(train_runs),
        "num_validation_runs": len(validation_runs),
        "training_label_counts": _label_counts(train_y),
        "validation_label_counts": _label_counts(validation_y),
        "training_labels_sha256": _canonical_sha256(train_label_records),
        "validation_labels_sha256": _canonical_sha256(validation_label_records),
        "training_manifest_runs_sha256": _canonical_sha256(
            [_run_key_record(key) for key in sorted(manifest.training_run_keys)]
        ),
        "validation_manifest_runs_sha256": _canonical_sha256(
            [_run_key_record(key) for key in sorted(manifest.validation_run_keys)]
        ),
        "hyperparameter_grid": list(
            _validate_depth_grid(tree_max_depth_grid)
            if policy_name == DECISION_TREE_METADATA
            else _validate_c_grid(logistic_c_grid)
        ),
        "selected_hyperparameter": selected_hyperparameter,
        "validation_selection_rule": (
            "maximize_run_level_best_expert_accuracy_then_choose_simpler_model"
        ),
        "validation_frontier": frontier,
        "budget": parsed_budget,
        "static_cost": costs,
        "static_cost_unit": unit,
        "static_cost_source": cost_source,
        "git_commit": _git_commit(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "artifacts": {
            "feature_manifest": FEATURE_MANIFEST_FILENAME,
            "model": MODEL_ARTIFACT_FILENAME,
            "validation_predictions": VALIDATION_PREDICTIONS_FILENAME,
            "failures": FAILURES_FILENAME,
        },
        "failures": [],
    }
    metadata_hash = _canonical_mapping_sha256(train_metadata)
    model_artifact = {
        "protocol_version": LEARNED_METADATA_PROTOCOL_VERSION,
        "policy_name": policy_name,
        "baseline_role": DIAGNOSTIC_BASELINE_ROLE,
        "diagnostic_baseline": True,
        "frozen": True,
        "fold": fold,
        "train_seeds": list(manifest.train_seeds),
        "validation_seeds": list(manifest.validation_seeds),
        "metric": metric,
        "label_source": "train_fold_run_level_best_expert",
        "selected_hyperparameter": selected_hyperparameter,
        "candidate_experts": list(CANDIDATE_EXPERTS),
        "budget": parsed_budget,
        "static_cost": costs,
        "static_cost_unit": unit,
        "feature_manifest_sha256": feature_manifest_sha256(feature_manifest),
        "train_metadata_sha256": metadata_hash,
        "model": model,
    }
    model_artifact = validate_model_artifact(model_artifact)

    output_dir = _artifact_output_dir(output_path)
    _ensure_safe_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_feature_manifest(feature_manifest, output_dir / FEATURE_MANIFEST_FILENAME)
    _write_json_atomic(output_dir / TRAIN_METADATA_FILENAME, train_metadata)
    _write_csv_atomic(
        output_dir / VALIDATION_PREDICTIONS_FILENAME,
        VALIDATION_PREDICTION_COLUMNS,
        validation_predictions,
    )
    _write_json_atomic(
        output_dir / FAILURES_FILENAME,
        {
            "protocol_version": LEARNED_METADATA_PROTOCOL_VERSION,
            "num_failed": 0,
            "failures": [],
        },
    )
    destination = output_dir / MODEL_ARTIFACT_FILENAME
    _write_json_atomic(destination, model_artifact)
    # Loading here is intentional: a calibration is only successful if the
    # exact frozen bundle that test replay will consume can be verified.
    policy_class = (
        DecisionTreeMetadataPolicy
        if policy_name == DECISION_TREE_METADATA
        else MultinomialLogisticMetadataPolicy
    )
    policy_class.load(destination)
    return destination


def calibrate_learned_metadata_artifacts(
    *,
    expert_quality_by_run: str | Path,
    fold_manifest: str | Path,
    output_root: str | Path,
    policy_name: str,
    metric: str = "image_auroc",
    folds: Sequence[str] | None = None,
    budget: int = 1,
    static_costs: Mapping[str, Any] | str | Path | None = None,
    static_cost_unit: str = DEFAULT_STATIC_COST_UNIT,
    tree_max_depth_grid: Sequence[int | None] = DEFAULT_TREE_MAX_DEPTH_GRID,
    logistic_c_grid: Sequence[float] = DEFAULT_LOGISTIC_C_GRID,
    training_seed: int = 0,
) -> list[Path]:
    """Train and freeze one independent artifact bundle per requested fold."""

    manifest_folds = read_learned_folds(fold_manifest, folds=folds)
    return [
        calibrate_learned_metadata_policy(
            expert_quality_by_run=expert_quality_by_run,
            fold_manifest=fold_manifest,
            fold=fold,
            output_path=Path(output_root) / policy_name / fold,
            policy_name=policy_name,
            metric=metric,
            budget=budget,
            static_costs=static_costs,
            static_cost_unit=static_cost_unit,
            tree_max_depth_grid=tree_max_depth_grid,
            logistic_c_grid=logistic_c_grid,
            training_seed=training_seed,
        )
        for fold in manifest_folds
    ]


def read_learned_folds(
    path: str | Path, *, folds: Sequence[str] | None = None
) -> dict[str, LearnedFold]:
    """Read run-level split assignments without exposing provenance as features."""

    source = Path(path)
    requested = _normalize_folds(folds)
    requested_set = set(requested or ())
    run_splits: dict[str, dict[RunKey, str]] = {}
    task_ids: dict[str, set[str]] = {}
    try:
        handle = source.open("r", newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise LearnedMetadataPolicyError(f"could not read fold manifest {source}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = [column for column in MANIFEST_COLUMNS if column not in fields]
        if missing:
            raise LearnedMetadataPolicyError(f"{source} is missing columns: {missing}")
        for line_number, raw in enumerate(reader, start=2):
            fold = str(raw.get("fold") or "").strip()
            if requested_set and fold not in requested_set:
                continue
            if not fold:
                raise LearnedMetadataPolicyError(f"{source}:{line_number} has empty fold")
            clean = {key: str(value or "").strip() for key, value in raw.items()}
            empty = [column for column in MANIFEST_COLUMNS if not clean.get(column)]
            if empty:
                raise LearnedMetadataPolicyError(
                    f"{source}:{line_number} has empty manifest values: {empty}"
                )
            split = clean["split"]
            if split not in {"train", "val", "test"}:
                raise LearnedMetadataPolicyError(
                    f"{source}:{line_number} has invalid split={split!r}"
                )
            run_key = _run_key(clean, source, line_number)
            previous = run_splits.setdefault(fold, {}).get(run_key)
            if previous is not None and previous != split:
                raise LearnedMetadataPolicyError(
                    f"{source} assigns run {run_key!r} to {previous!r} and {split!r}"
                )
            run_splits[fold][run_key] = split
            task_id = clean["task_id"]
            if task_id in task_ids.setdefault(fold, set()):
                raise LearnedMetadataPolicyError(
                    f"{source} duplicates task_id={task_id!r} in {fold}"
                )
            task_ids[fold].add(task_id)
    if requested_set - set(run_splits):
        raise LearnedMetadataPolicyError(
            f"manifest has no rows for folds: {sorted(requested_set - set(run_splits))}"
        )
    if not run_splits:
        raise LearnedMetadataPolicyError(f"{source} has no usable folds")
    result: dict[str, LearnedFold] = {}
    for fold in sorted(run_splits, key=_natural_fold_key):
        item = LearnedFold(fold=fold, run_splits=run_splits[fold])
        if not item.training_run_keys or not item.validation_run_keys:
            raise LearnedMetadataPolicyError(f"{fold} requires non-empty train and val runs")
        result[fold] = item
    return result


def read_train_validation_outcomes(
    path: str | Path,
    *,
    manifest_folds: Mapping[str, LearnedFold],
    metric: str = "image_auroc",
) -> dict[str, dict[str, list[ExpertOutcome]]]:
    """Read train/val outcomes for labels and never parse test metric cells."""

    if metric not in SUPPORTED_METRICS:
        raise LearnedMetadataPolicyError(
            f"metric must be one of {list(SUPPORTED_METRICS)!r}"
        )
    source = Path(path)
    result = {fold: {"train": [], "val": []} for fold in manifest_folds}
    seen = {
        fold: {"train": set(), "val": set()} for fold in manifest_folds
    }
    try:
        handle = source.open("r", newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise LearnedMetadataPolicyError(f"could not read outcome file {source}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = [
            column for column in (*QUALITY_IDENTITY_COLUMNS, metric) if column not in fields
        ]
        if missing:
            raise LearnedMetadataPolicyError(f"{source} is missing columns: {missing}")
        for line_number, raw in enumerate(reader, start=2):
            identity = {
                column: str(raw.get(column) or "").strip()
                for column in QUALITY_IDENTITY_COLUMNS
            }
            empty = [column for column, value in identity.items() if not value]
            if empty:
                raise LearnedMetadataPolicyError(
                    f"{source}:{line_number} has empty identity values: {empty}"
                )
            run_key = _run_key(identity, source, line_number)
            expert = _canonical_expert(identity["expert"], source, line_number)
            matches = [
                (fold, manifest.run_splits[run_key])
                for fold, manifest in manifest_folds.items()
                if run_key in manifest.run_splits
            ]
            if not matches:
                raise LearnedMetadataPolicyError(
                    f"{source}:{line_number} run {run_key!r} is absent from requested folds"
                )
            needed_splits = {split for _, split in matches if split in {"train", "val"}}
            if not needed_splits:
                # Isolation boundary: not even float-convert a test metric cell.
                continue
            value = _finite_unit_metric(raw.get(metric), metric, source, line_number)
            for fold, split in matches:
                if split == "test":
                    continue
                identity_key = (run_key, expert)
                if identity_key in seen[fold][split]:
                    raise LearnedMetadataPolicyError(
                        f"duplicate {split} row for fold={fold}, run={run_key!r}, expert={expert}"
                    )
                seen[fold][split].add(identity_key)
                result[fold][split].append(
                    ExpertOutcome(expert=expert, run_key=run_key, metric_value=value)
                )
    for fold, manifest in manifest_folds.items():
        for split, run_keys in (
            ("train", manifest.training_run_keys),
            ("val", manifest.validation_run_keys),
        ):
            expected = {
                (run_key, expert) for run_key in run_keys for expert in CANDIDATE_EXPERTS
            }
            actual = seen[fold][split]
            if expected != actual:
                raise LearnedMetadataPolicyError(
                    f"{split} outcome coverage mismatch for {fold}; "
                    f"missing={sorted(expected - actual)[:10]}, "
                    f"extra={sorted(actual - expected)[:10]}"
                )
    return result


def validate_model_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a frozen learned model artifact."""

    if not isinstance(value, Mapping) or set(value) != _MODEL_ARTIFACT_FIELDS:
        fields = set(value) if isinstance(value, Mapping) else set()
        raise LearnedMetadataPolicyError(
            "model artifact has invalid fields; "
            f"missing={sorted(_MODEL_ARTIFACT_FIELDS - fields)}, "
            f"extra={sorted(fields - _MODEL_ARTIFACT_FIELDS)}"
        )
    if value["protocol_version"] != LEARNED_METADATA_PROTOCOL_VERSION:
        raise LearnedMetadataPolicyError("model artifact protocol_version is invalid")
    policy_name = value["policy_name"]
    if policy_name not in LEARNED_METADATA_POLICY_NAMES:
        raise LearnedMetadataPolicyError("model artifact policy_name is invalid")
    if value["baseline_role"] != DIAGNOSTIC_BASELINE_ROLE:
        raise LearnedMetadataPolicyError("model artifact baseline role is invalid")
    if value["diagnostic_baseline"] is not True or value["frozen"] is not True:
        raise LearnedMetadataPolicyError("learned metadata artifact must be diagnostic and frozen")
    fold = _nonempty_string(value["fold"], "fold")
    train_seeds = _sorted_nonnegative_ints(value["train_seeds"], "train_seeds")
    validation_seeds = _sorted_nonnegative_ints(
        value["validation_seeds"], "validation_seeds"
    )
    if set(train_seeds).intersection(validation_seeds):
        raise LearnedMetadataPolicyError("model artifact train and validation seeds overlap")
    metric = value["metric"]
    if metric not in SUPPORTED_METRICS:
        raise LearnedMetadataPolicyError("model artifact metric is invalid")
    if value["label_source"] != "train_fold_run_level_best_expert":
        raise LearnedMetadataPolicyError("model artifact label_source is invalid")
    if value["candidate_experts"] != list(CANDIDATE_EXPERTS):
        raise LearnedMetadataPolicyError("model artifact expert order is invalid")
    budget = _nonnegative_int(value["budget"], "budget")
    if budget < 1:
        raise LearnedMetadataPolicyError("model artifact budget must be >= 1")
    costs = canonical_static_costs(value["static_cost"])
    unit = _nonempty_string(value["static_cost_unit"], "static_cost_unit")
    for field in ("feature_manifest_sha256", "train_metadata_sha256"):
        if not _is_sha256(value[field]):
            raise LearnedMetadataPolicyError(f"model artifact {field} is invalid")
    hyperparameter = value["selected_hyperparameter"]
    if not isinstance(hyperparameter, Mapping):
        raise LearnedMetadataPolicyError("selected_hyperparameter must be a mapping")
    if policy_name == DECISION_TREE_METADATA:
        if set(hyperparameter) != {"max_depth"}:
            raise LearnedMetadataPolicyError("decision tree requires selected max_depth")
        selected = hyperparameter["max_depth"]
        if selected is not None:
            _positive_int(selected, "max_depth")
    else:
        if set(hyperparameter) != {"C"}:
            raise LearnedMetadataPolicyError("logistic model requires selected C")
        if _finite_positive(hyperparameter["C"], "C") <= 0:
            raise LearnedMetadataPolicyError("C must be > 0")
    model = _validate_serialized_model(value["model"], policy_name=policy_name)
    return {
        "protocol_version": LEARNED_METADATA_PROTOCOL_VERSION,
        "policy_name": policy_name,
        "baseline_role": DIAGNOSTIC_BASELINE_ROLE,
        "diagnostic_baseline": True,
        "frozen": True,
        "fold": fold,
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "metric": metric,
        "label_source": "train_fold_run_level_best_expert",
        "selected_hyperparameter": dict(hyperparameter),
        "candidate_experts": list(CANDIDATE_EXPERTS),
        "budget": budget,
        "static_cost": costs,
        "static_cost_unit": unit,
        "feature_manifest_sha256": value["feature_manifest_sha256"],
        "train_metadata_sha256": value["train_metadata_sha256"],
        "model": model,
    }


def _load_frozen_bundle(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = Path(path)
    if source.is_dir():
        source = source / MODEL_ARTIFACT_FILENAME
    payload = _read_json(source)
    if payload.get("state_version") == LEARNED_METADATA_STATE_VERSION:
        if set(payload) != {
            "state_version",
            "policy_name",
            "model_artifact",
            "feature_manifest",
            "train_metadata",
        }:
            raise LearnedMetadataPolicyError("saved learned policy state has invalid fields")
        model_artifact = validate_model_artifact(payload["model_artifact"])
        feature_manifest = validate_feature_manifest(payload["feature_manifest"])
        train_metadata = _validate_train_metadata(payload["train_metadata"])
        if payload["policy_name"] != model_artifact["policy_name"]:
            raise LearnedMetadataPolicyError("saved policy name disagrees with model artifact")
    else:
        if source.name != MODEL_ARTIFACT_FILENAME:
            raise LearnedMetadataPolicyError(
                f"frozen learned model filename must be {MODEL_ARTIFACT_FILENAME}"
            )
        model_artifact = validate_model_artifact(payload)
        feature_manifest = validate_feature_manifest(
            _read_json(source.parent / FEATURE_MANIFEST_FILENAME)
        )
        train_metadata = _validate_train_metadata(
            _read_json(source.parent / TRAIN_METADATA_FILENAME)
        )
    if feature_manifest_sha256(feature_manifest) != model_artifact["feature_manifest_sha256"]:
        raise LearnedMetadataPolicyError("feature manifest hash does not match model artifact")
    if _canonical_mapping_sha256(train_metadata) != model_artifact["train_metadata_sha256"]:
        raise LearnedMetadataPolicyError("train metadata hash does not match model artifact")
    for field in ("policy_name", "fold"):
        if train_metadata.get(field) != model_artifact[field]:
            raise LearnedMetadataPolicyError(f"train metadata {field} disagrees with model")
    if train_metadata.get("diagnostic_baseline") is not True:
        raise LearnedMetadataPolicyError("train metadata must mark diagnostic_baseline=true")
    return model_artifact, feature_manifest, train_metadata


def _validate_train_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LearnedMetadataPolicyError("train metadata must be a mapping")
    required = {
        "protocol_version",
        "policy_name",
        "baseline_role",
        "diagnostic_baseline",
        "final_norm_route",
        "fold",
        "metric",
        "label_source",
        "config",
        "seed",
        "training_seed",
        "train_seeds",
        "validation_seeds",
        "selected_hyperparameter",
        "environment",
        "artifacts",
        "git_commit",
        "failures",
    }
    missing = sorted(required - set(value))
    if missing:
        raise LearnedMetadataPolicyError(f"train metadata is missing fields: {missing}")
    if value["protocol_version"] != LEARNED_METADATA_PROTOCOL_VERSION:
        raise LearnedMetadataPolicyError("train metadata protocol_version is invalid")
    if value["baseline_role"] != DIAGNOSTIC_BASELINE_ROLE:
        raise LearnedMetadataPolicyError("train metadata baseline role is invalid")
    if value["diagnostic_baseline"] is not True or value["final_norm_route"] is not False:
        raise LearnedMetadataPolicyError("train metadata diagnostic markers are invalid")
    if value["failures"] != []:
        raise LearnedMetadataPolicyError("successful frozen bundle must have no train failures")
    # Round-trip through JSON to detach arbitrary Mapping subclasses and ensure
    # the hash uses only JSON-safe metadata.
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise LearnedMetadataPolicyError("train metadata is not JSON-safe") from exc


def _build_run_labels(
    outcomes: Sequence[ExpertOutcome], *, split: str, fold: str
) -> list[LabeledRun]:
    by_run: dict[RunKey, dict[str, float]] = {}
    for row in outcomes:
        experts = by_run.setdefault(row.run_key, {})
        if row.expert in experts:
            raise LearnedMetadataPolicyError(
                f"duplicate {split} outcome for {fold}, run={row.run_key!r}, expert={row.expert}"
            )
        experts[row.expert] = row.metric_value
    if not by_run:
        raise LearnedMetadataPolicyError(f"{fold} has no {split} labeled runs")
    result: list[LabeledRun] = []
    for run_key in sorted(by_run):
        scores = by_run[run_key]
        missing = [expert for expert in CANDIDATE_EXPERTS if expert not in scores]
        if missing:
            raise LearnedMetadataPolicyError(
                f"{fold} {split} run={run_key!r} is missing experts: {missing}"
            )
        best = max(CANDIDATE_EXPERTS, key=lambda expert: (scores[expert], -CANDIDATE_EXPERTS.index(expert)))
        result.append(
            LabeledRun(
                run_key=run_key,
                category=run_key[1],
                k_shot=run_key[2],
                best_expert=best,
            )
        )
    return result


def _select_tree_depth(
    train_x: Sequence[Sequence[float]],
    train_y: Sequence[int],
    validation_x: Sequence[Sequence[float]],
    validation_y: Sequence[int],
    grid: Sequence[int | None],
) -> tuple[int | None, dict[str, Any], list[dict[str, Any]]]:
    candidates: list[tuple[float, int, int | None, dict[str, Any], list[list[float]]]] = []
    frontier: list[dict[str, Any]] = []
    for order, depth in enumerate(grid):
        model = _fit_decision_tree(train_x, train_y, max_depth=depth)
        probabilities = [_predict_probabilities(model, vector) for vector in validation_x]
        accuracy = _accuracy(probabilities, validation_y)
        complexity = depth if depth is not None else 10**9
        candidates.append((accuracy, complexity, depth, model, probabilities))
        frontier.append(
            {
                "max_depth": depth,
                "validation_accuracy": accuracy,
                "num_validation_runs": len(validation_y),
                "selected": False,
                "grid_order": order,
            }
        )
    selected = min(candidates, key=lambda item: (-item[0], item[1]))
    selected_depth = selected[2]
    for row in frontier:
        row["selected"] = row["max_depth"] == selected_depth
    return selected_depth, selected[3], frontier


def _select_logistic_c(
    train_x: Sequence[Sequence[float]],
    train_y: Sequence[int],
    validation_x: Sequence[Sequence[float]],
    validation_y: Sequence[int],
    grid: Sequence[float],
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    candidates: list[tuple[float, float, dict[str, Any]]] = []
    frontier: list[dict[str, Any]] = []
    for order, c_value in enumerate(grid):
        model = _fit_multinomial_logistic(train_x, train_y, c_value=c_value)
        probabilities = [_predict_probabilities(model, vector) for vector in validation_x]
        accuracy = _accuracy(probabilities, validation_y)
        candidates.append((accuracy, c_value, model))
        frontier.append(
            {
                "C": c_value,
                "validation_accuracy": accuracy,
                "num_validation_runs": len(validation_y),
                "selected": False,
                "grid_order": order,
            }
        )
    selected = min(candidates, key=lambda item: (-item[0], item[1]))
    for row in frontier:
        row["selected"] = math.isclose(row["C"], selected[1], rel_tol=0.0, abs_tol=0.0)
    return selected[1], selected[2], frontier


def _fit_decision_tree(
    x: Sequence[Sequence[float]], y: Sequence[int], *, max_depth: int | None
) -> dict[str, Any]:
    _validate_training_matrix(x, y)
    num_features = len(x[0])

    def build(indices: list[int], depth: int) -> dict[str, Any]:
        counts = [sum(y[index] == class_index for index in indices) for class_index in range(len(CANDIDATE_EXPERTS))]
        probabilities = _smoothed_probabilities(counts)
        leaf = {
            "node_type": "leaf",
            "class_counts": counts,
            "probabilities": probabilities,
        }
        if (
            len(indices) <= 1
            or sum(count > 0 for count in counts) <= 1
            or (max_depth is not None and depth >= max_depth)
        ):
            return leaf
        parent_impurity = _gini(counts)
        best: tuple[float, int, float, list[int], list[int]] | None = None
        for feature_index in range(num_features):
            values = sorted({float(x[index][feature_index]) for index in indices})
            for lower, upper in zip(values, values[1:]):
                threshold = (lower + upper) / 2.0
                left = [index for index in indices if x[index][feature_index] <= threshold]
                right = [index for index in indices if x[index][feature_index] > threshold]
                if not left or not right:
                    continue
                left_counts = [sum(y[index] == class_index for index in left) for class_index in range(len(CANDIDATE_EXPERTS))]
                right_counts = [sum(y[index] == class_index for index in right) for class_index in range(len(CANDIDATE_EXPERTS))]
                weighted = (len(left) * _gini(left_counts) + len(right) * _gini(right_counts)) / len(indices)
                gain = parent_impurity - weighted
                candidate = (gain, feature_index, threshold, left, right)
                if best is None or (gain, -feature_index, -threshold) > (
                    best[0], -best[1], -best[2]
                ):
                    best = candidate
        if best is None or best[0] <= 1e-15:
            return leaf
        return {
            "node_type": "split",
            "feature_index": best[1],
            "threshold": best[2],
            "gain": best[0],
            "probabilities": probabilities,
            "left": build(best[3], depth + 1),
            "right": build(best[4], depth + 1),
        }

    return {
        "kind": "decision_tree",
        "classes": list(CANDIDATE_EXPERTS),
        "num_features": num_features,
        "max_depth": max_depth,
        "laplace_smoothing": 1.0,
        "tree": build(list(range(len(x))), 0),
    }


def _fit_multinomial_logistic(
    x: Sequence[Sequence[float]], y: Sequence[int], *, c_value: float
) -> dict[str, Any]:
    _validate_training_matrix(x, y)
    c_value = _finite_positive(c_value, "C")
    num_samples = len(x)
    num_features = len(x[0])
    num_classes = len(CANDIDATE_EXPERTS)
    weights = [[0.0 for _ in range(num_features)] for _ in range(num_classes)]
    intercept = [0.0 for _ in range(num_classes)]
    regularization = 1.0 / (c_value * num_samples)
    max_iterations = 800
    tolerance = 1e-9
    iterations = 0
    previous_loss: float | None = None
    for iteration in range(max_iterations):
        gradient_w = [[0.0 for _ in range(num_features)] for _ in range(num_classes)]
        gradient_b = [0.0 for _ in range(num_classes)]
        loss = 0.0
        for vector, target in zip(x, y):
            logits = [
                intercept[class_index]
                + math.fsum(weight * value for weight, value in zip(weights[class_index], vector))
                for class_index in range(num_classes)
            ]
            probabilities = _softmax(logits)
            loss -= math.log(max(probabilities[target], 1e-300))
            for class_index in range(num_classes):
                residual = probabilities[class_index] - (1.0 if class_index == target else 0.0)
                gradient_b[class_index] += residual
                for feature_index, value in enumerate(vector):
                    gradient_w[class_index][feature_index] += residual * value
        squared_norm = math.fsum(
            weight * weight for class_weights in weights for weight in class_weights
        )
        loss = loss / num_samples + 0.5 * regularization * squared_norm
        learning_rate = 0.35 / math.sqrt(1.0 + iteration / 40.0)
        max_update = 0.0
        for class_index in range(num_classes):
            update_b = learning_rate * gradient_b[class_index] / num_samples
            intercept[class_index] -= update_b
            max_update = max(max_update, abs(update_b))
            for feature_index in range(num_features):
                gradient = (
                    gradient_w[class_index][feature_index] / num_samples
                    + regularization * weights[class_index][feature_index]
                )
                update = learning_rate * gradient
                weights[class_index][feature_index] -= update
                max_update = max(max_update, abs(update))
        iterations = iteration + 1
        if max_update < tolerance:
            break
        if previous_loss is not None and abs(previous_loss - loss) < tolerance * 0.1:
            break
        previous_loss = loss
    return {
        "kind": "multinomial_logistic",
        "classes": list(CANDIDATE_EXPERTS),
        "num_features": num_features,
        "C": c_value,
        "regularization": "l2",
        "optimizer": "deterministic_batch_gradient_descent",
        "iterations": iterations,
        "weights": weights,
        "intercept": intercept,
    }


def _predict_probabilities(model: Mapping[str, Any], vector: Sequence[float]) -> list[float]:
    kind = model.get("kind")
    if len(vector) != model.get("num_features"):
        raise LearnedMetadataPolicyError("feature vector length does not match frozen model")
    if kind == "decision_tree":
        node = model["tree"]
        while node["node_type"] == "split":
            node = (
                node["left"]
                if vector[node["feature_index"]] <= node["threshold"]
                else node["right"]
            )
        probabilities = node["probabilities"]
    elif kind == "multinomial_logistic":
        logits = [
            model["intercept"][class_index]
            + math.fsum(
                weight * value
                for weight, value in zip(model["weights"][class_index], vector)
            )
            for class_index in range(len(CANDIDATE_EXPERTS))
        ]
        probabilities = _softmax(logits)
    else:
        raise LearnedMetadataPolicyError(f"unknown frozen model kind {kind!r}")
    return _validate_probabilities(probabilities)


def _validate_serialized_model(value: Any, *, policy_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LearnedMetadataPolicyError("serialized model must be a mapping")
    expected_kind = (
        "decision_tree"
        if policy_name == DECISION_TREE_METADATA
        else "multinomial_logistic"
    )
    if value.get("kind") != expected_kind:
        raise LearnedMetadataPolicyError("serialized model kind is invalid")
    if value.get("classes") != list(CANDIDATE_EXPERTS):
        raise LearnedMetadataPolicyError("serialized model classes are invalid")
    num_features = _positive_int(value.get("num_features"), "model.num_features")
    if expected_kind == "decision_tree":
        required = {
            "kind",
            "classes",
            "num_features",
            "max_depth",
            "laplace_smoothing",
            "tree",
        }
        if set(value) != required:
            raise LearnedMetadataPolicyError("serialized decision tree fields are invalid")
        depth = value["max_depth"]
        if depth is not None:
            _positive_int(depth, "model.max_depth")
        if _finite_positive(value["laplace_smoothing"], "laplace_smoothing") <= 0:
            raise LearnedMetadataPolicyError("laplace_smoothing must be > 0")
        tree = _validate_tree_node(value["tree"], num_features=num_features)
        return {
            "kind": expected_kind,
            "classes": list(CANDIDATE_EXPERTS),
            "num_features": num_features,
            "max_depth": depth,
            "laplace_smoothing": float(value["laplace_smoothing"]),
            "tree": tree,
        }
    required = {
        "kind",
        "classes",
        "num_features",
        "C",
        "regularization",
        "optimizer",
        "iterations",
        "weights",
        "intercept",
    }
    if set(value) != required:
        raise LearnedMetadataPolicyError("serialized logistic model fields are invalid")
    c_value = _finite_positive(value["C"], "model.C")
    if value["regularization"] != "l2" or value["optimizer"] != "deterministic_batch_gradient_descent":
        raise LearnedMetadataPolicyError("serialized logistic training contract is invalid")
    iterations = _positive_int(value["iterations"], "model.iterations")
    weights = _finite_matrix(
        value["weights"], rows=len(CANDIDATE_EXPERTS), columns=num_features, field="weights"
    )
    intercept = _finite_vector(
        value["intercept"], length=len(CANDIDATE_EXPERTS), field="intercept"
    )
    return {
        "kind": expected_kind,
        "classes": list(CANDIDATE_EXPERTS),
        "num_features": num_features,
        "C": c_value,
        "regularization": "l2",
        "optimizer": "deterministic_batch_gradient_descent",
        "iterations": iterations,
        "weights": weights,
        "intercept": intercept,
    }


def _validate_tree_node(value: Any, *, num_features: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LearnedMetadataPolicyError("tree node must be a mapping")
    node_type = value.get("node_type")
    if node_type == "leaf":
        if set(value) != {"node_type", "class_counts", "probabilities"}:
            raise LearnedMetadataPolicyError("leaf node fields are invalid")
        counts = _nonnegative_int_vector(
            value["class_counts"], length=len(CANDIDATE_EXPERTS), field="class_counts"
        )
        if sum(counts) < 1:
            raise LearnedMetadataPolicyError("tree leaf has no training samples")
        probabilities = _validate_probabilities(value["probabilities"])
        return {
            "node_type": "leaf",
            "class_counts": counts,
            "probabilities": probabilities,
        }
    if node_type == "split":
        required = {
            "node_type",
            "feature_index",
            "threshold",
            "gain",
            "probabilities",
            "left",
            "right",
        }
        if set(value) != required:
            raise LearnedMetadataPolicyError("split node fields are invalid")
        feature_index = _nonnegative_int(value["feature_index"], "feature_index")
        if feature_index >= num_features:
            raise LearnedMetadataPolicyError("tree feature_index is out of range")
        gain = _finite_positive(value["gain"], "tree gain")
        return {
            "node_type": "split",
            "feature_index": feature_index,
            "threshold": _finite_number(value["threshold"], "tree threshold"),
            "gain": gain,
            "probabilities": _validate_probabilities(value["probabilities"]),
            "left": _validate_tree_node(value["left"], num_features=num_features),
            "right": _validate_tree_node(value["right"], num_features=num_features),
        }
    raise LearnedMetadataPolicyError(f"unknown tree node_type={node_type!r}")


def _validation_prediction_rows(
    *,
    fold: str,
    runs: Sequence[LabeledRun],
    vectors: Sequence[Sequence[float]],
    model: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run, vector in zip(runs, vectors):
        probabilities = _predict_probabilities(model, vector)
        selected_index = max(
            range(len(CANDIDATE_EXPERTS)), key=lambda index: (probabilities[index], -index)
        )
        ordered = sorted(probabilities, reverse=True)
        selected = CANDIDATE_EXPERTS[selected_index]
        rows.append(
            {
                "fold": fold,
                "split": "val",
                "run_identity_sha256": _canonical_mapping_sha256(
                    _run_key_record(run.run_key)
                ),
                "category": run.category,
                "k_shot": run.k_shot,
                "best_expert_label": run.best_expert,
                "predicted_expert": selected,
                "selected_probability": probabilities[selected_index],
                "margin": ordered[0] - ordered[1],
                "correct": selected == run.best_expert,
            }
        )
    return rows


def _resolve_static_costs(
    value: Mapping[str, Any] | str | Path | None,
) -> tuple[dict[str, float], str]:
    if value is None:
        return canonical_static_costs(DEFAULT_STATIC_COSTS), "default_equal_abstract_unit_cost"
    if isinstance(value, (str, Path)):
        source = Path(value)
        payload = _read_json(source)
        if "costs_ms" in payload:
            payload = payload["costs_ms"]
        return canonical_static_costs(payload), f"predeclared_cost_card:{source.name}"
    return canonical_static_costs(value), "predeclared_mapping"


def _artifact_output_dir(path: str | Path) -> Path:
    destination = Path(path)
    if destination.suffix:
        if destination.name != MODEL_ARTIFACT_FILENAME:
            raise LearnedMetadataPolicyError(
                f"model artifact filename must be {MODEL_ARTIFACT_FILENAME}"
            )
        return destination.parent
    return destination


def _validate_training_matrix(x: Sequence[Sequence[float]], y: Sequence[int]) -> None:
    if not x or len(x) != len(y):
        raise LearnedMetadataPolicyError("training matrix and labels must be non-empty and aligned")
    width = len(x[0])
    if width < 1 or any(len(row) != width for row in x):
        raise LearnedMetadataPolicyError("training matrix has inconsistent feature widths")
    for row in x:
        for value in row:
            _finite_number(value, "training feature")
    if any(
        isinstance(label, bool)
        or not isinstance(label, int)
        or not 0 <= label < len(CANDIDATE_EXPERTS)
        for label in y
    ):
        raise LearnedMetadataPolicyError("training labels are invalid")


def _accuracy(probabilities: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    if not labels or len(probabilities) != len(labels):
        raise LearnedMetadataPolicyError("validation predictions and labels must align")
    correct = 0
    for row, label in zip(probabilities, labels):
        predicted = max(range(len(row)), key=lambda index: (row[index], -index))
        correct += predicted == label
    return correct / len(labels)


def _gini(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return 1.0 - math.fsum((count / total) ** 2 for count in counts)


def _smoothed_probabilities(counts: Sequence[int]) -> list[float]:
    total = sum(counts) + len(counts)
    return [(count + 1.0) / total for count in counts]


def _softmax(logits: Sequence[float]) -> list[float]:
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    total = math.fsum(exponentials)
    return [value / total for value in exponentials]


def _validate_probabilities(value: Any) -> list[float]:
    probabilities = _finite_vector(
        value, length=len(CANDIDATE_EXPERTS), field="probabilities"
    )
    if any(item < 0.0 or item > 1.0 for item in probabilities):
        raise LearnedMetadataPolicyError("probabilities must be in [0, 1]")
    total = math.fsum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise LearnedMetadataPolicyError("probabilities must sum to one")
    return probabilities


def _validate_depth_grid(value: Sequence[int | None]) -> tuple[int | None, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise LearnedMetadataPolicyError("tree max_depth grid must be non-empty")
    parsed: list[int | None] = []
    for item in value:
        if item is None:
            parsed.append(None)
        else:
            parsed.append(_positive_int(item, "max_depth grid value"))
    if len(parsed) != len(set(parsed)):
        raise LearnedMetadataPolicyError("tree max_depth grid must contain unique values")
    finite = [item for item in parsed if item is not None]
    if finite != sorted(finite) or (None in parsed and parsed[-1] is not None):
        raise LearnedMetadataPolicyError("tree max_depth grid must be sorted with None last")
    return tuple(parsed)


def _validate_c_grid(value: Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise LearnedMetadataPolicyError("logistic C grid must be non-empty")
    parsed = tuple(_finite_positive(item, "C grid value") for item in value)
    if tuple(sorted(set(parsed))) != parsed:
        raise LearnedMetadataPolicyError("logistic C grid must be sorted and unique")
    return parsed


def _label_counts(labels: Sequence[int]) -> dict[str, int]:
    return {
        expert: sum(label == index for label in labels)
        for index, expert in enumerate(CANDIDATE_EXPERTS)
    }


def _labeled_run_hash_record(run: LabeledRun) -> dict[str, Any]:
    return {**_run_key_record(run.run_key), "best_expert": run.best_expert}


def _run_key_record(key: RunKey) -> dict[str, Any]:
    return {
        "dataset": key[0],
        "category": key[1],
        "k_shot": key[2],
        "seed": key[3],
        "support_set_id": key[4],
    }


def _run_key(row: Mapping[str, Any], path: Path, line_number: int) -> RunKey:
    dataset = str(row.get("dataset") or "").strip()
    category = str(row.get("category") or "").strip()
    support_set_id = str(row.get("support_set_id") or "").strip()
    if not dataset or not category or not support_set_id:
        raise LearnedMetadataPolicyError(f"{path}:{line_number} has empty run identity")
    return (
        dataset,
        category,
        _positive_csv_int(row.get("k_shot"), "k_shot", path, line_number),
        _nonnegative_csv_int(row.get("seed"), "seed", path, line_number),
        support_set_id,
    )


def _canonical_expert(value: Any, path: Path, line_number: int) -> str:
    normalized = "".join(character for character in str(value).lower() if character.isalnum())
    aliases = {
        "".join(character for character in expert.lower() if character.isalnum()): expert
        for expert in CANDIDATE_EXPERTS
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise LearnedMetadataPolicyError(
            f"{path}:{line_number} has unknown expert={value!r}"
        ) from exc


def _normalize_folds(value: Sequence[str] | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise LearnedMetadataPolicyError("folds must be a non-empty sequence")
    parsed = tuple(_nonempty_string(item, "fold") for item in value)
    if len(parsed) != len(set(parsed)):
        raise LearnedMetadataPolicyError("folds must be unique")
    return parsed


def _natural_fold_key(value: str) -> tuple[str, int, str]:
    prefix = value.rstrip("0123456789")
    suffix = value[len(prefix) :]
    return prefix, int(suffix) if suffix else -1, value


def _nonnegative_csv_int(value: Any, field: str, path: Path, line_number: int) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise LearnedMetadataPolicyError(
            f"{path}:{line_number} {field} must be an integer"
        ) from exc
    if parsed < 0:
        raise LearnedMetadataPolicyError(f"{path}:{line_number} {field} must be >= 0")
    return parsed


def _positive_csv_int(value: Any, field: str, path: Path, line_number: int) -> int:
    parsed = _nonnegative_csv_int(value, field, path, line_number)
    if parsed < 1:
        raise LearnedMetadataPolicyError(f"{path}:{line_number} {field} must be >= 1")
    return parsed


def _finite_unit_metric(value: Any, field: str, path: Path, line_number: int) -> float:
    try:
        parsed = float(str(value or "").strip())
    except ValueError as exc:
        raise LearnedMetadataPolicyError(
            f"{path}:{line_number} has invalid {field}={value!r}"
        ) from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise LearnedMetadataPolicyError(
            f"{path}:{line_number} {field} must be finite and in [0, 1]"
        )
    return parsed


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LearnedMetadataPolicyError(f"{field} must be a non-empty string")
    return value.strip()


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LearnedMetadataPolicyError(f"{field} must be an integer >= 0")
    return value


def _positive_int(value: Any, field: str) -> int:
    parsed = _nonnegative_int(value, field)
    if parsed < 1:
        raise LearnedMetadataPolicyError(f"{field} must be an integer >= 1")
    return parsed


def _sorted_nonnegative_ints(value: Any, field: str) -> list[int]:
    if not isinstance(value, list):
        raise LearnedMetadataPolicyError(f"{field} must be a list")
    parsed = [_nonnegative_int(item, f"{field} item") for item in value]
    if parsed != sorted(set(parsed)):
        raise LearnedMetadataPolicyError(f"{field} must be sorted and unique")
    return parsed


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise LearnedMetadataPolicyError(f"{field} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise LearnedMetadataPolicyError(f"{field} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise LearnedMetadataPolicyError(f"{field} must be a finite number")
    return parsed


def _finite_positive(value: Any, field: str) -> float:
    parsed = _finite_number(value, field)
    if parsed <= 0:
        raise LearnedMetadataPolicyError(f"{field} must be > 0")
    return parsed


def _finite_vector(value: Any, *, length: int, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise LearnedMetadataPolicyError(f"{field} must have length {length}")
    return [_finite_number(item, field) for item in value]


def _nonnegative_int_vector(value: Any, *, length: int, field: str) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise LearnedMetadataPolicyError(f"{field} must have length {length}")
    return [_nonnegative_int(item, field) for item in value]


def _finite_matrix(
    value: Any, *, rows: int, columns: int, field: str
) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise LearnedMetadataPolicyError(f"{field} must have {rows} rows")
    return [
        _finite_vector(row, length=columns, field=f"{field}[{index}]")
        for index, row in enumerate(value)
    ]


def _canonical_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    serialized = json.dumps(
        list(records), sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LearnedMetadataPolicyError(f"could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LearnedMetadataPolicyError(f"JSON artifact {path} must contain an object")
    return payload


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _write_csv_atomic(
    path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(columns), extrasaction="raise", lineterminator="\n"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        column: (
                            "true"
                            if row[column] is True
                            else "false"
                            if row[column] is False
                            else row[column]
                        )
                        for column in columns
                    }
                )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _ensure_safe_output(path: Path) -> None:
    lowered = {part.lower() for part in path.parts}
    if "evaluator_only" in lowered or "oracle" in lowered:
        raise LearnedMetadataPolicyError(
            "learned router artifacts must not be written under evaluator_only/oracle"
        )


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


decision_tree_metadata = DecisionTreeMetadataPolicy
multinomial_logistic_metadata = MultinomialLogisticMetadataPolicy
train_metadata_router = calibrate_learned_metadata_policy


__all__ = [
    "DECISION_TREE_METADATA",
    "DEFAULT_LOGISTIC_C_GRID",
    "DEFAULT_STATIC_COSTS",
    "DEFAULT_TREE_MAX_DEPTH_GRID",
    "DIAGNOSTIC_BASELINE_ROLE",
    "DecisionTreeMetadataPolicy",
    "LEARNED_METADATA_POLICY_NAMES",
    "LearnedMetadataPolicyError",
    "MODEL_ARTIFACT_FILENAME",
    "MULTINOMIAL_LOGISTIC_METADATA",
    "MultinomialLogisticMetadataPolicy",
    "calibrate_learned_metadata_artifacts",
    "calibrate_learned_metadata_policy",
    "decision_tree_metadata",
    "multinomial_logistic_metadata",
    "read_learned_folds",
    "read_train_validation_outcomes",
    "train_metadata_router",
    "validate_model_artifact",
]
