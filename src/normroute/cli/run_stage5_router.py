"""Train and evaluate one fold-safe Stage 5 BIR-AD Router.

Router features and evaluator supervision are loaded through separate paths.
Only train/validation outcomes can affect the model.  Test labels are accessed
after predictions have been frozen and only inside evaluator-side metrics.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from ..router.feature_bundle import (
    ROUTER_FEATURE_BUNDLE_COMPATIBLE_PROTOCOL_VERSIONS,
)
from ..router.learned_router import (
    Stage5RouterError,
    fit_linear_router,
)
from ..router.dataset import is_forbidden_inference_feature_name
from ..router.expert_bank import read_capability_bank
from ..router.teacher import (
    ExpertScoreCalibration,
    fit_expert_score_calibration,
    read_teacher_parquet,
)
from ..routing.quality_metrics import (
    compute_auroc,
    compute_average_precision,
    compute_f1_max,
)


STAGE5_ROUTER_RUN_PROTOCOL_VERSION = "stage5.router_run.v2"
STAGE5_ROUTER_PREDICTION_PROTOCOL_VERSION = "stage5.router_prediction.v2"
STAGE5_ROUTER_METRICS_PROTOCOL_VERSION = "stage5.router_metrics.v2"
STAGE5_ROUTER_PER_TASK_PROTOCOL_VERSION = "stage5.router_per_task_metrics.v2"
MODEL_NAME = "router_model.json"
PREDICTIONS_NAME = "predictions.jsonl"
FAILURES_NAME = "failures.json"
RUN_RECORD_NAME = "run.json"
METRICS_NAME = "metrics.json"
PER_TASK_METRICS_NAME = "per_task_metrics.csv"
_SPLITS = ("train", "val", "test")
@dataclass(frozen=True)
class FeatureArtifact:
    task_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: Any
    source_ablation: str
    sha256: str
    source_feature_view: str
    source_fbdp_ablation: str
    source_protocol_version: str


@dataclass(frozen=True)
class TaskOutcome:
    label: int
    scores: Mapping[str, float]
    runtimes_ms: Mapping[str, float | None]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-features", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--routing-matrix", required=True)
    parser.add_argument(
        "--teacher-data",
        help="Evaluator-only train teacher.parquet; enables soft supervision.",
    )
    parser.add_argument(
        "--capability-bank",
        help="Optional train-only capability_bank.json used by risk-cost selection.",
    )
    parser.add_argument(
        "--supervision",
        choices=("soft_teacher", "hard_oracle"),
        default="hard_oracle",
        help="hard_oracle is retained only as an ablation/legacy path.",
    )
    parser.add_argument("--fold", required=True, choices=tuple(f"fold{i}" for i in range(5)))
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--bir-ablation",
        help="Expected BIR-AD ablation provenance for views that include BIR.",
    )
    parser.add_argument(
        "--fbdp-ablation",
        help="Expected FBDP-AD ablation provenance for views that include FBDP.",
    )
    parser.add_argument(
        "--feature-view",
        choices=(
            "all",
            "normal_only",
            "normal_bir",
            "normal_fbdp",
            "normal_bir_fbdp",
        ),
        default="all",
        help="Select a strict causal prefix view from one Router bundle.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--l2-grid",
        type=float,
        nargs="+",
        default=(0.0, 1e-4, 1e-3),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capability-weight-grid", type=float, nargs="+", default=(0.0, 0.25, 0.5))
    parser.add_argument(
        "--uncertainty-weight-grid",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 1.0, 2.0),
    )
    parser.add_argument("--cost-weight-grid", type=float, nargs="+", default=(0.0, 0.05, 0.1))
    parser.add_argument(
        "--validation-runtime-tradeoff",
        type=float,
        default=0.05,
        help=(
            "Frozen coefficient in the validation empirical loss used to "
            "select capability/uncertainty/cost weights."
        ),
    )
    parser.add_argument(
        "--capability-mode",
        choices=("conditional", "static"),
        default="conditional",
    )
    parser.add_argument(
        "--uncertainty-mode",
        choices=("entropy_scaled_lcb", "legacy_interval_width"),
        default="entropy_scaled_lcb",
        help=(
            "How ECPB epistemic uncertainty enters routing. The default uses "
            "query predictive entropy times the expert-specific bootstrap "
            "lower-confidence-bound risk gap; legacy_interval_width is kept "
            "only for a reproducibility ablation."
        ),
    )
    parser.add_argument(
        "--capability-skills",
        nargs="+",
        choices=("boundary", "fgbg", "lowshot", "texture"),
        default=("boundary", "fgbg", "lowshot", "texture"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    evaluator_dir = output_dir / "evaluator_only"
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    outputs: dict[str, str | None] = {
        "model": None,
        "predictions": None,
        "metrics": None,
        "per_task_metrics": None,
    }
    runtime: dict[str, Any] = {}
    input_hashes: dict[str, str] = {}
    try:
        manifest_rows = read_fold_manifest(args.fold_manifest, args.fold)
        manifest_by_task = {row["task_id"]: row for row in manifest_rows}
        artifact = read_router_features(
            args.router_features,
            manifest_by_task,
            feature_view=args.feature_view,
        )
        selected_bir = args.feature_view in {
            "all", "normal_bir", "normal_bir_fbdp"
        } and artifact.source_feature_view in {"normal_bir", "normal_bir_fbdp"}
        selected_fbdp = args.feature_view in {
            "all", "normal_fbdp", "normal_bir_fbdp"
        } and artifact.source_feature_view in {"normal_fbdp", "normal_bir_fbdp"}
        expected_bir = args.bir_ablation
        expected_fbdp = args.fbdp_ablation
        if (
            args.feature_view == "normal_bir"
            or (
                args.feature_view == "all"
                and artifact.source_feature_view == "normal_bir"
            )
        ) and expected_bir is None:
            expected_bir = args.variant
        if (
            args.feature_view == "normal_fbdp"
            or (
                args.feature_view == "all"
                and artifact.source_feature_view == "normal_fbdp"
            )
        ) and expected_fbdp is None:
            expected_fbdp = args.variant
        if (
            args.feature_view == "all"
            and artifact.source_feature_view == "normal_bir_fbdp"
            and (expected_bir is None or expected_fbdp is None)
        ):
            raise Stage5RouterError(
                "combined all-feature view requires --bir-ablation and "
                "--fbdp-ablation"
            )
        if selected_bir and expected_bir and artifact.source_ablation != expected_bir:
            raise Stage5RouterError(
                "Router source BIR-AD ablation disagrees with --bir-ablation"
            )
        if (
            selected_fbdp
            and expected_fbdp
            and artifact.source_fbdp_ablation != expected_fbdp
        ):
            raise Stage5RouterError(
                "Router source FBDP-AD ablation disagrees with --fbdp-ablation"
            )
        if args.feature_view == "normal_only" and args.variant != "normal_only":
            raise Stage5RouterError(
                "normal_only feature view requires variant=normal_only"
            )
        if args.supervision == "soft_teacher" and not args.teacher_data:
            raise Stage5RouterError("soft_teacher supervision requires --teacher-data")
        if args.teacher_data and args.supervision != "soft_teacher":
            raise Stage5RouterError("--teacher-data requires --supervision soft_teacher")
        input_hashes = {
            "router_features": artifact.sha256,
            "fold_manifest": file_sha256(args.fold_manifest),
            "routing_matrix": file_sha256(args.routing_matrix),
        }
        if args.teacher_data:
            input_hashes["teacher_data"] = file_sha256(args.teacher_data)
        if args.capability_bank:
            input_hashes["capability_bank"] = file_sha256(args.capability_bank)
        np = _numpy()
        index_by_task = {
            task_id: index for index, task_id in enumerate(artifact.task_ids)
        }
        split_ids = {
            split: tuple(
                row["task_id"]
                for row in manifest_rows
                if row["split"] == split
            )
            for split in _SPLITS
        }
        split_indices = {
            split: np.asarray(
                [index_by_task[task_id] for task_id in split_ids[split]],
                dtype=np.int64,
            )
            for split in _SPLITS
        }
        supervision_task_ids = split_ids["train"] + split_ids["val"]
        supervision_manifest = {
            task_id: manifest_by_task[task_id]
            for task_id in supervision_task_ids
        }
        supervision_outcomes, experts = read_evaluator_outcomes(
            args.routing_matrix, supervision_manifest
        )
        validation_runtime_tradeoff = _nonnegative_float_value(
            args.validation_runtime_tradeoff,
            "validation_runtime_tradeoff",
        )
        train_weight_by_task = balanced_query_task_weights(
            split_ids["train"], manifest_by_task
        )
        validation_weight_by_task = balanced_query_task_weights(
            split_ids["val"], manifest_by_task
        )
        validation_sample_weights = np.asarray(
            [validation_weight_by_task[task_id] for task_id in split_ids["val"]],
            dtype=np.float64,
        )
        score_calibrations = fit_train_score_calibrations(
            split_ids["train"],
            supervision_outcomes,
            experts,
            task_weights=train_weight_by_task,
            fit_categories=sorted(
                {
                    manifest_by_task[task_id]["category"]
                    for task_id in split_ids["train"]
                }
            ),
        )
        empirical_runtime_scale = fit_train_runtime_scale(
            split_ids["train"], supervision_outcomes, experts
        )
        train_empirical_losses = empirical_routing_losses(
            split_ids["train"],
            supervision_outcomes,
            experts,
            score_calibrations,
            runtime_scale=empirical_runtime_scale,
            runtime_tradeoff=validation_runtime_tradeoff,
        )
        validation_empirical_losses = empirical_routing_losses(
            split_ids["val"],
            supervision_outcomes,
            experts,
            score_calibrations,
            runtime_scale=empirical_runtime_scale,
            runtime_tradeoff=validation_runtime_tradeoff,
        )
        train_sample_weights = None
        if args.supervision == "soft_teacher":
            train_targets, train_sample_weights, teacher_experts = read_soft_teacher_supervision(
                args.teacher_data,
                task_ids=split_ids["train"],
                fold=args.fold,
                train_categories={
                    row["category"] for row in manifest_rows if row["split"] == "train"
                },
            )
            if teacher_experts != experts:
                raise Stage5RouterError("teacher and routing matrix expert sets disagree")
            expected_weights = np.asarray(
                [train_weight_by_task[task_id] for task_id in split_ids["train"]],
                dtype=np.float64,
            )
            if not bool(
                np.allclose(
                    train_sample_weights,
                    expected_weights,
                    rtol=1e-10,
                    atol=1e-10,
                )
            ):
                raise Stage5RouterError(
                    "teacher weights disagree with fold grouped/inverse-frequency weights"
                )
        else:
            train_targets = np.argmin(train_empirical_losses, axis=1)
            train_sample_weights = np.asarray(
                [train_weight_by_task[task_id] for task_id in split_ids["train"]],
                dtype=np.float64,
            )
        validation_targets = np.argmin(validation_empirical_losses, axis=1)
        fit = fit_linear_router(
            artifact.values[split_indices["train"]],
            train_targets,
            artifact.values[split_indices["val"]],
            validation_targets,
            feature_names=artifact.feature_names,
            experts=experts,
            l2_grid=args.l2_grid,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            train_sample_weights=train_sample_weights,
            validation_sample_weights=validation_sample_weights,
        )

        capability_policy: dict[str, Any] | None = None
        capability_bank: dict[str, Any] | None = None
        if args.capability_bank:
            capability_bank = read_capability_bank(args.capability_bank)
            train_categories = {
                row["category"] for row in manifest_rows if row["split"] == "train"
            }
            if capability_bank.get("fold") != args.fold or set(
                capability_bank.get("train_categories", ())
            ) != train_categories:
                raise Stage5RouterError("capability bank fold/train categories disagree")
            capability_policy = select_capability_policy(
                fit.model.predict_proba(artifact.values[split_indices["val"]]),
                artifact.values[split_indices["val"]],
                split_ids["val"],
                manifest_by_task,
                experts,
                artifact.feature_names,
                capability_bank,
                capability_weight_grid=args.capability_weight_grid,
                uncertainty_weight_grid=args.uncertainty_weight_grid,
                cost_weight_grid=args.cost_weight_grid,
                capability_mode=args.capability_mode,
                enabled_skills=args.capability_skills,
                uncertainty_mode=args.uncertainty_mode,
                validation_empirical_losses=validation_empirical_losses,
                validation_runtime_tradeoff=validation_runtime_tradeoff,
                validation_sample_weights=validation_sample_weights,
            )

        # Freeze predictions before test labels are passed to evaluator metrics.
        test_values = artifact.values[split_indices["test"]]
        probabilities = fit.model.predict_proba(test_values)
        if capability_bank is None or capability_policy is None:
            selected_indices = np.argmax(probabilities, axis=1)
            selection_objectives = -np.log(np.maximum(probabilities, 1e-12))
        else:
            selection_objectives = capability_objectives(
                probabilities,
                test_values,
                split_ids["test"],
                manifest_by_task,
                experts,
                artifact.feature_names,
                capability_bank,
                capability_weight=capability_policy["capability_weight"],
                uncertainty_weight=capability_policy["uncertainty_weight"],
                cost_weight=capability_policy["cost_weight"],
                capability_mode=args.capability_mode,
                enabled_skills=args.capability_skills,
                uncertainty_mode=args.uncertainty_mode,
            )
            selected_indices = np.argmin(selection_objectives, axis=1)
        selected = {
            task_id: experts[int(index)]
            for task_id, index in zip(split_ids["test"], selected_indices)
        }
        model_path = _atomic_write_json(
            output_dir / MODEL_NAME, fit.model.to_dict()
        )
        prediction_rows = [
            {
                "protocol_version": STAGE5_ROUTER_PREDICTION_PROTOCOL_VERSION,
                "task_id": task_id,
                "fold": args.fold,
                "variant": args.variant,
                "selected_expert": selected[task_id],
                "expert_probabilities": {
                    expert: float(probability)
                    for expert, probability in zip(experts, row)
                },
                "selection_objectives": {
                    expert: float(objective)
                    for expert, objective in zip(experts, objective_row)
                },
            }
            for task_id, row, objective_row in zip(
                split_ids["test"], probabilities, selection_objectives
            )
        ]
        predictions_path = _atomic_write_jsonl(
            output_dir / PREDICTIONS_NAME, prediction_rows
        )

        # The evaluator channel is opened only after label-free test
        # predictions have been persisted.  Test labels and expert scores
        # therefore cannot affect model fitting, model selection, or routing.
        test_manifest = {
            task_id: manifest_by_task[task_id]
            for task_id in split_ids["test"]
        }
        test_outcomes, test_experts = read_evaluator_outcomes(
            args.routing_matrix, test_manifest
        )
        if test_experts != experts:
            raise Stage5RouterError(
                "train/validation and test evaluator expert sets disagree"
            )
        test_empirical_losses = empirical_routing_losses(
            split_ids["test"],
            test_outcomes,
            experts,
            score_calibrations,
            runtime_scale=empirical_runtime_scale,
            runtime_tradeoff=validation_runtime_tradeoff,
        )
        global_best = select_global_best_expert(
            split_ids["train"], supervision_outcomes, experts
        )
        learned_metrics, per_task_rows = evaluate_selection(
            split_ids["test"],
            test_outcomes,
            selected,
            experts,
            score_calibrations=score_calibrations,
            empirical_losses=test_empirical_losses,
        )
        global_metrics, _ = evaluate_selection(
            split_ids["test"],
            test_outcomes,
            {task_id: global_best for task_id in split_ids["test"]},
            experts,
            score_calibrations=score_calibrations,
            empirical_losses=test_empirical_losses,
        )
        oracle_selection = {
            task_id: experts[int(index)]
            for task_id, index in zip(
                split_ids["test"], np.argmin(test_empirical_losses, axis=1)
            )
        }
        oracle_metrics, _ = evaluate_selection(
            split_ids["test"],
            test_outcomes,
            oracle_selection,
            experts,
            score_calibrations=score_calibrations,
            empirical_losses=test_empirical_losses,
        )
        metrics = {
            "protocol_version": STAGE5_ROUTER_METRICS_PROTOCOL_VERSION,
            "fold": args.fold,
            "variant": args.variant,
            "feature_view": args.feature_view,
            "model_kind": "class_balanced_linear_softmax_soft_targets",
            "standardization": "train_only_feature_mean_std",
            "supervision_target": args.supervision,
            "test_evaluation_scope": (
                "evaluator_only_after_label_free_predictions"
            ),
            "runtime_source": "evaluator_only_routing_matrix.runtime_ms",
            "source_ablation": artifact.source_ablation,
            "source_fbdp_ablation": artifact.source_fbdp_ablation,
            "source_feature_view": artifact.source_feature_view,
            "source_feature_protocol": artifact.source_protocol_version,
            "feature_dimension": len(artifact.feature_names),
            "experts": list(experts),
            "task_counts": {
                split: len(split_ids[split]) for split in _SPLITS
            },
            "selected_l2": fit.model.l2,
            "validation_frontier": list(fit.frontier),
            "capability_policy": capability_policy,
            "capability_mode": args.capability_mode,
            "capability_skills": list(args.capability_skills),
            "uncertainty_mode": args.uncertainty_mode,
            "validation_empirical_utility": (
                "train_calibrated_zero_one_error_plus_normalized_runtime"
            ),
            "validation_runtime_tradeoff": validation_runtime_tradeoff,
            "repeat_weighting": (
                "equal_category_equal_query_inverse_variant_frequency"
            ),
            "evaluation_score_space": (
                "train_only_per_expert_weighted_platt_probability"
            ),
            "score_calibrations": {
                expert: score_calibrations[expert].to_dict()
                for expert in experts
            },
            "per_task_metrics_protocol_version": (
                STAGE5_ROUTER_PER_TASK_PROTOCOL_VERSION
            ),
            "global_best_expert_from_train": global_best,
            "methods": {
                "learned_router": learned_metrics,
                "global_best": global_metrics,
                "sample_oracle": oracle_metrics,
            },
        }
        metrics_path = _atomic_write_json(
            evaluator_dir / METRICS_NAME, metrics
        )
        per_task_path = write_per_task_metrics(
            evaluator_dir / PER_TASK_METRICS_NAME, per_task_rows
        )
        outputs = {
            "model": str(model_path),
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "per_task_metrics": str(per_task_path),
        }
        runtime = {
            "feature_dimension": len(artifact.feature_names),
            "experts": list(experts),
            "task_counts": metrics["task_counts"],
            "selected_l2": fit.model.l2,
            "training_backend": fit.model.backend,
            "supervision": args.supervision,
            "capability_policy": capability_policy,
            "uncertainty_mode": args.uncertainty_mode,
            "validation_runtime_tradeoff": validation_runtime_tradeoff,
            "empirical_runtime_scale_ms": empirical_runtime_scale,
            "source_feature_view": artifact.source_feature_view,
            "source_bir_ablation": artifact.source_ablation,
            "source_fbdp_ablation": artifact.source_fbdp_ablation,
            "source_feature_protocol": artifact.source_protocol_version,
            "runtime_source": "evaluator_only_routing_matrix.runtime_ms",
            "runtime_complete": True,
        }
    except Exception as exc:
        failures.append(
            {"code": type(exc).__name__, "message": str(exc)}
        )

    failures_path = _atomic_write_json(output_dir / FAILURES_NAME, failures)
    output_hashes = {
        name: file_sha256(path)
        for name, path in outputs.items()
        if path is not None and Path(path).is_file()
    }
    output_hashes["failures"] = file_sha256(failures_path)
    run_record = {
        "protocol_version": STAGE5_ROUTER_RUN_PROTOCOL_VERSION,
        "run_kind": "stage5_router_train_evaluate",
        "ok": not failures,
        "config": {
            "router_features": args.router_features,
            "fold_manifest": args.fold_manifest,
            "routing_matrix": args.routing_matrix,
            "teacher_data": args.teacher_data,
            "capability_bank": args.capability_bank,
            "supervision": args.supervision,
            "fold": args.fold,
            "variant": args.variant,
            "feature_view": args.feature_view,
            "bir_ablation": args.bir_ablation,
            "fbdp_ablation": args.fbdp_ablation,
            "device": args.device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "l2_grid": list(args.l2_grid),
            "seed": args.seed,
            "capability_weight_grid": list(args.capability_weight_grid),
            "uncertainty_weight_grid": list(args.uncertainty_weight_grid),
            "cost_weight_grid": list(args.cost_weight_grid),
            "capability_mode": args.capability_mode,
            "capability_skills": list(args.capability_skills),
            "uncertainty_mode": args.uncertainty_mode,
            "validation_runtime_tradeoff": args.validation_runtime_tradeoff,
            "validation_selection_metric": (
                "train_calibrated_zero_one_error_plus_normalized_runtime"
            ),
            "repeat_weighting": (
                "equal_category_equal_query_inverse_variant_frequency"
            ),
            "model_kind": "class_balanced_linear_softmax_soft_targets",
            "standardization": "train_only_feature_mean_std",
            "supervision_target": args.supervision,
            "test_prediction_contract": (
                "label_free_and_frozen_before_evaluator_join"
            ),
        },
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "seed": args.seed,
        "git_commit": git_commit(),
        "environment": environment_record(),
        "runtime_statistics": runtime,
        "outputs": {**outputs, "failures": str(failures_path)},
        "predictions": outputs["predictions"],
        "failures": failures,
    }
    _atomic_write_json(output_dir / RUN_RECORD_NAME, run_record)
    if failures:
        print(f"Stage 5 Router failed: {failures[0]['message']}", file=sys.stderr)
        return 1
    print(
        "Stage 5 Router PASS: "
        f"fold={args.fold} variant={args.variant} "
        f"metrics={outputs['metrics']}"
    )
    return 0


def read_fold_manifest(path: str | Path, fold: str) -> list[dict[str, str]]:
    source = Path(path)
    required = (
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
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise Stage5RouterError(
                f"{source} is missing fold columns: {missing}"
            )
        for line_number, row in enumerate(reader, start=2):
            clean = {key: (value or "").strip() for key, value in row.items()}
            if clean["fold"] != fold:
                continue
            if clean["split"] not in _SPLITS:
                raise Stage5RouterError(
                    f"{source}:{line_number} has invalid split"
                )
            task_id = clean["task_id"]
            if not task_id or task_id in seen:
                raise Stage5RouterError(
                    f"{source}:{line_number} has invalid/duplicate task_id"
                )
            seen.add(task_id)
            rows.append({name: clean[name] for name in required})
    if not rows or {row["split"] for row in rows} != set(_SPLITS):
        raise Stage5RouterError(
            f"{source} does not contain complete train/val/test rows for {fold}"
        )
    category_splits: dict[str, set[str]] = {}
    support_categories: dict[str, set[str]] = {}
    sample_categories: dict[str, set[str]] = {}
    for row in rows:
        category_splits.setdefault(row["category"], set()).add(row["split"])
        support_categories.setdefault(row["support_set_id"], set()).add(
            row["category"]
        )
        sample_categories.setdefault(row["sample_id"], set()).add(
            row["category"]
        )
    if any(len(values) != 1 for values in category_splits.values()):
        raise Stage5RouterError(
            f"{source} assigns a category to multiple splits in {fold}"
        )
    if any(len(values) != 1 for values in support_categories.values()):
        raise Stage5RouterError(
            f"{source} reuses a support_set_id across categories in {fold}"
        )
    if any(len(values) != 1 for values in sample_categories.values()):
        raise Stage5RouterError(
            f"{source} reuses a sample_id across categories in {fold}"
        )
    return rows


def read_router_features(
    path: str | Path,
    manifest_by_task: Mapping[str, Mapping[str, str]],
    *,
    feature_view: str,
) -> FeatureArtifact:
    np = _numpy()
    source = Path(path)
    task_ids = tuple(manifest_by_task)
    index_by_task = {task_id: index for index, task_id in enumerate(task_ids)}
    matrix = None
    source_names: tuple[str, ...] | None = None
    selected_names: tuple[str, ...] | None = None
    selected_indices = None
    source_ablation: str | None = None
    source_feature_view: str | None = None
    source_fbdp_ablation: str | None = None
    source_protocol_version: str | None = None
    seen: set[str] = set()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            digest.update(line)
            if not line.strip():
                raise Stage5RouterError(
                    f"{source}:{line_number} is blank"
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Stage5RouterError(
                    f"{source}:{line_number} is invalid JSON"
                ) from exc
            row_protocol = str(row.get("protocol_version", ""))
            if row_protocol not in ROUTER_FEATURE_BUNDLE_COMPATIBLE_PROTOCOL_VERSIONS:
                raise Stage5RouterError(
                    f"{source}:{line_number} has incompatible feature protocol"
                )
            task_id = str(row.get("task_id", ""))
            if task_id not in index_by_task or task_id in seen:
                raise Stage5RouterError(
                    f"{source}:{line_number} has unexpected/duplicate task_id"
                )
            seen.add(task_id)
            manifest = manifest_by_task[task_id]
            for field in (
                "dataset",
                "category",
                "k_shot",
                "seed",
                "support_set_id",
            ):
                if str(row.get(field, "")) != str(manifest[field]):
                    raise Stage5RouterError(
                        f"{source}:{line_number} feature provenance "
                        f"disagrees on {field}"
                    )
            names = tuple(str(item) for item in row.get("feature_names", ()))
            if (
                row_protocol == "stage5.router_feature_bundle.v2"
                and any(name.startswith("fbdp_") for name in names)
            ):
                raise Stage5RouterError(
                    f"{source}:{line_number} uses FBDP fields with the legacy protocol"
                )
            values = row.get("values")
            if selected_names is None:
                _assert_feature_names_safe(names)
                source_names = names
                if feature_view == "all":
                    selected_indices = np.arange(len(names), dtype=np.int64)
                else:
                    prefixes = {
                        "normal_only": ("normal_",),
                        "normal_bir": ("normal_", "bir_"),
                        "normal_fbdp": ("normal_", "fbdp_"),
                        "normal_bir_fbdp": ("normal_", "bir_", "fbdp_"),
                    }.get(feature_view)
                    if prefixes is None:
                        raise Stage5RouterError("unknown Router feature view")
                    selected_indices = np.asarray(
                        [
                            index
                            for index, name in enumerate(names)
                            if name.startswith(prefixes)
                        ],
                        dtype=np.int64,
                    )
                    required = set(prefixes)
                    observed = {
                        prefix
                        for prefix in prefixes
                        if any(name.startswith(prefix) for name in names)
                    }
                    if observed != required:
                        raise Stage5RouterError(
                            f"Router artifact cannot provide feature_view={feature_view!r}"
                        )
                if selected_indices.size == 0:
                    raise Stage5RouterError("Router feature view is empty")
                selected_names = tuple(names[int(index)] for index in selected_indices)
                matrix = np.empty(
                    (len(task_ids), len(selected_names)), dtype=np.float32
                )
                source_ablation = str(row.get("ablation_name", ""))
                source_feature_view = str(
                    row.get("feature_view") or _infer_feature_view(names)
                )
                source_fbdp_ablation = str(
                    row.get("fbdp_ablation_name", "not_applicable")
                )
                source_protocol_version = row_protocol
            elif names != source_names:
                raise Stage5RouterError("Router feature schema changed within file")
            if row_protocol != source_protocol_version:
                raise Stage5RouterError("Router feature file mixes protocol versions")
            if str(row.get("ablation_name", "")) != source_ablation:
                raise Stage5RouterError("Router feature file mixes ablations")
            row_feature_view = str(
                row.get("feature_view") or _infer_feature_view(names)
            )
            if row_feature_view != source_feature_view:
                raise Stage5RouterError("Router feature file mixes feature views")
            if str(row.get("fbdp_ablation_name", "not_applicable")) != source_fbdp_ablation:
                raise Stage5RouterError(
                    "Router feature file mixes FBDP-AD ablations"
                )
            try:
                value_array = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise Stage5RouterError(
                    f"{source}:{line_number} has invalid feature values"
                ) from exc
            if value_array.ndim != 1 or value_array.size != len(names):
                raise Stage5RouterError(
                    f"{source}:{line_number} feature values are misaligned"
                )
            if not bool(np.all(np.isfinite(value_array))):
                raise Stage5RouterError(
                    f"{source}:{line_number} has non-finite features"
                )
            assert matrix is not None and selected_indices is not None
            matrix[index_by_task[task_id]] = value_array[selected_indices]
    if seen != set(task_ids) or matrix is None or selected_names is None:
        missing = len(set(task_ids) - seen)
        raise Stage5RouterError(
            f"Router feature task coverage disagrees with fold manifest; missing={missing}"
        )
    return FeatureArtifact(
        task_ids=task_ids,
        feature_names=selected_names,
        values=matrix,
        source_ablation=source_ablation or "",
        sha256=digest.hexdigest(),
        source_feature_view=source_feature_view or "",
        source_fbdp_ablation=source_fbdp_ablation or "not_applicable",
        source_protocol_version=source_protocol_version or "",
    )


def _infer_feature_view(names: Sequence[str]) -> str:
    has_bir = any(name.startswith("bir_") for name in names)
    has_fbdp = any(name.startswith("fbdp_") for name in names)
    if has_bir and has_fbdp:
        return "normal_bir_fbdp"
    if has_bir:
        return "normal_bir"
    if has_fbdp:
        return "normal_fbdp"
    return "normal_only"


def read_evaluator_outcomes(
    path: str | Path,
    manifest_by_task: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, TaskOutcome], tuple[str, ...]]:
    source = Path(path)
    staged: dict[str, dict[str, Any]] = {}
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        expert_column = "expert_name" if "expert_name" in fieldnames else "expert"
        score_column = "final_score" if "final_score" in fieldnames else "image_score"
        required = (
            "image_id",
            "dataset",
            "category",
            "support_set_id",
            "k_shot",
            "seed",
            expert_column,
            score_column,
            "runtime_ms",
            "label",
        )
        missing = [name for name in required if name not in fieldnames]
        if missing:
            raise Stage5RouterError(
                f"{source} is missing evaluator columns: {missing}"
            )
        for line_number, row in enumerate(reader, start=2):
            clean = {key: (value or "").strip() for key, value in row.items()}
            task_id = (
                f"{clean['image_id']}|{clean['dataset']}|{clean['category']}|"
                f"{clean['support_set_id']}|{clean['k_shot']}|{clean['seed']}"
            )
            if task_id not in manifest_by_task:
                continue
            if "status" in fieldnames and clean["status"] != "ok":
                raise Stage5RouterError(
                    f"{source}:{line_number} contains failed expert output"
                )
            label = _label(clean["label"], source, line_number)
            expert = clean[expert_column].lower()
            score = _finite_float(
                clean[score_column], source, line_number, score_column
            )
            if not clean.get("runtime_ms"):
                raise Stage5RouterError(
                    f"{source}:{line_number} is missing runtime_ms; rebuild the "
                    "evaluator-only Stage 3 routing matrix"
                )
            runtime = _nonnegative_float(
                clean["runtime_ms"], source, line_number, "runtime_ms"
            )
            current = staged.setdefault(
                task_id,
                {"label": label, "scores": {}, "runtimes": {}},
            )
            if current["label"] != label:
                raise Stage5RouterError(
                    f"{source}:{line_number} has conflicting task labels"
                )
            if expert in current["scores"]:
                raise Stage5RouterError(
                    f"{source}:{line_number} duplicates an expert/task row"
                )
            current["scores"][expert] = score
            current["runtimes"][expert] = runtime
    if set(staged) != set(manifest_by_task):
        raise Stage5RouterError(
            "evaluator outcome task coverage disagrees with fold manifest"
        )
    expert_sets = {tuple(sorted(value["scores"])) for value in staged.values()}
    if len(expert_sets) != 1:
        raise Stage5RouterError("evaluator outcomes have inconsistent expert sets")
    experts = next(iter(expert_sets))
    if len(experts) < 2:
        raise Stage5RouterError("Router evaluation requires at least two experts")
    outcomes = {
        task_id: TaskOutcome(
            label=int(value["label"]),
            scores=dict(value["scores"]),
            runtimes_ms=dict(value["runtimes"]),
        )
        for task_id, value in staged.items()
    }
    return outcomes, experts


def fit_train_score_calibrations(
    task_ids: Sequence[str],
    outcomes: Mapping[str, TaskOutcome],
    experts: Sequence[str],
    *,
    task_weights: Mapping[str, float],
    fit_categories: Sequence[str],
) -> dict[str, ExpertScoreCalibration]:
    """Fit scale-invariant expert decisions using train categories only."""

    if not task_ids or len(experts) < 2:
        raise Stage5RouterError(
            "score calibration requires train rows and at least two experts"
        )
    labels = [outcomes[task_id].label for task_id in task_ids]
    if set(labels) != {0, 1}:
        raise Stage5RouterError(
            "score calibration requires both train labels"
        )
    if set(task_weights) != set(task_ids):
        raise Stage5RouterError("score calibration weights are misaligned")
    weights = [float(task_weights[task_id]) for task_id in task_ids]
    if any(not math.isfinite(value) or value <= 0.0 for value in weights):
        raise Stage5RouterError("score calibration weights are invalid")
    result: dict[str, ExpertScoreCalibration] = {}
    for expert in experts:
        scores = [float(outcomes[task_id].scores[expert]) for task_id in task_ids]
        result[expert] = fit_expert_score_calibration(
            expert_name=expert,
            labels=labels,
            scores=scores,
            weights=weights,
            fit_categories=fit_categories,
        )
    return result


def balanced_query_task_weights(
    task_ids: Sequence[str],
    manifest_by_task: Mapping[str, Mapping[str, str]],
) -> dict[str, float]:
    """Match teacher category/query balancing for repeated K/seed rows."""

    if not task_ids:
        raise Stage5RouterError("cannot weight an empty task split")
    grouped: dict[str, dict[tuple[str, str, str], list[str]]] = {}
    for task_id in task_ids:
        if task_id not in manifest_by_task:
            raise Stage5RouterError("task weighting manifest coverage disagrees")
        row = manifest_by_task[task_id]
        category = str(row["category"])
        query_id = str(row.get("sample_id") or row.get("image_id") or "").strip()
        if not query_id:
            raise Stage5RouterError("task weighting requires an opaque query id")
        group = (str(row.get("dataset", "")), category, query_id)
        grouped.setdefault(category, {}).setdefault(group, []).append(task_id)
    raw: dict[str, float] = {}
    for query_groups in grouped.values():
        group_mass = 1.0 / len(query_groups)
        for variants in query_groups.values():
            per_variant = group_mass / len(variants)
            for task_id in variants:
                raw[task_id] = per_variant
    return {task_id: raw[task_id] for task_id in task_ids}


def fit_train_runtime_scale(
    task_ids: Sequence[str],
    outcomes: Mapping[str, TaskOutcome],
    experts: Sequence[str],
) -> float:
    values = [
        float(outcomes[task_id].runtimes_ms[expert])
        for task_id in task_ids
        for expert in experts
        if outcomes[task_id].runtimes_ms[expert] is not None
    ]
    if len(values) != len(task_ids) * len(experts):
        raise Stage5RouterError("train runtime calibration is incomplete")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise Stage5RouterError("train runtime calibration is invalid")
    maximum = max(values, default=0.0)
    return maximum if maximum > 1e-12 else 1.0


def empirical_routing_losses(
    task_ids: Sequence[str],
    outcomes: Mapping[str, TaskOutcome],
    experts: Sequence[str],
    calibrations: Mapping[str, ExpertScoreCalibration],
    *,
    runtime_scale: float,
    runtime_tradeoff: float,
) -> Any:
    """Build scale-invariant observed loss without comparing raw scores."""

    if set(calibrations) != set(experts):
        raise Stage5RouterError("score calibration expert set disagrees")
    if not math.isfinite(runtime_scale) or runtime_scale <= 0.0:
        raise Stage5RouterError("empirical runtime scale must be positive")
    tradeoff = _nonnegative_float_value(
        runtime_tradeoff, "runtime_tradeoff"
    )
    rows: list[list[float]] = []
    for task_id in task_ids:
        outcome = outcomes[task_id]
        row = []
        for expert in experts:
            anomaly_probability = calibrations[expert].predict(
                outcome.scores[expert]
            )
            classification_error = float(
                int(anomaly_probability >= 0.5) != outcome.label
            )
            runtime = outcome.runtimes_ms[expert]
            if runtime is None or not math.isfinite(float(runtime)) or float(runtime) < 0.0:
                raise Stage5RouterError("empirical routing runtime is invalid")
            row.append(
                classification_error
                + tradeoff * float(runtime) / runtime_scale
            )
        rows.append(row)
    return _numpy().asarray(rows, dtype=_numpy().float64)


def read_soft_teacher_supervision(
    path: str | Path,
    *,
    task_ids: Sequence[str],
    fold: str,
    train_categories: set[str],
) -> tuple[Any, Any, tuple[str, ...]]:
    """Load only distilled distributions/weights, never raw scores, into training."""

    rows = read_teacher_parquet(path)
    staged: dict[str, dict[str, float]] = {}
    weights: dict[str, float] = {}
    for row in rows:
        if str(row.get("fold")) != fold or str(row.get("category")) not in train_categories:
            raise Stage5RouterError("teacher fold/train category scope disagrees")
        task_id = str(row["task_id"])
        expert = str(row["expert_name"])
        probability = float(row["soft_utility_probability"])
        weight = float(row["sample_weight"])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise Stage5RouterError("teacher contains invalid soft probability")
        if not math.isfinite(weight) or weight <= 0.0:
            raise Stage5RouterError("teacher contains invalid sample weight")
        if expert in staged.setdefault(task_id, {}):
            raise Stage5RouterError("teacher duplicates a task/expert row")
        staged[task_id][expert] = probability
        previous = weights.setdefault(task_id, weight)
        if not math.isclose(previous, weight, rel_tol=1e-12, abs_tol=1e-12):
            raise Stage5RouterError("teacher weight changes across experts")
    if set(staged) != set(task_ids):
        raise Stage5RouterError("teacher coverage disagrees with train split")
    expert_sets = {tuple(sorted(values)) for values in staged.values()}
    if len(expert_sets) != 1:
        raise Stage5RouterError("teacher expert coverage is inconsistent")
    experts = next(iter(expert_sets))
    np = _numpy()
    distributions = []
    sample_weights = []
    for task_id in task_ids:
        values = staged[task_id]
        total = sum(values.values())
        if not math.isclose(total, 1.0, rel_tol=1e-7, abs_tol=1e-7):
            raise Stage5RouterError("teacher distribution does not sum to one")
        distributions.append([values[expert] / total for expert in experts])
        sample_weights.append(weights[task_id])
    return (
        np.asarray(distributions, dtype=np.float64),
        np.asarray(sample_weights, dtype=np.float64),
        experts,
    )


def select_capability_policy(
    validation_probabilities: Any,
    validation_values: Any,
    validation_task_ids: Sequence[str],
    manifest_by_task: Mapping[str, Mapping[str, str]],
    experts: Sequence[str],
    feature_names: Sequence[str],
    bank: Mapping[str, Any],
    *,
    capability_weight_grid: Sequence[float],
    uncertainty_weight_grid: Sequence[float],
    cost_weight_grid: Sequence[float],
    capability_mode: str = "conditional",
    enabled_skills: Sequence[str] = ("boundary", "fgbg", "lowshot", "texture"),
    uncertainty_mode: str = "entropy_scaled_lcb",
    validation_empirical_losses: Any,
    validation_runtime_tradeoff: float,
    validation_sample_weights: Any,
) -> dict[str, Any]:
    """Choose weights by a frozen, scale-invariant validation loss."""

    grids = (
        _nonnegative_grid(capability_weight_grid, "capability_weight_grid"),
        _nonnegative_grid(uncertainty_weight_grid, "uncertainty_weight_grid"),
        _nonnegative_grid(cost_weight_grid, "cost_weight_grid"),
    )
    np = _numpy()
    empirical_losses = np.asarray(
        validation_empirical_losses, dtype=np.float64
    )
    if empirical_losses.shape != (len(validation_task_ids), len(experts)):
        raise Stage5RouterError(
            "validation empirical loss matrix has invalid shape"
        )
    if not bool(np.all(np.isfinite(empirical_losses))):
        raise Stage5RouterError("validation empirical losses must be finite")
    sample_weights = np.asarray(validation_sample_weights, dtype=np.float64)
    if (
        sample_weights.shape != (len(validation_task_ids),)
        or not bool(np.all(np.isfinite(sample_weights)))
        or bool(np.any(sample_weights <= 0.0))
    ):
        raise Stage5RouterError("validation sample weights are invalid")
    weight_sum = float(np.sum(sample_weights))
    frozen_runtime_tradeoff = _nonnegative_float_value(
        validation_runtime_tradeoff, "validation_runtime_tradeoff"
    )
    frontier: list[dict[str, float]] = []
    for capability_weight in grids[0]:
        for uncertainty_weight in grids[1]:
            for cost_weight in grids[2]:
                objectives = capability_objectives(
                    validation_probabilities,
                    validation_values,
                    validation_task_ids,
                    manifest_by_task,
                    experts,
                    feature_names,
                    bank,
                    capability_weight=capability_weight,
                    uncertainty_weight=uncertainty_weight,
                    cost_weight=cost_weight,
                    capability_mode=capability_mode,
                    enabled_skills=enabled_skills,
                    uncertainty_mode=uncertainty_mode,
                )
                indices = np.argmin(objectives, axis=1)
                agreements = []
                regrets = []
                selected_losses = []
                for row_index, selected_index in enumerate(indices):
                    best = int(np.argmin(empirical_losses[row_index]))
                    selected_loss = float(
                        empirical_losses[row_index, int(selected_index)]
                    )
                    best_loss = float(empirical_losses[row_index, best])
                    agreements.append(float(int(selected_index) == best))
                    regrets.append(selected_loss - best_loss)
                    selected_losses.append(selected_loss)
                frontier.append(
                    {
                        "capability_weight": capability_weight,
                        "uncertainty_weight": uncertainty_weight,
                        "cost_weight": cost_weight,
                        "validation_selection_accuracy": float(
                            np.dot(sample_weights, agreements) / weight_sum
                        ),
                        "validation_oracle_regret": float(
                            np.dot(sample_weights, regrets) / weight_sum
                        ),
                        "validation_empirical_loss": (
                            float(
                                np.dot(sample_weights, selected_losses)
                                / weight_sum
                            )
                        ),
                    }
                )
    selected = min(
        frontier,
        key=lambda row: (
            row["validation_empirical_loss"],
            row["validation_oracle_regret"],
            -row["validation_selection_accuracy"],
            row["capability_weight"] + row["uncertainty_weight"] + row["cost_weight"],
            row["capability_weight"],
            row["uncertainty_weight"],
            row["cost_weight"],
        ),
    )
    return {
        **selected,
        "selection_rule": "argmin_router_risk_plus_capability_uncertainty_cost",
        "selection_metric": (
            "mean_train_calibrated_zero_one_error_plus_normalized_runtime"
        ),
        "validation_runtime_tradeoff": frozen_runtime_tradeoff,
        "repeat_weighting": (
            "equal_category_equal_query_inverse_variant_frequency"
        ),
        "hyperparameter_split": "validation_only",
        "capability_mode": capability_mode,
        "enabled_skills": list(enabled_skills),
        "uncertainty_mode": uncertainty_mode,
        "frontier": frontier,
    }


def capability_objectives(
    probabilities: Any,
    feature_values: Any,
    task_ids: Sequence[str],
    manifest_by_task: Mapping[str, Mapping[str, str]],
    experts: Sequence[str],
    feature_names: Sequence[str],
    bank: Mapping[str, Any],
    *,
    capability_weight: float,
    uncertainty_weight: float,
    cost_weight: float,
    capability_mode: str = "conditional",
    enabled_skills: Sequence[str] = ("boundary", "fgbg", "lowshot", "texture"),
    uncertainty_mode: str = "entropy_scaled_lcb",
) -> Any:
    """Form the one-call conditional expert objective without evaluator inputs."""

    np = _numpy()
    probability_matrix = np.asarray(probabilities, dtype=np.float64)
    values = np.asarray(feature_values, dtype=np.float64)
    if len(experts) < 2:
        raise Stage5RouterError(
            "capability objectives require at least two experts"
        )
    if probability_matrix.shape != (len(task_ids), len(experts)):
        raise Stage5RouterError("Router probabilities have invalid shape")
    if values.ndim != 2 or values.shape[0] != len(task_ids) or values.shape[1] != len(feature_names):
        raise Stage5RouterError("capability feature matrix has invalid shape")
    profiles = bank.get("profiles", {})
    if set(profiles) != set(experts):
        raise Stage5RouterError("capability bank expert set disagrees")
    if capability_mode not in {"conditional", "static"}:
        raise Stage5RouterError("capability_mode must be conditional or static")
    if uncertainty_mode not in {"entropy_scaled_lcb", "legacy_interval_width"}:
        raise Stage5RouterError("unsupported uncertainty_mode")
    enabled = set(enabled_skills)
    if not enabled.issubset({"boundary", "fgbg", "lowshot", "texture"}):
        raise Stage5RouterError("enabled_skills contains an unsupported capability")
    thresholds = bank.get("difficulty_thresholds", {})
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    boundary_names = (
        "bir_query_bai",
        "bir_query_support_boundary_shift",
        "bir_absolute_boundary_shift",
    )
    fgbg_names = ("fbdp_fbc", "fbdp_foreground_background_confusion")
    texture_names = (
        "normal_niv_0002",
        "normal_niv_2",
        "normal_niv_structure",
        "normal_niv_texture",
        "normal_query_texture_complexity",
    )
    latency_values: list[float] = []
    failure_rates: list[float] = []
    for expert in experts:
        profile = profiles[expert]
        raw_latency = profile.get("latency_p95")
        if raw_latency is None:
            raw_latency = profile.get("latency")
        try:
            latency = float(raw_latency)
            failure_rate = float(profile.get("failure_rate"))
        except (TypeError, ValueError) as exc:
            raise Stage5RouterError(
                f"capability profile {expert!r} lacks numeric latency/failure_rate"
            ) from exc
        if not math.isfinite(latency) or latency <= 0.0:
            raise Stage5RouterError(
                f"capability profile {expert!r} requires positive train-only latency"
            )
        if not math.isfinite(failure_rate) or not 0.0 <= failure_rate <= 1.0:
            raise Stage5RouterError(
                f"capability profile {expert!r} has invalid failure_rate"
            )
        latency_values.append(latency)
        failure_rates.append(failure_rate)
    latency_scale = max(latency_values)
    result = np.zeros_like(probability_matrix)
    for row_index, task_id in enumerate(task_ids):
        row_probabilities = probability_matrix[row_index]
        if (
            not np.all(np.isfinite(row_probabilities))
            or np.any(row_probabilities < 0.0)
            or not math.isclose(
                float(np.sum(row_probabilities)),
                1.0,
                rel_tol=1e-7,
                abs_tol=1e-7,
            )
        ):
            raise Stage5RouterError("Router probabilities must be finite and sum to one")
        predictive_entropy = -sum(
            float(probability) * math.log(max(float(probability), 1e-12))
            for probability in row_probabilities
        ) / math.log(len(experts))
        predictive_entropy = min(max(predictive_entropy, 0.0), 1.0)
        boundary = _mean_named_feature(values[row_index], name_to_index, boundary_names, absolute=True)
        fgbg = _mean_named_feature(values[row_index], name_to_index, fgbg_names)
        texture = _mean_named_feature(values[row_index], name_to_index, texture_names, absolute=True)
        k_shot = int(manifest_by_task[task_id]["k_shot"])
        for expert_index, expert in enumerate(experts):
            profile = profiles[expert]
            active_skills = [float(profile["overall_skill"])]
            interval_names = ["overall_skill"]
            if (
                capability_mode == "conditional"
                and "boundary" in enabled
                and boundary is not None
                and boundary >= float(thresholds["boundary"])
            ):
                active_skills.append(float(profile["boundary_skill"]))
                interval_names.append("boundary_skill")
            if (
                capability_mode == "conditional"
                and "fgbg" in enabled
                and fgbg is not None
                and fgbg >= float(thresholds["fgbg"])
            ):
                active_skills.append(float(profile["fgbg_skill"]))
                interval_names.append("fgbg_skill")
            if (
                capability_mode == "conditional"
                and "lowshot" in enabled
                and k_shot == int(bank["lowshot_k"])
            ):
                active_skills.append(float(profile["lowshot_skill"]))
                interval_names.append("lowshot_skill")
            texture_threshold = thresholds.get("texture")
            if (
                texture is not None
                and capability_mode == "conditional"
                and "texture" in enabled
                and texture_threshold is not None
                and profile.get("texture_skill") is not None
                and texture >= float(texture_threshold)
            ):
                active_skills.append(float(profile["texture_skill"]))
                interval_names.append("texture_skill")
            capability_risk = -math.log(max(sum(active_skills) / len(active_skills), 1e-12))
            intervals = profile.get("confidence_intervals", {})
            uncertainty = _capability_uncertainty_risk(
                profile,
                active_skills,
                interval_names,
                predictive_entropy=predictive_entropy,
                mode=uncertainty_mode,
            )
            latency = latency_values[expert_index] / latency_scale
            reliability_cost = latency + failure_rates[expert_index]
            result[row_index, expert_index] = (
                -math.log(max(probability_matrix[row_index, expert_index], 1e-12))
                + capability_weight * capability_risk
                + uncertainty_weight * uncertainty
                + cost_weight * reliability_cost
            )
    return result


def _capability_uncertainty_risk(
    profile: Mapping[str, Any],
    active_skills: Sequence[float],
    interval_names: Sequence[str],
    *,
    predictive_entropy: float,
    mode: str,
) -> float:
    """Return an expert- and query-specific ECPB uncertainty penalty.

    ``entropy_scaled_lcb`` decomposes conservative capability risk into the
    point-estimate risk plus an uncertainty increment.  The increment is the
    gap to the category-bootstrap lower confidence bound, activated in
    proportion to the Router's normalized predictive entropy for this query.
    A failure-rate upper-bound excess is included as reliability uncertainty.
    All profile statistics were fitted from the current fold's train
    categories; no evaluator outcome is consumed here.
    """

    intervals = profile.get("confidence_intervals", {})
    if not isinstance(intervals, Mapping):
        raise Stage5RouterError("capability confidence_intervals must be a mapping")
    if mode == "legacy_interval_width":
        widths = []
        for name in interval_names:
            if name not in intervals:
                continue
            lower, upper = _confidence_bounds(intervals[name], name)
            widths.append(upper - lower)
        return sum(widths) / len(widths) if widths else 0.0

    point_skill = sum(active_skills) / len(active_skills)
    lower_skills: list[float] = []
    for point, name in zip(active_skills, interval_names):
        if name not in intervals:
            lower_skills.append(float(point))
            continue
        lower, _ = _confidence_bounds(intervals[name], name)
        # Bootstrap intervals may not be perfectly centred on the point
        # estimate.  Only uncertainty below the estimate is penalized.
        lower_skills.append(min(float(point), lower))
    lower_skill = sum(lower_skills) / len(lower_skills)
    lcb_risk_gap = max(
        0.0,
        -math.log(max(lower_skill, 1e-12))
        + math.log(max(point_skill, 1e-12)),
    )

    failure_excess = 0.0
    if "failure_rate" in intervals:
        _, failure_upper = _confidence_bounds(
            intervals["failure_rate"], "failure_rate"
        )
        failure_rate = float(profile.get("failure_rate"))
        failure_excess = max(0.0, failure_upper - failure_rate)
    return predictive_entropy * (lcb_risk_gap + failure_excess)


def _confidence_bounds(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, Mapping):
        raise Stage5RouterError(f"confidence interval {name!r} must be a mapping")
    try:
        lower = float(value["lower"])
        upper = float(value["upper"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage5RouterError(
            f"confidence interval {name!r} requires numeric lower/upper bounds"
        ) from exc
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower < 0.0
        or upper > 1.0
        or upper < lower
    ):
        raise Stage5RouterError(f"confidence interval {name!r} is invalid")
    return lower, upper


def _mean_named_feature(
    row: Any,
    name_to_index: Mapping[str, int],
    candidates: Sequence[str],
    *,
    absolute: bool = False,
) -> float | None:
    values = [float(row[name_to_index[name]]) for name in candidates if name in name_to_index]
    if not values:
        return None
    if absolute:
        values = [abs(value) for value in values]
    return sum(values) / len(values)


def _nonnegative_grid(values: Sequence[float], name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise Stage5RouterError(f"{name} must be numeric") from exc
    if not result or any(not math.isfinite(value) or value < 0.0 for value in result) or tuple(sorted(set(result))) != result:
        raise Stage5RouterError(f"{name} must be sorted, unique, finite, and non-negative")
    return result


def _nonnegative_float_value(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Stage5RouterError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise Stage5RouterError(f"{name} must be finite and non-negative")
    return result


def select_global_best_expert(
    task_ids: Sequence[str],
    outcomes: Mapping[str, TaskOutcome],
    experts: Sequence[str],
) -> str:
    labels = [outcomes[task_id].label for task_id in task_ids]
    if len(set(labels)) != 2:
        raise Stage5RouterError(
            "train split must contain normal and anomalous samples"
        )
    ranked = []
    for expert in experts:
        scores = [outcomes[task_id].scores[expert] for task_id in task_ids]
        runtime_values = [
            outcomes[task_id].runtimes_ms[expert] for task_id in task_ids
        ]
        average_runtime = (
            sum(float(value) for value in runtime_values if value is not None)
            / len(runtime_values)
            if runtime_values and all(value is not None for value in runtime_values)
            else float("inf")
        )
        ranked.append((compute_auroc(labels, scores), average_runtime, expert))
    return min(ranked, key=lambda item: (-item[0], item[1], item[2]))[2]


def evaluate_selection(
    task_ids: Sequence[str],
    outcomes: Mapping[str, TaskOutcome],
    selected: Mapping[str, str],
    experts: Sequence[str],
    *,
    score_calibrations: Mapping[str, ExpertScoreCalibration],
    empirical_losses: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    np = _numpy()
    loss_matrix = np.asarray(empirical_losses, dtype=np.float64)
    if loss_matrix.shape != (
        len(task_ids), len(experts)
    ):
        raise Stage5RouterError("evaluation empirical loss matrix is misaligned")
    if set(score_calibrations) != set(experts):
        raise Stage5RouterError("evaluation score calibrations disagree")
    labels: list[int] = []
    image_scores: list[float] = []
    regrets: list[float] = []
    normalized_utilities: list[float] = []
    agreements: list[float] = []
    runtimes: list[float | None] = []
    counts = {expert: 0 for expert in experts}
    rows: list[dict[str, Any]] = []
    for row_index, task_id in enumerate(task_ids):
        outcome = outcomes[task_id]
        expert = selected.get(task_id)
        if expert not in experts:
            raise Stage5RouterError(
                f"selection is missing/invalid for task_id={task_id!r}"
            )
        utilities = {
            name: -float(loss_matrix[row_index, index])
            for index, name in enumerate(experts)
        }
        oracle_expert = max(experts, key=lambda name: utilities[name])
        oracle_utility = utilities[oracle_expert]
        worst_utility = min(utilities.values())
        selected_utility = utilities[expert]
        regret = oracle_utility - selected_utility
        span = oracle_utility - worst_utility
        normalized = 1.0 if span <= 1e-12 else (
            selected_utility - worst_utility
        ) / span
        labels.append(outcome.label)
        calibrated_score = score_calibrations[expert].predict(
            outcome.scores[expert]
        )
        image_scores.append(calibrated_score)
        regrets.append(regret)
        normalized_utilities.append(normalized)
        agreements.append(1.0 if expert == oracle_expert else 0.0)
        runtimes.append(outcome.runtimes_ms[expert])
        counts[expert] += 1
        rows.append(
            {
                "protocol_version": STAGE5_ROUTER_PER_TASK_PROTOCOL_VERSION,
                "evaluator_only": True,
                "task_id": task_id,
                "label": outcome.label,
                "selected_expert": expert,
                "oracle_expert": oracle_expert,
                "selected_score": outcome.scores[expert],
                "selected_calibrated_score": calibrated_score,
                "selected_empirical_utility": selected_utility,
                "oracle_empirical_utility": oracle_utility,
                "empirical_regret": regret,
                "normalized_utility": normalized,
                "runtime_ms": outcome.runtimes_ms[expert],
            }
        )
    if len(set(labels)) != 2:
        raise Stage5RouterError(
            "test split must contain normal and anomalous samples"
        )
    f1_max, f1_threshold = compute_f1_max(labels, image_scores)
    metrics = {
        "num_samples": len(task_ids),
        "selection_accuracy": sum(agreements) / len(agreements),
        "normalized_utility_mean": (
            sum(normalized_utilities) / len(normalized_utilities)
        ),
        "oracle_regret_mean": sum(regrets) / len(regrets),
        "image_auroc": compute_auroc(labels, image_scores),
        "image_ap": compute_average_precision(labels, image_scores),
        "image_f1_max_evaluator_only": f1_max,
        "image_f1_max_threshold_evaluator_only": f1_threshold,
        "average_runtime_ms": (
            sum(float(value) for value in runtimes) / len(runtimes)
            if runtimes and all(value is not None for value in runtimes)
            else None
        ),
        "selection_counts": counts,
        "expert_calls_per_task": 1,
        "failure_rate": 0.0,
        "score_space": "train_only_per_expert_weighted_platt_probability",
        "utility_space": "negative_train_calibrated_zero_one_runtime_loss",
    }
    return metrics, rows


def write_per_task_metrics(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> Path:
    columns = (
        "protocol_version",
        "evaluator_only",
        "task_id",
        "label",
        "selected_expert",
        "oracle_expert",
        "selected_score",
        "selected_calibrated_score",
        "selected_empirical_utility",
        "oracle_empirical_utility",
        "empirical_regret",
        "normalized_utility",
        "runtime_ms",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        column: _csv_value(row.get(column))
                        for column in columns
                    }
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        np = _numpy()
        record["numpy_version"] = np.__version__
    except Stage5RouterError:
        record["numpy_version"] = None
    try:
        import torch

        record["torch_version"] = torch.__version__
        record["cuda_available"] = bool(torch.cuda.is_available())
        record["cuda_devices"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    except ImportError:
        record["torch_version"] = None
        record["cuda_available"] = False
        record["cuda_devices"] = []
    return record


def _assert_feature_names_safe(names: Sequence[str]) -> None:
    if not names or len(set(names)) != len(names):
        raise Stage5RouterError("Router feature names must be non-empty and unique")
    for name in names:
        if is_forbidden_inference_feature_name(name):
            raise Stage5RouterError(
                f"Router feature {name!r} violates the inference isolation policy "
                "(forbidden tokens)"
            )


def _label(value: str, path: Path, line_number: int) -> int:
    if value in {"0", "0.0"}:
        return 0
    if value in {"1", "1.0"}:
        return 1
    raise Stage5RouterError(
        f"{path}:{line_number} has invalid evaluator label"
    )


def _finite_float(
    value: str, path: Path, line_number: int, field: str
) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise Stage5RouterError(
            f"{path}:{line_number} has invalid {field}"
        ) from exc
    if not math.isfinite(parsed):
        raise Stage5RouterError(
            f"{path}:{line_number} has non-finite {field}"
        )
    return parsed


def _nonnegative_float(
    value: str, path: Path, line_number: int, field: str
) -> float:
    parsed = _finite_float(value, path, line_number, field)
    if parsed < 0.0:
        raise Stage5RouterError(
            f"{path}:{line_number} has negative {field}"
        )
    return parsed


def _atomic_write_json(path: Path, value: Any) -> Path:
    return _atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )


def _atomic_write_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> Path:
    return _atomic_write_text(
        path,
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n"
            for row in rows
        ),
    )


def _atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _temporary_path(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise Stage5RouterError("numpy is required for Stage 5 Router runs") from exc
    return np


__all__ = [
    "FeatureArtifact",
    "STAGE5_ROUTER_METRICS_PROTOCOL_VERSION",
    "STAGE5_ROUTER_PER_TASK_PROTOCOL_VERSION",
    "STAGE5_ROUTER_PREDICTION_PROTOCOL_VERSION",
    "STAGE5_ROUTER_RUN_PROTOCOL_VERSION",
    "TaskOutcome",
    "evaluate_selection",
    "main",
    "parse_args",
    "read_evaluator_outcomes",
    "read_fold_manifest",
    "read_router_features",
    "select_global_best_expert",
]


if __name__ == "__main__":
    raise SystemExit(main())
