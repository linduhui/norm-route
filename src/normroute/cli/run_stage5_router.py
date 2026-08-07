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
from ..router.expert_bank import read_capability_bank
from ..router.teacher import read_teacher_parquet
from ..routing.quality_metrics import (
    compute_auroc,
    compute_average_precision,
    compute_f1_max,
)


STAGE5_ROUTER_RUN_PROTOCOL_VERSION = "stage5.router_run.v1"
STAGE5_ROUTER_PREDICTION_PROTOCOL_VERSION = "stage5.router_prediction.v1"
STAGE5_ROUTER_METRICS_PROTOCOL_VERSION = "stage5.router_metrics.v1"
MODEL_NAME = "router_model.json"
PREDICTIONS_NAME = "predictions.jsonl"
FAILURES_NAME = "failures.json"
RUN_RECORD_NAME = "run.json"
METRICS_NAME = "metrics.json"
PER_TASK_METRICS_NAME = "per_task_metrics.csv"
_SPLITS = ("train", "val", "test")
_FORBIDDEN_FEATURE_TOKENS = frozenset(
    {
        "label",
        "labels",
        "mask",
        "masks",
        "defect",
        "defecttype",
        "oracle",
        "expert",
        "utility",
        "utilities",
        "target",
        "targets",
        "score",
        "scores",
    }
)


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
    parser.add_argument("--uncertainty-weight-grid", type=float, nargs="+", default=(0.0, 0.1))
    parser.add_argument("--cost-weight-grid", type=float, nargs="+", default=(0.0, 0.05, 0.1))
    parser.add_argument(
        "--capability-mode",
        choices=("conditional", "static"),
        default="conditional",
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
        else:
            train_targets = oracle_targets(
                split_ids["train"], supervision_outcomes, experts
            )
        validation_targets = oracle_targets(
            split_ids["val"], supervision_outcomes, experts
        )
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
                supervision_outcomes,
                experts,
                artifact.feature_names,
                capability_bank,
                capability_weight_grid=args.capability_weight_grid,
                uncertainty_weight_grid=args.uncertainty_weight_grid,
                cost_weight_grid=args.cost_weight_grid,
                capability_mode=args.capability_mode,
                enabled_skills=args.capability_skills,
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
        global_best = select_global_best_expert(
            split_ids["train"], supervision_outcomes, experts
        )
        learned_metrics, per_task_rows = evaluate_selection(
            split_ids["test"], test_outcomes, selected, experts
        )
        global_metrics, _ = evaluate_selection(
            split_ids["test"],
            test_outcomes,
            {task_id: global_best for task_id in split_ids["test"]},
            experts,
        )
        oracle_selection = {
            task_id: experts[
                int(oracle_targets((task_id,), test_outcomes, experts)[0])
            ]
            for task_id in split_ids["test"]
        }
        oracle_metrics, _ = evaluate_selection(
            split_ids["test"], test_outcomes, oracle_selection, experts
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
            "source_feature_view": artifact.source_feature_view,
            "source_bir_ablation": artifact.source_ablation,
            "source_fbdp_ablation": artifact.source_fbdp_ablation,
            "source_feature_protocol": artifact.source_protocol_version,
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
            runtime = (
                _nonnegative_float(
                    clean["runtime_ms"], source, line_number, "runtime_ms"
                )
                if "runtime_ms" in fieldnames and clean.get("runtime_ms")
                else None
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


def oracle_targets(
    task_ids: Sequence[str],
    outcomes: Mapping[str, TaskOutcome],
    experts: Sequence[str],
) -> Any:
    np = _numpy()
    targets = []
    for task_id in task_ids:
        outcome = outcomes[task_id]
        utilities = [
            _oriented_utility(outcome.label, outcome.scores[expert])
            for expert in experts
        ]
        targets.append(max(range(len(experts)), key=lambda index: utilities[index]))
    return np.asarray(targets, dtype=np.int64)


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
    outcomes: Mapping[str, TaskOutcome],
    experts: Sequence[str],
    feature_names: Sequence[str],
    bank: Mapping[str, Any],
    *,
    capability_weight_grid: Sequence[float],
    uncertainty_weight_grid: Sequence[float],
    cost_weight_grid: Sequence[float],
    capability_mode: str = "conditional",
    enabled_skills: Sequence[str] = ("boundary", "fgbg", "lowshot", "texture"),
) -> dict[str, Any]:
    """Choose risk/capability/uncertainty/cost weights on validation only."""

    grids = (
        _nonnegative_grid(capability_weight_grid, "capability_weight_grid"),
        _nonnegative_grid(uncertainty_weight_grid, "uncertainty_weight_grid"),
        _nonnegative_grid(cost_weight_grid, "cost_weight_grid"),
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
                )
                np = _numpy()
                indices = np.argmin(objectives, axis=1)
                agreements = []
                regrets = []
                for task_id, selected_index in zip(validation_task_ids, indices):
                    outcome = outcomes[task_id]
                    utilities = [
                        _oriented_utility(outcome.label, outcome.scores[expert])
                        for expert in experts
                    ]
                    best = max(range(len(experts)), key=lambda index: (utilities[index], -index))
                    agreements.append(int(selected_index) == best)
                    regrets.append(utilities[best] - utilities[int(selected_index)])
                frontier.append(
                    {
                        "capability_weight": capability_weight,
                        "uncertainty_weight": uncertainty_weight,
                        "cost_weight": cost_weight,
                        "validation_selection_accuracy": sum(agreements) / len(agreements),
                        "validation_oracle_regret": sum(regrets) / len(regrets),
                    }
                )
    selected = min(
        frontier,
        key=lambda row: (
            -row["validation_selection_accuracy"],
            row["validation_oracle_regret"],
            row["capability_weight"] + row["uncertainty_weight"] + row["cost_weight"],
            row["capability_weight"],
            row["uncertainty_weight"],
            row["cost_weight"],
        ),
    )
    return {
        **selected,
        "selection_rule": "argmin_router_risk_plus_capability_uncertainty_cost",
        "hyperparameter_split": "validation_only",
        "capability_mode": capability_mode,
        "enabled_skills": list(enabled_skills),
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
) -> Any:
    """Form the one-call conditional expert objective without evaluator inputs."""

    np = _numpy()
    probability_matrix = np.asarray(probabilities, dtype=np.float64)
    values = np.asarray(feature_values, dtype=np.float64)
    if probability_matrix.shape != (len(task_ids), len(experts)):
        raise Stage5RouterError("Router probabilities have invalid shape")
    if values.ndim != 2 or values.shape[0] != len(task_ids) or values.shape[1] != len(feature_names):
        raise Stage5RouterError("capability feature matrix has invalid shape")
    profiles = bank.get("profiles", {})
    if set(profiles) != set(experts):
        raise Stage5RouterError("capability bank expert set disagrees")
    if capability_mode not in {"conditional", "static"}:
        raise Stage5RouterError("capability_mode must be conditional or static")
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
    latency_values = [
        float(profiles[expert].get("latency_p95") or profiles[expert].get("latency") or 0.0)
        for expert in experts
    ]
    latency_scale = max(max(latency_values), 1e-12)
    result = np.zeros_like(probability_matrix)
    for row_index, task_id in enumerate(task_ids):
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
            widths = [
                max(0.0, float(intervals[name]["upper"]) - float(intervals[name]["lower"]))
                for name in interval_names if name in intervals
            ]
            uncertainty = sum(widths) / len(widths) if widths else 0.0
            latency = latency_values[expert_index] / latency_scale
            reliability_cost = latency + float(profile["failure_rate"])
            result[row_index, expert_index] = (
                -math.log(max(probability_matrix[row_index, expert_index], 1e-12))
                + capability_weight * capability_risk
                + uncertainty_weight * uncertainty
                + cost_weight * reliability_cost
            )
    return result


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
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels: list[int] = []
    image_scores: list[float] = []
    regrets: list[float] = []
    normalized_utilities: list[float] = []
    agreements: list[float] = []
    runtimes: list[float | None] = []
    counts = {expert: 0 for expert in experts}
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        outcome = outcomes[task_id]
        expert = selected.get(task_id)
        if expert not in experts:
            raise Stage5RouterError(
                f"selection is missing/invalid for task_id={task_id!r}"
            )
        utilities = {
            name: _oriented_utility(outcome.label, outcome.scores[name])
            for name in experts
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
        image_scores.append(outcome.scores[expert])
        regrets.append(regret)
        normalized_utilities.append(normalized)
        agreements.append(1.0 if expert == oracle_expert else 0.0)
        runtimes.append(outcome.runtimes_ms[expert])
        counts[expert] += 1
        rows.append(
            {
                "evaluator_only": True,
                "task_id": task_id,
                "label": outcome.label,
                "selected_expert": expert,
                "oracle_expert": oracle_expert,
                "selected_score": outcome.scores[expert],
                "selected_oriented_utility": selected_utility,
                "oracle_oriented_utility": oracle_utility,
                "oracle_regret": regret,
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
    }
    return metrics, rows


def write_per_task_metrics(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> Path:
    columns = (
        "evaluator_only",
        "task_id",
        "label",
        "selected_expert",
        "oracle_expert",
        "selected_score",
        "selected_oriented_utility",
        "oracle_oriented_utility",
        "oracle_regret",
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
        normalized = "".join(
            character.lower() if character.isalnum() else "_"
            for character in name
        )
        tokens = {token for token in normalized.split("_") if token}
        forbidden = sorted(tokens.intersection(_FORBIDDEN_FEATURE_TOKENS))
        if forbidden:
            raise Stage5RouterError(
                f"Router feature {name!r} contains forbidden tokens {forbidden}"
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


def _oriented_utility(label: int, score: float) -> float:
    return score if label == 1 else -score


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
    "STAGE5_ROUTER_PREDICTION_PROTOCOL_VERSION",
    "STAGE5_ROUTER_RUN_PROTOCOL_VERSION",
    "TaskOutcome",
    "evaluate_selection",
    "main",
    "oracle_targets",
    "parse_args",
    "read_evaluator_outcomes",
    "read_fold_manifest",
    "read_router_features",
    "select_global_best_expert",
]


if __name__ == "__main__":
    raise SystemExit(main())
