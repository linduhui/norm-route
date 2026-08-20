"""Strict final audit for the Stage 5 teacher/ECPB v3 three-seed closure."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping
import xml.etree.ElementTree as ET

from ..router.expert_bank import read_capability_bank
from ..router.teacher import (
    TEACHER_PROTOCOL_VERSION,
    ensure_evaluator_only_routing_matrix,
    read_teacher_parquet,
)
from .run_stage5_router import (
    STAGE5_ROUTER_METRICS_PROTOCOL_VERSION,
    STAGE5_ROUTER_PER_TASK_PROTOCOL_VERSION,
    STAGE5_ROUTER_PREDICTION_PROTOCOL_VERSION,
    STAGE5_ROUTER_RUN_PROTOCOL_VERSION,
)
from .stage5_artifacts import atomic_write_json, file_sha256, git_commit
from .summarize_stage5_router import METRICS


VARIANTS = (
    "hard_oracle",
    "soft_teacher_legacy_target",
    "soft_teacher_no_bank",
    "soft_teacher_static",
    "soft_teacher_full",
    "soft_no_capability",
    "soft_no_uncertainty",
    "soft_no_cost",
)
FOLDS = tuple(f"fold{index}" for index in range(5))
SEEDS = (0, 1, 2)
COMPARISON_NAMES = (
    "hard_to_full",
    "sharpness_gain",
    "ecpb_gain",
    "conditional_profile_gain",
    "capability_gain",
    "uncertainty_gain",
    "latency_cost_gain",
)
_FORBIDDEN_PREDICTION_TOKENS = frozenset(
    {
        "label",
        "labels",
        "mask",
        "masks",
        "defect",
        "defecttype",
        "anomalytype",
        "score",
        "scores",
        "utility",
        "utilities",
        "outcome",
        "outcomes",
        "teacher",
    }
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--routing-matrix", required=True)
    parser.add_argument("--output")
    return parser.parse_args(argv)


def audit_closure(run_root: str | Path, routing_matrix: str | Path) -> dict[str, Any]:
    root = Path(run_root)
    matrix = ensure_evaluator_only_routing_matrix(routing_matrix)
    failures: list[str] = []
    commits: set[str] = set()
    fold_manifest_hashes: set[str] = set()
    stage2_summary_hashes: set[str] = set()
    feature_hash_by_fold: dict[str, str] = {}
    audit_fold_manifest_hashes: set[str] = set()
    teacher_calibration_diagnostics: list[dict[str, Any]] = []
    counts = {
        "teacher_folds": 0,
        "legacy_teacher_folds": 0,
        "capability_banks": 0,
        "router_runs": 0,
        "prediction_rows": 0,
        "seed_summaries": 0,
        "multiseed_summaries": 0,
    }

    matrix_hash = file_sha256(matrix)
    for fold in FOLDS:
        v3_diagnostics: dict[str, float] | None = None
        teacher_path = root / "evaluator_only" / fold / "teacher.parquet"
        legacy_path = (
            root
            / "ablations"
            / "legacy_target"
            / "evaluator_only"
            / fold
            / "teacher.parquet"
        )
        bank_path = root / "capability_bank" / fold / "capability_bank.json"
        audit_path = root / "evaluator_only" / fold / "dataset_audit.json"
        try:
            rows = read_teacher_parquet(teacher_path)
            metadata = _json(root / "evaluator_only" / fold / "teacher_metadata.json")
            v3_diagnostics = _audit_teacher_rows(
                rows, metadata, fold, legacy=False
            )
            counts["teacher_folds"] += 1
        except Exception as exc:
            failures.append(f"teacher {fold}: {type(exc).__name__}: {exc}")
        try:
            legacy_rows = read_teacher_parquet(legacy_path)
            legacy_metadata = _json(legacy_path.with_name("teacher_metadata.json"))
            legacy_diagnostics = _audit_teacher_rows(
                legacy_rows, legacy_metadata, fold, legacy=True
            )
            if v3_diagnostics is not None:
                teacher_calibration_diagnostics.append(
                    {
                        "fold": fold,
                        **{
                            f"v3_{name}": value
                            for name, value in v3_diagnostics.items()
                        },
                        **{
                            f"legacy_{name}": value
                            for name, value in legacy_diagnostics.items()
                        },
                        **{
                            f"delta_{name}": v3_diagnostics[name] - value
                            for name, value in legacy_diagnostics.items()
                            if name in v3_diagnostics
                            and isinstance(value, (int, float))
                        },
                    }
                )
            counts["legacy_teacher_folds"] += 1
        except Exception as exc:
            failures.append(f"legacy teacher {fold}: {type(exc).__name__}: {exc}")
        try:
            bank = read_capability_bank(bank_path)
            _audit_bank(bank, fold)
            bank_run = _audit_clean_run(
                bank_path.with_name("run.json"),
                required_output_names={"capability_bank", "failures"},
            )
            _record_commit(commits, bank_run, bank_path.with_name("run.json"))
            bank_config = bank_run.get("config", {})
            if (
                bank_config.get("fold") != fold
                or int(bank_config.get("seed", -1)) != 0
                or not _same_optional_path(
                    bank_config.get("teacher_data"), teacher_path
                )
            ):
                raise ValueError("capability-bank build config mismatch")
            bank_hashes = bank_run.get("input_hashes", {})
            if bank_hashes.get("teacher_data") != file_sha256(teacher_path):
                raise ValueError("capability-bank teacher hash mismatch")
            feature_hash = str(bank_hashes.get("router_features", ""))
            stage2_hash = str(bank_hashes.get("stage2_summary", ""))
            if not feature_hash or not stage2_hash:
                raise ValueError("capability-bank input provenance is incomplete")
            if (
                file_sha256(bank_config.get("router_features", ""))
                != feature_hash
                or file_sha256(bank_config.get("stage2_summary", ""))
                != stage2_hash
            ):
                raise ValueError("capability-bank configured inputs are stale")
            feature_hash_by_fold[fold] = feature_hash
            stage2_summary_hashes.add(stage2_hash)
            audit = _json(audit_path)
            if audit.get("ok") is not True or audit.get("failures") != []:
                raise ValueError("dataset audit is not clean")
            audit_hashes = audit.get("input_hashes", {})
            if (
                audit_hashes.get("teacher_data") != file_sha256(teacher_path)
                or audit_hashes.get("capability_bank") != file_sha256(bank_path)
            ):
                raise ValueError("dataset audit input hashes are stale")
            audit_fold_hash = str(audit_hashes.get("fold_manifest", ""))
            if not audit_fold_hash:
                raise ValueError("dataset audit fold-manifest hash is missing")
            audit_fold_manifest_hashes.add(audit_fold_hash)
            counts["capability_banks"] += 1
        except Exception as exc:
            failures.append(f"capability bank {fold}: {type(exc).__name__}: {exc}")

    for teacher_run in (
        root / "evaluator_only" / "run.json",
        root / "ablations" / "legacy_target" / "evaluator_only" / "run.json",
    ):
        try:
            run = _audit_clean_run(
                teacher_run,
                required_output_names={
                    *(f"{fold}.teacher" for fold in FOLDS),
                    *(f"{fold}.metadata" for fold in FOLDS),
                    "failures",
                },
            )
            _record_commit(commits, run, teacher_run)
            if run.get("input_hashes", {}).get("routing_matrix") != matrix_hash:
                raise ValueError("teacher routing-matrix hash mismatch")
            teacher_config = run.get("config", {})
            expected_strategy = (
                "legacy_raw_objective_soft_ce"
                if "legacy_target" in teacher_run.parts
                else "train_robust_gap_validation_oracle_nll"
            )
            if (
                teacher_config.get("fold") != "all"
                or teacher_config.get("calibration_strategy")
                != "leave_one_train_category_out"
                or teacher_config.get("repeat_weighting")
                != "equal_category_equal_query_inverse_variant_frequency"
                or teacher_config.get("sharpness_strategy") != expected_strategy
            ):
                raise ValueError("teacher build config mismatch")
            fold_hash = str(run.get("input_hashes", {}).get("fold_manifest", ""))
            if not fold_hash:
                raise ValueError("teacher fold-manifest hash is missing")
            fold_manifest_hashes.add(fold_hash)
        except Exception as exc:
            failures.append(f"teacher run {teacher_run}: {type(exc).__name__}: {exc}")

    if len(fold_manifest_hashes) != 1:
        failures.append("teacher runs do not share one fold-manifest hash")
    if len(stage2_summary_hashes) != 1:
        failures.append("capability-bank runs do not share one Stage2 summary hash")
    if audit_fold_manifest_hashes != fold_manifest_hashes:
        failures.append("dataset audits and teacher runs use different fold manifests")
    if set(feature_hash_by_fold) != set(FOLDS):
        failures.append("capability-bank feature provenance is incomplete")
    fold_manifest_hash = next(iter(fold_manifest_hashes), "")

    for seed in SEEDS:
        for fold in FOLDS:
            for variant in VARIANTS:
                run_dir = root / "router" / f"seed{seed}" / fold / variant
                expected_teacher = root / "evaluator_only" / fold / "teacher.parquet"
                expected_bank = root / "capability_bank" / fold / "capability_bank.json"
                if variant == "hard_oracle":
                    expected_teacher = None
                    expected_bank = None
                elif variant == "soft_teacher_legacy_target":
                    expected_teacher = (
                        root
                        / "ablations"
                        / "legacy_target"
                        / "evaluator_only"
                        / fold
                        / "teacher.parquet"
                    )
                    expected_bank = None
                elif variant == "soft_teacher_no_bank":
                    expected_bank = None
                try:
                    prediction_count, router_run = _audit_router_run(
                        run_dir,
                        seed=seed,
                        fold=fold,
                        variant=variant,
                        matrix_hash=matrix_hash,
                        expected_teacher=expected_teacher,
                        expected_bank=expected_bank,
                        expected_fold_manifest_hash=fold_manifest_hash,
                        expected_feature_hash=feature_hash_by_fold.get(fold, ""),
                    )
                    _record_commit(commits, router_run, run_dir / "run.json")
                    counts["router_runs"] += 1
                    counts["prediction_rows"] += prediction_count
                except Exception as exc:
                    failures.append(
                        f"router seed={seed} fold={fold} variant={variant}: "
                        f"{type(exc).__name__}: {exc}"
                    )
        try:
            summary_run = _audit_summary(
                root / "router" / f"seed{seed}" / "summary_v3",
                expected_rows=(120, 24, 49),
            )
            _record_commit(
                commits,
                summary_run,
                root / "router" / f"seed{seed}" / "summary_v3" / "run.json",
            )
            counts["seed_summaries"] += 1
        except Exception as exc:
            failures.append(f"seed{seed} summary: {type(exc).__name__}: {exc}")

    try:
        multiseed_run = _audit_multiseed_summary(
            root / "router" / "summary_multiseed_v3"
        )
        _record_commit(
            commits,
            multiseed_run,
            root / "router" / "summary_multiseed_v3" / "run.json",
        )
        counts["multiseed_summaries"] = 1
    except Exception as exc:
        failures.append(f"multi-seed summary: {type(exc).__name__}: {exc}")

    expected_counts = {
        "teacher_folds": 5,
        "legacy_teacher_folds": 5,
        "capability_banks": 5,
        "router_runs": 120,
        "seed_summaries": 3,
        "multiseed_summaries": 1,
    }
    for name, expected in expected_counts.items():
        if counts[name] != expected:
            failures.append(f"{name}: expected {expected}, observed {counts[name]}")
    try:
        _audit_pytest_evidence(root / "pytest_full.xml")
    except Exception as exc:
        failures.append(f"pytest evidence: {type(exc).__name__}: {exc}")
    current_commit = git_commit()
    if len(commits) != 1 or not current_commit or commits != {current_commit}:
        failures.append(
            "teacher/bank/router/summary commits are missing, mixed, or differ "
            "from the auditing checkout"
        )
    try:
        _audit_completion_state(root, current_commit)
    except Exception as exc:
        failures.append(f"completion marker: {type(exc).__name__}: {exc}")
    return {
        "protocol_version": "stage5.teacher_ecpb_v3_closure_audit.v1",
        "ok": not failures,
        "run_root": str(root),
        "routing_matrix": str(matrix),
        "routing_matrix_sha256": matrix_hash,
        "counts": counts,
        "git_commit": next(iter(commits), None) if len(commits) == 1 else None,
        "teacher_calibration_diagnostics": {
            "per_fold": teacher_calibration_diagnostics,
            "aggregate": _aggregate_teacher_diagnostics(
                teacher_calibration_diagnostics
            ),
        },
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit_closure(args.run_root, args.routing_matrix)
    except Exception as exc:
        report = {
            "protocol_version": "stage5.teacher_ecpb_v3_closure_audit.v1",
            "ok": False,
            "counts": {},
            "failures": [f"{type(exc).__name__}: {exc}"],
        }
    output = Path(args.output) if args.output else Path(args.run_root) / "final_acceptance_v3.json"
    atomic_write_json(output, report)
    if report["ok"] is not True:
        print(f"Teacher/ECPB v3 closure audit failed: {report['failures'][:3]}", file=sys.stderr)
        return 1
    print(f"Teacher/ECPB v3 closure audit PASS: output={output}")
    return 0


def _audit_teacher_rows(
    rows: list[dict[str, Any]],
    metadata: Mapping[str, Any],
    fold: str,
    *,
    legacy: bool,
) -> dict[str, float]:
    if not rows or any(
        row.get("protocol_version") != TEACHER_PROTOCOL_VERSION
        or row.get("evaluator_only") is not True
        or row.get("split") != "train"
        or row.get("fold") != fold
        for row in rows
    ):
        raise ValueError("teacher rows violate protocol/fold/train isolation")
    expected_strategy = (
        "legacy_raw_objective_soft_ce"
        if legacy
        else "train_robust_gap_validation_oracle_nll"
    )
    if metadata.get("sharpness_strategy") != expected_strategy:
        raise ValueError("teacher sharpness strategy mismatch")
    if metadata.get("objective_scale_scope") != "train_categories_only":
        raise ValueError("teacher objective scale is not train-only")
    scale = float(metadata.get("objective_scale"))
    floor = float(metadata.get("minimum_probability"))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("teacher objective scale is invalid")
    if legacy:
        if floor != 0.0:
            raise ValueError("legacy teacher must use zero probability floor")
    elif not 0.0 < floor < 1.0:
        raise ValueError("v3 teacher must select a positive probability floor")
    by_task: dict[str, list[float]] = {}
    weight_by_task: dict[str, float] = {}
    for row in rows:
        probability = float(row["soft_utility_probability"])
        if not math.isfinite(probability) or probability < floor - 1e-12:
            raise ValueError("teacher contains invalid soft probability")
        task_id = str(row["task_id"])
        by_task.setdefault(task_id, []).append(probability)
        weight = float(row["sample_weight"])
        previous_weight = weight_by_task.setdefault(task_id, weight)
        if (
            not math.isfinite(weight)
            or weight <= 0.0
            or not math.isclose(
                previous_weight, weight, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise ValueError("teacher task sample weights are invalid/inconsistent")
    if any(not math.isclose(sum(values), 1.0, rel_tol=1e-7, abs_tol=1e-7) for values in by_task.values()):
        raise ValueError("teacher distribution does not sum to one")
    selected = metadata.get("selected_sharpness_diagnostics")
    if not isinstance(selected, Mapping):
        raise ValueError("teacher selected sharpness diagnostics are missing")
    required_diagnostics = (
        "validation_empirical_nll",
        "validation_multiclass_brier",
        "validation_ece",
        "validation_distribution_entropy",
        "validation_effective_class_count",
        "validation_selection_accuracy",
    )
    diagnostics = {
        name: float(selected[name]) for name in required_diagnostics
    }
    if any(not math.isfinite(value) for value in diagnostics.values()):
        raise ValueError("teacher validation diagnostics are non-finite")
    weighted_entropies = []
    weighted_top1_values = []
    total_weight = sum(weight_by_task.values())
    for task_id, probabilities in by_task.items():
        task_weight = weight_by_task[task_id]
        weighted_entropies.append(
            task_weight
            * -sum(
                value * math.log(max(value, 1e-12))
                for value in probabilities
            )
        )
        weighted_top1_values.append(task_weight * max(probabilities))
    diagnostics.update(
        {
            "train_teacher_entropy": sum(weighted_entropies) / total_weight,
            "train_teacher_top1_probability": (
                sum(weighted_top1_values) / total_weight
            ),
            "temperature": float(metadata["temperature"]),
            "minimum_probability": floor,
        }
    )
    category_diagnostics = metadata.get("teacher_entropy_by_category")
    if not isinstance(category_diagnostics, Mapping) or not category_diagnostics:
        raise ValueError("teacher entropy-by-category diagnostics are missing")
    return diagnostics


def _audit_bank(bank: Mapping[str, Any], fold: str) -> None:
    profiles = bank.get("profiles")
    if bank.get("fold") != fold or not isinstance(profiles, Mapping) or len(profiles) < 2:
        raise ValueError("capability bank fold/profiles mismatch")
    provenance = bank.get("runtime_provenance", {})
    if provenance.get("category_scope") != "train_only" or provenance.get("stage2_summary_cross_check") is not True:
        raise ValueError("capability runtime provenance is not accepted")
    for profile in profiles.values():
        for name in (
            "boundary_skill",
            "fgbg_skill",
            "lowshot_skill",
            "texture_skill",
            "failure_rate",
            "latency_p50",
            "latency_p95",
            "confidence_intervals",
        ):
            if name not in profile:
                raise ValueError(f"capability profile lacks {name}")
        if float(profile["latency_p50"]) <= 0.0 or float(profile["latency_p95"]) <= 0.0:
            raise ValueError("capability latency is not positive")


def _audit_router_run(
    run_dir: Path,
    *,
    seed: int,
    fold: str,
    variant: str,
    matrix_hash: str,
    expected_teacher: Path | None,
    expected_bank: Path | None,
    expected_fold_manifest_hash: str,
    expected_feature_hash: str,
) -> tuple[int, dict[str, Any]]:
    run = _audit_clean_run(
        run_dir / "run.json",
        required_output_names={
            "model",
            "predictions",
            "metrics",
            "per_task_metrics",
            "failures",
        },
    )
    failures = _json_list(run_dir / "failures.json")
    metrics = _json(run_dir / "evaluator_only" / "metrics.json")
    if run.get("protocol_version") != STAGE5_ROUTER_RUN_PROTOCOL_VERSION:
        raise ValueError("Router run protocol mismatch")
    if metrics.get("protocol_version") != STAGE5_ROUTER_METRICS_PROTOCOL_VERSION:
        raise ValueError("Router metrics protocol mismatch")
    if run.get("ok") is not True or failures != [] or run.get("failures") != []:
        raise ValueError("run/failures are not clean")
    config = run.get("config", {})
    if (
        config.get("fold") != fold
        or config.get("variant") != variant
        or int(config.get("seed", -1)) != seed
        or config.get("uncertainty_mode") != "entropy_scaled_lcb"
        or not math.isclose(
            float(config.get("validation_runtime_tradeoff", -1.0)),
            0.05,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or config.get("validation_selection_metric")
        != "train_calibrated_zero_one_error_plus_normalized_runtime"
        or config.get("repeat_weighting")
        != "equal_category_equal_query_inverse_variant_frequency"
        or config.get("feature_view") != "normal_bir_fbdp"
    ):
        raise ValueError("run config provenance mismatch")
    expected_supervision = "hard_oracle" if variant == "hard_oracle" else "soft_teacher"
    if (
        config.get("supervision") != expected_supervision
        or config.get("supervision_target") != expected_supervision
        or not _same_optional_path(config.get("teacher_data"), expected_teacher)
        or not _same_optional_path(config.get("capability_bank"), expected_bank)
    ):
        raise ValueError("run supervision/teacher/bank provenance mismatch")
    expected_capability_mode = "static" if variant == "soft_teacher_static" else "conditional"
    expected_capability_grid = [0.0] if variant == "soft_no_capability" else [0.0, 0.25, 0.5]
    expected_uncertainty_grid = [0.0] if variant == "soft_no_uncertainty" else [0.0, 0.25, 0.5, 1.0, 2.0]
    expected_cost_grid = [0.0] if variant == "soft_no_cost" else [0.0, 0.05, 0.1]
    if (
        config.get("capability_mode") != expected_capability_mode
        or not _same_numbers(config.get("capability_weight_grid"), expected_capability_grid)
        or not _same_numbers(config.get("uncertainty_weight_grid"), expected_uncertainty_grid)
        or not _same_numbers(config.get("cost_weight_grid"), expected_cost_grid)
    ):
        raise ValueError("run ablation grid provenance mismatch")
    run_hashes = run.get("input_hashes", {})
    if run_hashes.get("routing_matrix") != matrix_hash:
        raise ValueError("run routing matrix hash mismatch")
    if run_hashes.get("fold_manifest") != expected_fold_manifest_hash:
        raise ValueError("run fold-manifest hash mismatch")
    if run_hashes.get("router_features") != expected_feature_hash:
        raise ValueError("run Router-feature hash mismatch")
    if expected_teacher is not None and run_hashes.get("teacher_data") != file_sha256(expected_teacher):
        raise ValueError("run teacher hash mismatch")
    if expected_bank is not None and run_hashes.get("capability_bank") != file_sha256(expected_bank):
        raise ValueError("run capability-bank hash mismatch")
    expected_input_names = {"router_features", "fold_manifest", "routing_matrix"}
    if expected_teacher is not None:
        expected_input_names.add("teacher_data")
    if expected_bank is not None:
        expected_input_names.add("capability_bank")
    if set(run_hashes) != expected_input_names:
        raise ValueError("run input hash key set mismatch")
    runtime = run.get("runtime_statistics", {})
    if runtime.get("runtime_complete") is not True or runtime.get("runtime_source") != "evaluator_only_routing_matrix.runtime_ms":
        raise ValueError("runtime provenance is incomplete")
    if metrics.get("runtime_source") != "evaluator_only_routing_matrix.runtime_ms":
        raise ValueError("metrics runtime provenance mismatch")
    policy = metrics.get("capability_policy")
    if expected_bank is None:
        if policy is not None:
            raise ValueError("bank-free variant unexpectedly has a capability policy")
    else:
        if not isinstance(policy, Mapping):
            raise ValueError("bank-backed variant lacks a capability policy")
        for name in (
            "capability_weight",
            "uncertainty_weight",
            "cost_weight",
            "validation_empirical_loss",
            "validation_selection_accuracy",
            "validation_oracle_regret",
        ):
            if not math.isfinite(float(policy.get(name))):
                raise ValueError(f"capability policy lacks finite {name}")
        if (
            policy.get("uncertainty_mode") != "entropy_scaled_lcb"
            or policy.get("selection_metric")
            != "mean_train_calibrated_zero_one_error_plus_normalized_runtime"
        ):
            raise ValueError("capability policy selection provenance mismatch")
    score_calibrations = metrics.get("score_calibrations")
    if not isinstance(score_calibrations, Mapping) or set(score_calibrations) != set(
        metrics.get("experts", ())
    ):
        raise ValueError("metrics train-only score calibration is incomplete")
    if metrics.get("per_task_metrics_protocol_version") != STAGE5_ROUTER_PER_TASK_PROTOCOL_VERSION:
        raise ValueError("per-task metrics protocol provenance mismatch")
    for values in metrics.get("methods", {}).values():
        runtime_ms = float(values["average_runtime_ms"])
        if not math.isfinite(runtime_ms) or runtime_ms <= 0.0:
            raise ValueError("method runtime is not positive")
    predictions_path = run_dir / "predictions.jsonl"
    if not _same_optional_path(run.get("outputs", {}).get("predictions"), predictions_path):
        raise ValueError("run prediction output path mismatch")
    count = 0
    seen_task_ids: set[str] = set()
    expert_names = set(metrics.get("experts", ()))
    prediction_keys = {
        "protocol_version",
        "task_id",
        "fold",
        "variant",
        "selected_expert",
        "expert_probabilities",
        "selection_objectives",
    }
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            findings = _forbidden_prediction_paths(row)
            if findings:
                raise ValueError(
                    f"prediction {line_number} leaks forbidden fields: {findings[:3]}"
                )
            if set(row) != prediction_keys:
                raise ValueError(f"prediction {line_number} schema mismatch")
            task_id = str(row.get("task_id", ""))
            probabilities = row.get("expert_probabilities")
            objectives = row.get("selection_objectives")
            if (
                row.get("protocol_version")
                != STAGE5_ROUTER_PREDICTION_PROTOCOL_VERSION
                or row.get("fold") != fold
                or row.get("variant") != variant
                or not task_id
                or task_id in seen_task_ids
                or row.get("selected_expert") not in expert_names
                or not isinstance(probabilities, Mapping)
                or set(probabilities) != expert_names
                or not isinstance(objectives, Mapping)
                or set(objectives) != expert_names
            ):
                raise ValueError(f"prediction {line_number} provenance mismatch")
            probability_values = [float(value) for value in probabilities.values()]
            objective_values = [float(value) for value in objectives.values()]
            if (
                any(not math.isfinite(value) or value < 0.0 for value in probability_values)
                or not math.isclose(sum(probability_values), 1.0, rel_tol=1e-7, abs_tol=1e-7)
                or any(not math.isfinite(value) for value in objective_values)
            ):
                raise ValueError(f"prediction {line_number} values are invalid")
            seen_task_ids.add(task_id)
            count += 1
    expected_count = int(metrics.get("task_counts", {}).get("test", -1))
    learned_count = int(
        metrics.get("methods", {}).get("learned_router", {}).get("num_samples", -1)
    )
    if count <= 0 or count != expected_count or count != learned_count:
        raise ValueError("prediction count disagrees with evaluator metrics")
    _audit_per_task_metrics(
        run_dir / "evaluator_only" / "per_task_metrics.csv",
        expected_count=count,
    )
    return count, run


def _audit_summary(
    path: Path, expected_rows: tuple[int, int, int]
) -> dict[str, Any]:
    run = _audit_clean_run(
        path / "run.json",
        required_output_names={
            "per_fold_metrics",
            "summary_metrics",
            "paired_deltas",
            "paper_tables",
            "failures",
        },
    )
    observed = tuple(
        _csv_row_count(path / name)
        for name in (
            "per_fold_metrics.csv",
            "summary_metrics.csv",
            "paired_ablation_deltas.csv",
        )
    )
    if observed != expected_rows:
        raise ValueError(f"summary row counts {observed} != {expected_rows}")
    config = run.get("config", {})
    if (
        tuple(config.get("folds", ())) != FOLDS
        or tuple(config.get("variants", ())) != VARIANTS
        or config.get("experiment_family") != "teacher_ecpb"
        or config.get("reference_variant") != "soft_teacher_full"
    ):
        raise ValueError("single-seed summary config mismatch")
    _audit_metric_grid(
        path / "per_fold_metrics.csv",
        expected={
            (fold, variant, method)
            for fold in FOLDS
            for variant in VARIANTS
            for method in ("learned_router", "global_best", "sample_oracle")
        },
        keys=("fold", "variant", "method"),
    )
    _audit_metric_grid(
        path / "summary_metrics.csv",
        expected={
            (variant, method)
            for variant in VARIANTS
            for method in ("learned_router", "global_best", "sample_oracle")
        },
        keys=("variant", "method"),
        required_integer=("fold_count", 5),
    )
    _audit_paired_grid(path / "paired_ablation_deltas.csv", multiseed=False)
    return run


def _audit_multiseed_summary(path: Path) -> dict[str, Any]:
    run = _audit_clean_run(
        path / "run.json",
        required_output_names={
            "per_seed_fold_metrics",
            "summary_metrics",
            "paired_deltas",
            "selected_policies",
            "route_changes",
            "failures",
        },
    )
    observed = tuple(
        _csv_row_count(path / name)
        for name in (
            "per_seed_fold_metrics.csv",
            "multiseed_summary_metrics.csv",
            "multiseed_paired_deltas.csv",
            "selected_capability_policies.csv",
            "route_change_diagnostics.csv",
        )
    )
    if observed != (360, 168, 49, 120, 45):
        raise ValueError(f"multi-seed summary row counts are invalid: {observed}")
    _audit_metric_grid(
        path / "per_seed_fold_metrics.csv",
        expected={
            (str(seed), fold, variant, method)
            for seed in SEEDS
            for fold in FOLDS
            for variant in VARIANTS
            for method in ("learned_router", "global_best", "sample_oracle")
        },
        keys=("seed", "fold", "variant", "method"),
    )
    _audit_metric_grid(
        path / "multiseed_summary_metrics.csv",
        expected={
            (variant, method, metric)
            for variant in VARIANTS
            for method in ("learned_router", "global_best", "sample_oracle")
            for metric in METRICS
        },
        keys=("variant", "method", "metric"),
        required_integer=("block_count", 15),
    )
    _audit_paired_grid(path / "multiseed_paired_deltas.csv", multiseed=True)
    _audit_metric_grid(
        path / "selected_capability_policies.csv",
        expected={
            (str(seed), fold, variant)
            for seed in SEEDS
            for fold in FOLDS
            for variant in VARIANTS
        },
        keys=("seed", "fold", "variant"),
    )
    _audit_metric_grid(
        path / "route_change_diagnostics.csv",
        expected={
            (str(seed), fold, variant)
            for seed in SEEDS
            for fold in FOLDS
            for variant in (
                "soft_no_capability",
                "soft_no_uncertainty",
                "soft_no_cost",
            )
        },
        keys=("seed", "fold", "ablation_variant"),
    )
    return run


def _forbidden_prediction_paths(value: Any, path: str = "root") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
            tokens = set(normalized.split("_"))
            if tokens.intersection(_FORBIDDEN_PREDICTION_TOKENS):
                findings.append(f"{path}.{key}")
            findings.extend(_forbidden_prediction_paths(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_forbidden_prediction_paths(child, f"{path}[{index}]"))
    return findings


def _json(path: Path) -> dict[str, Any]:
    value = _json_value(path)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _json_list(path: Path) -> list[Any]:
    value = _json_value(path)
    if not isinstance(value, list):
        raise ValueError(f"JSON root must be an array: {path}")
    return value


def _json_value(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_clean_run(
    path: Path,
    *,
    required_output_names: set[str] | None = None,
) -> dict[str, Any]:
    run = _json(path)
    failures_path = path.with_name("failures.json")
    if (
        run.get("ok") is not True
        or run.get("failures") != []
        or _json_list(failures_path) != []
    ):
        raise ValueError(f"run is not clean: {path}")
    outputs = run.get("outputs")
    hashes = run.get("output_hashes")
    if not isinstance(outputs, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError(f"run output provenance is invalid: {path}")
    if not hashes or set(hashes) != set(outputs):
        raise ValueError(f"run output/hash key sets disagree: {path}")
    if required_output_names is not None and set(outputs) != required_output_names:
        raise ValueError(f"run required output key set disagrees: {path}")
    for name, digest in hashes.items():
        output = outputs.get(name)
        if output is None or file_sha256(output) != digest:
            raise ValueError(f"output hash mismatch for {name}: {path}")
    return run


def _record_commit(commits: set[str], run: Mapping[str, Any], path: Path) -> None:
    commit = run.get("git_commit")
    if not isinstance(commit, str) or not commit.strip():
        raise ValueError(f"run is missing git commit: {path}")
    commits.add(commit)


def _audit_per_task_metrics(path: Path, *, expected_count: int) -> None:
    expected_columns = {
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
    }
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != expected_columns:
            raise ValueError("per-task metrics schema mismatch")
        rows = list(reader)
    if len(rows) != expected_count or len({row["task_id"] for row in rows}) != expected_count:
        raise ValueError("per-task metrics count/identity mismatch")
    if any(
        row["protocol_version"] != STAGE5_ROUTER_PER_TASK_PROTOCOL_VERSION
        or row["evaluator_only"].casefold() != "true"
        for row in rows
    ):
        raise ValueError("per-task metrics protocol/isolation mismatch")


def _audit_metric_grid(
    path: Path,
    *,
    expected: set[tuple[str, ...]],
    keys: tuple[str, ...],
    required_integer: tuple[str, int] | None = None,
) -> None:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    observed = {tuple(str(row[key]) for key in keys) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError(f"CSV grid is incomplete/duplicated: {path}")
    if required_integer is not None:
        name, expected_value = required_integer
        if any(int(row[name]) != expected_value for row in rows):
            raise ValueError(f"CSV {name} provenance mismatch: {path}")


def _audit_paired_grid(path: Path, *, multiseed: bool) -> None:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    observed = {(row["comparison"], row["metric"]) for row in rows}
    expected = {
        (comparison, metric)
        for comparison in COMPARISON_NAMES
        for metric in METRICS
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError("paired comparison grid is incomplete/duplicated")
    numeric_fields = (
        (
            "mean_paired_delta",
            "std_paired_delta",
            "positive_fold_fraction",
            "two_sided_sign_flip_p_value",
            "ci95_lower",
            "ci95_upper",
        )
        if multiseed
        else (
            "mean_paired_delta",
            "std_paired_delta",
            "positive_fold_count",
            "negative_fold_count",
            "exact_sign_flip_p_two_sided",
        )
    )
    for row in rows:
        if any(not math.isfinite(float(row[name])) for name in numeric_fields):
            raise ValueError("paired comparison contains non-finite statistics")
        if multiseed and (
            int(row["seed_count"]) != 3
            or int(row["fold_count"]) != 5
            or row["pairing_unit"] != "fold_after_seed_mean"
        ):
            raise ValueError("multi-seed paired inference provenance mismatch")


def _audit_pytest_evidence(path: Path) -> None:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise ValueError("pytest JUnit report has no test suites")
    tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", "0")) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", "0")) for suite in suites)
    if tests <= 0 or failures != 0 or errors != 0:
        raise ValueError(
            f"pytest did not pass cleanly: tests={tests}, failures={failures}, errors={errors}"
        )


def _key_value_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        result[name] = value
    return result


def _audit_completion_state(root: Path, current_commit: str) -> None:
    """Accept an in-flight audit and a cryptographically closed prior audit."""

    status = _key_value_file(root / "STATUS")
    state = status.get("status")
    if state not in {"auditing", "complete"}:
        raise ValueError("audit-state marker must be auditing or complete")
    if not current_commit or status.get("git_commit") != current_commit:
        raise ValueError("audit-state marker commit mismatch")
    if state == "auditing":
        return

    complete = _key_value_file(root / "COMPLETE.txt")
    if complete.get("git_commit") != current_commit:
        raise ValueError("completion marker commit mismatch")
    if complete.get("run_root") is None or (
        Path(complete["run_root"]).resolve() != root.resolve()
    ):
        raise ValueError("completion marker run root mismatch")
    if complete.get("run_tag") != root.name or complete.get("router_runs") != "120":
        raise ValueError("completion marker run identity mismatch")
    required_hashes = {
        "audit_sha256": root / "final_acceptance_v3.json",
        "multiseed_paired_deltas_sha256": (
            root
            / "router"
            / "summary_multiseed_v3"
            / "multiseed_paired_deltas.csv"
        ),
    }
    for name, path in required_hashes.items():
        if complete.get(name) != file_sha256(path):
            raise ValueError(f"completion marker {name} mismatch")


def _aggregate_teacher_diagnostics(
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    if len(rows) != len(FOLDS):
        return {}
    numeric_names = sorted(
        name
        for name in rows[0]
        if name != "fold" and isinstance(rows[0][name], (int, float))
    )
    return {
        name: sum(float(row[name]) for row in rows) / len(rows)
        for name in numeric_names
    }


def _same_optional_path(observed: Any, expected: Path | None) -> bool:
    if expected is None:
        return observed is None
    if observed is None:
        return False
    return Path(str(observed)).resolve() == expected.resolve()


def _same_numbers(observed: Any, expected: list[float]) -> bool:
    if not isinstance(observed, list) or len(observed) != len(expected):
        return False
    try:
        return all(
            math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-12)
            for left, right in zip(observed, expected)
        )
    except (TypeError, ValueError):
        return False


def _csv_row_count(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


if __name__ == "__main__":
    raise SystemExit(main())
