from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.normroute.router.expert_bank import (
    ExpertBankInputError,
    build_expert_capability_bank,
    read_stage2_runtime_summary,
    write_capability_bank,
)
from src.normroute.router.teacher import (
    TeacherInputError,
    TeacherIsolationError,
    build_teacher,
    read_teacher_parquet,
    write_teacher_parquet,
)
from src.normroute.cli.audit_teacher_artifacts import main as audit_main
from src.normroute.cli.build_expert_capability_bank import main as bank_main
from src.normroute.cli.build_teacher_targets import main as teacher_main
from src.normroute.cli.run_stage5_router import balanced_query_task_weights


def _inputs(
    root: Path,
    *,
    test_score: str = "0.5",
    test_runtime: object | None = None,
) -> tuple[Path, Path]:
    manifest = root / "fold_manifest.csv"
    matrix = root / "evaluator_only" / "routing_matrix_long.csv"
    matrix.parent.mkdir(parents=True)
    manifest_rows = []
    routing_rows = []
    specifications = (
        ("train", "bottle", "train-normal", 0, 1, 0),
        ("train", "bottle", "train-normal", 0, 4, 2),
        ("train", "bottle", "train-anomaly", 1, 2, 1),
        ("train", "carpet", "train-carpet-normal", 0, 1, 0),
        ("train", "carpet", "train-carpet-anomaly", 1, 2, 1),
        ("val", "cable", "val-normal", 0, 1, 0),
        ("val", "cable", "val-anomaly", 1, 2, 1),
        ("test", "capsule", "test-query", 1, 1, 0),
    )
    for split, category, image_id, label, k_shot, seed in specifications:
        support_id = f"{category}-k{k_shot}-s{seed}"
        task_id = f"{image_id}|mvtec|{category}|{support_id}|{k_shot}|{seed}"
        manifest_rows.append(
            {
                "fold": "fold0",
                "split": split,
                "task_id": task_id,
                "sample_id": image_id,
                "dataset": "mvtec",
                "category": category,
                "k_shot": k_shot,
                "seed": seed,
                "support_set_id": support_id,
            }
        )
        for expert, normal_score, anomaly_score, runtime in (
            ("PatchCore", 0.1, 0.9, 10.0),
            ("WinCLIP", 0.3, 0.7, 20.0),
        ):
            score = normal_score if label == 0 else anomaly_score
            if split == "test":
                score = test_score
            routing_rows.append(
                {
                    "image_id": image_id,
                    "dataset": "mvtec",
                    "category": category,
                    "support_set_id": support_id,
                    "k_shot": k_shot,
                    "seed": seed,
                    "expert_name": expert,
                    "final_score": score,
                    "runtime_ms": (
                        test_runtime
                        if split == "test" and test_runtime is not None
                        else runtime
                    ),
                    "label": label,
                    "status": "ok",
                    "error_message": "",
                }
            )
    _write_csv(manifest, manifest_rows)
    _write_csv(matrix, routing_rows)
    return manifest, matrix


def _stage2_summary(
    path: Path,
    matrix: Path,
    *,
    poison_test_runtime: bool = False,
) -> Path:
    with matrix.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["expert_name"].casefold(),
            row["dataset"],
            row["category"],
            row["support_set_id"],
            row["k_shot"],
            row["seed"],
        )
        grouped.setdefault(key, []).append(row)
    summary_rows = []
    for key, condition_rows in sorted(grouped.items()):
        expert, dataset, category, support_set_id, k_shot, seed = key
        failed = sum(row["status"] != "ok" for row in condition_rows)
        runtime: object = sum(
            float(row["runtime_ms"]) for row in condition_rows
        ) / len(condition_rows)
        if poison_test_runtime and category == "capsule":
            runtime = "test-category-poison"
        summary_rows.append(
            {
                "expert": expert,
                "dataset": dataset,
                "category": category,
                "k_shot": k_shot,
                "seed": seed,
                "support_set_id": support_set_id,
                "metrics_path": f"stage2/{expert}/{category}/metrics.json",
                "num_predictions": len(condition_rows),
                "num_success": len(condition_rows) - failed,
                "num_failed": failed,
                "average_tool_calls": 1.0,
                "average_runtime_ms": runtime,
                "abstention_rate": 0.0,
            }
        )
    _write_csv(path, summary_rows)
    return path


def test_teacher_requires_evaluator_only_routing_matrix(tmp_path: Path) -> None:
    manifest, matrix = _inputs(tmp_path)
    unsafe = tmp_path / "routing_matrix_long.csv"
    unsafe.write_bytes(matrix.read_bytes())

    with pytest.raises(TeacherIsolationError, match="evaluator_only"):
        build_teacher(
            routing_matrix_path=unsafe,
            fold_manifest_path=manifest,
            fold="fold0",
        )


def test_teacher_rejects_parent_traversal_out_of_evaluator_only(
    tmp_path: Path,
) -> None:
    manifest, safe_matrix = _inputs(tmp_path / "inputs")
    stage3 = tmp_path / "stage3"
    (stage3 / "evaluator_only").mkdir(parents=True)
    outside_matrix = stage3 / "routing_matrix_long.csv"
    outside_matrix.write_bytes(safe_matrix.read_bytes())
    traversal = stage3 / "evaluator_only" / ".." / "routing_matrix_long.csv"

    with pytest.raises(TeacherIsolationError, match="evaluator_only"):
        build_teacher(
            routing_matrix_path=traversal,
            fold_manifest_path=manifest,
            fold="fold0",
        )


def test_teacher_rejects_evaluator_only_symlink_or_junction_to_outside(
    tmp_path: Path,
) -> None:
    manifest, safe_matrix = _inputs(tmp_path / "inputs")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "routing_matrix_long.csv").write_bytes(safe_matrix.read_bytes())
    alias_parent = tmp_path / "stage3"
    alias_parent.mkdir()
    evaluator_alias = alias_parent / "evaluator_only"
    try:
        evaluator_alias.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlink/junction creation is unavailable: {exc}")

    with pytest.raises(TeacherIsolationError, match="evaluator_only"):
        build_teacher(
            routing_matrix_path=evaluator_alias / "routing_matrix_long.csv",
            fold_manifest_path=manifest,
            fold="fold0",
        )


def test_teacher_output_rejects_traversal_and_symlink_escape(
    tmp_path: Path,
) -> None:
    manifest, matrix = _inputs(tmp_path / "inputs")
    artifact = build_teacher(
        routing_matrix_path=matrix,
        fold_manifest_path=manifest,
        fold="fold0",
    )
    stage5 = tmp_path / "stage5"
    evaluator_only = stage5 / "evaluator_only"
    evaluator_only.mkdir(parents=True)
    with pytest.raises(TeacherIsolationError, match="evaluator_only"):
        write_teacher_parquet(
            artifact,
            evaluator_only / ".." / "teacher.parquet",
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    alias_root = tmp_path / "alias"
    alias_root.mkdir()
    alias = alias_root / "evaluator_only"
    try:
        alias.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlink/junction creation is unavailable: {exc}")
    with pytest.raises(TeacherIsolationError, match="evaluator_only"):
        write_teacher_parquet(artifact, alias / "teacher.parquet")


def test_teacher_cli_checks_boundary_before_any_input_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, safe_matrix = _inputs(tmp_path / "inputs")
    stage3 = tmp_path / "stage3"
    (stage3 / "evaluator_only").mkdir(parents=True)
    outside_matrix = stage3 / "routing_matrix_long.csv"
    outside_matrix.write_bytes(safe_matrix.read_bytes())
    traversal = stage3 / "evaluator_only" / ".." / "routing_matrix_long.csv"
    hashed_paths: list[Path] = []

    def record_hash(path: str | Path) -> str:
        hashed_paths.append(Path(path))
        return "0" * 64

    monkeypatch.setattr(
        "src.normroute.cli.build_teacher_targets.file_sha256",
        record_hash,
    )
    output_dir = tmp_path / "outputs" / "evaluator_only" / "fold0"
    assert teacher_main(
        [
            "--routing-matrix", str(traversal),
            "--fold-manifest", str(manifest),
            "--fold", "fold0",
            "--output-dir", str(output_dir),
        ]
    ) == 1

    # The CLI may hash the failures artifact it just wrote, but neither input
    # may be inspected before the evaluator-only boundary has passed.
    assert all(path.name == "failures.json" for path in hashed_paths)
    run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert run["ok"] is False
    assert run["input_hashes"] == {}
    assert run["failures"][0]["code"] == "TeacherIsolationError"


def test_teacher_requires_runtime_on_allowed_categories(tmp_path: Path) -> None:
    manifest, matrix = _inputs(tmp_path)
    with matrix.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["category"] == "bottle":
            row["runtime_ms"] = ""
            break
    _write_csv(matrix, rows)

    with pytest.raises(TeacherInputError, match="missing runtime_ms"):
        build_teacher(
            routing_matrix_path=matrix,
            fold_manifest_path=manifest,
            fold="fold0",
        )


def test_teacher_ablation_paths_are_explicit_and_still_train_only(tmp_path: Path) -> None:
    manifest, matrix = _inputs(tmp_path)
    artifact = build_teacher(
        routing_matrix_path=matrix,
        fold_manifest_path=manifest,
        fold="fold0",
        calibration_strategy="full_train",
        repeat_weighting="uniform_task",
    )

    assert artifact.calibration_strategy == "full_train"
    assert artifact.repeat_weighting == "uniform_task"
    assert {row["split"] for row in artifact.rows} == {"train"}
    assert {row["calibration_scope"] for row in artifact.rows} == {"full_train"}
    assert {row["sample_weight"] for row in artifact.rows} == {1.0}


def test_teacher_and_router_share_grouped_repeat_weights(tmp_path: Path) -> None:
    manifest, matrix = _inputs(tmp_path)
    artifact = build_teacher(
        routing_matrix_path=matrix,
        fold_manifest_path=manifest,
        fold="fold0",
    )
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    by_task = {row["task_id"]: row for row in manifest_rows}
    train_ids = tuple(
        row["task_id"] for row in manifest_rows if row["split"] == "train"
    )
    router_weights = balanced_query_task_weights(train_ids, by_task)
    teacher_weights = {
        row["task_id"]: float(row["sample_weight"])
        for row in artifact.rows
    }

    assert teacher_weights == pytest.approx(router_weights)


def test_teacher_v3_uses_train_gap_scale_and_validation_sharpness_grid(
    tmp_path: Path,
) -> None:
    manifest, matrix = _inputs(tmp_path)
    artifact = build_teacher(
        routing_matrix_path=matrix,
        fold_manifest_path=manifest,
        fold="fold0",
    )
    metadata = artifact.metadata()

    assert artifact.protocol_version == "stage5.teacher.v3"
    assert artifact.objective_scale > 0.0
    assert artifact.minimum_probability in {0.005, 0.01, 0.025}
    assert metadata["objective_scale_scope"] == "train_categories_only"
    assert metadata["sharpness_strategy"] == (
        "train_robust_gap_validation_oracle_nll"
    )
    assert metadata["proper_loss"] == (
        "validation_train_calibrated_zero_one_cost_multiclass_log_loss"
    )
    assert len(artifact.temperature_frontier) == 15
    assert all(
        item["validation_calibration_loss"] >= 0.0
        and item["validation_empirical_nll"] >= 0.0
        and item["validation_multiclass_brier"] >= 0.0
        and 0.0 <= item["validation_ece"] <= 1.0
        and item["validation_effective_class_count"] >= 1.0
        and 0.0 <= item["validation_selection_accuracy"] <= 1.0
        for item in artifact.temperature_frontier
    )
    assert metadata["sharpness_target"] == (
        "train_calibrated_zero_one_error_plus_runtime_and_failure"
    )
    assert metadata["selected_sharpness_diagnostics"]
    assert set(metadata["teacher_entropy_by_category"]) == {"bottle", "carpet"}

    by_task: dict[str, list[dict]] = {}
    for row in artifact.rows:
        by_task.setdefault(row["task_id"], []).append(row)
    assert all(
        all(row["soft_utility_probability"] >= artifact.minimum_probability for row in rows)
        and sum(row["soft_utility_probability"] for row in rows)
        == pytest.approx(1.0)
        for rows in by_task.values()
    )


def test_teacher_train_scale_is_invariant_to_validation_mutation(
    tmp_path: Path,
) -> None:
    manifest_a, matrix_a = _inputs(tmp_path / "a")
    manifest_b, matrix_b = _inputs(tmp_path / "b")
    with matrix_b.open("r", newline="", encoding="utf-8") as handle:
        changed = list(csv.DictReader(handle))
    for row in changed:
        if row["category"] == "cable":
            # Validation is allowed to select sharpness, but it cannot refit
            # train-only Platt parameters or the train objective-gap scale.
            row["final_score"] = str(100.0 - float(row["final_score"]))
    _write_csv(matrix_b, changed)

    first = build_teacher(
        routing_matrix_path=matrix_a,
        fold_manifest_path=manifest_a,
        fold="fold0",
    )
    second = build_teacher(
        routing_matrix_path=matrix_b,
        fold_manifest_path=manifest_b,
        fold="fold0",
    )

    assert first.calibrations == second.calibrations
    assert first.objective_scale == pytest.approx(second.objective_scale)
    assert first.train_categories == second.train_categories


def test_teacher_soft_targets_are_per_expert_affine_score_scale_invariant(
    tmp_path: Path,
) -> None:
    manifest_a, matrix_a = _inputs(tmp_path / "a")
    manifest_b, matrix_b = _inputs(tmp_path / "b")
    with matrix_b.open("r", newline="", encoding="utf-8") as handle:
        scaled = list(csv.DictReader(handle))
    transforms = {
        "PatchCore": (10.0, 7.0),
        "WinCLIP": (3.25, -11.0),
    }
    for row in scaled:
        if row["category"] != "capsule":
            multiplier, offset = transforms[row["expert_name"]]
            row["final_score"] = str(
                multiplier * float(row["final_score"]) + offset
            )
    _write_csv(matrix_b, scaled)

    first = build_teacher(
        routing_matrix_path=matrix_a,
        fold_manifest_path=manifest_a,
        fold="fold0",
    )
    second = build_teacher(
        routing_matrix_path=matrix_b,
        fold_manifest_path=manifest_b,
        fold="fold0",
    )
    first_probabilities = {
        (row["task_id"], row["expert_name"]): row["soft_utility_probability"]
        for row in first.rows
    }
    second_probabilities = {
        (row["task_id"], row["expert_name"]): row["soft_utility_probability"]
        for row in second.rows
    }

    assert first.objective_scale == pytest.approx(second.objective_scale)
    assert first.temperature == second.temperature
    assert first.minimum_probability == second.minimum_probability
    assert first_probabilities == pytest.approx(second_probabilities)


def test_teacher_is_train_only_soft_and_parquet_is_evaluator_only(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    manifest, matrix = _inputs(tmp_path)
    artifact = build_teacher(
        routing_matrix_path=matrix,
        fold_manifest_path=manifest,
        fold="fold0",
    )

    assert {row["category"] for row in artifact.rows} == {"bottle", "carpet"}
    assert {row["split"] for row in artifact.rows} == {"train"}
    by_task: dict[str, list[dict]] = {}
    for row in artifact.rows:
        by_task.setdefault(row["task_id"], []).append(row)
    assert all(
        sum(row["soft_utility_probability"] for row in rows) == pytest.approx(1.0)
        for rows in by_task.values()
    )
    assert all(sum(bool(row["hard_oracle"]) for row in rows) == 1 for rows in by_task.values())
    assert all(row["calibration_scope"] == "leave_one_train_category_out" for row in artifact.rows)
    assert all(row["category"] not in row["calibration_fit_categories"] for row in artifact.rows)
    assert all(row["sample_weight"] > 0.0 for row in artifact.rows)
    task_rows = [rows[0] for rows in by_task.values()]
    category_mass: dict[str, float] = {}
    query_mass: dict[tuple[str, str], float] = {}
    for row in task_rows:
        category_mass[row["category"]] = category_mass.get(row["category"], 0.0) + row["sample_weight"]
        key = (row["category"], row["query_group_id"])
        query_mass[key] = query_mass.get(key, 0.0) + row["sample_weight"]
    assert len(set(round(value, 10) for value in category_mass.values())) == 1
    bottle_query_mass = [value for (category, _), value in query_mass.items() if category == "bottle"]
    assert len(set(round(value, 10) for value in bottle_query_mass)) == 1

    output = tmp_path / "run" / "evaluator_only" / "teacher.parquet"
    assert write_teacher_parquet(artifact, output) == output
    assert len(read_teacher_parquet(output)) == len(artifact.rows)
    with pytest.raises(TeacherIsolationError, match="evaluator_only"):
        write_teacher_parquet(artifact, tmp_path / "teacher.parquet")


def test_test_category_changes_cannot_change_teacher_or_capability_bank(tmp_path: Path) -> None:
    manifest_a, matrix_a = _inputs(tmp_path / "a", test_score="0.0")
    manifest_b, matrix_b = _inputs(
        tmp_path / "b",
        test_score="not-even-numeric",
        test_runtime="not-even-numeric",
    )
    first = build_teacher(
        routing_matrix_path=matrix_a,
        fold_manifest_path=manifest_a,
        fold="fold0",
    )
    second = build_teacher(
        routing_matrix_path=matrix_b,
        fold_manifest_path=manifest_b,
        fold="fold0",
    )
    assert first.metadata() == second.metadata()
    assert first.rows == second.rows

    task_ids = sorted({row["task_id"] for row in first.rows})
    boundary = {task_id: float(index) for index, task_id in enumerate(task_ids)}
    fgbg = {task_id: float(len(task_ids) - index) for index, task_id in enumerate(task_ids)}
    bank = build_expert_capability_bank(
        first,
        boundary_signals=boundary,
        fgbg_signals=fgbg,
    )
    payload = bank.to_dict()
    assert set(payload["profiles"]) == {"patchcore", "winclip"}
    for profile in payload["profiles"].values():
        assert {
            "boundary_skill",
            "fgbg_skill",
            "lowshot_skill",
            "latency",
            "failure_rate",
        }.issubset(profile)
        assert len(profile["boundary_curve"]) >= 2
        assert len(profile["fgbg_curve"]) >= 2
        assert set(profile["confidence_intervals"]) >= {
            "overall_skill",
            "boundary_skill",
            "fgbg_skill",
            "lowshot_skill",
            "latency_p50",
            "latency_p95",
            "failure_rate",
        }
        assert profile["latency_p50"] > 0.0
        assert profile["latency_p95"] >= profile["latency_p50"]
        assert len(profile["capability_vector"]) == len(payload["capability_feature_names"])
    output = tmp_path / "capability_bank.json"
    assert write_capability_bank(bank, output) == output
    assert output.is_file()


def test_capability_bank_does_not_parse_test_feature_records(tmp_path: Path) -> None:
    manifest, matrix = _inputs(tmp_path)
    artifact = build_teacher(
        routing_matrix_path=matrix,
        fold_manifest_path=manifest,
        fold="fold0",
    )
    task_ids = sorted({row["task_id"] for row in artifact.rows})
    train_features = [
        {
            "task_id": task_id,
            "feature_names": ["bir_query_bai", "fbdp_fbc", "normal_niv_2"],
            "values": [float(index), float(index + 1), float(index + 2)],
        }
        for index, task_id in enumerate(task_ids)
    ]
    malformed_test_feature = {
        "task_id": "test-query|mvtec|capsule|test-support|1|0",
        "feature_names": "intentionally malformed and test-only",
        "values": "not numeric",
    }

    clean = build_expert_capability_bank(artifact, feature_records=train_features)
    with_test = build_expert_capability_bank(
        artifact,
        feature_records=[*train_features, malformed_test_feature],
    )

    assert clean.to_dict() == with_test.to_dict()
    assert all(
        profile["texture_skill"] is not None
        and profile["texture_curve"]
        for profile in clean.to_dict()["profiles"].values()
    )


def test_stage2_runtime_summary_ignores_test_poison_and_crosschecks_bank(
    tmp_path: Path,
) -> None:
    manifest, matrix = _inputs(tmp_path / "inputs")
    artifact = build_teacher(
        routing_matrix_path=matrix,
        fold_manifest_path=manifest,
        fold="fold0",
    )
    clean_path = _stage2_summary(tmp_path / "clean_summary.csv", matrix)
    poisoned_path = _stage2_summary(
        tmp_path / "poisoned_summary.csv",
        matrix,
        poison_test_runtime=True,
    )
    train_categories = {"bottle", "carpet"}
    clean = read_stage2_runtime_summary(
        clean_path, allowed_categories=train_categories
    )
    poisoned = read_stage2_runtime_summary(
        poisoned_path, allowed_categories=train_categories
    )
    assert clean == poisoned

    task_ids = sorted({row["task_id"] for row in artifact.rows})
    boundary = {task_id: float(index) for index, task_id in enumerate(task_ids)}
    fgbg = {
        task_id: float(len(task_ids) - index)
        for index, task_id in enumerate(task_ids)
    }
    bank = build_expert_capability_bank(
        artifact,
        boundary_signals=boundary,
        fgbg_signals=fgbg,
        runtime_summary_records=poisoned,
        bootstrap_replicates=10,
    ).to_dict()
    assert bank["runtime_provenance"]["stage2_summary_cross_check"] is True
    assert bank["runtime_provenance"]["train_categories"] == ["bottle", "carpet"]
    assert all(
        profile["latency_p50"] is not None
        and profile["latency_p95"] is not None
        for profile in bank["profiles"].values()
    )

    with clean_path.open("r", newline="", encoding="utf-8") as handle:
        mismatched_rows = list(csv.DictReader(handle))
    for row in mismatched_rows:
        if row["category"] == "bottle":
            row["average_runtime_ms"] = str(float(row["average_runtime_ms"]) + 1.0)
            break
    mismatched_path = tmp_path / "mismatched_summary.csv"
    _write_csv(mismatched_path, mismatched_rows)
    mismatched = read_stage2_runtime_summary(
        mismatched_path, allowed_categories=train_categories
    )
    with pytest.raises(ExpertBankInputError, match="average_runtime_ms disagrees"):
        build_expert_capability_bank(
            artifact,
            boundary_signals=boundary,
            fgbg_signals=fgbg,
            runtime_summary_records=mismatched,
            bootstrap_replicates=0,
        )


def test_teacher_bank_and_audit_clis_form_reproducible_pipeline(tmp_path: Path) -> None:
    manifest, matrix = _inputs(tmp_path / "inputs")
    teacher_dir = tmp_path / "outputs" / "evaluator_only" / "fold0"
    assert teacher_main(
        [
            "--routing-matrix", str(matrix),
            "--fold-manifest", str(manifest),
            "--fold", "fold0",
            "--output-dir", str(teacher_dir),
            "--seed", "7",
        ]
    ) == 0
    teacher_path = teacher_dir / "teacher.parquet"
    assert teacher_path.is_file()
    teacher_run = json.loads(
        (teacher_dir / "run.json").read_text(encoding="utf-8")
    )
    assert teacher_run["config"]["sharpness_strategy"] == (
        "train_robust_gap_validation_oracle_nll"
    )
    assert teacher_run["config"]["minimum_probabilities"] == [
        0.005,
        0.01,
        0.025,
    ]
    assert teacher_run["fold_metadata"]["fold0"]["objective_scale_scope"] == (
        "train_categories_only"
    )

    feature_path = tmp_path / "router_features.jsonl"
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    with feature_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(manifest_rows):
            handle.write(json.dumps({
                "task_id": row["task_id"],
                "feature_names": ["bir_query_bai", "fbdp_fbc", "normal_niv_2"],
                "values": [float(index), float(index + 1), float(index + 2)],
            }) + "\n")

    bank_dir = tmp_path / "outputs" / "capability_bank" / "fold0"
    stage2_summary = _stage2_summary(tmp_path / "stage2_summary.csv", matrix)
    assert bank_main(
        [
            "--teacher-data", str(teacher_path),
            "--router-features", str(feature_path),
            "--stage2-summary", str(stage2_summary),
            "--fold", "fold0",
            "--output-dir", str(bank_dir),
            "--bootstrap-replicates", "20",
            "--seed", "7",
        ]
    ) == 0
    bank_path = bank_dir / "capability_bank.json"
    audit_path = teacher_dir / "dataset_audit.json"
    assert audit_main(
        [
            "--teacher-data", str(teacher_path),
            "--capability-bank", str(bank_path),
            "--fold-manifest", str(manifest),
            "--fold", "fold0",
            "--output", str(audit_path),
        ]
    ) == 0
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert all(report["checks"].values())
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    assert bank["runtime_provenance"]["stage2_summary_cross_check"] is True
    assert all(
        profile["latency_p50"] > 0.0 and profile["latency_p95"] > 0.0
        for profile in bank["profiles"].values()
    )
    assert json.loads((teacher_dir / "run.json").read_text(encoding="utf-8"))["ok"] is True
    assert json.loads((bank_dir / "run.json").read_text(encoding="utf-8"))["ok"] is True


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
