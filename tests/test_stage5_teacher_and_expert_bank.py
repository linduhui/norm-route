from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.normroute.router.expert_bank import (
    build_expert_capability_bank,
    write_capability_bank,
)
from src.normroute.router.teacher import (
    TeacherIsolationError,
    build_teacher,
    read_teacher_parquet,
    write_teacher_parquet,
)
from src.normroute.cli.audit_teacher_artifacts import main as audit_main
from src.normroute.cli.build_expert_capability_bank import main as bank_main
from src.normroute.cli.build_teacher_targets import main as teacher_main


def _inputs(root: Path, *, test_score: str = "0.5") -> tuple[Path, Path]:
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
                    "runtime_ms": runtime,
                    "label": label,
                    "status": "ok",
                    "error_message": "",
                }
            )
    _write_csv(manifest, manifest_rows)
    _write_csv(matrix, routing_rows)
    return manifest, matrix


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
    manifest_b, matrix_b = _inputs(tmp_path / "b", test_score="not-even-numeric")
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
            "failure_rate",
        }
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
    assert bank_main(
        [
            "--teacher-data", str(teacher_path),
            "--router-features", str(feature_path),
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
    assert json.loads((teacher_dir / "run.json").read_text(encoding="utf-8"))["ok"] is True
    assert json.loads((bank_dir / "run.json").read_text(encoding="utf-8"))["ok"] is True


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
