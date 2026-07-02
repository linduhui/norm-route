import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.generate_stage1_gate_report import build_report


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else ["dataset", "support_set_id", "k_shot", "seed", "category"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _audit(dataset: str) -> dict[str, object]:
    return {
        "dataset": dataset,
        "categories": ["bottle"],
        "counts": {
            "categories": 1,
            "train_good_images": 2,
            "test_good_images": 1,
            "test_anomaly_images": 1,
            "missing_masks": 0,
            "broken_image_files": 0,
            "broken_mask_files": 0,
            "duplicate_image_ids": 0,
            "mask_size_mismatches": 0,
        },
        "run": {
            "git_commit": "abc123",
            "seed": None,
            "predictions": "not_applicable_manifest_only",
        },
    }


def _leakage(support_rows: int = 2) -> dict[str, object]:
    return {
        "ok": True,
        "error_count": 0,
        "warning_count": 0,
        "counts": {
            "agent_input_rows": 4,
            "evaluator_rows": 4,
            "support_rows": support_rows,
            "support_sets": 1,
        },
        "issues": [],
    }


def _support_rows(dataset: str, category: str, k_shot: int, seed: int) -> list[dict[str, object]]:
    return [
        {
            "support_set_id": f"{dataset}_{category}_k{k_shot}_seed{seed}",
            "dataset": dataset,
            "category": category,
            "k_shot": k_shot,
            "seed": seed,
            "support_rank": rank,
            "image_id": f"{dataset}-{category}-train-good-{rank}",
            "image_path": f"/datasets/{dataset}/{category}/train/good/{rank:03d}.png",
        }
        for rank in range(1, k_shot + 1)
    ]


def _write_references(project_root: Path) -> None:
    (project_root / "docs").mkdir(parents=True, exist_ok=True)
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    (project_root / "docs" / "task_spec.md").write_text("# Task\n", encoding="utf-8")
    (project_root / "docs" / "data_leakage_policy.md").write_text("# Policy\n", encoding="utf-8")
    (project_root / "configs" / "folds.yaml").write_text("fold_1:\n  - bottle\n", encoding="utf-8")


def _write_summary_layout(stage1_dir: Path, project_root: Path) -> None:
    _write_json(stage1_dir / "mvtec_full_audit.json", _audit("mvtec"))
    _write_json(stage1_dir / "visa_full_audit.json", _audit("visa"))
    _write_csv(stage1_dir / "mvtec_support_summary.csv", _support_rows("mvtec", "bottle", 2, 0))
    _write_csv(stage1_dir / "visa_support_summary.csv", _support_rows("visa", "candle", 2, 0))
    _write_json(stage1_dir / "mvtec_leakage_report.json", _leakage())
    _write_json(stage1_dir / "visa_leakage_report.json", _leakage())
    _write_json(
        stage1_dir / "fake_expert_eval" / "metrics_stub.json",
        {
            "num_predictions": 1,
            "num_joined": 1,
            "num_missing_labels": 0,
            "num_anomaly": 0,
            "num_normal": 1,
            "num_failed_predictions": 0,
        },
    )
    _write_json(
        stage1_dir / "fake_expert_eval" / "evaluation_failures.json",
        {"missing_predictions": [], "extra_predictions": [], "failed_predictions": []},
    )
    _write_csv(stage1_dir / "fake_expert_eval" / "per_sample_joined.csv", [{"image_id": "img-1", "has_evaluator_label": 1}])
    (stage1_dir / "pytest_server.log").write_text("========== 4 passed in 0.12s ==========\n", encoding="utf-8")
    _write_references(project_root)


def _write_actual_layout(project_root: Path, *, closed_evaluation: bool) -> Path:
    stage1_dir = project_root / "outputs" / "stage1_gate"
    _write_json(stage1_dir / "mvtec_full_audit.json", _audit("mvtec"))
    _write_json(stage1_dir / "visa_full_audit.json", _audit("visa"))
    for dataset, category in [("mvtec", "bottle"), ("visa", "candle")]:
        for k_shot in [1, 2]:
            for seed in [0, 1]:
                _write_csv(
                    project_root / "data" / "support_sets" / f"{dataset}_k{k_shot}_seed{seed}.csv",
                    _support_rows(dataset, category, k_shot, seed),
                )
                _write_json(
                    stage1_dir / "leakage" / f"{dataset}_k{k_shot}_seed{seed}_leakage.json",
                    _leakage(support_rows=k_shot),
                )
    _write_json(
        project_root / "outputs" / "evaluation_fake_expert" / "metrics_stub.json",
        {
            "num_predictions": 1,
            "num_joined": 1 if closed_evaluation else 0,
            "num_missing_labels": 0 if closed_evaluation else 1,
            "num_anomaly": 0,
            "num_normal": 1 if closed_evaluation else 0,
            "num_failed_predictions": 0,
        },
    )
    _write_json(
        project_root / "outputs" / "evaluation_fake_expert" / "evaluation_failures.json",
        {
            "missing_predictions": [] if closed_evaluation else ["img-1"],
            "extra_predictions": [] if closed_evaluation else ["smoke/image_000"],
            "failed_predictions": [],
        },
    )
    _write_csv(
        project_root / "outputs" / "evaluation_fake_expert" / "per_sample_joined.csv",
        [{"image_id": "img-1", "has_evaluator_label": 1 if closed_evaluation else 0}],
    )
    (stage1_dir / "pytest_server.log").write_text("========== 56 passed in 0.49s ==========\n", encoding="utf-8")
    _write_references(project_root)
    return stage1_dir


def test_build_report_marks_missing_inputs_and_does_not_invent_counts(tmp_path: Path) -> None:
    report = build_report(stage1_dir=tmp_path / "outputs" / "stage1_gate", project_root=tmp_path)

    assert "| `mvtec_full_audit.json` | missing | MVTec full audit |" in report
    assert "- audit file: missing" in report
    assert "- support summary/source files: missing" in report
    assert "- leakage report/source files: missing" in report
    assert "- fake expert metrics file: missing" in report
    assert "Gate结论: **NOT MET**" in report
    assert "FakeExpert metrics are a protocol smoke/evaluation isolation check" in report


def test_build_report_summarizes_summary_layout_and_passes_gate(tmp_path: Path) -> None:
    stage1_dir = tmp_path / "outputs" / "stage1_gate"
    _write_summary_layout(stage1_dir, tmp_path)

    report = build_report(stage1_dir=stage1_dir, project_root=tmp_path)

    assert "## 3. MVTec数据统计" in report
    assert "- train_good_images: 2" in report
    assert "- unique support_set_id: 1 (mvtec_bottle_k2_seed0)" in report
    assert "- ok: true" in report
    assert "| `num_predictions` | 1 |" in report
    assert "- missing_predictions: 0" in report
    assert "- pytest passed: true" in report
    assert "- none recorded from the provided artifacts." in report
    assert "Gate结论: **PASS**" in report


def test_build_report_summarizes_actual_directory_layout(tmp_path: Path) -> None:
    stage1_dir = _write_actual_layout(tmp_path, closed_evaluation=True)

    report = build_report(stage1_dir=stage1_dir, project_root=tmp_path)

    assert "`data/support_sets/mvtec_k*_seed*.csv` | present" in report
    assert "`outputs/stage1_gate/leakage/mvtec_*_leakage.json` | present" in report
    assert "`outputs/evaluation_fake_expert/metrics_stub.json` | present" in report
    assert "- layout: support_set_directory" in report
    assert "- source_files: 4" in report
    assert "- unique k_shot: 2 (1, 2)" in report
    assert "- layout: leakage_directory" in report
    assert "- report_files: 4" in report
    assert "- pytest passed: true" in report
    assert "Gate结论: **PASS**" in report


def test_build_report_keeps_gate_blocked_when_fake_expert_evaluation_is_not_closed(tmp_path: Path) -> None:
    stage1_dir = _write_actual_layout(tmp_path, closed_evaluation=False)

    report = build_report(stage1_dir=stage1_dir, project_root=tmp_path)

    assert "num_missing_labels=1" in report
    assert "num_joined=0 differs from num_predictions=1" in report
    assert "missing_predictions=1" in report
    assert "extra_predictions=1" in report
    assert "Gate结论: **NOT MET**" in report


def test_build_report_flags_parse_errors(tmp_path: Path) -> None:
    stage1_dir = tmp_path / "outputs" / "stage1_gate"
    _write_summary_layout(stage1_dir, tmp_path)
    (stage1_dir / "mvtec_full_audit.json").write_text("{not json", encoding="utf-8")

    report = build_report(stage1_dir=stage1_dir, project_root=tmp_path)

    assert "parse_error" in report
    assert "Gate结论: **NOT MET**" in report


def test_stage1_gate_report_cli_writes_output(tmp_path: Path) -> None:
    stage1_dir = _write_actual_layout(tmp_path, closed_evaluation=True)
    output = stage1_dir / "STAGE1_GATE_REPORT.md"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/generate_stage1_gate_report.py",
            "--stage1-dir",
            str(stage1_dir),
            "--project-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert output.is_file()
    assert "Wrote" in completed.stdout
    assert "Gate结论: **PASS**" in output.read_text(encoding="utf-8")
