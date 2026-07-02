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


def _leakage() -> dict[str, object]:
    return {
        "ok": True,
        "error_count": 0,
        "warning_count": 0,
        "counts": {
            "agent_input_rows": 4,
            "evaluator_rows": 4,
            "support_rows": 2,
            "support_sets": 1,
        },
        "issues": [],
    }


def _write_complete_inputs(stage1_dir: Path, project_root: Path) -> None:
    _write_json(stage1_dir / "mvtec_full_audit.json", _audit("mvtec"))
    _write_json(stage1_dir / "visa_full_audit.json", _audit("visa"))
    _write_csv(
        stage1_dir / "mvtec_support_summary.csv",
        [
            {
                "dataset": "mvtec",
                "support_set_id": "mvtec_bottle_k2_seed0",
                "k_shot": 2,
                "seed": 0,
                "category": "bottle",
            }
        ],
    )
    _write_csv(
        stage1_dir / "visa_support_summary.csv",
        [
            {
                "dataset": "visa",
                "support_set_id": "visa_candle_k2_seed0",
                "k_shot": 2,
                "seed": 0,
                "category": "candle",
            }
        ],
    )
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

    (project_root / "docs").mkdir(parents=True, exist_ok=True)
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    (project_root / "docs" / "task_spec.md").write_text("# Task\n", encoding="utf-8")
    (project_root / "docs" / "data_leakage_policy.md").write_text("# Policy\n", encoding="utf-8")
    (project_root / "configs" / "folds.yaml").write_text("fold_1:\n  - bottle\n", encoding="utf-8")


def test_build_report_marks_missing_inputs_and_does_not_invent_counts(tmp_path: Path) -> None:
    report = build_report(stage1_dir=tmp_path / "outputs" / "stage1_gate", project_root=tmp_path)

    assert "| `mvtec_full_audit.json` | missing | MVTec full audit |" in report
    assert "- audit file: missing" in report
    assert "- support summary file: missing" in report
    assert "- leakage report file: missing" in report
    assert "- fake expert metrics file: missing" in report
    assert "Gate结论: **NOT MET**" in report
    assert "FakeExpert metrics are a protocol smoke/evaluation isolation check" in report


def test_build_report_summarizes_complete_artifacts_and_passes_gate(tmp_path: Path) -> None:
    stage1_dir = tmp_path / "outputs" / "stage1_gate"
    _write_complete_inputs(stage1_dir, tmp_path)

    report = build_report(stage1_dir=stage1_dir, project_root=tmp_path)

    assert "## 3. MVTec数据统计" in report
    assert "- train_good_images: 2" in report
    assert "- test_anomaly_images: 1" in report
    assert "- unique support_set_id: 1 (mvtec_bottle_k2_seed0)" in report
    assert "- ok: true" in report
    assert "| `num_predictions` | 1 |" in report
    assert "FakeExpert 只用于闭环 smoke test" in report
    assert "- none recorded from the provided artifacts." in report
    assert "Gate结论: **PASS**" in report


def test_build_report_flags_parse_errors_and_failed_fake_expert_metrics(tmp_path: Path) -> None:
    stage1_dir = tmp_path / "outputs" / "stage1_gate"
    _write_complete_inputs(stage1_dir, tmp_path)
    (stage1_dir / "mvtec_full_audit.json").write_text("{not json", encoding="utf-8")
    _write_json(
        stage1_dir / "fake_expert_eval" / "metrics_stub.json",
        {
            "num_predictions": 1,
            "num_joined": 0,
            "num_missing_labels": 1,
            "num_anomaly": 0,
            "num_normal": 0,
            "num_failed_predictions": 0,
        },
    )

    report = build_report(stage1_dir=stage1_dir, project_root=tmp_path)

    assert "parse_error" in report
    assert "num_missing_labels=1" in report
    assert "Gate结论: **NOT MET**" in report


def test_stage1_gate_report_cli_writes_output(tmp_path: Path) -> None:
    stage1_dir = tmp_path / "outputs" / "stage1_gate"
    output = stage1_dir / "STAGE1_GATE_REPORT.md"
    _write_complete_inputs(stage1_dir, tmp_path)

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
