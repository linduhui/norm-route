import csv
import json
import subprocess
import sys
from pathlib import Path

from src.normroute.cli.validate_outputs import validate_output_tree
from src.normroute.evaluation.export import PREDICTION_COLUMNS


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_predictions(
    path: Path,
    rows: list[dict[str, object]],
    columns: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _prediction_row(
    *,
    expert_name: str = "dummy",
    support_set_id: str = "mvtec_bottle_k1_seed0",
    status: str = "ok",
) -> dict[str, object]:
    return {
        "image_id": "test-good-0",
        "expert_name": expert_name,
        "dataset": "mvtec",
        "category": "bottle",
        "support_set_id": support_set_id,
        "k_shot": 1,
        "seed": 0,
        "final_score": 0.1,
        "final_decision": "normal",
        "anomaly_map_path": "",
        "pixel_score_path": "",
        "actions": "",
        "tool_calls": 1,
        "runtime_ms": 2.0,
        "status": status,
        "error_message": "",
    }


def _write_run(
    run_dir: Path,
    *,
    expert_name: str = "dummy",
    support_set_id: str = "mvtec_bottle_k1_seed0",
    num_failed_samples: int = 0,
    failed_predictions: list[dict[str, object]] | None = None,
) -> None:
    _write_predictions(
        run_dir / "predictions.csv",
        [_prediction_row(expert_name=expert_name, support_set_id=support_set_id)],
    )
    _write_json(
        run_dir / "metrics.json",
        {
            "num_predictions": 1,
            "num_success": 1 - num_failed_samples,
            "num_failed_samples": num_failed_samples,
        },
    )
    _write_json(
        run_dir / "failures.json",
        {"failed_predictions": failed_predictions or []},
    )


def test_validate_outputs_accepts_consistent_stage2_tree(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    support_set_id = "mvtec_bottle_k1_seed0"
    _write_run(
        output_root / "dummy" / "mvtec" / "bottle" / "k1" / "seed0" / support_set_id,
        expert_name="dummy",
        support_set_id=support_set_id,
    )
    _write_run(
        output_root / "patchcore" / "mvtec" / "bottle" / "k1" / "seed0" / support_set_id,
        expert_name="patchcore",
        support_set_id=support_set_id,
    )

    assert validate_output_tree(output_root) == []


def test_validate_outputs_reports_schema_leakage_support_and_failure_errors(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"
    first_dir = (
        output_root
        / "dummy"
        / "mvtec"
        / "bottle"
        / "k1"
        / "seed0"
        / "mvtec_bottle_k1_seed0"
    )
    second_dir = (
        output_root
        / "patchcore"
        / "mvtec"
        / "bottle"
        / "k1"
        / "seed0"
        / "mvtec_bottle_k1_seed99"
    )
    forbidden_columns = [*PREDICTION_COLUMNS, "label"]
    row = _prediction_row(expert_name="dummy", support_set_id="mvtec_bottle_k1_seed0")
    row["label"] = "good"
    _write_predictions(first_dir / "predictions.csv", [row], columns=forbidden_columns)
    _write_json(first_dir / "metrics.json", {"num_failed_samples": 1})
    _write_json(first_dir / "failures.json", {"failed_predictions": []})
    _write_run(
        second_dir,
        expert_name="patchcore",
        support_set_id="mvtec_bottle_k1_seed99",
    )

    errors = validate_output_tree(output_root)

    assert any("forbidden prediction columns" in error for error in errors)
    assert any("failed_predictions" in error and "reports 1" in error for error in errors)
    assert any("Inconsistent support_set_id across experts" in error for error in errors)


def test_validate_outputs_reports_missing_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "dummy" / "mvtec" / "bottle" / "k1" / "seed0" / "sid"
    _write_json(run_dir / "metrics.json", {"num_failed_samples": 0})

    errors = validate_output_tree(tmp_path / "outputs")

    assert any("missing required artifact predictions.csv" in error for error in errors)
    assert any("missing required artifact failures.json" in error for error in errors)


def test_validate_outputs_cli_returns_nonzero_on_errors(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _write_run(
        output_root / "dummy" / "mvtec" / "bottle" / "k1" / "seed0" / "sid",
        num_failed_samples=1,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.normroute.cli.validate_outputs",
            str(output_root),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "ERROR:" in completed.stderr
