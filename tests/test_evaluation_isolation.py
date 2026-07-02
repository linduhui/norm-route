import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.evaluation.join_predictions import EvaluationJoinError, join_predictions_with_evaluator
from src.evaluation.metrics_stub import compute_metrics_stub
from src.evaluation.reporting import write_evaluation_outputs
from src.experts.results import ExpertResult


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _prediction(image_id: str, *, status: str = "ok") -> ExpertResult:
    return ExpertResult(
        schema_version="1.0",
        expert_name="fake_expert",
        image_id=image_id,
        support_set_id="support-1",
        raw_score=0.5,
        normalized_score=0.5,
        anomaly_map_path=f"maps/{image_id}.png",
        runtime_ms=0.0,
        peak_memory_mb=0.0,
        model_version="fake-0",
        weight_hash="no-weights",
        status=status,
        error_message="" if status == "ok" else "failed",
    )


def _write_predictions(path: Path, image_ids: list[str], *, failed: set[str] | None = None) -> None:
    failed = failed or set()
    path.write_text(
        "".join(_prediction(image_id, status="error" if image_id in failed else "ok").to_jsonl_line() for image_id in image_ids),
        encoding="utf-8",
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    predictions = tmp_path / "predictions.jsonl"
    evaluator = tmp_path / "evaluator.csv"
    agent_input = tmp_path / "agent_input.csv"
    output_dir = tmp_path / "eval"

    _write_predictions(predictions, ["img-normal", "img-anomaly"])
    _write_csv(
        evaluator,
        ["image_id", "label", "mask_path", "defect_type"],
        [
            {"image_id": "img-normal", "label": 0, "mask_path": "", "defect_type": "good"},
            {"image_id": "img-anomaly", "label": 1, "mask_path": "/masks/img-anomaly.png", "defect_type": "crack"},
        ],
    )
    _write_csv(
        agent_input,
        ["image_id", "dataset", "category", "split", "image_path"],
        [
            {"image_id": "img-normal", "dataset": "mvtec", "category": "bottle", "split": "test", "image_path": "/images/normal.png"},
            {"image_id": "img-anomaly", "dataset": "mvtec", "category": "bottle", "split": "test", "image_path": "/images/anomaly.png"},
        ],
    )
    return predictions, evaluator, agent_input, output_dir


def test_join_uses_only_image_id_and_writes_debug_csv_under_output_dir(tmp_path: Path) -> None:
    predictions, evaluator, agent_input, output_dir = _write_inputs(tmp_path)

    joined = join_predictions_with_evaluator(
        predictions_jsonl=predictions,
        evaluator_csv=evaluator,
        agent_input_csv=agent_input,
    )
    metrics = compute_metrics_stub(joined)
    metrics_path, joined_path, failures_path = write_evaluation_outputs(
        output_dir=output_dir,
        joined=joined,
        metrics=metrics,
    )

    assert metrics == {
        "num_predictions": 2,
        "num_joined": 2,
        "num_missing_labels": 0,
        "num_anomaly": 1,
        "num_normal": 1,
        "num_failed_predictions": 0,
    }
    assert metrics_path.parent == output_dir
    assert joined_path.parent == output_dir
    assert failures_path.parent == output_dir

    rows = list(csv.DictReader(joined_path.open("r", newline="", encoding="utf-8")))
    assert [row["image_id"] for row in rows] == ["img-normal", "img-anomaly"]
    assert rows[1]["label"] == "1"
    assert rows[1]["mask_path"] == "/masks/img-anomaly.png"


def test_prediction_jsonl_forbidden_ground_truth_fields_fail_evaluation(tmp_path: Path) -> None:
    predictions, evaluator, agent_input, _ = _write_inputs(tmp_path)
    payload = json.loads(_prediction("img-normal").to_jsonl_line())
    payload["label"] = 0
    predictions.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(EvaluationJoinError, match="Forbidden ExpertResult fields"):
        join_predictions_with_evaluator(
            predictions_jsonl=predictions,
            evaluator_csv=evaluator,
            agent_input_csv=agent_input,
        )


def test_missing_predictions_are_recorded_not_silently_ignored(tmp_path: Path) -> None:
    predictions, evaluator, agent_input, output_dir = _write_inputs(tmp_path)
    _write_predictions(predictions, ["img-normal"])

    joined = join_predictions_with_evaluator(
        predictions_jsonl=predictions,
        evaluator_csv=evaluator,
        agent_input_csv=agent_input,
    )
    metrics = compute_metrics_stub(joined)
    _, _, failures_path = write_evaluation_outputs(output_dir=output_dir, joined=joined, metrics=metrics)
    failures = json.loads(failures_path.read_text(encoding="utf-8"))

    assert joined.missing_predictions == ["img-anomaly"]
    assert failures["missing_predictions"] == ["img-anomaly"]
    assert metrics["num_predictions"] == 1
    assert metrics["num_joined"] == 1


def test_failed_prediction_status_is_counted(tmp_path: Path) -> None:
    predictions, evaluator, agent_input, _ = _write_inputs(tmp_path)
    _write_predictions(predictions, ["img-normal", "img-anomaly"], failed={"img-anomaly"})

    joined = join_predictions_with_evaluator(
        predictions_jsonl=predictions,
        evaluator_csv=evaluator,
        agent_input_csv=agent_input,
    )

    assert compute_metrics_stub(joined)["num_failed_predictions"] == 1


def test_evaluate_predictions_cli_writes_outputs_and_returns_nonzero_for_missing(tmp_path: Path) -> None:
    predictions, evaluator, agent_input, output_dir = _write_inputs(tmp_path)
    _write_predictions(predictions, ["img-normal"])

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_predictions.py",
            "--predictions-jsonl",
            str(predictions),
            "--evaluator-csv",
            str(evaluator),
            "--agent-input-csv",
            str(agent_input),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert (output_dir / "metrics_stub.json").is_file()
    assert (output_dir / "per_sample_joined.csv").is_file()
    failures = json.loads((output_dir / "evaluation_failures.json").read_text(encoding="utf-8"))
    assert failures["missing_predictions"] == ["img-anomaly"]
