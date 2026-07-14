import csv
from pathlib import Path

import pytest

from src.normroute.agent.replay_executor import SELECTED_PREDICTION_COLUMNS
from src.normroute.evaluation.stage4 import (
    Stage4EvaluationError,
    evaluate_selected_predictions,
    write_stage4_evaluation,
)


def _selected_row(index: int, label: int) -> dict[str, object]:
    return {
        "task_id": f"task-{index}",
        "fold": "fold0",
        "split": "test",
        "policy_name": "always_patchcore",
        "selected_expert": "PatchCore",
        "stage2_run_dir": "stage2/run",
        "image_id": f"sample-{index}",
        "expert_name": "patchcore",
        "dataset": "mvtec",
        "category": "bottle",
        "support_set_id": "mvtec_bottle_k1_seed0",
        "k_shot": 1,
        "seed": 0,
        "final_score": 0.9 if label else 0.1,
        "final_decision": "anomaly" if label else "normal",
        "anomaly_map_path": "",
        "pixel_score_path": "",
        "actions": "PATCHCORE_GLOBAL",
        "tool_calls": 1,
        "runtime_ms": 5.0,
        "status": "ok",
        "error_message": "",
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_evaluate_selected_predictions_reports_fold_metrics_without_test_tuning(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected_predictions.csv"
    rows = [_selected_row(0, 0), _selected_row(1, 1)]
    _write_csv(selected, list(SELECTED_PREDICTION_COLUMNS), rows)
    evaluator = tmp_path / "evaluator.csv"
    _write_csv(
        evaluator,
        ["image_id", "label", "mask_path", "defect_type"],
        [
            {"image_id": "sample-0", "label": 0, "mask_path": "", "defect_type": "good"},
            {"image_id": "sample-1", "label": 1, "mask_path": "mask.png", "defect_type": "scratch"},
        ],
    )

    result = evaluate_selected_predictions(
        selected_predictions=selected,
        evaluator_csv=evaluator,
    )
    metrics = result.metric_rows[0]
    assert metrics["fold"] == "fold0"
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["ap"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["estimated_runtime_ms"] == pytest.approx(10.0)
    assert metrics["tool_calls"] == 2

    outputs = write_stage4_evaluation(
        result=result,
        output_dir=tmp_path / "evaluator_only",
        selected_predictions=selected,
        evaluator_csv=evaluator,
    )
    assert all(path.is_file() for path in outputs.values())


def test_evaluator_refuses_to_silently_drop_failed_selected_prediction(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected_predictions.csv"
    row = _selected_row(0, 0)
    row["status"] = "error"
    row["error_message"] = "explicit failure"
    _write_csv(selected, list(SELECTED_PREDICTION_COLUMNS), [row])
    evaluator = tmp_path / "evaluator.csv"
    _write_csv(
        evaluator,
        ["image_id", "label"],
        [{"image_id": "sample-0", "label": 0}],
    )

    with pytest.raises(Stage4EvaluationError, match="cannot be silently excluded"):
        evaluate_selected_predictions(
            selected_predictions=selected,
            evaluator_csv=evaluator,
        )
