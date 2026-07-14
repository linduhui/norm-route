import csv
import json
from pathlib import Path

from src.normroute.agent.policy import AlwaysAnomalyDINOPolicy
from src.normroute.agent.protocol import AgentTask
from src.normroute.agent.replay_executor import execute_replay
from src.normroute.evaluation.export import PREDICTION_COLUMNS


def _inputs(tmp_path: Path, *, prediction_tool_calls: int = 1):
    task = AgentTask(
        task_id="task-budget",
        sample_id="sample-budget",
        dataset="mvtec",
        category="bottle",
        k_shot=1,
        seed=0,
        support_set_id="mvtec_bottle_k1_seed0",
    )
    manifest = [
        {
            "fold": "fold0",
            "split": "test",
            "task_id": task.task_id,
            "sample_id": task.sample_id,
            "dataset": task.dataset,
            "category": task.category,
            "k_shot": task.k_shot,
            "seed": task.seed,
            "support_set_id": task.support_set_id,
        }
    ]
    run_dir = (
        tmp_path
        / "stage2"
        / "anomalydino"
        / "mvtec"
        / "bottle"
        / "k1"
        / "seed0"
        / task.support_set_id
    )
    run_dir.mkdir(parents=True)
    prediction = {
        "image_id": task.sample_id,
        "expert_name": "anomalydino",
        "dataset": task.dataset,
        "category": task.category,
        "support_set_id": task.support_set_id,
        "k_shot": task.k_shot,
        "seed": task.seed,
        "final_score": 0.5,
        "final_decision": "anomaly",
        "anomaly_map_path": "",
        "pixel_score_path": "",
        "actions": "ANOMALYDINO_TOKEN_NN",
        "tool_calls": prediction_tool_calls,
        "runtime_ms": 10.0,
        "status": "ok",
        "error_message": "",
    }
    with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerow(prediction)
    return task, manifest, tmp_path / "stage2"


def test_replay_stops_before_stage2_lookup_when_decision_exceeds_budget(
    tmp_path: Path,
) -> None:
    task, manifest, stage2_root = _inputs(tmp_path)
    result = execute_replay(
        tasks=[task],
        manifest_rows=manifest,
        policy=AlwaysAnomalyDINOPolicy(),
        stage2_root=stage2_root,
        output_dir=tmp_path / "stage4-budget-zero",
        fold="fold0",
        tool_budget=0,
    )

    summary = json.loads(result.budget_summary_path.read_text(encoding="utf-8"))
    assert result.num_failures == 1
    assert result.num_selected_predictions == 0
    assert summary["num_budget_failures"] == 1
    assert summary["within_budget"] is False
    assert summary["total_tool_call_budget"] == 0
    assert summary["planned_tool_calls"] == 1


def test_replay_rejects_stage2_prediction_that_exceeds_budget(tmp_path: Path) -> None:
    task, manifest, stage2_root = _inputs(tmp_path, prediction_tool_calls=2)
    result = execute_replay(
        tasks=[task],
        manifest_rows=manifest,
        policy=AlwaysAnomalyDINOPolicy(),
        stage2_root=stage2_root,
        output_dir=tmp_path / "stage4-stage2-over-budget",
        fold="fold0",
        tool_budget=1,
    )

    failures = json.loads(result.failures_path.read_text(encoding="utf-8"))
    summary = json.loads(result.budget_summary_path.read_text(encoding="utf-8"))
    assert result.num_failures == 1
    assert result.num_selected_predictions == 0
    assert failures["failed_tasks"][0]["failure_type"] == "budget_exceeded"
    assert summary["num_budget_failures"] == 1
    assert summary["within_budget"] is False
