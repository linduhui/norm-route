import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.normroute.agent.policy import AlwaysAnomalyDINOPolicy
from src.normroute.agent.policy_registry import create_policy, list_policies
from src.normroute.agent.protocol import AgentTask, CANDIDATE_EXPERTS
from src.normroute.agent.replay_executor import (
    ReplayExecutionError,
    ROUTE_DECISION_COLUMNS,
    SELECTED_PREDICTION_COLUMNS,
    execute_replay,
)
from src.normroute.evaluation.export import PREDICTION_COLUMNS


ROOT = Path(__file__).resolve().parents[1]


def _task() -> AgentTask:
    return AgentTask(
        task_id="task-0",
        sample_id="sample-0",
        dataset="mvtec",
        category="bottle",
        k_shot=1,
        seed=0,
        support_set_id="mvtec_bottle_k1_seed0",
    )


def _manifest(task: AgentTask) -> list[dict[str, object]]:
    return [
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


def _prediction(expert_name: str = "anomalydino") -> dict[str, object]:
    return {
        "image_id": "sample-0",
        "expert_name": expert_name,
        "dataset": "mvtec",
        "category": "bottle",
        "support_set_id": "mvtec_bottle_k1_seed0",
        "k_shot": 1,
        "seed": 0,
        "final_score": 0.25,
        "final_decision": "normal",
        "anomaly_map_path": "maps/sample-0.png",
        "pixel_score_path": "scores/sample-0.csv",
        "actions": "ANOMALYDINO_TOKEN_NN",
        "tool_calls": 1,
        "runtime_ms": 12.5,
        "status": "ok",
        "error_message": "",
    }


def _write_stage2_run(
    stage2_root: Path, rows: list[dict[str, object]] | None = None
) -> Path:
    run_dir = (
        stage2_root
        / "anomalydino"
        / "mvtec"
        / "bottle"
        / "k1"
        / "seed0"
        / "mvtec_bottle_k1_seed0"
    )
    run_dir.mkdir(parents=True)
    with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows or [_prediction()])
    return run_dir


def test_always_anomalydino_policy_registry_and_persistence(tmp_path: Path) -> None:
    assert "always_anomalydino" in list_policies()
    policy = create_policy("always_anomalydino")
    decision = policy.select(_task())

    assert decision.selected_expert == "AnomalyDINO"
    assert decision.selected_expert in CANDIDATE_EXPERTS
    assert decision.tool_calls == 1
    assert {"label", "mask", "mask_path", "oracle_answer"}.isdisjoint(decision.to_dict())

    state_path = policy.save(tmp_path / "policy.json")
    loaded = AlwaysAnomalyDINOPolicy.load(state_path)
    assert loaded.select(_task()).to_dict() == decision.to_dict()


def test_replay_writes_required_outputs_from_selected_stage2_run(tmp_path: Path) -> None:
    task = _task()
    run_dir = _write_stage2_run(tmp_path / "outputs" / "stage2")
    output_dir = tmp_path / "outputs" / "stage4" / "run"

    result = execute_replay(
        tasks=[task],
        manifest_rows=_manifest(task),
        policy=AlwaysAnomalyDINOPolicy(),
        stage2_root=tmp_path / "outputs" / "stage2",
        output_dir=output_dir,
        fold="fold0",
        split="test",
        tool_budget=1,
    )

    assert result.num_tasks == 1
    assert result.num_selected_predictions == 1
    assert result.num_failures == 0
    for file_name in (
        "route_decisions.csv",
        "selected_predictions.csv",
        "failures.json",
        "run_metadata.json",
        "budget_summary.json",
    ):
        assert (output_dir / file_name).is_file()

    with result.route_decisions_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        decisions = list(reader)
    assert reader.fieldnames == list(ROUTE_DECISION_COLUMNS)
    assert decisions[0]["selected_expert"] == "AnomalyDINO"
    assert decisions[0]["fold"] == "fold0"
    assert decisions[0]["split"] == "test"

    with result.selected_predictions_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        selected = list(reader)
    assert reader.fieldnames == list(SELECTED_PREDICTION_COLUMNS)
    assert selected[0]["image_id"] == task.sample_id
    assert selected[0]["stage2_run_dir"] == str(run_dir)
    assert selected[0]["expert_name"] == "anomalydino"
    assert selected[0]["runtime_source"] == "estimated_runtime"
    assert selected[0]["estimated_runtime_ms"] == "12.5"
    assert selected[0]["actual_runtime_ms"] == ""

    failures = json.loads(result.failures_path.read_text(encoding="utf-8"))
    assert failures["failed_tasks"] == []
    metadata = json.loads(result.run_metadata_path.read_text(encoding="utf-8"))
    assert metadata["config"]["policy_name"] == "always_anomalydino"
    assert metadata["git_commit"]
    assert metadata["environment"]["python_version"]


def test_replay_records_mixed_expert_stage2_run_as_failure(tmp_path: Path) -> None:
    task = _task()
    second = _prediction(expert_name="patchcore")
    second["image_id"] = "another-sample"
    _write_stage2_run(tmp_path / "stage2", [_prediction(), second])

    result = execute_replay(
        tasks=[task],
        manifest_rows=_manifest(task),
        policy=AlwaysAnomalyDINOPolicy(),
        stage2_root=tmp_path / "stage2",
        output_dir=tmp_path / "stage4",
        fold="fold0",
    )

    assert result.num_failures == 1
    assert result.num_selected_predictions == 0
    failures = json.loads(result.failures_path.read_text(encoding="utf-8"))
    assert failures["failed_tasks"][0]["failure_type"] == "replay_error"
    assert "exactly one expert" in failures["failed_tasks"][0]["error_message"]


def test_run_agent_cli_completes_always_anomalydino_smoke(tmp_path: Path) -> None:
    task = _task()
    stage2_root = tmp_path / "outputs" / "stage2"
    _write_stage2_run(stage2_root)
    tasks_path = tmp_path / "pre_route_tasks.jsonl"
    tasks_path.write_text(json.dumps(task.to_dict()) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "fold_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_manifest(task)[0]))
        writer.writeheader()
        writer.writerows(_manifest(task))
    output_dir = tmp_path / "outputs" / "stage4" / "cli-run"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.normroute.cli.run_agent",
            "--policy",
            "always_anomalydino",
            "--tasks",
            str(tasks_path),
            "--fold-manifest",
            str(manifest_path),
            "--stage2-root",
            str(stage2_root),
            "--output-dir",
            str(output_dir),
            "--fold",
            "fold0",
            "--split",
            "test",
            "--budget",
            "1",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "route_decisions.csv" in completed.stdout
    assert (output_dir / "route_decisions.csv").is_file()
    assert (output_dir / "selected_predictions.csv").is_file()


def test_replay_rejects_evaluator_fields_in_direct_manifest_input(tmp_path: Path) -> None:
    task = _task()
    _write_stage2_run(tmp_path / "stage2")
    manifest = _manifest(task)
    manifest[0]["label"] = "anomaly"

    with pytest.raises(ReplayExecutionError, match="forbidden fields"):
        execute_replay(
            tasks=[task],
            manifest_rows=manifest,
            policy=AlwaysAnomalyDINOPolicy(),
            stage2_root=tmp_path / "stage2",
            output_dir=tmp_path / "stage4",
            fold="fold0",
        )
