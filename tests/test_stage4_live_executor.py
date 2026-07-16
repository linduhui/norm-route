import csv
import json
from pathlib import Path
import subprocess

from src.normroute.agent.live_executor import (
    LIVE_EXECUTION_COLUMNS,
    LIVE_SELECTED_PREDICTION_COLUMNS,
    execute_live,
)
from src.normroute.agent.policy import AlwaysAnomalyDINOPolicy
from src.normroute.agent.protocol import AgentTask
from src.normroute.agent.replay_executor import SELECTED_PREDICTION_COLUMNS
from src.normroute.evaluation import stage4 as stage4_evaluation
from src.normroute.evaluation.export import PREDICTION_COLUMNS


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


def _write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_live_inputs(root: Path) -> None:
    _write_csv(
        root / "data" / "manifests" / "mvtec_agent_input.csv",
        ["image_id", "dataset", "category", "split", "image_path"],
        [
            {
                "image_id": "another-sample",
                "dataset": "mvtec",
                "category": "bottle",
                "split": "test",
                "image_path": "another.png",
            },
            {
                "image_id": "sample-0",
                "dataset": "mvtec",
                "category": "bottle",
                "split": "test",
                "image_path": "query.png",
            },
        ],
    )
    _write_csv(
        root / "data" / "support_sets" / "mvtec_k1_seed0.csv",
        [
            "support_set_id",
            "dataset",
            "category",
            "k_shot",
            "seed",
            "support_rank",
            "image_id",
            "image_path",
        ],
        [
            {
                "support_set_id": "mvtec_bottle_k1_seed0",
                "dataset": "mvtec",
                "category": "bottle",
                "k_shot": 1,
                "seed": 0,
                "support_rank": 0,
                "image_id": "support-0",
                "image_path": "support.png",
            }
        ],
    )


def _prediction() -> dict[str, object]:
    return {
        "image_id": "sample-0",
        "expert_name": "anomalydino",
        "dataset": "mvtec",
        "category": "bottle",
        "support_set_id": "mvtec_bottle_k1_seed0",
        "k_shot": 1,
        "seed": 0,
        "final_score": 0.2,
        "final_decision": "normal",
        "anomaly_map_path": "",
        "pixel_score_path": "",
        "actions": "ANOMALYDINO_TOKEN_NN",
        "tool_calls": 1,
        "runtime_ms": 17.0,
        "status": "ok",
        "error_message": "",
    }


def test_live_and_replay_use_identical_selected_prediction_schema() -> None:
    assert tuple(LIVE_SELECTED_PREDICTION_COLUMNS) == tuple(
        SELECTED_PREDICTION_COLUMNS
    )


def test_live_executor_mock_subprocess_smoke_and_evaluator_order(
    tmp_path: Path, monkeypatch
) -> None:
    task = _task()
    _write_live_inputs(tmp_path)
    evaluator_csv = tmp_path / "data" / "manifests" / "mvtec_evaluator.csv"
    _write_csv(
        evaluator_csv,
        ["image_id", "label"],
        [{"image_id": "sample-0", "label": 0}],
    )
    state = {"expert_completed": False, "evaluator_reads": 0}
    calls: list[list[str]] = []

    original_read_evaluator = stage4_evaluation._read_evaluator_labels

    def tracked_read_evaluator(path):
        assert state["expert_completed"] is True
        state["evaluator_reads"] += 1
        return original_read_evaluator(path)

    monkeypatch.setattr(
        stage4_evaluation, "_read_evaluator_labels", tracked_read_evaluator
    )

    def mock_subprocess(command, **kwargs):
        del kwargs
        calls.append(command)
        assert state["evaluator_reads"] == 0
        assert command[1:3] == ["-m", "normroute.cli.run_expert"]
        task_manifest = Path(command[command.index("--agent-input-csv") + 1])
        with task_manifest.open(newline="", encoding="utf-8") as handle:
            scoped_rows = list(csv.DictReader(handle))
        assert [row["image_id"] for row in scoped_rows] == ["sample-0"]
        output_dir = Path(command[command.index("--output-dir") + 1])
        _write_csv(output_dir / "predictions.csv", PREDICTION_COLUMNS, [_prediction()])
        state["expert_completed"] = True
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout="mock expert stdout\n",
            stderr="mock expert stderr\n",
        )

    ticks = iter([10.0, 10.25])
    output_dir = tmp_path / "outputs" / "stage4" / "live"
    result = execute_live(
        tasks=[task],
        manifest_rows=_manifest(task),
        policy=AlwaysAnomalyDINOPolicy(),
        data_root="data",
        output_dir=output_dir,
        fold="fold0",
        evaluator_csv=evaluator_csv,
        project_root=tmp_path,
        python_executable="python",
        subprocess_runner=mock_subprocess,
        clock=lambda: next(ticks),
    )

    assert len(calls) == 1
    assert result.num_tasks == 1
    assert result.num_selected_predictions == 1
    assert result.num_failures == 0
    assert result.num_evaluation_failures == 0
    assert state["evaluator_reads"] == 1
    assert result.evaluation_paths["metrics_path"].is_file()
    assert result.stdout_path.read_text(encoding="utf-8").endswith(
        "mock expert stdout\n"
    )
    assert result.stderr_path.read_text(encoding="utf-8").endswith(
        "mock expert stderr\n"
    )

    with result.live_executions_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        executions = list(reader)
    assert reader.fieldnames == list(LIVE_EXECUTION_COLUMNS)
    assert float(executions[0]["actual_runtime_ms"]) == 250.0
    assert float(executions[0]["estimated_runtime_ms"]) == 0.0
    assert Path(executions[0]["stdout_log"]).is_file()
    assert Path(executions[0]["stderr_log"]).is_file()

    with result.selected_predictions_path.open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)
        selected = list(reader)
    assert reader.fieldnames == list(LIVE_SELECTED_PREDICTION_COLUMNS)
    assert selected[0]["runtime_source"] == "actual_runtime"
    assert float(selected[0]["actual_runtime_ms"]) == 250.0
    assert selected[0]["image_id"] == "sample-0"


def test_live_executor_records_nonzero_subprocess_once(tmp_path: Path) -> None:
    task = _task()
    _write_live_inputs(tmp_path)
    calls: list[list[str]] = []

    def failed_subprocess(command, **kwargs):
        del kwargs
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            returncode=7,
            stdout="partial stdout\n",
            stderr="expert crashed\n",
        )

    ticks = iter([20.0, 20.1])
    result = execute_live(
        tasks=[task],
        manifest_rows=_manifest(task),
        policy=AlwaysAnomalyDINOPolicy(),
        data_root="data",
        output_dir=tmp_path / "live-failed",
        fold="fold0",
        project_root=tmp_path,
        subprocess_runner=failed_subprocess,
        clock=lambda: next(ticks),
    )

    assert len(calls) == 1
    assert result.num_failures == 1
    assert result.num_selected_predictions == 0
    failures = json.loads(result.failures_path.read_text(encoding="utf-8"))
    failure = failures["failed_tasks"][0]
    assert failure["failure_type"] == "expert_subprocess_failed"
    assert failure["returncode"] == 7
    assert Path(failure["stdout_log"]).read_text(encoding="utf-8") == "partial stdout\n"
    assert Path(failure["stderr_log"]).read_text(encoding="utf-8") == "expert crashed\n"
