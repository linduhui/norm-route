import csv
import json
from pathlib import Path

from src.normroute.agent.protocol import AgentTask
from src.normroute.cli.run_stage4_grid import Stage4GridConfig, run_stage4_grid
from src.normroute.evaluation.export import PREDICTION_COLUMNS


def _task(seed: int) -> AgentTask:
    return AgentTask(
        task_id=f"task-{seed}",
        sample_id=f"sample-{seed}",
        dataset="mvtec",
        category="bottle",
        k_shot=1,
        seed=seed,
        support_set_id=f"mvtec_bottle_k1_seed{seed}",
    )


def _manifest_row(task: AgentTask, fold: str, split: str) -> dict[str, object]:
    return {
        "fold": fold,
        "split": split,
        "task_id": task.task_id,
        "sample_id": task.sample_id,
        "dataset": task.dataset,
        "category": task.category,
        "k_shot": task.k_shot,
        "seed": task.seed,
        "support_set_id": task.support_set_id,
    }


def _write_stage2(stage2_root: Path, task: AgentTask, expert: str) -> None:
    normalized = "".join(character for character in expert.lower() if character.isalnum())
    run_dir = (
        stage2_root
        / normalized
        / task.dataset
        / task.category
        / f"k{task.k_shot}"
        / f"seed{task.seed}"
        / task.support_set_id
    )
    run_dir.mkdir(parents=True)
    row = {
        "image_id": task.sample_id,
        "expert_name": normalized,
        "dataset": task.dataset,
        "category": task.category,
        "support_set_id": task.support_set_id,
        "k_shot": task.k_shot,
        "seed": task.seed,
        "final_score": 0.5,
        "final_decision": "anomaly",
        "anomaly_map_path": "",
        "pixel_score_path": "",
        "actions": normalized.upper(),
        "tool_calls": 1,
        "runtime_ms": 2.0,
        "status": "ok",
        "error_message": "",
    }
    with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def test_stage4_grid_runs_multiple_folds_and_policies(tmp_path: Path) -> None:
    tasks = [_task(0), _task(1)]
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "".join(json.dumps(task.to_dict()) + "\n" for task in tasks),
        encoding="utf-8",
    )
    manifest_rows = [
        _manifest_row(tasks[0], "fold0", "test"),
        _manifest_row(tasks[1], "fold0", "train"),
        _manifest_row(tasks[0], "fold1", "train"),
        _manifest_row(tasks[1], "fold1", "test"),
    ]
    manifest_path = tmp_path / "fold_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    stage2_root = tmp_path / "stage2"
    for task in tasks:
        for expert in ("PatchCore", "WinCLIP"):
            _write_stage2(stage2_root, task, expert)

    result = run_stage4_grid(
        config=Stage4GridConfig(
            seed=11,
            folds=("fold0", "fold1"),
            policies=("always_patchcore", "always_winclip"),
            split="test",
            tool_budget=1,
        ),
        tasks_path=tasks_path,
        fold_manifest_path=manifest_path,
        stage2_root=stage2_root,
        output_root=tmp_path / "stage4",
    )

    assert result["num_total"] == 4
    assert result["num_failed"] == 0
    assert {row["fold"] for row in result["records"]} == {"fold0", "fold1"}
    assert {row["policy_name"] for row in result["records"]} == {
        "always_patchcore",
        "always_winclip",
    }
    for row in result["records"]:
        output_dir = Path(row["output_dir"])
        assert (output_dir / "selected_predictions.csv").is_file()
        assert (output_dir / "failures.json").is_file()
