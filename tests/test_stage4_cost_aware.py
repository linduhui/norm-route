import csv
import json
from pathlib import Path

import pytest

from src.normroute.agent.policy_registry import create_policy, list_policies
from src.normroute.agent.protocol import AgentTask
from src.normroute.policies.cost_aware import (
    DEFAULT_LAMBDA_GRID,
    ESTIMATED_RUNTIME,
    CostAwarePolicy,
    CostAwarePolicyError,
    calibrate_cost_aware_policy,
    cost_aware_utility,
)


QUALITY_COLUMNS = [
    "expert",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
    "image_auroc",
    "image_ap",
    "average_runtime_ms",
]
MANIFEST_COLUMNS = [
    "fold",
    "split",
    "task_id",
    "sample_id",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
]
EXPERTS = ("PatchCore", "WinCLIP", "AnomalyDINO")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _inputs(
    tmp_path: Path,
    *,
    validation_quality: tuple[float, float, float] = (0.4, 0.9, 0.3),
    test_quality: tuple[object, object, object] = ("invalid", "invalid", "invalid"),
) -> tuple[Path, Path]:
    split_by_seed = {0: "train", 3: "train", 1: "val", 2: "test"}
    manifest_rows = []
    for seed, split in split_by_seed.items():
        manifest_rows.append(
            {
                "fold": "fold0",
                "split": split,
                "task_id": f"task-{seed}",
                "sample_id": f"sample-{seed}",
                "dataset": "mvtec",
                "category": "bottle",
                "k_shot": 1,
                "seed": seed,
                "support_set_id": f"support-{seed}",
            }
        )
    manifest = tmp_path / "fold_manifest.csv"
    _write_csv(manifest, MANIFEST_COLUMNS, manifest_rows)

    quality_rows: list[dict[str, object]] = []
    train_values = {
        0: ((0.8, 80.0), (0.85, 8.0), (0.5, 18.0)),
        3: ((1.0, 120.0), (0.85, 12.0), (0.5, 22.0)),
    }
    for seed, values in train_values.items():
        for expert, (quality, runtime) in zip(EXPERTS, values):
            quality_rows.append(_quality_row(seed, expert, quality, runtime))
    for expert, quality in zip(EXPERTS, validation_quality):
        # Validation runtime is deliberately invalid: calibration must use the
        # train-fold estimate for cost and parse validation quality only.
        quality_rows.append(_quality_row(1, expert, quality, "not-read"))
    for expert, quality in zip(EXPERTS, test_quality):
        # Neither test quality nor test runtime may be parsed or hashed.
        quality_rows.append(_quality_row(2, expert, quality, "not-read"))
    quality = tmp_path / "expert_quality_by_run.csv"
    _write_csv(quality, QUALITY_COLUMNS, quality_rows)
    return manifest, quality


def _quality_row(
    seed: int, expert: str, quality: object, runtime: object
) -> dict[str, object]:
    return {
        "expert": expert,
        "dataset": "mvtec",
        "category": "bottle",
        "k_shot": 1,
        "seed": seed,
        "support_set_id": f"support-{seed}",
        "image_auroc": quality,
        "image_ap": quality,
        "average_runtime_ms": runtime,
    }


def _task() -> AgentTask:
    return AgentTask(
        task_id="test-task",
        sample_id="test-sample",
        dataset="mvtec",
        category="bottle",
        k_shot=1,
        seed=2,
        support_set_id="support-2",
    )


def test_cost_aware_utility_uses_required_formula() -> None:
    assert DEFAULT_LAMBDA_GRID == (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
    value = cost_aware_utility(
        0.75,
        30.0,
        quality_values=(0.5, 0.75, 1.0),
        runtime_values=(10.0, 30.0, 50.0),
        lambda_value=0.2,
    )
    assert value == pytest.approx(0.5 - 0.2 * 0.5)


def test_train_averages_validation_lambda_and_frontier_are_persisted(
    tmp_path: Path,
) -> None:
    manifest, quality = _inputs(tmp_path)
    artifact_path, frontier_path = calibrate_cost_aware_policy(
        expert_quality_by_run=quality,
        fold_manifest=manifest,
        fold="fold0",
        output_path=tmp_path / "policy_artifact.json",
    )

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["selected_lambda"] == pytest.approx(0.2)
    assert artifact["train_seeds"] == [0, 3]
    assert artifact["validation_seeds"] == [1]
    assert artifact["runtime_source"] == ESTIMATED_RUNTIME
    exact = artifact["statistics"]["category_shot"]
    patchcore = next(row for row in exact if row["expert"] == "PatchCore")
    assert patchcore["average_quality"] == pytest.approx(0.9)
    assert patchcore["average_runtime_ms"] == pytest.approx(100.0)

    with frontier_path.open(newline="", encoding="utf-8") as handle:
        frontier = list(csv.DictReader(handle))
    assert [float(row["lambda"]) for row in frontier] == list(DEFAULT_LAMBDA_GRID)
    assert sum(row["selected"] == "true" for row in frontier) == 1
    assert all(row["runtime_source"] == ESTIMATED_RUNTIME for row in frontier)

    assert "cost_aware" in list_policies()
    policy = create_policy("cost_aware", artifact=artifact_path)
    decision = policy.select(_task())
    assert decision.selected_expert == "WinCLIP"
    assert decision.estimated_cost_ms == pytest.approx(10.0)
    assert ESTIMATED_RUNTIME in decision.decision_reason
    assert "utility=" in decision.decision_reason


def test_test_quality_cannot_change_final_policy_artifact(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    manifest_a, quality_a = _inputs(first_dir, test_quality=(0.0, 0.0, 1.0))
    manifest_b, quality_b = _inputs(second_dir, test_quality=(1.0, 1.0, 0.0))
    artifact_a, _ = calibrate_cost_aware_policy(
        expert_quality_by_run=quality_a,
        fold_manifest=manifest_a,
        fold="fold0",
        output_path=first_dir / "policy_artifact.json",
    )
    artifact_b, _ = calibrate_cost_aware_policy(
        expert_quality_by_run=quality_b,
        fold_manifest=manifest_b,
        fold="fold0",
        output_path=second_dir / "policy_artifact.json",
    )
    assert json.loads(artifact_a.read_text(encoding="utf-8")) == json.loads(
        artifact_b.read_text(encoding="utf-8")
    )


def test_lambda_selection_uses_validation_and_not_test(tmp_path: Path) -> None:
    manifest, quality = _inputs(
        tmp_path,
        validation_quality=(0.95, 0.5, 0.3),
        test_quality=(0.0, 1.0, 0.0),
    )
    artifact_path, _ = calibrate_cost_aware_policy(
        expert_quality_by_run=quality,
        fold_manifest=manifest,
        fold="fold0",
        output_path=tmp_path / "policy_artifact.json",
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["selected_lambda"] == pytest.approx(0.0)


def test_max_runtime_ms_is_a_hard_route_constraint(tmp_path: Path) -> None:
    manifest, quality = _inputs(tmp_path)
    artifact_path, _ = calibrate_cost_aware_policy(
        expert_quality_by_run=quality,
        fold_manifest=manifest,
        fold="fold0",
        output_path=tmp_path / "policy_artifact.json",
        max_runtime_ms=50.0,
    )
    decision = CostAwarePolicy.load(artifact_path).select(_task())
    assert decision.selected_expert == "WinCLIP"
    assert decision.estimated_cost_ms <= 50.0
    assert "max_runtime_ms=50" in decision.decision_reason

    with pytest.raises(CostAwarePolicyError, match="No lambda.*feasible"):
        calibrate_cost_aware_policy(
            expert_quality_by_run=quality,
            fold_manifest=manifest,
            fold="fold0",
            output_path=tmp_path / "impossible" / "policy_artifact.json",
            max_runtime_ms=5.0,
        )
