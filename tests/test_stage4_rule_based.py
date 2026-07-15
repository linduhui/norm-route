import csv
import json
from pathlib import Path

import pytest

from src.normroute.agent.protocol import AgentTask, CANDIDATE_EXPERTS
from src.normroute.agent.replay_executor import ReplayExecutionError, execute_replay
from src.normroute.cli.calibrate_policy import (
    PolicyCalibrationError,
    calibrate_policy,
    calibrate_policy_artifacts,
)
from src.normroute.policies.rule_based import (
    CategoryPriorPolicy,
    CategoryShotPriorPolicy,
    RUNTIME_TIE_BREAK,
)


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
QUALITY_COLUMNS = [
    "expert",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
    "image_auroc",
    "image_ap",
    "runtime_ms",
]


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _manifest_row(
    *,
    category: str,
    k_shot: int,
    seed: int,
    split: str = "train",
    fold: str = "fold0",
) -> dict[str, object]:
    support_set_id = f"mvtec_{category}_k{k_shot}_seed{seed}"
    return {
        "fold": fold,
        "split": split,
        "task_id": f"{fold}-{category}-k{k_shot}-seed{seed}",
        "sample_id": f"sample-{category}-k{k_shot}-seed{seed}",
        "dataset": "mvtec",
        "category": category,
        "k_shot": k_shot,
        "seed": seed,
        "support_set_id": support_set_id,
    }


def _quality_rows(
    manifest_row: dict[str, object],
    *,
    auroc: tuple[object, object, object],
    image_ap: tuple[object, object, object] | None = None,
    runtimes: tuple[object, object, object] = (5.0, 3.0, 7.0),
) -> list[dict[str, object]]:
    ap_values = image_ap or auroc
    return [
        {
            "expert": expert,
            "dataset": manifest_row["dataset"],
            "category": manifest_row["category"],
            "k_shot": manifest_row["k_shot"],
            "seed": manifest_row["seed"],
            "support_set_id": manifest_row["support_set_id"],
            "image_auroc": auroc[index],
            "image_ap": ap_values[index],
            "runtime_ms": runtimes[index],
        }
        for index, expert in enumerate(CANDIDATE_EXPERTS)
    ]


def _task(category: str, k_shot: int, index: int) -> AgentTask:
    return AgentTask(
        task_id=f"task-{index}",
        sample_id=f"sample-{index}",
        dataset="mvtec",
        category=category,
        k_shot=k_shot,
        seed=99,
        support_set_id=f"mvtec_{category}_k{k_shot}_seed99",
    )


def test_calibrator_never_reads_or_hashes_test_quality(tmp_path: Path) -> None:
    train = _manifest_row(category="bottle", k_shot=1, seed=0, split="train")
    test = _manifest_row(category="bottle", k_shot=1, seed=1, split="test")
    manifest = tmp_path / "fold_manifest.csv"
    quality_a = tmp_path / "quality_a.csv"
    quality_b = tmp_path / "quality_b.csv"
    _write_csv(manifest, MANIFEST_COLUMNS, [train, test])
    train_rows = _quality_rows(train, auroc=(0.9, 0.8, 0.7))
    _write_csv(
        quality_a,
        QUALITY_COLUMNS,
        train_rows + _quality_rows(test, auroc=(0.0, 0.0, 1.0)),
    )
    _write_csv(
        quality_b,
        QUALITY_COLUMNS,
        train_rows
        + _quality_rows(
            test,
            auroc=("poison-test", "poison-test", "poison-test"),
            image_ap=("poison-test", "poison-test", "poison-test"),
            runtimes=("poison-test", "poison-test", "poison-test"),
        ),
    )

    first = calibrate_policy(
        expert_quality_by_run=quality_a,
        fold_manifest=manifest,
        fold="fold0",
        output_path=tmp_path / "first" / "policy_artifact.json",
    )
    second = calibrate_policy(
        expert_quality_by_run=quality_b,
        fold_manifest=manifest,
        fold="fold0",
        output_path=tmp_path / "second" / "policy_artifact.json",
    )
    first_payload = json.loads(first.read_text(encoding="utf-8"))
    second_payload = json.loads(second.read_text(encoding="utf-8"))

    assert first_payload == second_payload
    assert first_payload["train_seeds"] == [0]
    assert first_payload["metric"] == "image_auroc"
    assert first_payload["tie_break"] == RUNTIME_TIE_BREAK
    assert first_payload["git_commit"]
    serialized = json.dumps(first_payload).lower()
    assert "poison-test" not in serialized
    assert '"label"' not in serialized
    assert '"test_quality"' not in serialized


def test_calibrator_writes_one_artifact_per_manifest_fold(tmp_path: Path) -> None:
    fold0_train = _manifest_row(
        category="bottle", k_shot=1, seed=0, split="train", fold="fold0"
    )
    fold0_test = _manifest_row(
        category="bottle", k_shot=1, seed=1, split="test", fold="fold0"
    )
    fold1_test = _manifest_row(
        category="bottle", k_shot=1, seed=0, split="test", fold="fold1"
    )
    fold1_train = _manifest_row(
        category="bottle", k_shot=1, seed=1, split="train", fold="fold1"
    )
    manifest = tmp_path / "fold_manifest.csv"
    quality = tmp_path / "expert_quality_by_run.csv"
    _write_csv(
        manifest,
        MANIFEST_COLUMNS,
        [fold0_train, fold0_test, fold1_test, fold1_train],
    )
    _write_csv(
        quality,
        QUALITY_COLUMNS,
        _quality_rows(fold0_train, auroc=(0.9, 0.8, 0.7))
        + _quality_rows(fold1_train, auroc=(0.7, 0.8, 0.9)),
    )

    paths = calibrate_policy_artifacts(
        expert_quality_by_run=quality,
        fold_manifest=manifest,
        output_root=tmp_path / "policies",
    )

    assert paths == [
        tmp_path
        / "policies"
        / "category_shot_prior"
        / "fold0"
        / "policy_artifact.json",
        tmp_path
        / "policies"
        / "category_shot_prior"
        / "fold1"
        / "policy_artifact.json",
    ]
    assert [json.loads(path.read_text(encoding="utf-8"))["fold"] for path in paths] == [
        "fold0",
        "fold1",
    ]


def test_category_shot_policy_uses_exact_fallback_order(tmp_path: Path) -> None:
    manifest_rows = [
        _manifest_row(category="bottle", k_shot=1, seed=0),
        _manifest_row(category="bottle", k_shot=2, seed=0),
        _manifest_row(category="cable", k_shot=1, seed=0),
        _manifest_row(category="cable", k_shot=2, seed=0),
    ]
    quality_rows: list[dict[str, object]] = []
    for row, metrics in zip(
        manifest_rows,
        (
            (0.9, 0.1, 0.1),
            (0.1, 1.0, 0.1),
            (0.1, 0.1, 1.0),
            (0.1, 0.1, 1.0),
        ),
    ):
        quality_rows.extend(_quality_rows(row, auroc=metrics))
    manifest = tmp_path / "fold_manifest.csv"
    quality = tmp_path / "expert_quality_by_run.csv"
    _write_csv(manifest, MANIFEST_COLUMNS, manifest_rows)
    _write_csv(quality, QUALITY_COLUMNS, quality_rows)

    artifact = calibrate_policy(
        expert_quality_by_run=quality,
        fold_manifest=manifest,
        fold="fold0",
        output_path=tmp_path / "policy_artifact.json",
        policy_name="category_shot_prior",
    )
    policy = CategoryShotPriorPolicy.load(artifact)
    policy.fit([])

    exact = policy.select(_task("bottle", 1, 1))
    category_fallback = policy.select(_task("bottle", 8, 2))
    global_fallback = policy.select(_task("unseen", 8, 3))

    assert exact.selected_expert == "PatchCore"
    assert "category+shot" in exact.decision_reason
    assert category_fallback.selected_expert == "WinCLIP"
    assert "category fallback" in category_fallback.decision_reason
    assert global_fallback.selected_expert == "AnomalyDINO"
    assert "global best fallback" in global_fallback.decision_reason
    assert all(decision.tool_calls == 1 for decision in (exact, category_fallback, global_fallback))


def test_metric_is_configurable_and_category_prior_has_global_fallback(tmp_path: Path) -> None:
    train = _manifest_row(category="bottle", k_shot=1, seed=3)
    manifest = tmp_path / "fold_manifest.csv"
    quality = tmp_path / "expert_quality_by_run.csv"
    _write_csv(manifest, MANIFEST_COLUMNS, [train])
    _write_csv(
        quality,
        QUALITY_COLUMNS,
        _quality_rows(
            train,
            auroc=(0.95, 0.5, 0.4),
            image_ap=(0.4, 0.5, 0.99),
        ),
    )
    artifact = calibrate_policy(
        expert_quality_by_run=quality,
        fold_manifest=manifest,
        fold="fold0",
        output_path=tmp_path / "policy_artifact.json",
        policy_name="category_prior",
        metric="image_ap",
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    policy = CategoryPriorPolicy.load(artifact)
    policy.fit([])

    assert payload["metric"] == "image_ap"
    assert payload["category_shot_rules"] == []
    assert payload["fallback_order"] == ["category", "global_best"]
    assert policy.select(_task("bottle", 1, 1)).selected_expert == "AnomalyDINO"
    assert policy.select(_task("unseen", 8, 2)).selected_expert == "AnomalyDINO"


def test_training_runtime_breaks_metric_ties(tmp_path: Path) -> None:
    train = _manifest_row(category="bottle", k_shot=1, seed=2)
    manifest = tmp_path / "fold_manifest.csv"
    quality = tmp_path / "expert_quality_by_run.csv"
    _write_csv(manifest, MANIFEST_COLUMNS, [train])
    _write_csv(
        quality,
        QUALITY_COLUMNS,
        _quality_rows(
            train,
            auroc=(0.8, 0.8, 0.8),
            runtimes=(9.0, 2.0, 5.0),
        ),
    )

    artifact = calibrate_policy(
        expert_quality_by_run=quality,
        fold_manifest=manifest,
        fold="fold0",
        output_path=tmp_path / "policy_artifact.json",
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert payload["global_best"] == "WinCLIP"
    assert {row["selected_expert"] for row in payload["category_rules"]} == {"WinCLIP"}
    assert {row["selected_expert"] for row in payload["category_shot_rules"]} == {"WinCLIP"}


def test_metric_tie_without_training_runtime_fails_explicitly(tmp_path: Path) -> None:
    train = _manifest_row(category="bottle", k_shot=1, seed=0)
    manifest = tmp_path / "fold_manifest.csv"
    quality = tmp_path / "expert_quality_by_run.csv"
    _write_csv(manifest, MANIFEST_COLUMNS, [train])
    rows = _quality_rows(train, auroc=(0.8, 0.8, 0.7))
    for row in rows:
        row["runtime_ms"] = ""
    _write_csv(quality, QUALITY_COLUMNS, rows)

    with pytest.raises(PolicyCalibrationError, match="lacks complete training runtime"):
        calibrate_policy(
            expert_quality_by_run=quality,
            fold_manifest=manifest,
            fold="fold0",
            output_path=tmp_path / "policy_artifact.json",
        )


def test_replay_rejects_policy_artifact_from_a_different_fold(tmp_path: Path) -> None:
    train = _manifest_row(category="bottle", k_shot=1, seed=0, split="train")
    manifest = tmp_path / "fold_manifest.csv"
    quality = tmp_path / "expert_quality_by_run.csv"
    _write_csv(manifest, MANIFEST_COLUMNS, [train])
    _write_csv(quality, QUALITY_COLUMNS, _quality_rows(train, auroc=(0.9, 0.8, 0.7)))
    artifact = calibrate_policy(
        expert_quality_by_run=quality,
        fold_manifest=manifest,
        fold="fold0",
        output_path=tmp_path / "policy_artifact.json",
    )
    policy = CategoryShotPriorPolicy.load(artifact)
    task = _task("bottle", 1, 7)
    stage2_root = tmp_path / "stage2"
    stage2_root.mkdir()
    fold1_manifest = [
        {
            "fold": "fold1",
            "split": "test",
            "task_id": task.task_id,
            "sample_id": task.sample_id,
            "dataset": task.dataset,
            "category": task.category,
            "k_shot": str(task.k_shot),
            "seed": str(task.seed),
            "support_set_id": task.support_set_id,
        }
    ]

    with pytest.raises(ReplayExecutionError, match="not replay fold"):
        execute_replay(
            tasks=[task],
            manifest_rows=fold1_manifest,
            policy=policy,
            stage2_root=stage2_root,
            output_dir=tmp_path / "replay",
            fold="fold1",
            split="test",
        )
