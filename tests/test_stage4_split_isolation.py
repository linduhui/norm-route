import csv
import json
from pathlib import Path

import pytest

from src.normroute.agent.task_builder import build_pre_route_task
from src.normroute.cli.build_stage4_splits import (
    EXPECTED_FOLDS,
    Stage4SplitError,
    build_stage4_splits,
    load_seed_cv_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "stage4" / "seed_cv.yaml"


def _pre_route_task(seed: int) -> dict[str, object]:
    return build_pre_route_task(
        {
            "task_id": f"task-{seed}",
            "sample_id": "sample-1",
            "dataset": "mvtec",
            "category": "bottle",
            "support_set_id": f"mvtec_bottle_k1_seed{seed}",
            "k_shot": 1,
            "seed": seed,
        }
    )


def _write_tasks(path: Path, seeds: range) -> None:
    path.write_text(
        "".join(json.dumps(_pre_route_task(seed)) + "\n" for seed in seeds),
        encoding="utf-8",
    )


def test_seed_cv_config_matches_frozen_five_folds() -> None:
    config = load_seed_cv_config(CONFIG)

    assert tuple(config.folds) == tuple(EXPECTED_FOLDS)
    for fold, expected in EXPECTED_FOLDS.items():
        spec = config.folds[fold]
        assert {split: getattr(spec, split) for split in ("train", "val", "test")} == expected


def test_fold_manifest_assigns_each_task_once_with_disjoint_seed_sets(tmp_path: Path) -> None:
    tasks_path = tmp_path / "pre_route_tasks.jsonl"
    _write_tasks(tasks_path, range(5))
    manifest_path = tmp_path / "fold_manifest.csv"
    audit_path = tmp_path / "split_audit.json"

    build_stage4_splits(
        tasks_path=tasks_path,
        config_path=CONFIG,
        manifest_path=manifest_path,
        audit_path=audit_path,
    )

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 25
    for fold, expected in EXPECTED_FOLDS.items():
        fold_rows = [row for row in rows if row["fold"] == fold]
        assert len(fold_rows) == 5
        assert len({row["task_id"] for row in fold_rows}) == 5
        for split, seeds in expected.items():
            assert {int(row["seed"]) for row in fold_rows if row["split"] == split} == set(seeds)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["all_folds_valid"] is True
    assert audit["source_task_count"] == 5
    assert audit["manifest_row_count"] == 25
    assert audit["failures"] == []
    for fold in EXPECTED_FOLDS:
        assert audit["folds"][fold]["isolation_ok"] is True
        assert all(not overlap for overlap in audit["folds"][fold]["seed_overlaps"].values())


def test_split_builder_fails_when_a_configured_seed_is_missing(tmp_path: Path) -> None:
    tasks_path = tmp_path / "pre_route_tasks.jsonl"
    _write_tasks(tasks_path, range(4))

    with pytest.raises(Stage4SplitError, match=r"missing=\[4\]"):
        build_stage4_splits(
            tasks_path=tasks_path,
            config_path=CONFIG,
            manifest_path=tmp_path / "fold_manifest.csv",
            audit_path=tmp_path / "split_audit.json",
        )
