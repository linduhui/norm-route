import csv
import json
from pathlib import Path

import pytest

from src.normroute.cli.build_stage5_splits import (
    EXPECTED_CATEGORIES,
    EXPECTED_FOLDS,
    EXPECTED_GROUPS,
    Stage5SplitError,
    build_stage5_splits,
    load_category_cv_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "stage5" / "category_cv_v2.yaml"


def _tasks() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category in EXPECTED_CATEGORIES:
        for k_shot, seed in ((1, 0), (2, 1)):
            support_set_id = f"mvtec_{category}_k{k_shot}_seed{seed}"
            for query_index in range(2):
                rows.append(
                    {
                        "protocol_version": "stage4.pre_route.v1",
                        "task_id": f"{category}-k{k_shot}-s{seed}-q{query_index}",
                        "sample_id": f"{category}-query-{query_index}",
                        "dataset": "mvtec",
                        "category": category,
                        "k_shot": k_shot,
                        "seed": seed,
                        "support_set_id": support_set_id,
                        "candidate_experts": ["PatchCore", "WinCLIP", "AnomalyDINO"],
                        "policy_features": {
                            "dataset": "mvtec",
                            "category": category,
                            "k_shot": k_shot,
                        },
                    }
                )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_category_cv_config_matches_frozen_groups_and_cyclic_folds() -> None:
    config = load_category_cv_config(CONFIG)

    assert config.protocol_version == "stage5.category_cv.v2"
    assert config.split_unit == "category"
    assert config.groups == EXPECTED_GROUPS
    assert tuple(config.expected_categories) == EXPECTED_CATEGORIES
    assert tuple(config.folds) == tuple(EXPECTED_FOLDS)
    for fold, expected in EXPECTED_FOLDS.items():
        spec = config.folds[fold]
        assert {
            split: spec.categories_for(split) for split in ("train", "val", "test")
        } == expected


def test_manifest_keeps_complete_category_bundles_in_one_split(tmp_path: Path) -> None:
    tasks = _tasks()
    tasks_path = tmp_path / "stage5_tasks.jsonl"
    _write_jsonl(tasks_path, tasks)
    manifest_path = tmp_path / "fold_manifest.csv"
    audit_path = tmp_path / "split_audit.json"

    build_stage5_splits(
        tasks_path=tasks_path,
        config_path=CONFIG,
        manifest_path=manifest_path,
        audit_path=audit_path,
    )

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(tasks) * 5
    for fold, expected in EXPECTED_FOLDS.items():
        fold_rows = [row for row in rows if row["fold"] == fold]
        assert len(fold_rows) == len(tasks)
        assert len({row["task_id"] for row in fold_rows}) == len(tasks)
        for split, categories in expected.items():
            assert {row["category"] for row in fold_rows if row["split"] == split} == set(categories)
        for category in EXPECTED_CATEGORIES:
            category_rows = [row for row in fold_rows if row["category"] == category]
            assert len({row["split"] for row in category_rows}) == 1
            assert {row["k_shot"] for row in category_rows} == {"1", "2"}
            assert {row["seed"] for row in category_rows} == {"0", "1"}
            assert len({row["sample_id"] for row in category_rows}) == 2
            assert len({row["support_set_id"] for row in category_rows}) == 2

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["all_folds_valid"] is True
    assert audit["manifest_row_count"] == len(tasks) * 5
    assert audit["failures"] == []
    for fold in EXPECTED_FOLDS:
        fold_audit = audit["folds"][fold]
        assert fold_audit["isolation_ok"] is True
        assert fold_audit["category_bundle_violations"] == {}
        assert all(not overlap for overlap in fold_audit["category_overlaps"].values())


def test_split_builder_rejects_incomplete_category_grid(tmp_path: Path) -> None:
    tasks = [row for row in _tasks() if row["category"] != "zipper"]
    tasks_path = tmp_path / "stage5_tasks.jsonl"
    _write_jsonl(tasks_path, tasks)

    with pytest.raises(Stage5SplitError, match=r"missing=\['zipper'\]"):
        build_stage5_splits(
            tasks_path=tasks_path,
            config_path=CONFIG,
            manifest_path=tmp_path / "fold_manifest.csv",
            audit_path=tmp_path / "split_audit.json",
        )


def test_split_builder_rejects_multiple_support_ids_for_one_signature(tmp_path: Path) -> None:
    tasks = _tasks()
    tasks[1]["support_set_id"] = "different-support-id"
    tasks_path = tmp_path / "stage5_tasks.jsonl"
    _write_jsonl(tasks_path, tasks)

    with pytest.raises(Stage5SplitError, match="multiple support_set_id"):
        build_stage5_splits(
            tasks_path=tasks_path,
            config_path=CONFIG,
            manifest_path=tmp_path / "fold_manifest.csv",
            audit_path=tmp_path / "split_audit.json",
        )
