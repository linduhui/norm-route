import csv
import json
from pathlib import Path

import pytest

from src.normroute.cli.audit_stage5_inputs import (
    audit_inference_visible_file,
    audit_stage5_inputs,
    forbidden_field_reason,
)
from src.normroute.cli.build_stage5_splits import (
    EXPECTED_CATEGORIES,
    Stage5SplitError,
    build_stage5_splits,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "stage5" / "category_cv_v2.yaml"


@pytest.mark.parametrize(
    ("field", "family"),
    [
        ("label", "label_or_ground_truth"),
        ("targetLabel", "label_or_ground_truth"),
        ("mask_path", "mask"),
        ("defect_type", "defect_or_anomaly_type"),
        ("expert_scores", "expert_score_or_utility"),
        ("PATCHCORE_SCORE", "expert_score_or_utility"),
        ("OracleBestExpert", "oracle"),
        ("teacherUtility", "teacher_score_target_or_utility"),
    ],
)
def test_semantic_forbidden_fields_are_case_insensitive(field: str, family: str) -> None:
    assert forbidden_field_reason(field) == family


def test_jsonl_audit_rejects_nested_expert_or_evaluator_information(tmp_path: Path) -> None:
    path = tmp_path / "inference_tasks.jsonl"
    path.write_text(
        json.dumps({"task_id": "safe", "features": {"category": "bottle"}})
        + "\n"
        + json.dumps(
            {
                "task_id": "leaked",
                "features": {
                    "nested": {"expert_scores": {"PatchCore": 0.9}},
                    "teacherUtility": 1.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    failures = audit_inference_visible_file(path)

    assert {failure["field_family"] for failure in failures} == {
        "expert_score_or_utility",
        "teacher_score_target_or_utility",
    }
    assert all(failure["code"] == "FORBIDDEN_INFERENCE_FIELD" for failure in failures)


def test_csv_audit_rejects_label_mask_oracle_and_expert_score_headers(tmp_path: Path) -> None:
    path = tmp_path / "inference.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["task_id", "label", "mask", "winclip_score", "oracle_answer"]
        )
        writer.writerow(["task", "1", "path", "0.4", "WinCLIP"])

    failures = audit_inference_visible_file(path)

    assert len(failures) == 4
    assert {failure["field_family"] for failure in failures} == {
        "label_or_ground_truth",
        "mask",
        "expert_score_or_utility",
        "oracle",
    }


def test_clean_input_audit_writes_reproducible_failure_aware_report(tmp_path: Path) -> None:
    input_path = tmp_path / "safe.json"
    input_path.write_text(
        json.dumps(
            {
                "task_id": "task",
                "query_id": "query",
                "features": {"dataset": "mvtec", "category": "bottle", "k_shot": 1},
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "audit.json"

    report = audit_stage5_inputs(
        [input_path], report_path=report_path, config_path=CONFIG
    )

    assert report["ok"] is True
    assert report["files_scanned"] == 1
    assert report["failures"] == []
    assert report["files"][0]["sha256"]
    assert report["config"]["sha256"]
    assert report["provenance"]["git_commit"]
    assert json.loads(report_path.read_text(encoding="utf-8"))["ok"] is True


def test_audit_rejects_oracle_named_inference_location(tmp_path: Path) -> None:
    input_path = tmp_path / "oracle_outputs" / "tasks.json"
    input_path.parent.mkdir()
    input_path.write_text(json.dumps({"task_id": "task"}), encoding="utf-8")

    failures = audit_inference_visible_file(input_path)

    assert any(failure["code"] == "FORBIDDEN_INFERENCE_PATH" for failure in failures)


def test_split_builder_rejects_nested_leakage_before_writing(tmp_path: Path) -> None:
    rows = []
    for category in EXPECTED_CATEGORIES:
        row = {
            "task_id": f"task-{category}",
            "sample_id": f"sample-{category}",
            "dataset": "mvtec",
            "category": category,
            "k_shot": 1,
            "seed": 0,
            "support_set_id": f"mvtec_{category}_k1_seed0",
            "features": {"category": category},
        }
        rows.append(row)
    rows[0]["features"] = {"nested": {"Oracle": "PatchCore"}}
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest_path = tmp_path / "fold_manifest.csv"

    with pytest.raises(Stage5SplitError, match="forbidden inference-visible fields"):
        build_stage5_splits(
            tasks_path=tasks_path,
            config_path=CONFIG,
            manifest_path=manifest_path,
            audit_path=tmp_path / "split_audit.json",
        )
    assert not manifest_path.exists()
