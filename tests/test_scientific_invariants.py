from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _parse_simple_fold_yaml(text: str) -> dict[str, list[str]]:
    folds: dict[str, list[str]] = {}
    current_fold: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_fold = line[:-1]
            folds[current_fold] = []
            continue
        if line.startswith("  - "):
            assert current_fold is not None
            folds[current_fold].append(line.removeprefix("  - "))

    return folds


def test_mvtec_folds_cover_15_categories_without_duplicates():
    folds = _parse_simple_fold_yaml(_read_text("configs/folds.yaml"))
    categories = [category for fold in folds.values() for category in fold]

    assert len(categories) == 15
    assert len(set(categories)) == 15
    assert set(folds) == {"fold_1", "fold_2", "fold_3", "fold_4", "fold_5"}


def test_forbidden_fields_include_ground_truth_fields():
    task_spec = _read_text("docs/task_spec.md")

    for forbidden_field in ["label", "mask_path", "defect_type"]:
        assert forbidden_field in task_spec


def test_action_space_contains_stop_and_abstain_actions():
    task_spec = _read_text("docs/task_spec.md")

    assert "STOP_NORMAL" in task_spec
    assert "STOP_ANOMALY" in task_spec
    assert "ABSTAIN" in task_spec


def test_task_spec_contains_key_input_and_prediction_fields():
    task_spec = _read_text("docs/task_spec.md")
    required_fields = [
        "InspectionInput",
        "image_id",
        "query_path",
        "dataset",
        "category",
        "support_set_id",
        "k_shot",
        "seed",
        "budget",
        "final_score",
        "final_decision",
        "anomaly_map_path",
        "actions",
        "tool_calls",
        "runtime_ms",
        "status",
    ]

    missing = [field for field in required_fields if field not in task_spec]

    assert missing == []
