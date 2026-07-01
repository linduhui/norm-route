import csv
from pathlib import Path

import pytest
from PIL import Image

from scripts.audit_dataset import AUDIT_COLUMNS, audit_dataset, write_audit_csv
from src.data.manifest import AGENT_INPUT_COLUMNS, EVALUATOR_COLUMNS


def _png(path: Path, size: tuple[int, int] = (8, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 20, 30)).save(path)


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _manifest_pair(tmp_path: Path) -> tuple[Path, Path]:
    agent_csv = tmp_path / "manifests" / "mvtec_agent_input.csv"
    evaluator_csv = tmp_path / "manifests" / "mvtec_evaluator.csv"

    agent_rows = [
        {
            "image_id": "train-good",
            "dataset": "mvtec",
            "category": "bottle",
            "split": "train",
            "image_path": "bottle/train/good/000.png",
        },
        {
            "image_id": "test-good",
            "dataset": "mvtec",
            "category": "bottle",
            "split": "test",
            "image_path": "bottle/test/good/001.png",
        },
        {
            "image_id": "test-crack",
            "dataset": "mvtec",
            "category": "bottle",
            "split": "test",
            "image_path": "bottle/test/crack/002.png",
        },
        {
            "image_id": "test-hole",
            "dataset": "mvtec",
            "category": "bottle",
            "split": "test",
            "image_path": "bottle/test/hole/003.png",
        },
    ]
    evaluator_rows = [
        {"image_id": "train-good", "label": 0, "mask_path": "", "defect_type": "good"},
        {"image_id": "test-good", "label": 0, "mask_path": "", "defect_type": "good"},
        {
            "image_id": "test-crack",
            "label": 1,
            "mask_path": "bottle/ground_truth/crack/002_mask.png",
            "defect_type": "crack",
        },
        {"image_id": "test-hole", "label": 1, "mask_path": "", "defect_type": "hole"},
    ]

    _write_csv(agent_csv, AGENT_INPUT_COLUMNS, agent_rows)
    _write_csv(evaluator_csv, EVALUATOR_COLUMNS, evaluator_rows)
    return agent_csv, evaluator_csv


def test_audit_dataset_summarizes_mvtec_category(tmp_path: Path) -> None:
    data_root = tmp_path / "mvtec"
    _png(data_root / "bottle" / "train" / "good" / "000.png")
    _png(data_root / "bottle" / "test" / "good" / "001.png")
    _png(data_root / "bottle" / "test" / "crack" / "002.png")
    _png(data_root / "bottle" / "ground_truth" / "crack" / "002_mask.png")
    _png(data_root / "bottle" / "test" / "hole" / "003.png")
    agent_csv, evaluator_csv = _manifest_pair(tmp_path)

    rows = audit_dataset(
        agent_input_csv=agent_csv,
        evaluator_csv=evaluator_csv,
        path_root=data_root,
    )

    assert rows == [
        {
            "dataset": "mvtec",
            "category": "bottle",
            "train_images": 1,
            "test_good_images": 1,
            "test_anomaly_images": 2,
            "anomaly_types": "crack;hole",
            "missing_masks": 1,
            "broken_files": 0,
        }
    ]


def test_audit_dataset_counts_broken_images_and_masks(tmp_path: Path) -> None:
    data_root = tmp_path / "mvtec"
    _png(data_root / "bottle" / "train" / "good" / "000.png")
    _png(data_root / "bottle" / "test" / "good" / "001.png")
    _png(data_root / "bottle" / "test" / "crack" / "002.png")
    _png(data_root / "bottle" / "test" / "hole" / "003.png")
    broken_mask = data_root / "bottle" / "ground_truth" / "crack" / "002_mask.png"
    broken_mask.parent.mkdir(parents=True, exist_ok=True)
    broken_mask.write_text("not an image", encoding="utf-8")
    agent_csv, evaluator_csv = _manifest_pair(tmp_path)

    rows = audit_dataset(
        agent_input_csv=agent_csv,
        evaluator_csv=evaluator_csv,
        path_root=data_root,
    )

    assert rows[0]["missing_masks"] == 1
    assert rows[0]["broken_files"] == 1


def test_audit_writer_uses_required_columns(tmp_path: Path) -> None:
    output_csv = tmp_path / "audit.csv"

    write_audit_csv(
        output_csv,
        [
            {
                "dataset": "mvtec",
                "category": "bottle",
                "train_images": 1,
                "test_good_images": 1,
                "test_anomaly_images": 1,
                "anomaly_types": "crack",
                "missing_masks": 0,
                "broken_files": 0,
            }
        ],
    )

    with output_csv.open(newline="", encoding="utf-8") as handle:
        assert csv.DictReader(handle).fieldnames == AUDIT_COLUMNS


def test_audit_rejects_manifest_field_changes(tmp_path: Path) -> None:
    agent_csv = tmp_path / "agent.csv"
    evaluator_csv = tmp_path / "evaluator.csv"
    _write_csv(agent_csv, AGENT_INPUT_COLUMNS + ["label"], [])
    _write_csv(evaluator_csv, EVALUATOR_COLUMNS, [])

    with pytest.raises(ValueError, match="unexpected columns"):
        audit_dataset(agent_input_csv=agent_csv, evaluator_csv=evaluator_csv, path_root=tmp_path)


def test_audit_rejects_unmatched_image_ids(tmp_path: Path) -> None:
    agent_csv = tmp_path / "agent.csv"
    evaluator_csv = tmp_path / "evaluator.csv"
    _write_csv(
        agent_csv,
        AGENT_INPUT_COLUMNS,
        [
            {
                "image_id": "agent-only",
                "dataset": "mvtec",
                "category": "bottle",
                "split": "test",
                "image_path": "bottle/test/good/001.png",
            }
        ],
    )
    _write_csv(
        evaluator_csv,
        EVALUATOR_COLUMNS,
        [{"image_id": "evaluator-only", "label": 0, "mask_path": "", "defect_type": "good"}],
    )

    with pytest.raises(ValueError, match="image_id mismatch"):
        audit_dataset(agent_input_csv=agent_csv, evaluator_csv=evaluator_csv, path_root=tmp_path)
