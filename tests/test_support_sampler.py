import csv
from pathlib import Path

import pytest

from src.data.support_sampler import (
    SUPPORT_SET_COLUMNS,
    build_support_set_rows,
    write_support_set_csv,
)


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _agent_rows() -> list[dict[str, object]]:
    return [
        {
            "image_id": "mvtec_bottle_train_000",
            "dataset": "mvtec",
            "category": "bottle",
            "split": "train",
            "image_path": "/datasets/mvtec/bottle/train/good/000.png",
        },
        {
            "image_id": "mvtec_bottle_train_001",
            "dataset": "mvtec",
            "category": "bottle",
            "split": "train",
            "image_path": "/datasets/mvtec/bottle/train/good/001.png",
        },
        {
            "image_id": "mvtec_bottle_test_good",
            "dataset": "mvtec",
            "category": "bottle",
            "split": "test",
            "image_path": "/datasets/mvtec/bottle/test/good/002.png",
        },
        {
            "image_id": "mvtec_bottle_test_bad",
            "dataset": "mvtec",
            "category": "bottle",
            "split": "test",
            "image_path": "/datasets/mvtec/bottle/test/crack/003.png",
        },
        {
            "image_id": "mvtec_cable_train_000",
            "dataset": "mvtec",
            "category": "cable",
            "split": "train",
            "image_path": "/datasets/mvtec/cable/train/good/000.png",
        },
        {
            "image_id": "mvtec_cable_train_001",
            "dataset": "mvtec",
            "category": "cable",
            "split": "train",
            "image_path": "/datasets/mvtec/cable/train/good/001.png",
        },
    ]


def test_samples_train_good_only_and_writes_required_columns(tmp_path: Path) -> None:
    manifest = tmp_path / "mvtec_agent_input.csv"
    _write_csv(
        manifest,
        ["image_id", "dataset", "category", "split", "image_path"],
        _agent_rows(),
    )

    rows = build_support_set_rows(manifest, dataset="mvtec", k_shot=2, seed=0)
    output_path = write_support_set_csv(tmp_path / "support" / "mvtec_k2_seed0.csv", rows)

    assert len(rows) == 4
    assert {row["category"] for row in rows} == {"bottle", "cable"}
    assert all(row["k_shot"] == "2" and row["seed"] == "0" for row in rows)
    assert all("/train/good/" in row["image_path"] for row in rows)
    assert "mvtec_bottle_test_good" not in {row["image_id"] for row in rows}
    assert "mvtec_bottle_test_bad" not in {row["image_id"] for row in rows}

    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == SUPPORT_SET_COLUMNS


def test_sampling_is_deterministic_and_changes_with_seed(tmp_path: Path) -> None:
    manifest = tmp_path / "mvtec_agent_input.csv"
    rows = _agent_rows() + [
        {
            "image_id": f"mvtec_bottle_train_{index:03d}",
            "dataset": "mvtec",
            "category": "bottle",
            "split": "train",
            "image_path": f"/datasets/mvtec/bottle/train/good/{index:03d}.png",
        }
        for index in range(2, 10)
    ]
    _write_csv(manifest, ["image_id", "dataset", "category", "split", "image_path"], rows)

    first = build_support_set_rows(manifest, dataset="mvtec", k_shot=1, seed=0)
    second = build_support_set_rows(manifest, dataset="mvtec", k_shot=1, seed=0)
    different_seed = build_support_set_rows(manifest, dataset="mvtec", k_shot=1, seed=1)

    assert first == second
    assert first != different_seed


def test_raises_when_category_has_too_few_train_good_images(tmp_path: Path) -> None:
    manifest = tmp_path / "mvtec_agent_input.csv"
    _write_csv(
        manifest,
        ["image_id", "dataset", "category", "split", "image_path"],
        _agent_rows(),
    )

    with pytest.raises(ValueError, match="Category 'bottle'.*requested K=3"):
        build_support_set_rows(manifest, dataset="mvtec", k_shot=3, seed=0)


def test_rejects_forbidden_evaluator_columns(tmp_path: Path) -> None:
    manifest = tmp_path / "leaky_agent_input.csv"
    _write_csv(
        manifest,
        ["image_id", "dataset", "category", "split", "image_path", "label"],
        [
            {
                "image_id": "x",
                "dataset": "mvtec",
                "category": "bottle",
                "split": "train",
                "image_path": "/datasets/mvtec/bottle/train/good/000.png",
                "label": 0,
            }
        ],
    )

    with pytest.raises(ValueError, match="forbidden fields"):
        build_support_set_rows(manifest, dataset="mvtec", k_shot=1, seed=0)


def test_visa_support_uses_train_normal_images_only(tmp_path: Path) -> None:
    manifest = tmp_path / "visa_agent_input.csv"
    _write_csv(
        manifest,
        ["image_id", "dataset", "category", "split", "image_path"],
        [
            {
                "image_id": "visa_candle_train_normal",
                "dataset": "visa",
                "category": "candle",
                "split": "train",
                "image_path": "/datasets/VisA/candle/Data/Images/Normal/000.JPG",
            },
            {
                "image_id": "visa_candle_test_normal",
                "dataset": "visa",
                "category": "candle",
                "split": "test",
                "image_path": "/datasets/VisA/candle/Data/Images/Normal/001.JPG",
            },
            {
                "image_id": "visa_candle_test_anomaly",
                "dataset": "visa",
                "category": "candle",
                "split": "test",
                "image_path": "/datasets/VisA/candle/Data/Images/Anomaly/100.JPG",
            },
        ],
    )

    rows = build_support_set_rows(manifest, dataset="visa", k_shot=1, seed=0)

    assert [row["image_id"] for row in rows] == ["visa_candle_train_normal"]
    assert rows[0]["support_set_id"] == "visa_candle_k1_seed0"
