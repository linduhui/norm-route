import csv
from pathlib import Path

from PIL import Image

from src.data.manifest import AGENT_INPUT_COLUMNS, EVALUATOR_COLUMNS, write_manifest_outputs
from src.data.visa import build_visa_manifest


def _png(path: Path, size: tuple[int, int] = (8, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 20, 30)).save(path)


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _fake_current_visa(root: Path) -> None:
    _png(root / "candle" / "Data" / "Images" / "Normal" / "000.JPG")
    _png(root / "candle" / "Data" / "Images" / "Normal" / "001.JPG")
    _png(root / "candle" / "Data" / "Images" / "Anomaly" / "100.JPG")
    _png(root / "candle" / "Data" / "Masks" / "Anomaly" / "100.png")
    _write_csv(
        root / "split_csv" / "1cls.csv",
        ["object", "split", "label", "image", "mask"],
        [
            {
                "object": "candle",
                "split": "train",
                "label": "normal",
                "image": "candle/Data/Images/Normal/000.JPG",
                "mask": "",
            },
            {
                "object": "candle",
                "split": "test",
                "label": "normal",
                "image": "candle/Data/Images/Normal/001.JPG",
                "mask": "",
            },
            {
                "object": "candle",
                "split": "test",
                "label": "anomaly",
                "image": "candle/Data/Images/Anomaly/100.JPG",
                "mask": "candle/Data/Masks/Anomaly/100.png",
            },
        ],
    )


def test_visa_manifest_reads_detected_split_csv_layout(tmp_path: Path) -> None:
    _fake_current_visa(tmp_path)

    rows = build_visa_manifest(tmp_path, path_style="relative")

    assert list(rows.agent_input[0]) == AGENT_INPUT_COLUMNS
    assert list(rows.evaluator[0]) == EVALUATOR_COLUMNS
    assert rows.audit["split_csv"] == "split_csv/1cls.csv"
    assert rows.audit["counts"]["train_good_images"] == 1
    assert rows.audit["counts"]["test_good_images"] == 1
    assert rows.audit["counts"]["test_anomaly_images"] == 1
    assert rows.audit["counts"]["missing_masks"] == 0
    assert {row["dataset"] for row in rows.agent_input} == {"visa"}
    assert all("label" not in row and "mask_path" not in row and "defect_type" not in row for row in rows.agent_input)

    evaluator_by_id = {row["image_id"]: row for row in rows.evaluator}
    anomaly_agent_row = next(row for row in rows.agent_input if "Anomaly" in row["image_path"])
    anomaly_eval_row = evaluator_by_id[anomaly_agent_row["image_id"]]
    assert anomaly_eval_row == {
        "image_id": anomaly_agent_row["image_id"],
        "label": 1,
        "mask_path": "candle/Data/Masks/Anomaly/100.png",
        "defect_type": "Anomaly",
    }


def test_visa_manifest_supports_explicit_train_test_fake_layout(tmp_path: Path) -> None:
    _png(tmp_path / "pcb" / "train" / "good" / "000.png")
    _png(tmp_path / "pcb" / "test" / "good" / "001.png")
    _png(tmp_path / "pcb" / "test" / "scratch" / "002.png")
    _png(tmp_path / "pcb" / "ground_truth" / "scratch" / "002_mask.png")
    broken = tmp_path / "pcb" / "test" / "scratch" / "003.png"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("not an image", encoding="utf-8")

    rows = build_visa_manifest(tmp_path, path_style="relative", categories=["pcb"])

    assert rows.audit["categories"] == ["pcb"]
    assert rows.audit["counts"]["train_good_images"] == 1
    assert rows.audit["counts"]["test_good_images"] == 1
    assert rows.audit["counts"]["test_anomaly_images"] == 2
    assert rows.audit["counts"]["missing_masks"] == 1
    assert rows.audit["counts"]["broken_image_files"] == 1
    assert rows.audit["missing_masks"] == [
        {"category": "pcb", "defect_type": "scratch", "image_path": "pcb/test/scratch/003.png"}
    ]


def test_visa_image_ids_are_deterministic_and_unique(tmp_path: Path) -> None:
    _fake_current_visa(tmp_path)

    first = build_visa_manifest(tmp_path, path_style="relative")
    second = build_visa_manifest(tmp_path, path_style="relative")

    first_ids = [row["image_id"] for row in first.agent_input]
    second_ids = [row["image_id"] for row in second.agent_input]
    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))
    assert first.audit["counts"]["duplicate_image_ids"] == 0


def test_visa_manifest_writer_outputs_visa_files(tmp_path: Path) -> None:
    data_root = tmp_path / "visa"
    _fake_current_visa(data_root)
    rows = build_visa_manifest(data_root, path_style="relative")

    agent_path, evaluator_path, audit_path = write_manifest_outputs(
        agent_rows=rows.agent_input,
        evaluator_rows=rows.evaluator,
        audit=rows.audit,
        manifest_dir=tmp_path / "data" / "manifests",
        audit_path=tmp_path / "outputs" / "stage1_gate" / "visa_full_audit.json",
        dataset="visa",
    )

    assert agent_path.name == "visa_agent_input.csv"
    assert evaluator_path.name == "visa_evaluator.csv"
    assert audit_path.name == "visa_full_audit.json"
    with agent_path.open(newline="", encoding="utf-8") as handle:
        assert csv.DictReader(handle).fieldnames == AGENT_INPUT_COLUMNS
    with evaluator_path.open(newline="", encoding="utf-8") as handle:
        assert csv.DictReader(handle).fieldnames == EVALUATOR_COLUMNS
