import json
from pathlib import Path
import sys

import pytest
from PIL import Image

from scripts.inspect_visa_structure import inspect_visa_structure, main, write_report


def _png(path: Path, size: tuple[int, int] = (8, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 20, 30)).save(path)


def test_inspect_visa_structure_reports_observed_layout(tmp_path: Path) -> None:
    visa_root = tmp_path / "visa"
    _png(visa_root / "candle" / "train" / "good" / "000.JPG")
    _png(visa_root / "candle" / "test" / "bad" / "001.png")
    _png(visa_root / "candle" / "ground_truth" / "bad" / "001_mask.png")
    _png(visa_root / "capsules" / "Images" / "normal" / "002.tiff")
    (visa_root / "split_csv").mkdir(parents=True)

    report = inspect_visa_structure(visa_root)

    assert report["top_level_folders"] == ["candle", "capsules", "split_csv"]
    assert report["image_file_counts"] == {
        "total": 4,
        "by_extension": {".jpg": 1, ".png": 2, ".tiff": 1},
    }
    assert report["detected_folders"]["train"] == ["candle/train"]
    assert report["detected_folders"]["test"] == ["candle/test"]
    assert report["detected_folders"]["good"] == [
        "candle/train/good",
        "capsules/Images/normal",
    ]
    assert "candle/test/bad" in report["detected_folders"]["bad_or_anomaly"]
    assert "candle/ground_truth" in report["detected_folders"]["mask_like"]

    categories = {item["name"]: item for item in report["category_folders"]}
    assert sorted(categories) == ["candle", "capsules"]
    assert categories["candle"]["image_file_counts"]["total"] == 3
    assert categories["candle"]["train_folders"] == ["candle/train"]
    assert categories["candle"]["test_folders"] == ["candle/test"]
    assert categories["candle"]["mask_like_folders"] == ["candle/ground_truth"]
    assert categories["capsules"]["good_folders"] == ["capsules/Images/normal"]


def test_inspect_visa_structure_detects_split_first_categories(tmp_path: Path) -> None:
    visa_root = tmp_path / "visa"
    _png(visa_root / "train" / "pcb" / "good" / "000.png")
    _png(visa_root / "test" / "pcb" / "anomaly" / "001.jpeg")

    report = inspect_visa_structure(visa_root)

    assert report["top_level_folders"] == ["test", "train"]
    assert report["image_file_counts"]["by_extension"] == {".jpeg": 1, ".png": 1}
    assert report["detected_folders"]["train"] == ["train"]
    assert report["detected_folders"]["test"] == ["test"]
    assert report["category_folders"] == [
        {
            "name": "pcb",
            "relative_path": "test/pcb",
            "source": "test_child",
            "direct_children": ["anomaly"],
            "train_folders": [],
            "test_folders": [],
            "good_folders": [],
            "bad_or_anomaly_folders": ["test/pcb/anomaly"],
            "mask_like_folders": [],
            "image_file_counts": {
                "total": 1,
                "by_extension": {".jpeg": 1},
            },
        }
    ]


def test_inspect_visa_structure_rejects_missing_root(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="VisA root does not exist"):
        inspect_visa_structure(missing_root)


def test_cli_reports_missing_root_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_root = tmp_path / "missing"
    monkeypatch.setattr(sys, "argv", ["inspect_visa_structure.py", "--visa-root", str(missing_root)])

    with pytest.raises(SystemExit) as exc_info:
        main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "VisA root does not exist" in captured.err
    assert "Traceback" not in captured.err


def test_write_report_writes_json_without_touching_visa_root(tmp_path: Path) -> None:
    visa_root = tmp_path / "visa"
    _png(visa_root / "candle" / "train" / "good" / "000.png")
    before = sorted(path.relative_to(visa_root).as_posix() for path in visa_root.rglob("*"))
    report = inspect_visa_structure(visa_root)

    output_path = tmp_path / "outputs" / "stage1_gate" / "visa_structure_report.json"
    write_report(output_path, report)

    after = sorted(path.relative_to(visa_root).as_posix() for path in visa_root.rglob("*"))
    assert before == after
    assert json.loads(output_path.read_text(encoding="utf-8"))["dataset"] == "visa"
