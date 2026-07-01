import csv
from pathlib import Path

from PIL import Image

from src.data.manifest import AGENT_INPUT_COLUMNS, EVALUATOR_COLUMNS, write_manifest_outputs
from src.data.mvtec import build_mvtec_manifest


def _png(path: Path, size: tuple[int, int] = (8, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 20, 30)).save(path)


def _fake_mvtec(root: Path) -> None:
    _png(root / "bottle" / "train" / "good" / "000.png")
    _png(root / "bottle" / "test" / "good" / "001.png")
    _png(root / "bottle" / "test" / "crack" / "002.png")
    _png(root / "bottle" / "ground_truth" / "crack" / "002_mask.png")
    _png(root / "bottle" / "test" / "crack" / "003.png")


def test_agent_input_has_no_forbidden_fields(tmp_path: Path) -> None:
    _fake_mvtec(tmp_path)

    rows = build_mvtec_manifest(tmp_path, path_style="relative")

    assert list(rows.agent_input[0]) == AGENT_INPUT_COLUMNS
    for forbidden_field in ["label", "mask_path", "defect_type"]:
        assert forbidden_field not in rows.agent_input[0]


def test_evaluator_has_labels(tmp_path: Path) -> None:
    _fake_mvtec(tmp_path)

    rows = build_mvtec_manifest(tmp_path, path_style="relative")

    assert list(rows.evaluator[0]) == EVALUATOR_COLUMNS
    labels = {row["defect_type"]: row["label"] for row in rows.evaluator}
    assert labels["good"] == 0
    assert labels["crack"] == 1


def test_deterministic_image_id(tmp_path: Path) -> None:
    _fake_mvtec(tmp_path)

    first = build_mvtec_manifest(tmp_path, path_style="relative")
    second = build_mvtec_manifest(tmp_path, path_style="relative")

    assert [row["image_id"] for row in first.agent_input] == [
        row["image_id"] for row in second.agent_input
    ]
    assert first.audit["counts"]["duplicate_image_ids"] == 0


def test_missing_mask_is_recorded(tmp_path: Path) -> None:
    _fake_mvtec(tmp_path)

    rows = build_mvtec_manifest(tmp_path, path_style="relative")

    assert rows.audit["counts"]["missing_masks"] == 1
    assert rows.audit["missing_masks"] == [
        {
            "category": "bottle",
            "defect_type": "crack",
            "image_path": "bottle/test/crack/003.png",
        }
    ]


def test_can_scan_selected_category(tmp_path: Path) -> None:
    _fake_mvtec(tmp_path)
    _png(tmp_path / "cable" / "train" / "good" / "000.png")

    rows = build_mvtec_manifest(tmp_path, path_style="relative", categories=["bottle"])

    assert rows.audit["categories"] == ["bottle"]
    assert {row["category"] for row in rows.agent_input} == {"bottle"}


def test_manifest_writer_outputs_expected_files(tmp_path: Path) -> None:
    data_root = tmp_path / "mvtec"
    _fake_mvtec(data_root)
    rows = build_mvtec_manifest(data_root, path_style="relative")

    agent_path, evaluator_path, audit_path = write_manifest_outputs(
        agent_rows=rows.agent_input,
        evaluator_rows=rows.evaluator,
        audit=rows.audit,
        manifest_dir=tmp_path / "data" / "manifests",
        audit_path=tmp_path / "outputs" / "stage1_gate" / "mvtec_audit.json",
    )

    assert agent_path.name == "mvtec_agent_input.csv"
    assert evaluator_path.name == "mvtec_evaluator.csv"
    assert audit_path.name == "mvtec_audit.json"

    with agent_path.open(newline="", encoding="utf-8") as handle:
        assert csv.DictReader(handle).fieldnames == AGENT_INPUT_COLUMNS
    with evaluator_path.open(newline="", encoding="utf-8") as handle:
        assert csv.DictReader(handle).fieldnames == EVALUATOR_COLUMNS
