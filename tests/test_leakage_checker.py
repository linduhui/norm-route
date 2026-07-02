import csv
import json
import subprocess
import sys
from pathlib import Path

from src.data.leakage_checker import check_leakage, write_json_report, write_summary
from src.data.manifest import AGENT_INPUT_COLUMNS, EVALUATOR_COLUMNS
from src.data.support_sampler import SUPPORT_SET_COLUMNS


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _valid_agent_rows() -> list[dict[str, object]]:
    return [
        {
            "image_id": "train-good-0",
            "dataset": "mvtec",
            "category": "bottle",
            "split": "train",
            "image_path": "/datasets/mvtec/bottle/train/good/000.png",
        },
        {
            "image_id": "train-good-1",
            "dataset": "mvtec",
            "category": "bottle",
            "split": "train",
            "image_path": "/datasets/mvtec/bottle/train/good/001.png",
        },
        {
            "image_id": "test-good",
            "dataset": "mvtec",
            "category": "bottle",
            "split": "test",
            "image_path": "/datasets/mvtec/bottle/test/good/002.png",
        },
        {
            "image_id": "test-crack",
            "dataset": "mvtec",
            "category": "bottle",
            "split": "test",
            "image_path": "/datasets/mvtec/bottle/test/crack/003.png",
        },
    ]


def _valid_evaluator_rows() -> list[dict[str, object]]:
    return [
        {"image_id": "train-good-0", "label": 0, "mask_path": "", "defect_type": "good"},
        {"image_id": "train-good-1", "label": 0, "mask_path": "", "defect_type": "good"},
        {"image_id": "test-good", "label": 0, "mask_path": "", "defect_type": "good"},
        {
            "image_id": "test-crack",
            "label": 1,
            "mask_path": "/datasets/mvtec/bottle/ground_truth/crack/003_mask.png",
            "defect_type": "crack",
        },
    ]


def _valid_support_rows() -> list[dict[str, object]]:
    return [
        {
            "support_set_id": "mvtec_bottle_k2_seed0",
            "dataset": "mvtec",
            "category": "bottle",
            "k_shot": 2,
            "seed": 0,
            "support_rank": 1,
            "image_id": "train-good-0",
            "image_path": "/datasets/mvtec/bottle/train/good/000.png",
        },
        {
            "support_set_id": "mvtec_bottle_k2_seed0",
            "dataset": "mvtec",
            "category": "bottle",
            "k_shot": 2,
            "seed": 0,
            "support_rank": 2,
            "image_id": "train-good-1",
            "image_path": "/datasets/mvtec/bottle/train/good/001.png",
        },
    ]


def _write_valid_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    agent_csv = tmp_path / "agent.csv"
    evaluator_csv = tmp_path / "evaluator.csv"
    support_csv = tmp_path / "support.csv"
    _write_csv(agent_csv, AGENT_INPUT_COLUMNS, _valid_agent_rows())
    _write_csv(evaluator_csv, EVALUATOR_COLUMNS, _valid_evaluator_rows())
    _write_csv(support_csv, SUPPORT_SET_COLUMNS, _valid_support_rows())
    return agent_csv, evaluator_csv, support_csv


def _issue_codes(report: dict[str, object]) -> set[str]:
    return {issue["code"] for issue in report["issues"]}  # type: ignore[index]


def test_leakage_checker_accepts_valid_manifests_and_support_set(tmp_path: Path) -> None:
    agent_csv, evaluator_csv, support_csv = _write_valid_inputs(tmp_path)

    report = check_leakage(
        agent_input_csv=agent_csv,
        evaluator_csv=evaluator_csv,
        support=support_csv,
    )

    assert report["ok"] is True
    assert report["error_count"] == 0
    assert "PASS" in report["summary"]


def test_leakage_checker_rejects_agent_input_label_leak(tmp_path: Path) -> None:
    agent_csv, evaluator_csv, support_csv = _write_valid_inputs(tmp_path)
    leaky_rows = [dict(row, label=0) for row in _valid_agent_rows()]
    _write_csv(agent_csv, AGENT_INPUT_COLUMNS + ["label"], leaky_rows)

    report = check_leakage(
        agent_input_csv=agent_csv,
        evaluator_csv=evaluator_csv,
        support=support_csv,
    )

    assert report["ok"] is False
    assert "AGENT_INPUT_FORBIDDEN_COLUMNS" in _issue_codes(report)


def test_leakage_checker_rejects_duplicate_ids_and_join_mismatch(tmp_path: Path) -> None:
    agent_csv, evaluator_csv, support_csv = _write_valid_inputs(tmp_path)
    agent_rows = _valid_agent_rows() + [
        {
            "image_id": "train-good-0",
            "dataset": "mvtec",
            "category": "bottle",
            "split": "train",
            "image_path": "/datasets/mvtec/bottle/train/good/004.png",
        }
    ]
    evaluator_rows = _valid_evaluator_rows()[:-1]
    _write_csv(agent_csv, AGENT_INPUT_COLUMNS, agent_rows)
    _write_csv(evaluator_csv, EVALUATOR_COLUMNS, evaluator_rows)

    report = check_leakage(
        agent_input_csv=agent_csv,
        evaluator_csv=evaluator_csv,
        support=support_csv,
    )

    assert "AGENT_IMAGE_ID_NOT_UNIQUE" in _issue_codes(report)
    assert "AGENT_EVALUATOR_JOIN_ROW_COUNT_CHANGED" in _issue_codes(report)


def test_leakage_checker_rejects_support_using_test_image(tmp_path: Path) -> None:
    agent_csv, evaluator_csv, support_csv = _write_valid_inputs(tmp_path)
    support_rows = _valid_support_rows()
    support_rows[1]["image_id"] = "test-good"
    support_rows[1]["image_path"] = "/datasets/mvtec/bottle/test/good/002.png"
    _write_csv(support_csv, SUPPORT_SET_COLUMNS, support_rows)

    report = check_leakage(
        agent_input_csv=agent_csv,
        evaluator_csv=evaluator_csv,
        support=support_csv,
    )

    codes = _issue_codes(report)
    assert "SUPPORT_NOT_TRAIN_SPLIT" in codes
    assert "SUPPORT_USES_TEST_IMAGE" in codes
    assert "SUPPORT_NOT_NORMAL_TRAIN_GOOD" in codes


def test_leakage_checker_rejects_support_missing_from_agent_input(tmp_path: Path) -> None:
    agent_csv, evaluator_csv, support_csv = _write_valid_inputs(tmp_path)
    support_rows = _valid_support_rows()
    support_rows[0]["image_id"] = "not-in-agent"
    _write_csv(support_csv, SUPPORT_SET_COLUMNS, support_rows)

    report = check_leakage(
        agent_input_csv=agent_csv,
        evaluator_csv=evaluator_csv,
        support=support_csv,
    )

    assert "SUPPORT_IMAGE_ID_NOT_IN_AGENT_INPUT" in _issue_codes(report)


def test_leakage_checker_rejects_duplicate_support_and_k_mismatch(tmp_path: Path) -> None:
    agent_csv, evaluator_csv, support_csv = _write_valid_inputs(tmp_path)
    support_rows = _valid_support_rows()
    support_rows[1]["image_id"] = "train-good-0"
    _write_csv(support_csv, SUPPORT_SET_COLUMNS, support_rows)

    report = check_leakage(
        agent_input_csv=agent_csv,
        evaluator_csv=evaluator_csv,
        support=support_csv,
    )

    codes = _issue_codes(report)
    assert "SUPPORT_SET_DUPLICATE_IMAGE_ID" in codes
    assert "SUPPORT_K_MISMATCH" not in codes

    support_rows = _valid_support_rows()[:1]
    _write_csv(support_csv, SUPPORT_SET_COLUMNS, support_rows)
    report = check_leakage(
        agent_input_csv=agent_csv,
        evaluator_csv=evaluator_csv,
        support=support_csv,
    )
    assert "SUPPORT_K_MISMATCH" in _issue_codes(report)


def test_leakage_checker_rejects_multiple_ids_for_same_support_signature(tmp_path: Path) -> None:
    agent_csv, evaluator_csv, _ = _write_valid_inputs(tmp_path)
    support_dir = tmp_path / "support_sets"
    rows_a = _valid_support_rows()
    rows_b = [dict(row, support_set_id="mvtec_bottle_alt_k2_seed0") for row in _valid_support_rows()]
    _write_csv(support_dir / "a.csv", SUPPORT_SET_COLUMNS, rows_a)
    _write_csv(support_dir / "b.csv", SUPPORT_SET_COLUMNS, rows_b)

    report = check_leakage(
        agent_input_csv=agent_csv,
        evaluator_csv=evaluator_csv,
        support=support_dir,
    )

    assert "SUPPORT_SET_ID_NOT_UNIQUE_FOR_SIGNATURE" in _issue_codes(report)


def test_leakage_checker_accepts_manifest_proof_when_path_is_not_canonical(tmp_path: Path) -> None:
    agent_csv, evaluator_csv, support_csv = _write_valid_inputs(tmp_path)
    agent_rows = _valid_agent_rows()
    agent_rows[0]["image_path"] = "/opaque/storage/000.png"
    support_rows = _valid_support_rows()
    support_rows[0]["image_path"] = "/opaque/storage/000.png"
    _write_csv(agent_csv, AGENT_INPUT_COLUMNS, agent_rows)
    _write_csv(support_csv, SUPPORT_SET_COLUMNS, support_rows)

    report = check_leakage(
        agent_input_csv=agent_csv,
        evaluator_csv=evaluator_csv,
        support=support_csv,
    )

    assert report["ok"] is True


def test_report_writers_and_cli_return_nonzero_on_leak(tmp_path: Path) -> None:
    agent_csv, evaluator_csv, support_csv = _write_valid_inputs(tmp_path)
    leaky_rows = [dict(row, defect_type="good") for row in _valid_agent_rows()]
    _write_csv(agent_csv, AGENT_INPUT_COLUMNS + ["defect_type"], leaky_rows)
    json_report = tmp_path / "report.json"
    summary_output = tmp_path / "summary.txt"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_leakage.py",
            "--agent-input-csv",
            str(agent_csv),
            "--evaluator-csv",
            str(evaluator_csv),
            "--support",
            str(support_csv),
            "--json-report",
            str(json_report),
            "--summary-output",
            str(summary_output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    report = json.loads(json_report.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert "FAIL" in summary_output.read_text(encoding="utf-8")

    write_json_report(tmp_path / "direct.json", report)
    write_summary(tmp_path / "direct.txt", report)
    assert (tmp_path / "direct.json").is_file()
    assert (tmp_path / "direct.txt").is_file()
