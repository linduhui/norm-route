import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.normroute.evaluation.export import METRICS_FIELDS, PREDICTION_COLUMNS
from src.normroute.experts.base import ExpertInputError, validate_expert_input_fields


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def test_dummy_expert_cli_writes_stage2_output_schema(tmp_path: Path) -> None:
    agent_input = tmp_path / "agent_input.csv"
    support = tmp_path / "support.csv"
    output_dir = tmp_path / "stage2"
    _write_csv(
        agent_input,
        ["image_id", "dataset", "category", "split", "image_path"],
        [
            {
                "image_id": "train-good-0",
                "dataset": "mvtec",
                "category": "bottle",
                "split": "train",
                "image_path": "/datasets/mvtec/bottle/train/good/000.png",
            },
            {
                "image_id": "test-good-0",
                "dataset": "mvtec",
                "category": "bottle",
                "split": "test",
                "image_path": "/datasets/mvtec/bottle/test/good/001.png",
            },
        ],
    )
    _write_csv(
        support,
        [
            "support_set_id",
            "dataset",
            "category",
            "k_shot",
            "seed",
            "support_rank",
            "image_id",
            "image_path",
        ],
        [
            {
                "support_set_id": "mvtec_bottle_k1_seed0",
                "dataset": "mvtec",
                "category": "bottle",
                "k_shot": 1,
                "seed": 0,
                "support_rank": 1,
                "image_id": "train-good-0",
                "image_path": "/datasets/mvtec/bottle/train/good/000.png",
            }
        ],
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.normroute.cli.run_expert",
            "--expert",
            "dummy",
            "--agent-input-csv",
            str(agent_input),
            "--support-set-csv",
            str(support),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "anomaly_maps").is_dir()
    assert (output_dir / "run_metadata.json").is_file()

    with (output_dir / "predictions.csv").open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == PREDICTION_COLUMNS
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["image_id"] == "test-good-0"
    assert rows[0]["support_set_id"] == "mvtec_bottle_k1_seed0"
    for forbidden in ["label", "mask_path", "defect_type", "anomaly_type"]:
        assert forbidden not in rows[0]

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert list(metrics) == sorted(METRICS_FIELDS)
    assert metrics["num_predictions"] == 1
    assert metrics["num_failed"] == 0

    failures = json.loads((output_dir / "failures.json").read_text(encoding="utf-8"))
    assert failures == {"failed_predictions": []}

    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["stage"] == "stage2"
    assert metadata["expert_name"] == "dummy"
    assert metadata["support_set_csv"] == str(support)


def test_expert_input_rejects_evaluator_only_fields() -> None:
    for forbidden in ["label", "mask_path", "defect_type", "anomaly_type"]:
        with pytest.raises(ExpertInputError, match="forbidden fields"):
            validate_expert_input_fields({"image_id": "x", forbidden: "leak"})
