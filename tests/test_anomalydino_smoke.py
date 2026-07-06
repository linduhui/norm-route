import csv
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from src.normroute.cli.run_grid import run_grid
from src.normroute.evaluation.export import PREDICTION_COLUMNS
from src.normroute.experts.anomalydino import AnomalyDINOConfig, AnomalyDINOExpert
from src.normroute.experts.base import Expert


ROOT = Path(__file__).resolve().parents[1]


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def test_anomalydino_implements_stage2_expert_interface() -> None:
    expert: Expert = AnomalyDINOExpert(output_dir="unused")

    assert expert.name == "anomalydino"


def test_anomalydino_config_loads_stage2_yaml() -> None:
    config = AnomalyDINOConfig.from_yaml(
        ROOT / "configs" / "stage2" / "anomalydino_mvtec.yaml"
    )

    assert config.image_size == 64
    assert config.token_grid_size == 8
    assert config.image_score_top_fraction == 0.1


def test_anomalydino_cli_writes_required_outputs_without_ground_truth(tmp_path: Path) -> None:
    support_image = tmp_path / "mvtec" / "bottle" / "train" / "good" / "000.png"
    query_image = tmp_path / "mvtec" / "bottle" / "test" / "good" / "001.png"
    _write_png(support_image, (20, 40, 60))
    _write_png(query_image, (24, 44, 64))

    agent_input = tmp_path / "agent_input.csv"
    support = tmp_path / "support.csv"
    output_dir = tmp_path / "stage2" / "anomalydino"
    support_set_id = "mvtec_bottle_k1_seed0"
    _write_csv(
        agent_input,
        ["image_id", "dataset", "category", "split", "image_path"],
        [
            {
                "image_id": "train-good-0",
                "dataset": "mvtec",
                "category": "bottle",
                "split": "train",
                "image_path": str(support_image),
            },
            {
                "image_id": "test-good-0",
                "dataset": "mvtec",
                "category": "bottle",
                "split": "test",
                "image_path": str(query_image),
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
                "support_set_id": support_set_id,
                "dataset": "mvtec",
                "category": "bottle",
                "k_shot": 1,
                "seed": 0,
                "support_rank": 1,
                "image_id": "train-good-0",
                "image_path": str(support_image),
            }
        ],
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.normroute.cli.run_expert",
            "--expert",
            "anomalydino",
            "--config",
            str(ROOT / "configs" / "stage2" / "anomalydino_mvtec.yaml"),
            "--agent-input-csv",
            str(agent_input),
            "--support-set-csv",
            str(support),
            "--output-dir",
            str(output_dir),
            "--dataset",
            "mvtec",
            "--category",
            "bottle",
            "--k-shot",
            "1",
            "--seed",
            "0",
            "--support-set-id",
            support_set_id,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "failures.json").is_file()

    with (output_dir / "predictions.csv").open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == PREDICTION_COLUMNS
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert row["expert_name"] == "anomalydino"
    assert row["support_set_id"] == support_set_id
    assert row["k_shot"] == "1"
    assert row["seed"] == "0"
    assert row["status"] == "ok"
    assert 0.0 <= float(row["final_score"]) <= 1.0
    assert Path(row["anomaly_map_path"]).is_file()
    assert Path(row["pixel_score_path"]).is_file()
    for forbidden in ["label", "mask_path", "defect_type", "anomaly_type"]:
        assert forbidden not in row

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["num_predictions"] == 1
    assert metrics["num_failed"] == 0

    failures = json.loads((output_dir / "failures.json").read_text(encoding="utf-8"))
    assert failures == {"failed_predictions": []}


def test_run_grid_invokes_anomalydino_expert(tmp_path: Path) -> None:
    support_image = tmp_path / "mvtec" / "bottle" / "train" / "good" / "000.png"
    query_image = tmp_path / "mvtec" / "bottle" / "test" / "good" / "001.png"
    _write_png(support_image, (20, 40, 60))
    _write_png(query_image, (24, 44, 64))

    agent_input = tmp_path / "manifests" / "mvtec_agent_input.csv"
    support_dir = tmp_path / "support_sets"
    output_root = tmp_path / "outputs"
    support_set_id = "mvtec_bottle_k1_seed0"
    _write_csv(
        agent_input,
        ["image_id", "dataset", "category", "split", "image_path"],
        [
            {
                "image_id": "test-good-0",
                "dataset": "mvtec",
                "category": "bottle",
                "split": "test",
                "image_path": str(query_image),
            }
        ],
    )
    _write_csv(
        support_dir / "mvtec_k1_seed0.csv",
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
                "support_set_id": support_set_id,
                "dataset": "mvtec",
                "category": "bottle",
                "k_shot": 1,
                "seed": 0,
                "support_rank": 1,
                "image_id": "train-good-0",
                "image_path": str(support_image),
            }
        ],
    )

    result = run_grid(
        experts=["anomalydino"],
        dataset="mvtec",
        categories=["bottle"],
        k_shots=[1],
        seeds=[0],
        agent_input_csv=agent_input,
        support_set_dir=support_dir,
        support_set_csv_template="{dataset}_k{k_shot}_seed{seed}.csv",
        output_root=output_root,
    )

    combo_dir = (
        output_root
        / "anomalydino"
        / "mvtec"
        / "bottle"
        / "k1"
        / "seed0"
        / support_set_id
    )
    assert result["num_ok"] == 1
    assert (combo_dir / "predictions.csv").is_file()
    assert (combo_dir / "metrics.json").is_file()
    assert (combo_dir / "failures.json").is_file()
