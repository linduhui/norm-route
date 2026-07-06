import csv
import json
from pathlib import Path

from src.normroute.cli.run_grid import run_grid


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def test_grid_does_not_overwrite_existing_results(tmp_path: Path) -> None:
    agent_input = tmp_path / "manifests" / "mvtec_agent_input.csv"
    support_dir = tmp_path / "support_sets"
    output_root = tmp_path / "outputs"
    support_set_id = "mvtec_bottle_k1_seed0"
    existing_dir = (
        output_root
        / "dummy"
        / "mvtec"
        / "bottle"
        / "k1"
        / "seed0"
        / support_set_id
    )
    existing_dir.mkdir(parents=True)
    existing_predictions = existing_dir / "predictions.csv"
    existing_predictions.write_text("sentinel\n", encoding="utf-8")
    _write_csv(
        agent_input,
        ["image_id", "dataset", "category", "split", "image_path"],
        [
            {
                "image_id": "test-good-0",
                "dataset": "mvtec",
                "category": "bottle",
                "split": "test",
                "image_path": "/datasets/mvtec/bottle/test/good/001.png",
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
                "image_path": "/datasets/mvtec/bottle/train/good/000.png",
            }
        ],
    )

    result = run_grid(
        experts=["dummy"],
        dataset="mvtec",
        categories=["bottle"],
        k_shots=[1],
        seeds=[0],
        agent_input_csv=agent_input,
        support_set_dir=support_dir,
        support_set_csv_template="{dataset}_k{k_shot}_seed{seed}.csv",
        output_root=output_root,
    )

    assert existing_predictions.read_text(encoding="utf-8") == "sentinel\n"
    assert result["num_skipped_existing"] == 1
    assert result["num_failed"] == 0

    grid_log = json.loads((output_root / "grid_log.json").read_text(encoding="utf-8"))
    assert grid_log["records"][0]["status"] == "skipped_existing"
    assert grid_log["records"][0]["output_dir"] == str(existing_dir)


def test_grid_writes_combination_output_dir(tmp_path: Path) -> None:
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
                "image_path": "/datasets/mvtec/bottle/test/good/001.png",
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
                "image_path": "/datasets/mvtec/bottle/train/good/000.png",
            }
        ],
    )

    result = run_grid(
        experts=["dummy"],
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
        / "dummy"
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
