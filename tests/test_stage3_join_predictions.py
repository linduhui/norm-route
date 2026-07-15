import csv
import json
import subprocess
import sys
from pathlib import Path

from src.normroute.evaluation.export import PREDICTION_COLUMNS
from src.normroute.routing.join_predictions import (
    build_routing_matrices,
    write_routing_matrices,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prediction_row(expert_name: str, score: float) -> dict[str, object]:
    return {
        "image_id": "sample-1",
        "expert_name": expert_name,
        "dataset": "mvtec",
        "category": "bottle",
        "support_set_id": "mvtec_bottle_k1_seed0",
        "k_shot": 1,
        "seed": 0,
        "final_score": score,
        "final_decision": "anomaly",
        "anomaly_map_path": "",
        "pixel_score_path": "",
        "actions": "score",
        "tool_calls": 1,
        "runtime_ms": 2.0,
        "status": "ok",
        "error_message": "",
    }


def _write_run(stage2_root: Path, expert_name: str, score: float) -> None:
    run_dir = (
        stage2_root
        / expert_name
        / "mvtec"
        / "bottle"
        / "k1"
        / "seed0"
        / "mvtec_bottle_k1_seed0"
    )
    run_dir.mkdir(parents=True)
    _write_csv(run_dir / "predictions.csv", PREDICTION_COLUMNS, [_prediction_row(expert_name, score)])
    _write_json(run_dir / "metrics.json", {"num_predictions": 1, "num_failed": 0})
    _write_json(run_dir / "failures.json", {"failed_predictions": []})
    _write_json(run_dir / "run_metadata.json", {"expert_name": expert_name, "seed": 0})


def test_build_routing_matrices_long_and_wide(tmp_path: Path) -> None:
    stage2_root = tmp_path / "outputs" / "stage2"
    for expert_name, score in [
        ("patchcore", 0.1),
        ("winclip", 0.2),
        ("anomalydino", 0.3),
    ]:
        _write_run(stage2_root, expert_name, score)
    evaluator_csv = tmp_path / "data" / "manifests" / "mvtec_evaluator.csv"
    _write_csv(
        evaluator_csv,
        ["image_id", "label", "mask_path", "defect_type"],
        [
            {
                "image_id": "sample-1",
                "label": 1,
                "mask_path": "/masks/sample-1.png",
                "defect_type": "scratch",
            }
        ],
    )

    matrices = build_routing_matrices(stage2_root=stage2_root, evaluator_csv=evaluator_csv)

    assert len(matrices.long_rows) == 3
    assert {row["expert_name"] for row in matrices.long_rows} == {
        "patchcore",
        "winclip",
        "anomalydino",
    }
    assert all(row["label"] == "1" for row in matrices.long_rows)
    assert all(row["mask_path"] == "/masks/sample-1.png" for row in matrices.long_rows)
    assert all(row["runtime_ms"] == "2.0" for row in matrices.long_rows)
    assert len(matrices.wide_rows) == 1
    assert matrices.wide_rows[0]["patchcore_score"] == "0.1"
    assert matrices.wide_rows[0]["winclip_score"] == "0.2"
    assert matrices.wide_rows[0]["anomalydino_score"] == "0.3"

    output_dir = tmp_path / "outputs" / "stage3" / "evaluator_only"
    long_path, wide_path = write_routing_matrices(matrices, output_dir)
    assert long_path.is_file()
    assert wide_path.is_file()


def test_build_routing_matrix_cli_writes_default_named_files(tmp_path: Path) -> None:
    stage2_root = tmp_path / "outputs" / "stage2"
    _write_run(stage2_root, "patchcore", 0.1)
    evaluator_csv = tmp_path / "data" / "manifests" / "mvtec_evaluator.csv"
    _write_csv(evaluator_csv, ["image_id", "label", "mask_path"], [{"image_id": "sample-1", "label": 0, "mask_path": ""}])
    output_dir = tmp_path / "outputs" / "stage3" / "evaluator_only"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.normroute.cli.build_routing_matrix",
            "--stage2-root",
            str(stage2_root),
            "--evaluator-csv",
            str(evaluator_csv),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "routing_matrix_long.csv").is_file()
    assert (output_dir / "routing_matrix_wide.csv").is_file()
