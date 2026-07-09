import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.normroute.evaluation.export import PREDICTION_COLUMNS
from src.normroute.routing.join_predictions import (
    RoutingMatrixError,
    audit_stage2_output_tree,
    write_routing_matrices,
    RoutingMatrices,
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


def _prediction_row() -> dict[str, object]:
    return {
        "image_id": "sample-1",
        "expert_name": "patchcore",
        "dataset": "mvtec",
        "category": "bottle",
        "support_set_id": "mvtec_bottle_k1_seed0",
        "k_shot": 1,
        "seed": 0,
        "final_score": 0.1,
        "final_decision": "normal",
        "anomaly_map_path": "",
        "pixel_score_path": "",
        "actions": "score",
        "tool_calls": 1,
        "runtime_ms": 2.0,
        "status": "ok",
        "error_message": "",
    }


def _write_minimal_run(run_dir: Path, *, columns: list[str] | None = None) -> None:
    row = _prediction_row()
    if columns and "label" in columns:
        row["label"] = 0
    _write_csv(run_dir / "predictions.csv", columns or PREDICTION_COLUMNS, [row])
    _write_json(run_dir / "metrics.json", {"num_predictions": 1, "num_failed": 0})
    _write_json(run_dir / "failures.json", {"failed_predictions": []})
    _write_json(run_dir / "run_metadata.json", {"expert_name": "patchcore"})


def test_audit_stage2_outputs_requires_run_metadata_and_rejects_label_leakage(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "stage2" / "patchcore" / "mvtec" / "bottle" / "k1" / "seed0" / "sid"
    _write_minimal_run(run_dir, columns=[*PREDICTION_COLUMNS, "label"])
    (run_dir / "run_metadata.json").unlink()

    audit = audit_stage2_output_tree(tmp_path / "stage2")

    assert any("run_metadata.json" in error for error in audit.errors)
    assert any("forbidden prediction columns" in error for error in audit.errors)


def test_audit_stage2_outputs_cli_returns_nonzero_on_leakage(tmp_path: Path) -> None:
    run_dir = tmp_path / "stage2" / "patchcore" / "mvtec" / "bottle" / "k1" / "seed0" / "sid"
    _write_minimal_run(run_dir, columns=[*PREDICTION_COLUMNS, "mask_path"])

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.normroute.cli.audit_stage2_outputs",
            str(tmp_path / "stage2"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "forbidden prediction columns" in completed.stderr


def test_routing_matrices_refuse_agent_visible_output_dir(tmp_path: Path) -> None:
    matrices = RoutingMatrices(long_rows=[], wide_rows=[])

    with pytest.raises(RoutingMatrixError, match="evaluator_only"):
        write_routing_matrices(matrices, tmp_path / "outputs" / "stage3" / "agent_visible")

