import csv
import subprocess
import sys
from pathlib import Path

import pytest

from src.normroute.routing.complementarity import compute_complementarity_summary
from src.normroute.routing.join_predictions import RoutingMatrixError
from src.normroute.routing.oracle import compute_oracle_outputs, write_oracle_outputs


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _routing_row(
    *,
    image_id: str,
    label: int,
    expert_name: str,
    score: float,
    decision: str,
    seed: int,
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "dataset": "mvtec",
        "category": "bottle",
        "support_set_id": f"mvtec_bottle_k1_seed{seed}",
        "k_shot": 1,
        "seed": seed,
        "expert_name": expert_name,
        "final_score": score,
        "final_decision": decision,
        "status": "ok",
        "error_message": "",
        "anomaly_map_path": "",
        "pixel_score_path": "",
        "label": label,
        "mask_path": "",
        "run_dir": f"outputs/stage2/{expert_name}/seed{seed}",
    }


def _quality_row(
    *,
    expert: str,
    seed: int,
    image_auroc: float,
    image_ap: float,
) -> dict[str, object]:
    return {
        "expert": expert,
        "dataset": "mvtec",
        "category": "bottle",
        "k_shot": 1,
        "seed": seed,
        "support_set_id": f"mvtec_bottle_k1_seed{seed}",
        "image_auroc": image_auroc,
        "image_ap": image_ap,
        "image_f1_max": image_auroc,
        "best_threshold": 0.5,
        "num_samples": 2,
        "num_normal": 1,
        "num_anomaly": 1,
    }


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    routing_path = tmp_path / "outputs" / "stage3" / "evaluator_only" / "routing_matrix_long.csv"
    quality_path = tmp_path / "outputs" / "stage3" / "quality" / "expert_quality_by_run.csv"
    routing_columns = [
        "image_id",
        "dataset",
        "category",
        "support_set_id",
        "k_shot",
        "seed",
        "expert_name",
        "final_score",
        "final_decision",
        "status",
        "error_message",
        "anomaly_map_path",
        "pixel_score_path",
        "label",
        "mask_path",
        "run_dir",
    ]
    rows: list[dict[str, object]] = []
    rows.extend(
        [
            _routing_row(
                image_id="seed0-normal",
                label=0,
                expert_name="patchcore",
                score=0.1,
                decision="normal",
                seed=0,
            ),
            _routing_row(
                image_id="seed0-normal",
                label=0,
                expert_name="winclip",
                score=0.7,
                decision="anomaly",
                seed=0,
            ),
            _routing_row(
                image_id="seed0-normal",
                label=0,
                expert_name="anomalydino",
                score=0.4,
                decision="normal",
                seed=0,
            ),
            _routing_row(
                image_id="seed0-anomaly",
                label=1,
                expert_name="patchcore",
                score=0.8,
                decision="anomaly",
                seed=0,
            ),
            _routing_row(
                image_id="seed0-anomaly",
                label=1,
                expert_name="winclip",
                score=0.2,
                decision="normal",
                seed=0,
            ),
            _routing_row(
                image_id="seed0-anomaly",
                label=1,
                expert_name="anomalydino",
                score=0.6,
                decision="anomaly",
                seed=0,
            ),
            _routing_row(
                image_id="seed1-normal",
                label=0,
                expert_name="patchcore",
                score=0.8,
                decision="anomaly",
                seed=1,
            ),
            _routing_row(
                image_id="seed1-normal",
                label=0,
                expert_name="winclip",
                score=0.2,
                decision="normal",
                seed=1,
            ),
            _routing_row(
                image_id="seed1-normal",
                label=0,
                expert_name="anomalydino",
                score=0.5,
                decision="anomaly",
                seed=1,
            ),
            _routing_row(
                image_id="seed1-anomaly",
                label=1,
                expert_name="patchcore",
                score=0.3,
                decision="normal",
                seed=1,
            ),
            _routing_row(
                image_id="seed1-anomaly",
                label=1,
                expert_name="winclip",
                score=0.9,
                decision="anomaly",
                seed=1,
            ),
            _routing_row(
                image_id="seed1-anomaly",
                label=1,
                expert_name="anomalydino",
                score=0.4,
                decision="normal",
                seed=1,
            ),
        ]
    )
    _write_csv(routing_path, routing_columns, rows)

    quality_columns = [
        "expert",
        "dataset",
        "category",
        "k_shot",
        "seed",
        "support_set_id",
        "image_auroc",
        "image_ap",
        "image_f1_max",
        "best_threshold",
        "num_samples",
        "num_normal",
        "num_anomaly",
    ]
    _write_csv(
        quality_path,
        quality_columns,
        [
            _quality_row(expert="patchcore", seed=0, image_auroc=0.9, image_ap=0.9),
            _quality_row(expert="winclip", seed=0, image_auroc=0.7, image_ap=0.7),
            _quality_row(expert="anomalydino", seed=0, image_auroc=0.6, image_ap=0.6),
            _quality_row(expert="patchcore", seed=1, image_auroc=0.4, image_ap=0.4),
            _quality_row(expert="winclip", seed=1, image_auroc=0.95, image_ap=0.95),
            _quality_row(expert="anomalydino", seed=1, image_auroc=0.8, image_ap=0.8),
        ],
    )
    return routing_path, quality_path


def test_compute_oracle_outputs_and_selection_counts_are_evaluator_only(tmp_path: Path) -> None:
    routing_path, quality_path = _write_fixture(tmp_path)

    result = compute_oracle_outputs(
        routing_matrix_long=routing_path,
        expert_quality_by_run=quality_path,
    )
    summary = {row["method"]: row for row in result.summary_rows}

    assert summary["best_single"]["evaluator_only"] is True
    assert summary["best_single"]["selected_expert"] == "winclip"
    assert summary["best_single"]["image_auroc_mean"] == pytest.approx(0.825)
    assert summary["run_level_oracle"]["image_auroc_mean"] == pytest.approx(0.925)
    assert summary["sample_level_oracle"]["image_auroc_mean"] == pytest.approx(1.0)

    counts = {
        (row["category"], row["k_shot"], row["expert"]): row["selection_count"]
        for row in result.selection_count_rows
    }
    assert counts[("bottle", "1", "patchcore")] == 1
    assert counts[("bottle", "1", "winclip")] == 1

    output_dir = tmp_path / "outputs" / "stage3" / "evaluator_only"
    summary_path, counts_path = write_oracle_outputs(result, output_dir)
    written_summary = _read_csv(summary_path)
    assert {row["evaluator_only"] for row in written_summary} == {"True"}
    assert "sample_level_oracle" in {row["method"] for row in written_summary}
    assert not (output_dir / "sample_level_oracle_predictions.csv").exists()
    assert {row["evaluator_only"] for row in _read_csv(counts_path)} == {"True"}

    with pytest.raises(RoutingMatrixError, match="evaluator_only"):
        write_oracle_outputs(result, tmp_path / "outputs" / "stage3" / "agent_visible")


def test_compute_complementarity_summary_contains_required_analyses(tmp_path: Path) -> None:
    routing_path, quality_path = _write_fixture(tmp_path)

    rows = compute_complementarity_summary(
        routing_matrix_long=routing_path,
        expert_quality_by_run=quality_path,
    )
    by_analysis = {row["analysis"]: row for row in rows}

    assert {
        "best_expert_by_category",
        "best_expert_by_k_shot",
        "expert_score_margin",
        "expert_disagreement_rate",
    }.issubset(by_analysis)
    assert by_analysis["best_expert_by_category"]["expert"] == "winclip"
    assert by_analysis["best_expert_by_k_shot"]["expert"] == "winclip"
    assert by_analysis["expert_score_margin"]["expert"] == "winclip"
    assert by_analysis["expert_score_margin"]["comparison_expert"] == "anomalydino"
    assert by_analysis["expert_score_margin"]["value"] == pytest.approx(0.125)
    assert by_analysis["expert_disagreement_rate"]["value"] == pytest.approx(1.0)
    assert all(row["evaluator_only"] is True for row in rows)


def test_compute_oracle_cli_writes_expected_files(tmp_path: Path) -> None:
    routing_path, quality_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "outputs" / "stage3" / "evaluator_only"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.normroute.cli.compute_oracle",
            "--routing-matrix-long",
            str(routing_path),
            "--expert-quality-by-run",
            str(quality_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "oracle_summary.csv").is_file()
    assert (output_dir / "oracle_selection_counts.csv").is_file()
    assert (output_dir / "complementarity_summary.csv").is_file()
    assert {row["evaluator_only"] for row in _read_csv(output_dir / "complementarity_summary.csv")} == {
        "True"
    }
