import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.normroute.routing.quality_metrics import (
    QualityMetricError,
    compute_average_precision,
    compute_auroc,
    compute_f1_max,
    evaluate_expert_quality,
    write_expert_quality_outputs,
)


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
    image_id: str,
    label: int,
    score: float,
    *,
    expert_name: str = "patchcore",
    category: str = "bottle",
    seed: int = 0,
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "dataset": "mvtec",
        "category": category,
        "support_set_id": f"mvtec_{category}_k1_seed{seed}",
        "k_shot": 1,
        "seed": seed,
        "expert_name": expert_name,
        "final_score": score,
        "final_decision": "anomaly" if score >= 0.5 else "normal",
        "status": "ok",
        "error_message": "",
        "anomaly_map_path": "",
        "pixel_score_path": "",
        "label": label,
        "mask_path": "",
        "run_dir": "outputs/stage2/run",
    }


def test_metric_primitives_treat_larger_scores_as_more_anomalous() -> None:
    labels = [0, 1, 0, 1]
    scores = [0.1, 0.8, 0.4, 0.9]

    assert compute_auroc(labels, scores) == pytest.approx(1.0)
    assert compute_average_precision(labels, scores) == pytest.approx(1.0)
    f1, threshold = compute_f1_max(labels, scores)
    assert f1 == pytest.approx(1.0)
    assert threshold == pytest.approx(0.8)


def test_evaluate_expert_quality_by_run_summary_and_warnings(tmp_path: Path) -> None:
    matrix_path = tmp_path / "outputs" / "stage3" / "evaluator_only" / "routing_matrix_long.csv"
    columns = [
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
    _write_csv(
        matrix_path,
        columns,
        [
            _routing_row("normal-1", 0, 0.1),
            _routing_row("normal-2", 0, 0.4),
            _routing_row("anom-1", 1, 0.8),
            _routing_row("anom-2", 1, 0.9),
            _routing_row("only-normal", 0, 0.2, expert_name="winclip"),
        ],
    )

    result = evaluate_expert_quality(matrix_path)

    by_run = {row["expert"]: row for row in result.by_run_rows}
    assert by_run["patchcore"]["image_auroc"] == pytest.approx(1.0)
    assert by_run["patchcore"]["image_ap"] == pytest.approx(1.0)
    assert by_run["patchcore"]["image_f1_max"] == pytest.approx(1.0)
    assert by_run["patchcore"]["best_threshold"] == pytest.approx(0.8)
    assert by_run["patchcore"]["num_samples"] == 4
    assert by_run["patchcore"]["num_normal"] == 2
    assert by_run["patchcore"]["num_anomaly"] == 2
    assert by_run["winclip"]["image_auroc"] is None
    assert by_run["winclip"]["image_ap"] is None
    assert len(result.warnings) == 1
    assert "Single-label group" in result.warnings[0]

    output_dir = tmp_path / "outputs" / "stage3" / "evaluator_only"
    reports_dir = tmp_path / "reports" / "stage3"
    by_run_path, summary_path, warnings_path = write_expert_quality_outputs(
        result,
        output_dir,
        reports_dir,
    )
    assert by_run_path.is_file()
    assert summary_path.is_file()
    assert (reports_dir / "expert_quality_summary.csv").is_file()

    written_by_run = {row["expert"]: row for row in _read_csv(by_run_path)}
    assert written_by_run["patchcore"]["image_auroc"] == "1"
    assert written_by_run["winclip"]["image_auroc"] == ""
    warnings_payload = json.loads(warnings_path.read_text(encoding="utf-8"))
    assert warnings_payload["warnings"] == result.warnings


def test_evaluate_expert_quality_cli_writes_expected_files(tmp_path: Path) -> None:
    matrix_path = tmp_path / "outputs" / "stage3" / "evaluator_only" / "routing_matrix_long.csv"
    _write_csv(
        matrix_path,
        [
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
        ],
        [
            _routing_row("normal-1", 0, 0.1),
            _routing_row("anom-1", 1, 0.8),
        ],
    )
    output_dir = tmp_path / "outputs" / "stage3" / "evaluator_only"
    reports_dir = tmp_path / "reports" / "stage3"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.normroute.cli.evaluate_expert_quality",
            "--routing-matrix-long",
            str(matrix_path),
            "--output-dir",
            str(output_dir),
            "--reports-dir",
            str(reports_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "expert_quality_by_run.csv").is_file()
    assert (output_dir / "expert_quality_summary.csv").is_file()
    assert (output_dir / "expert_quality_warnings.json").is_file()
    assert (reports_dir / "expert_quality_summary.csv").is_file()


def test_evaluate_expert_quality_rejects_invalid_label_without_silent_skip(tmp_path: Path) -> None:
    matrix_path = tmp_path / "routing_matrix_long.csv"
    _write_csv(
        matrix_path,
        ["dataset", "category", "support_set_id", "k_shot", "seed", "expert_name", "final_score", "label"],
        [
            {
                "dataset": "mvtec",
                "category": "bottle",
                "support_set_id": "sid",
                "k_shot": 1,
                "seed": 0,
                "expert_name": "patchcore",
                "final_score": 0.1,
                "label": "unknown",
            }
        ],
    )

    with pytest.raises(QualityMetricError, match="invalid binary label"):
        evaluate_expert_quality(matrix_path)
