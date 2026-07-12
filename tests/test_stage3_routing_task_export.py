import csv
import json
import subprocess
import sys
from pathlib import Path

from src.normroute.cli.export_routing_tasks import export_routing_tasks
from src.normroute.cli.validate_stage3_outputs import validate_stage3_outputs


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _routing_wide_row(image_id: str = "sample-1") -> dict[str, object]:
    return {
        "image_id": image_id,
        "dataset": "mvtec",
        "category": "bottle",
        "support_set_id": "mvtec_bottle_k1_seed0",
        "k_shot": 1,
        "seed": 0,
        "label": 1,
        "mask_path": "/masks/sample-1.png",
        "patchcore_score": 0.1,
        "winclip_score": 0.2,
        "anomalydino_score": 0.3,
    }


def _write_stage3_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    evaluator_only_dir = tmp_path / "outputs" / "stage3" / "evaluator_only"
    agent_visible_dir = tmp_path / "outputs" / "stage3" / "agent_visible"
    oracle_dir = tmp_path / "outputs" / "stage3" / "oracle"
    wide_path = evaluator_only_dir / "routing_matrix_wide.csv"
    wide_columns = [
        "image_id",
        "dataset",
        "category",
        "support_set_id",
        "k_shot",
        "seed",
        "label",
        "mask_path",
        "patchcore_score",
        "winclip_score",
        "anomalydino_score",
    ]
    _write_csv(wide_path, wide_columns, [_routing_wide_row()])
    long_rows = []
    for expert, score in [
        ("patchcore", 0.1),
        ("winclip", 0.2),
        ("anomalydino", 0.3),
    ]:
        long_rows.append(
            {
                "image_id": "sample-1",
                "dataset": "mvtec",
                "category": "bottle",
                "support_set_id": "mvtec_bottle_k1_seed0",
                "k_shot": 1,
                "seed": 0,
                "expert_name": expert,
                "final_score": score,
                "final_decision": "anomaly",
                "status": "ok",
                "error_message": "",
                "anomaly_map_path": "",
                "pixel_score_path": "",
                "label": 1,
                "mask_path": "/masks/sample-1.png",
                "run_dir": f"outputs/stage2/{expert}",
            }
        )
    _write_csv(
        evaluator_only_dir / "routing_matrix_long.csv",
        list(long_rows[0]),
        long_rows,
    )
    _write_csv(
        oracle_dir / "oracle_summary.csv",
        ["evaluator_only", "method", "selection_metric"],
        [{"evaluator_only": "True", "method": "best_single", "selection_metric": "image_auroc"}],
    )
    _write_csv(
        oracle_dir / "oracle_selection_counts.csv",
        ["evaluator_only", "expert", "selection_count"],
        [{"evaluator_only": "True", "expert": "patchcore", "selection_count": 1}],
    )
    export_routing_tasks(routing_matrix_wide=wide_path, output_dir=agent_visible_dir)
    return wide_path, evaluator_only_dir, agent_visible_dir, oracle_dir


def test_export_routing_tasks_removes_evaluator_only_fields(tmp_path: Path) -> None:
    wide_path, _, agent_visible_dir, _ = _write_stage3_fixture(tmp_path)

    task_path, card_path = export_routing_tasks(
        routing_matrix_wide=wide_path,
        output_dir=agent_visible_dir,
    )

    tasks = [json.loads(line) for line in task_path.read_text(encoding="utf-8").splitlines()]
    assert len(tasks) == 1
    assert tasks[0]["sample_id"] == "sample-1"
    assert tasks[0]["expert_scores"] == {
        "PatchCore": 0.1,
        "WinCLIP": 0.2,
        "AnomalyDINO": 0.3,
    }
    forbidden = {"label", "mask_path", "defect_type", "anomaly_type", "oracle_best_expert"}
    assert forbidden.isdisjoint(tasks[0])

    cards = json.loads(card_path.read_text(encoding="utf-8"))
    assert set(cards) == {"PatchCore", "WinCLIP", "AnomalyDINO"}
    for card in cards.values():
        assert {"method_description", "input_requirements", "compute_cost"}.issubset(card)
        assert forbidden.isdisjoint(card)


def test_validate_stage3_outputs_accepts_clean_fixture(tmp_path: Path) -> None:
    _, evaluator_only_dir, agent_visible_dir, oracle_dir = _write_stage3_fixture(tmp_path)

    errors = validate_stage3_outputs(
        stage3_root=tmp_path / "outputs" / "stage3",
        evaluator_only_dir=evaluator_only_dir,
        agent_visible_dir=agent_visible_dir,
        oracle_dir=oracle_dir,
    )

    assert errors == []


def test_validate_stage3_outputs_rejects_agent_visible_leakage(tmp_path: Path) -> None:
    _, evaluator_only_dir, agent_visible_dir, oracle_dir = _write_stage3_fixture(tmp_path)
    with (agent_visible_dir / "agent_routing_tasks.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"sample_id": "sample-2", "label": 1}) + "\n")

    errors = validate_stage3_outputs(
        stage3_root=tmp_path / "outputs" / "stage3",
        evaluator_only_dir=evaluator_only_dir,
        agent_visible_dir=agent_visible_dir,
        oracle_dir=oracle_dir,
    )

    assert any("forbidden agent-visible fields" in error for error in errors)


def test_validate_stage3_outputs_rejects_unaligned_expert_samples(tmp_path: Path) -> None:
    _, evaluator_only_dir, agent_visible_dir, oracle_dir = _write_stage3_fixture(tmp_path)
    rows = []
    with (evaluator_only_dir / "routing_matrix_long.csv").open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        for row in reader:
            if row["expert_name"] == "winclip":
                row["image_id"] = "different-sample"
            rows.append(row)
    _write_csv(evaluator_only_dir / "routing_matrix_long.csv", columns, rows)

    errors = validate_stage3_outputs(
        stage3_root=tmp_path / "outputs" / "stage3",
        evaluator_only_dir=evaluator_only_dir,
        agent_visible_dir=agent_visible_dir,
        oracle_dir=oracle_dir,
    )

    assert any("not aligned" in error for error in errors)


def test_export_routing_tasks_cli_writes_expected_files(tmp_path: Path) -> None:
    wide_path, _, _, _ = _write_stage3_fixture(tmp_path)
    output_dir = tmp_path / "outputs" / "stage3" / "routing_matrix"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.normroute.cli.export_routing_tasks",
            "--routing-matrix-wide",
            str(wide_path),
            "--output-dir",
            str(output_dir),
            "--dataset",
            "mvtec",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "agent_routing_tasks.jsonl").is_file()
    assert (output_dir / "expert_cards.json").is_file()


def test_validate_stage3_outputs_cli_supports_day14_paths(tmp_path: Path) -> None:
    wide_path, evaluator_only_dir, agent_visible_dir, oracle_dir = _write_stage3_fixture(tmp_path)
    routing_matrix_dir = tmp_path / "outputs" / "stage3" / "routing_matrix"
    export_routing_tasks(routing_matrix_wide=wide_path, output_dir=routing_matrix_dir, dataset="mvtec")
    report_dir = tmp_path / "reports" / "stage3"
    report_dir.mkdir(parents=True)
    (report_dir / "stage3_report.md").write_text("# Stage 3\n", encoding="utf-8")
    (report_dir / "routing_matrix_schema.md").write_text("# Schema\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.normroute.cli.validate_stage3_outputs",
            "--stage3-root",
            str(tmp_path / "outputs" / "stage3"),
            "--evaluator-only-dir",
            str(evaluator_only_dir),
            "--oracle-dir",
            str(oracle_dir),
            "--report-dir",
            str(report_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert agent_visible_dir.is_dir()
