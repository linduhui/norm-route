import csv
import json
from pathlib import Path

import pytest

from src.normroute.agent.replay_executor import SELECTED_PREDICTION_COLUMNS
from src.normroute.evaluation.stage4_summary import (
    AGGREGATIONS,
    REPORT_SUMMARY_FILENAMES,
    Stage4SummaryError,
    summarize_stage4,
    write_stage4_summary,
)


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _prediction(
    *,
    fold: str,
    policy: str,
    expert: str,
    task_id: str,
    image_id: str,
    category: str,
    label: int,
    score: float,
    seed: int,
) -> dict[str, object]:
    support_set_id = f"mvtec_{category}_k1_seed{seed}"
    return {
        "task_id": task_id,
        "fold": fold,
        "split": "test",
        "policy_name": policy,
        "selected_expert": expert,
        "stage2_run_dir": f"outputs/stage2/{expert}/{support_set_id}",
        "runtime_source": "estimated_runtime",
        "image_id": image_id,
        "expert_name": expert,
        "dataset": "mvtec",
        "category": category,
        "support_set_id": support_set_id,
        "k_shot": 1,
        "seed": seed,
        "final_score": score,
        "final_decision": "anomaly" if score >= 0.5 else "normal",
        "anomaly_map_path": "",
        "pixel_score_path": "",
        "actions": "TEST",
        "tool_calls": 1,
        "runtime_ms": 10.0 if expert == "PatchCore" else 20.0,
        "status": "ok",
        "error_message": "",
        "_label": label,
    }


def _fixture(tmp_path: Path) -> tuple[list[Path], Path, Path]:
    rows_by_combo: dict[tuple[str, str], list[dict[str, object]]] = {}
    evaluator_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for fold_index, fold in enumerate(("fold0", "fold1")):
        seed = fold_index
        for category in ("bottle", "cable"):
            for label in (0, 1):
                image_id = f"{fold}-{category}-{label}"
                task_id = f"task-{fold}-{category}-{label}"
                evaluator_rows.append({"image_id": image_id, "label": label})
                manifest_rows.append(
                    {
                        "fold": fold,
                        "split": "test",
                        "task_id": task_id,
                        "sample_id": image_id,
                        "dataset": "mvtec",
                        "category": category,
                        "k_shot": 1,
                        "seed": seed,
                        "support_set_id": f"mvtec_{category}_k1_seed{seed}",
                    }
                )
                patch_good = fold == "fold1" or category == "bottle"
                patch_score = (0.9 if label else 0.1) if patch_good else (0.2 if label else 0.8)
                win_score = (0.2 if label else 0.8) if patch_good else (0.9 if label else 0.1)
                for policy, expert, score in (
                    ("always_patchcore", "PatchCore", patch_score),
                    ("always_winclip", "WinCLIP", win_score),
                ):
                    rows_by_combo.setdefault((policy, fold), []).append(
                        _prediction(
                            fold=fold,
                            policy=policy,
                            expert=expert,
                            task_id=task_id,
                            image_id=image_id,
                            category=category,
                            label=label,
                            score=score,
                            seed=seed,
                        )
                    )
                router_expert = "PatchCore" if patch_good else "WinCLIP"
                router_score = patch_score if router_expert == "PatchCore" else win_score
                rows_by_combo.setdefault(("router", fold), []).append(
                    _prediction(
                        fold=fold,
                        policy="router",
                        expert=router_expert,
                        task_id=task_id,
                        image_id=image_id,
                        category=category,
                        label=label,
                        score=router_score,
                        seed=seed,
                    )
                )

    paths: list[Path] = []
    for (policy, fold), rows in rows_by_combo.items():
        path = tmp_path / "outputs" / "stage4" / "runs" / policy / fold / "test" / "selected_predictions.csv"
        clean_rows = []
        for row in rows:
            clean_rows.append({key: value for key, value in row.items() if key != "_label"})
        _write_csv(path, list(SELECTED_PREDICTION_COLUMNS), clean_rows)
        path.with_name("failures.json").write_text(
            json.dumps({"num_failed": 0, "failed_tasks": []}), encoding="utf-8"
        )
        paths.append(path)
    evaluator = tmp_path / "data" / "evaluator.csv"
    _write_csv(evaluator, ["image_id", "label"], evaluator_rows)
    manifest = tmp_path / "outputs" / "stage4" / "splits" / "fold_manifest.csv"
    _write_csv(
        manifest,
        [
            "fold",
            "split",
            "task_id",
            "sample_id",
            "dataset",
            "category",
            "k_shot",
            "seed",
            "support_set_id",
        ],
        manifest_rows,
    )
    return paths, evaluator, manifest


def test_full_stage4_summary_has_fold_local_references_and_macro_metrics(
    tmp_path: Path,
) -> None:
    paths, evaluator, manifest = _fixture(tmp_path)

    result = summarize_stage4(
        selected_predictions=paths,
        evaluator_csv=evaluator,
        fold_manifest_csv=manifest,
    )
    names = {row["policy_name"] for row in result.fold_metric_rows}
    assert {"best_single_expert", "run_level_oracle"}.issubset(names)
    assert "sample_level_oracle" not in names
    assert {row["aggregation"] for row in result.fold_metric_rows} == set(AGGREGATIONS)

    by_key = {
        (row["policy_name"], row["fold"], row["aggregation"]): row
        for row in result.fold_metric_rows
    }
    assert by_key[("run_level_oracle", "fold0", "macro_run")]["auroc"] == pytest.approx(1.0)
    assert by_key[("router", "fold0", "macro_run")]["selection_agreement_with_oracle"] == pytest.approx(1.0)
    assert by_key[("always_patchcore", "fold0", "macro_run")]["auroc"] == pytest.approx(0.5)
    assert by_key[("best_single_expert", "fold0", "macro_run")]["selected_expert"] == "PatchCore"
    assert by_key[("run_level_oracle", "fold0", "macro_run")]["policy_kind"] == "evaluator_only_reference"
    assert result.failure_cases == []

    summary = {
        (row["policy_name"], row["aggregation"]): row
        for row in result.policy_summary_rows
    }
    patch = summary[("always_patchcore", "macro_run")]
    assert patch["auroc_mean"] == pytest.approx(0.75)
    assert patch["auroc_std"] == pytest.approx(0.25)
    assert "oracle_regret_mean" in patch
    assert "estimated_runtime_ms_mean" in patch
    assert "tool_calls_mean" in patch
    assert "abstention_rate_mean" in patch


def test_stage4_summary_writes_evaluation_outputs_and_only_compact_reports(
    tmp_path: Path,
) -> None:
    paths, evaluator, manifest = _fixture(tmp_path)
    result = summarize_stage4(
        selected_predictions=paths,
        evaluator_csv=evaluator,
        fold_manifest_csv=manifest,
    )
    output = tmp_path / "outputs" / "stage4" / "evaluation"
    reports = tmp_path / "reports" / "stage4"

    written = write_stage4_summary(
        result=result,
        output_dir=output,
        reports_dir=reports,
    )

    assert all(path.is_file() for path in written.values())
    assert {path.name for path in reports.iterdir()} == set(REPORT_SUMMARY_FILENAMES)
    failures = json.loads(written["failure_cases"].read_text(encoding="utf-8"))
    assert failures["failure_cases"] == []
    config = json.loads(written["config"].read_text(encoding="utf-8"))
    assert config["sample_level_oracle_exported"] is False
    assert config["oracle_scope"].startswith("one expert per complete test run")

    with pytest.raises(Stage4SummaryError, match="outputs/stage4/evaluation"):
        write_stage4_summary(
            result=result,
            output_dir=tmp_path / "reports" / "stage4",
            reports_dir=None,
        )


def test_stage4_summary_rejects_sample_level_oracle_as_policy(tmp_path: Path) -> None:
    paths, evaluator, manifest = _fixture(tmp_path)
    path = paths[0]
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["policy_name"] = "sample_level_oracle"
    _write_csv(path, list(SELECTED_PREDICTION_COLUMNS), rows)

    with pytest.raises(Stage4SummaryError, match="cannot be supplied as realizable"):
        summarize_stage4(
            selected_predictions=paths,
            evaluator_csv=evaluator,
            fold_manifest_csv=manifest,
        )
