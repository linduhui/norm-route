from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.normroute.cli.summarize_stage5_router import METRICS
from src.normroute.cli.summarize_stage5_router_multiseed import (
    Stage5RouterMultiseedSummaryError,
    _verify_multiseed_provenance,
    aggregate_multiseed_records,
    load_mechanism_diagnostics,
    paired_multiseed_deltas,
)


def _records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for seed in (0, 1):
        for fold_index, fold in enumerate(("fold0", "fold1")):
            for variant in ("reference", "candidate"):
                for method_index, method in enumerate(
                    ("learned_router", "global_best", "sample_oracle")
                ):
                    base = 10.0 + seed + method_index
                    delta = (1.0, 3.0)[fold_index] if variant == "candidate" else 0.0
                    row: dict[str, object] = {
                        "seed": seed,
                        "fold": fold,
                        "variant": variant,
                        "method": method,
                        "feature_dimension": 4,
                        "num_samples": 8,
                    }
                    for metric_index, metric in enumerate(METRICS):
                        row[metric] = base + metric_index + delta
                    records.append(row)
    return records


def test_multiseed_summary_uses_all_seed_fold_blocks() -> None:
    summary = aggregate_multiseed_records(
        _records(),
        seeds=(0, 1),
        folds=("fold0", "fold1"),
        variants=("reference", "candidate"),
    )

    row = next(
        item
        for item in summary
        if item["variant"] == "reference"
        and item["method"] == "learned_router"
        and item["metric"] == METRICS[0]
    )
    assert row["seed_count"] == 2
    assert row["fold_count"] == 2
    assert row["block_count"] == 4
    assert row["mean"] == pytest.approx(10.5)


def test_multiseed_pairing_averages_seeds_before_fold_inference() -> None:
    paired = paired_multiseed_deltas(
        _records(),
        seeds=(0, 1),
        folds=("fold0", "fold1"),
        variants=("reference", "candidate"),
        comparisons=(("reference", "candidate", "gain"),),
        bootstrap_replicates=100,
        bootstrap_seed=7,
    )

    row = next(
        item
        for item in paired
        if item["method"] == "learned_router" and item["metric"] == METRICS[0]
    )
    assert row["pairing_unit"] == "fold_after_seed_mean"
    assert row["mean_paired_delta"] == pytest.approx(2.0)
    assert row["positive_fold_fraction"] == 1.0
    assert row["two_sided_sign_flip_p_value"] == pytest.approx(0.5)


def test_multiseed_summary_rejects_incomplete_grid() -> None:
    with pytest.raises(Stage5RouterMultiseedSummaryError, match="incomplete"):
        aggregate_multiseed_records(
            _records()[:-1],
            seeds=(0, 1),
            folds=("fold0", "fold1"),
            variants=("reference", "candidate"),
        )


def _write_router_run(
    root: Path,
    *,
    seed: int,
    commit: str = "a" * 40,
    config_updates: dict[str, object] | None = None,
    teacher_hash: str = "b" * 64,
    bank_hash: str = "c" * 64,
) -> None:
    config: dict[str, object] = {
        "router_features": "/artifacts/fold0/features.jsonl",
        "fold_manifest": "/artifacts/fold_manifest.csv",
        "routing_matrix": "/artifacts/evaluator_only/routing_matrix.csv",
        "teacher_data": "/artifacts/fold0/teacher.parquet",
        "capability_bank": "/artifacts/fold0/capability_bank.json",
        "supervision": "soft_teacher",
        "supervision_target": "soft_teacher",
        "fold": "fold0",
        "variant": "full",
        "feature_view": "normal_bir_fbdp",
        "bir_ablation": "full",
        "fbdp_ablation": "full",
        "device": "cuda:0",
        "epochs": 40,
        "batch_size": 2048,
        "learning_rate": 0.05,
        "l2_grid": [0.0, 0.0001, 0.001],
        "seed": seed,
        "capability_weight_grid": [0.0, 0.25, 0.5],
        "uncertainty_weight_grid": [0.0, 0.25, 0.5, 1.0, 2.0],
        "cost_weight_grid": [0.0, 0.05, 0.1],
        "capability_mode": "conditional",
        "capability_skills": ["boundary", "fgbg", "lowshot", "texture"],
        "uncertainty_mode": "entropy_scaled_lcb",
        "validation_runtime_tradeoff": 0.05,
        "validation_selection_metric": "train_calibrated_zero_one_error_plus_normalized_runtime",
        "repeat_weighting": "equal_category_equal_query_inverse_variant_frequency",
        "model_kind": "class_balanced_linear_softmax_soft_targets",
        "standardization": "train_only_feature_mean_std",
        "test_prediction_contract": "label_free_and_frozen_before_evaluator_join",
    }
    config.update(config_updates or {})
    run = {
        "seed": seed,
        "git_commit": commit,
        "config": config,
        "input_hashes": {
            "router_features": "d" * 64,
            "fold_manifest": "e" * 64,
            "routing_matrix": "f" * 64,
            "teacher_data": teacher_hash,
            "capability_bank": bank_hash,
        },
    }
    path = root / f"seed{seed}" / "fold0" / "full" / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run), encoding="utf-8")


def test_multiseed_provenance_allows_only_seed_to_differ(tmp_path: Path) -> None:
    _write_router_run(tmp_path, seed=0)
    _write_router_run(tmp_path, seed=1)

    provenance = _verify_multiseed_provenance(
        tmp_path,
        seeds=(0, 1),
        folds=("fold0",),
        variants=("full",),
    )

    assert provenance["git_commit"] == "a" * 40
    assert provenance["teacher_data:fold0:full"] == "b" * 64
    assert provenance["capability_bank:fold0:full"] == "c" * 64


def test_multiseed_provenance_rejects_mixed_git_commits(tmp_path: Path) -> None:
    _write_router_run(tmp_path, seed=0)
    _write_router_run(tmp_path, seed=1, commit="9" * 40)

    with pytest.raises(Stage5RouterMultiseedSummaryError, match="git commits"):
        _verify_multiseed_provenance(
            tmp_path,
            seeds=(0, 1),
            folds=("fold0",),
            variants=("full",),
        )


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("epochs", 41),
        ("batch_size", 1024),
        ("learning_rate", 0.01),
        ("l2_grid", [0.0, 0.01]),
        ("device", "cpu"),
        ("model_kind", "different_model"),
        ("standardization", "none"),
        ("test_prediction_contract", "different_contract"),
        ("feature_view", "normal_only"),
        ("bir_ablation", "single_bank"),
        ("fbdp_ablation", "without_directional_evidence"),
        ("capability_weight_grid", [0.0]),
        ("uncertainty_weight_grid", [0.0]),
        ("cost_weight_grid", [0.0]),
        ("capability_mode", "static"),
        ("capability_skills", ["boundary"]),
        ("uncertainty_mode", "legacy_interval_width"),
    ),
)
def test_multiseed_provenance_rejects_non_seed_training_config_drift(
    tmp_path: Path, field: str, changed: object
) -> None:
    _write_router_run(tmp_path, seed=0)
    _write_router_run(tmp_path, seed=1, config_updates={field: changed})

    with pytest.raises(
        Stage5RouterMultiseedSummaryError,
        match="non-seed training config|redesigned uncertainty",
    ):
        _verify_multiseed_provenance(
            tmp_path,
            seeds=(0, 1),
            folds=("fold0",),
            variants=("full",),
        )


@pytest.mark.parametrize(
    ("hash_name", "changed_hash", "message"),
    (
        ("teacher_hash", "1" * 64, "teacher_data"),
        ("bank_hash", "2" * 64, "capability_bank"),
    ),
)
def test_multiseed_provenance_rejects_per_fold_teacher_or_bank_hash_drift(
    tmp_path: Path, hash_name: str, changed_hash: str, message: str
) -> None:
    _write_router_run(tmp_path, seed=0)
    _write_router_run(tmp_path, seed=1, **{hash_name: changed_hash})

    with pytest.raises(Stage5RouterMultiseedSummaryError, match=message):
        _verify_multiseed_provenance(
            tmp_path,
            seeds=(0, 1),
            folds=("fold0",),
            variants=("full",),
        )


def test_mechanism_diagnostics_report_selected_weights_and_route_changes(
    tmp_path: Path,
) -> None:
    variants = (
        "soft_teacher_full",
        "soft_no_capability",
        "soft_no_uncertainty",
        "soft_no_cost",
    )
    for variant in variants:
        run_dir = tmp_path / "seed0" / "fold0" / variant
        (run_dir / "evaluator_only").mkdir(parents=True)
        policy = {
            "capability_weight": 0.25,
            "uncertainty_weight": (
                0.0 if variant == "soft_no_uncertainty" else 0.5
            ),
            "cost_weight": 0.0 if variant == "soft_no_cost" else 0.1,
            "validation_empirical_loss": 0.2,
            "validation_selection_accuracy": 0.8,
            "validation_oracle_regret": 0.1,
        }
        (run_dir / "evaluator_only" / "metrics.json").write_text(
            json.dumps(
                {
                    "capability_policy": policy,
                    "validation_runtime_tradeoff": 0.05,
                    "uncertainty_mode": "entropy_scaled_lcb",
                }
            ),
            encoding="utf-8",
        )
        selected = "b" if variant == "soft_no_uncertainty" else "a"
        (run_dir / "predictions.jsonl").write_text(
            json.dumps({"task_id": "task", "selected_expert": selected})
            + "\n",
            encoding="utf-8",
        )

    policies, changes = load_mechanism_diagnostics(
        tmp_path,
        seeds=(0,),
        folds=("fold0",),
        variants=variants,
    )

    assert len(policies) == 4
    assert next(
        row for row in policies if row["variant"] == "soft_teacher_full"
    )["uncertainty_weight"] == 0.5
    assert len(changes) == 3
    assert next(
        row
        for row in changes
        if row["ablation_variant"] == "soft_no_uncertainty"
    )["route_change_rate"] == 1.0
