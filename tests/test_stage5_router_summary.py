from __future__ import annotations

import pytest

from src.normroute.cli.summarize_stage5_router import (
    aggregate_records,
    exact_sign_flip_pvalue,
    paired_ablation_deltas,
    render_paper_tables,
)


def _record(
    fold: str,
    variant: str,
    *,
    auroc: float,
    method: str = "learned_router",
) -> dict:
    return {
        "fold": fold,
        "variant": variant,
        "method": method,
        "feature_dimension": 10 if variant == "normal_only" else 15,
        "num_samples": 20,
        "selection_accuracy": auroc,
        "normalized_utility_mean": auroc,
        "oracle_regret_mean": 1.0 - auroc,
        "image_auroc": auroc,
        "image_ap": auroc,
        "image_f1_max_evaluator_only": auroc,
        "average_runtime_ms": 10.0,
    }


def test_exact_sign_flip_permutation_and_paired_fold_deltas() -> None:
    folds = tuple(f"fold{index}" for index in range(5))
    records = []
    for index, fold in enumerate(folds):
        records.append(
            _record(fold, "normal_only", auroc=0.60 + index * 0.01)
        )
        records.append(
            _record(fold, "full", auroc=0.70 + index * 0.01)
        )

    paired = paired_ablation_deltas(
        records,
        folds=folds,
        variants=("normal_only", "full"),
    )
    auroc = next(
        row
        for row in paired
        if row["comparison"] == "full_vs_normal"
        and row["metric"] == "image_auroc"
    )

    assert auroc["mean_paired_delta"] == pytest.approx(0.1)
    assert auroc["positive_fold_count"] == 5
    assert auroc["negative_fold_count"] == 0
    assert auroc["exact_sign_flip_p_two_sided"] == pytest.approx(0.0625)
    assert exact_sign_flip_pvalue([0.1] * 5) == pytest.approx(0.0625)


def test_router_summary_aggregation_and_markdown_are_academic_tables() -> None:
    records = []
    for fold, value in (("fold0", 0.7), ("fold1", 0.8)):
        for variant in ("normal_only", "full"):
            for method in ("learned_router", "global_best", "sample_oracle"):
                records.append(
                    _record(
                        fold,
                        variant,
                        auroc=(
                            value
                            if method != "sample_oracle"
                            else min(value + 0.1, 1.0)
                        ),
                        method=method,
                    )
                )
    summary = aggregate_records(records)
    paired = paired_ablation_deltas(
        records,
        folds=("fold0", "fold1"),
        variants=("normal_only", "full"),
    )
    markdown = render_paper_tables(summary, paired)

    full = next(
        row
        for row in summary
        if row["variant"] == "full" and row["method"] == "learned_router"
    )
    assert full["fold_count"] == 2
    assert full["image_auroc_mean"] == pytest.approx(0.75)
    assert "Strict ablation and downstream Router" in markdown
    assert "Full BIR-AD Router against evaluator references" in markdown
    assert "Exact sign-flip p" in markdown
