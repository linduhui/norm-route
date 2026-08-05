"""Summarize strict five-fold Stage 5 Router experiments.

The summary requires every requested fold/variant artifact.  Missing or failed
runs are reported as failures instead of being silently omitted.  Paired
ablation deltas use the same fold on both sides and include an exact two-sided
sign-flip permutation test.
"""

from __future__ import annotations

import argparse
import csv
from itertools import product
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from .run_stage5_router import (
    STAGE5_ROUTER_METRICS_PROTOCOL_VERSION,
    file_sha256,
)


STAGE5_ROUTER_SUMMARY_PROTOCOL_VERSION = "stage5.router_summary.v1"
DEFAULT_FOLDS = tuple(f"fold{index}" for index in range(5))
DEFAULT_VARIANTS = (
    "normal_only",
    "sigma_l2",
    "plus_sobel",
    "plus_structural_boundary",
    "plus_directional_evidence",
    "plus_cross_modal_disagreement",
    "plus_support_consistency",
    "representations_only",
    "full",
)
METHODS = ("learned_router", "global_best", "sample_oracle")
METRICS = (
    "selection_accuracy",
    "normalized_utility_mean",
    "oracle_regret_mean",
    "image_auroc",
    "image_ap",
    "image_f1_max_evaluator_only",
    "average_runtime_ms",
)
PAIRED_COMPARISONS = (
    ("normal_only", "sigma_l2", "add_sigma_l2"),
    ("sigma_l2", "plus_sobel", "add_sobel"),
    ("plus_sobel", "plus_structural_boundary", "add_structural_boundary"),
    (
        "plus_structural_boundary",
        "plus_directional_evidence",
        "add_directional_evidence",
    ),
    (
        "plus_directional_evidence",
        "plus_cross_modal_disagreement",
        "add_cross_modal_disagreement",
    ),
    (
        "plus_cross_modal_disagreement",
        "plus_support_consistency",
        "add_support_consistency",
    ),
    ("plus_support_consistency", "full", "add_dual_representations"),
    ("normal_only", "representations_only", "representations_vs_normal"),
    ("normal_only", "full", "full_vs_normal"),
)


class Stage5RouterSummaryError(ValueError):
    """Raised when the Router cross-validation summary is incomplete."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        default="outputs/stage5/router_cv_strict_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/stage5/router_cv_strict_v1/summary",
    )
    parser.add_argument("--folds", nargs="+", default=DEFAULT_FOLDS)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument(
        "--paired-comparison",
        nargs=3,
        action="append",
        metavar=("REFERENCE", "CANDIDATE", "NAME"),
        help="Repeatable paired comparison; defaults to the frozen BIR chain.",
    )
    parser.add_argument(
        "--reference-variant",
        default="full",
        help="Variant shown against evaluator reference methods in the table.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    outputs: dict[str, str | None] = {
        "per_fold_metrics": None,
        "summary_metrics": None,
        "paired_deltas": None,
        "paper_tables": None,
    }
    input_hashes: dict[str, str] = {}
    prediction_paths: list[str] = []
    try:
        records, input_hashes, prediction_paths = load_cv_records(
            args.input_root,
            folds=tuple(args.folds),
            variants=tuple(args.variants),
        )
        summary = aggregate_records(records)
        paired = paired_ablation_deltas(
            records,
            folds=tuple(args.folds),
            variants=tuple(args.variants),
            comparisons=(
                tuple(tuple(item) for item in args.paired_comparison)
                if args.paired_comparison
                else PAIRED_COMPARISONS
            ),
        )
        outputs = {
            "per_fold_metrics": str(
                write_csv(
                    output_dir / "per_fold_metrics.csv",
                    records,
                    fieldnames=_per_fold_columns(),
                )
            ),
            "summary_metrics": str(
                write_csv(
                    output_dir / "summary_metrics.csv",
                    summary,
                    fieldnames=_summary_columns(),
                )
            ),
            "paired_deltas": str(
                write_csv(
                    output_dir / "paired_ablation_deltas.csv",
                    paired,
                    fieldnames=_paired_columns(),
                )
            ),
            "paper_tables": str(
                atomic_write_text(
                    output_dir / "router_cv_tables.md",
                    render_paper_tables(
                        summary,
                        paired,
                        reference_variant=args.reference_variant,
                    ),
                )
            ),
        }
    except Exception as exc:
        failures.append({"code": type(exc).__name__, "message": str(exc)})

    failures_path = atomic_write_json(output_dir / "failures.json", failures)
    run = {
        "protocol_version": STAGE5_ROUTER_SUMMARY_PROTOCOL_VERSION,
        "run_kind": "stage5_router_cv_summary",
        "ok": not failures,
        "config": {
            "input_root": args.input_root,
            "output_dir": args.output_dir,
            "folds": list(args.folds),
            "variants": list(args.variants),
            "paired_comparisons": (
                args.paired_comparison
                if args.paired_comparison
                else [list(item) for item in PAIRED_COMPARISONS]
            ),
            "reference_variant": args.reference_variant,
        },
        "input_hashes": input_hashes,
        "seed": None,
        "git_commit": git_commit(),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "predictions": prediction_paths,
        "outputs": {**outputs, "failures": str(failures_path)},
        "failures": failures,
    }
    atomic_write_json(output_dir / "run.json", run)
    if failures:
        print(
            f"Stage 5 Router summary failed: {failures[0]['message']}",
            file=sys.stderr,
        )
        return 1
    print(
        "Stage 5 Router summary PASS: "
        f"table={outputs['paper_tables']}"
    )
    return 0


def load_cv_records(
    input_root: str | Path,
    *,
    folds: Sequence[str],
    variants: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, str], list[str]]:
    root = Path(input_root)
    if not folds or len(set(folds)) != len(folds):
        raise Stage5RouterSummaryError("folds must be non-empty and unique")
    if not variants or len(set(variants)) != len(variants):
        raise Stage5RouterSummaryError("variants must be non-empty and unique")
    records: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    predictions: list[str] = []
    comparison_hashes: dict[str, str] = {}
    comparison_settings: dict[str, str] = {}
    comparison_experts: tuple[str, ...] | None = None
    for fold in folds:
        for variant in variants:
            run_dir = root / fold / variant
            run_path = run_dir / "run.json"
            failures_path = run_dir / "failures.json"
            predictions_path = run_dir / "predictions.jsonl"
            metrics_path = run_dir / "evaluator_only" / "metrics.json"
            for path in (
                run_path,
                failures_path,
                predictions_path,
                metrics_path,
            ):
                if not path.is_file():
                    raise Stage5RouterSummaryError(
                        f"required Router artifact does not exist: {path}"
                    )
                hashes[str(path)] = file_sha256(path)
            run = _read_json(run_path)
            failures = _read_json(failures_path)
            metrics = _read_json(metrics_path)
            if run.get("ok") is not True or failures != []:
                raise Stage5RouterSummaryError(
                    f"Router run is failed or incomplete: {run_dir}"
                )
            config = run.get("config")
            if not isinstance(config, Mapping):
                raise Stage5RouterSummaryError(
                    f"Router run has invalid config: {run_path}"
                )
            if config.get("fold") != fold or config.get("variant") != variant:
                raise Stage5RouterSummaryError(
                    f"Router run config provenance mismatch: {run_path}"
                )
            expected_view = "normal_only" if variant == "normal_only" else "all"
            if config.get("feature_view") != expected_view:
                raise Stage5RouterSummaryError(
                    f"Router run has invalid feature view: {run_path}"
                )
            for name in (
                "epochs",
                "batch_size",
                "learning_rate",
                "l2_grid",
                "seed",
                "device",
                "model_kind",
                "standardization",
                "supervision_target",
                "test_prediction_contract",
            ):
                value = json.dumps(
                    config.get(name),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                prior = comparison_settings.setdefault(name, value)
                if value != prior:
                    raise Stage5RouterSummaryError(
                        f"Router comparison mixes training setting {name}"
                    )
            run_hashes = run.get("input_hashes")
            if not isinstance(run_hashes, Mapping):
                raise Stage5RouterSummaryError(
                    f"Router run has invalid input hashes: {run_path}"
                )
            for name in ("fold_manifest", "routing_matrix"):
                value = str(run_hashes.get(name, ""))
                if not value:
                    raise Stage5RouterSummaryError(
                        f"Router run is missing {name} hash: {run_path}"
                    )
                prior = comparison_hashes.setdefault(name, value)
                if value != prior:
                    raise Stage5RouterSummaryError(
                        f"Router comparison mixes {name} artifacts"
                    )
            if (
                metrics.get("protocol_version")
                != STAGE5_ROUTER_METRICS_PROTOCOL_VERSION
                or metrics.get("fold") != fold
                or metrics.get("variant") != variant
            ):
                raise Stage5RouterSummaryError(
                    f"Router metrics provenance mismatch: {metrics_path}"
                )
            experts = tuple(str(item) for item in metrics.get("experts", ()))
            if len(experts) < 2 or len(set(experts)) != len(experts):
                raise Stage5RouterSummaryError(
                    f"Router metrics expert set is invalid: {metrics_path}"
                )
            if comparison_experts is None:
                comparison_experts = experts
            elif experts != comparison_experts:
                raise Stage5RouterSummaryError(
                    "Router comparison mixes expert sets"
                )
            dimension = _positive_int(
                metrics.get("feature_dimension"),
                f"{metrics_path}:feature_dimension",
            )
            methods = metrics.get("methods")
            if not isinstance(methods, Mapping):
                raise Stage5RouterSummaryError(
                    f"Router metrics methods are invalid: {metrics_path}"
                )
            for method in METHODS:
                values = methods.get(method)
                if not isinstance(values, Mapping):
                    raise Stage5RouterSummaryError(
                        f"Router metrics missing method {method}: {metrics_path}"
                    )
                record: dict[str, Any] = {
                    "fold": fold,
                    "variant": variant,
                    "method": method,
                    "feature_dimension": dimension,
                    "num_samples": _positive_int(
                        values.get("num_samples"),
                        f"{metrics_path}:{method}:num_samples",
                    ),
                }
                for metric in METRICS:
                    value = values.get(metric)
                    record[metric] = (
                        None
                        if metric == "average_runtime_ms" and value is None
                        else _finite_float(
                            value, f"{metrics_path}:{method}:{metric}"
                        )
                    )
                records.append(record)
            predictions.append(str(predictions_path))
    return records, hashes, predictions


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in records:
        grouped.setdefault(
            (str(row["variant"]), str(row["method"])), []
        ).append(row)
    output: list[dict[str, Any]] = []
    for (variant, method), rows in grouped.items():
        dimensions = {int(row["feature_dimension"]) for row in rows}
        if len(dimensions) != 1:
            raise Stage5RouterSummaryError(
                f"feature dimension drifted across folds for {variant}/{method}"
            )
        item: dict[str, Any] = {
            "variant": variant,
            "method": method,
            "fold_count": len(rows),
            "feature_dimension": next(iter(dimensions)),
        }
        for metric in METRICS:
            values = [row[metric] for row in rows]
            if any(value is None for value in values):
                if not all(value is None for value in values):
                    raise Stage5RouterSummaryError(
                        f"partial runtime coverage for {variant}/{method}"
                    )
                item[f"{metric}_mean"] = None
                item[f"{metric}_std"] = None
            else:
                numeric = [float(value) for value in values]
                item[f"{metric}_mean"] = statistics.fmean(numeric)
                item[f"{metric}_std"] = (
                    statistics.stdev(numeric) if len(numeric) > 1 else 0.0
                )
        output.append(item)
    return sorted(
        output,
        key=lambda row: (
            DEFAULT_VARIANTS.index(row["variant"])
            if row["variant"] in DEFAULT_VARIANTS
            else len(DEFAULT_VARIANTS),
            METHODS.index(row["method"]),
        ),
    )


def paired_ablation_deltas(
    records: Sequence[Mapping[str, Any]],
    *,
    folds: Sequence[str],
    variants: Sequence[str],
    comparisons: Sequence[tuple[str, str, str]] = PAIRED_COMPARISONS,
) -> list[dict[str, Any]]:
    by_key = {
        (str(row["fold"]), str(row["variant"]), str(row["method"])): row
        for row in records
    }
    output: list[dict[str, Any]] = []
    requested = set(variants)
    for reference, candidate, comparison in comparisons:
        if reference not in requested or candidate not in requested:
            continue
        for metric in METRICS:
            differences: list[float] = []
            unavailable = False
            for fold in folds:
                reference_value = by_key[
                    (fold, reference, "learned_router")
                ][metric]
                candidate_value = by_key[
                    (fold, candidate, "learned_router")
                ][metric]
                if reference_value is None or candidate_value is None:
                    unavailable = True
                    break
                differences.append(
                    float(candidate_value) - float(reference_value)
                )
            output.append(
                {
                    "comparison": comparison,
                    "reference_variant": reference,
                    "candidate_variant": candidate,
                    "metric": metric,
                    "fold_count": len(folds),
                    "mean_paired_delta": (
                        None if unavailable else statistics.fmean(differences)
                    ),
                    "std_paired_delta": (
                        None
                        if unavailable
                        else (
                            statistics.stdev(differences)
                            if len(differences) > 1
                            else 0.0
                        )
                    ),
                    "positive_fold_count": (
                        None
                        if unavailable
                        else sum(value > 0.0 for value in differences)
                    ),
                    "negative_fold_count": (
                        None
                        if unavailable
                        else sum(value < 0.0 for value in differences)
                    ),
                    "exact_sign_flip_p_two_sided": (
                        None
                        if unavailable
                        else exact_sign_flip_pvalue(differences)
                    ),
                }
            )
    return output


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    if not differences or any(
        not math.isfinite(float(value)) for value in differences
    ):
        raise Stage5RouterSummaryError(
            "paired differences must be non-empty and finite"
        )
    observed = abs(statistics.fmean(float(value) for value in differences))
    tolerance = 1e-15
    extreme = 0
    total = 0
    for signs in product((-1.0, 1.0), repeat=len(differences)):
        permuted = abs(
            statistics.fmean(
                sign * float(value)
                for sign, value in zip(signs, differences)
            )
        )
        extreme += int(permuted + tolerance >= observed)
        total += 1
    return extreme / total


def render_paper_tables(
    summary: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    *,
    reference_variant: str = "full",
) -> str:
    learned = [row for row in summary if row["method"] == "learned_router"]
    reference_methods = [
        row for row in summary if row["variant"] == reference_variant
    ]
    if not reference_methods:
        raise Stage5RouterSummaryError(
            f"reference variant {reference_variant!r} is absent from summary"
        )
    lines = [
        "# Stage 5 Router five-fold results",
        "",
        "Values are fold mean +/- sample standard deviation. F1-max is evaluator-only.",
        "",
        "## Strict ablation and downstream Router",
        "",
        "| Variant | Dim. | Selection acc. (higher) | Normalized utility (higher) | Regret (lower) | Image AUROC (higher) | Image AP (higher) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in learned:
        lines.append(
            "| {variant} | {dimension} | {selection} | {utility} | "
            "{regret} | {auroc} | {ap} |".format(
                variant=row["variant"],
                dimension=row["feature_dimension"],
                selection=_mean_std(row, "selection_accuracy"),
                utility=_mean_std(row, "normalized_utility_mean"),
                regret=_mean_std(row, "oracle_regret_mean"),
                auroc=_mean_std(row, "image_auroc"),
                ap=_mean_std(row, "image_ap"),
            )
        )
    lines.extend(
        [
            "",
            f"## {reference_variant} Router against evaluator references",
            "",
            "| Method | Selection acc. (higher) | Normalized utility (higher) | Regret (lower) | Image AUROC (higher) | Image AP (higher) | Runtime ms (lower) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in reference_methods:
        lines.append(
            "| {method} | {selection} | {utility} | {regret} | "
            "{auroc} | {ap} | {runtime} |".format(
                method=row["method"],
                selection=_mean_std(row, "selection_accuracy"),
                utility=_mean_std(row, "normalized_utility_mean"),
                regret=_mean_std(row, "oracle_regret_mean"),
                auroc=_mean_std(row, "image_auroc"),
                ap=_mean_std(row, "image_ap"),
                runtime=_mean_std(row, "average_runtime_ms"),
            )
        )
    primary_pairs = [
        row for row in paired if row["metric"] == "image_auroc"
    ]
    lines.extend(
        [
            "",
            "## Paired fold ablation deltas (Image AUROC)",
            "",
            "| Added component | Candidate - reference | Positive folds | Exact sign-flip p |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in primary_pairs:
        lines.append(
            "| {comparison} | {delta} | {positive}/{folds} | {pvalue} |".format(
                comparison=row["comparison"],
                delta=_format_optional(row["mean_paired_delta"]),
                positive=row["positive_fold_count"],
                folds=row["fold_count"],
                pvalue=_format_optional(
                    row["exact_sign_flip_p_two_sided"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "A positive paired delta means the candidate has higher AUROC on the same fold. "
            "With five folds, the exact two-sided sign-flip test is descriptive and has coarse resolution; "
            "effect size and fold consistency should be reported together with p-values.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        name: _csv_value(row.get(name))
                        for name in fieldnames
                    }
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def atomic_write_json(path: Path, value: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )


def atomic_write_text(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage5RouterSummaryError(
            f"could not read Router artifact {path}: {exc}"
        ) from exc


def _per_fold_columns() -> tuple[str, ...]:
    return (
        "fold",
        "variant",
        "method",
        "feature_dimension",
        "num_samples",
        *METRICS,
    )


def _summary_columns() -> tuple[str, ...]:
    columns = ["variant", "method", "fold_count", "feature_dimension"]
    for metric in METRICS:
        columns.extend((f"{metric}_mean", f"{metric}_std"))
    return tuple(columns)


def _paired_columns() -> tuple[str, ...]:
    return (
        "comparison",
        "reference_variant",
        "candidate_variant",
        "metric",
        "fold_count",
        "mean_paired_delta",
        "std_paired_delta",
        "positive_fold_count",
        "negative_fold_count",
        "exact_sign_flip_p_two_sided",
    )


def _mean_std(row: Mapping[str, Any], metric: str) -> str:
    mean = row[f"{metric}_mean"]
    std = row[f"{metric}_std"]
    if mean is None or std is None:
        return "NA"
    return f"{float(mean):.4f} ± {float(std):.4f}"


def _format_optional(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


def _finite_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise Stage5RouterSummaryError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise Stage5RouterSummaryError(f"{name} must be finite")
    return parsed


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise Stage5RouterSummaryError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise Stage5RouterSummaryError(
            f"{name} must be a positive integer"
        ) from exc
    if parsed <= 0 or parsed != value:
        raise Stage5RouterSummaryError(f"{name} must be a positive integer")
    return parsed


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _temporary_path(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


__all__ = [
    "DEFAULT_FOLDS",
    "DEFAULT_VARIANTS",
    "METRICS",
    "STAGE5_ROUTER_SUMMARY_PROTOCOL_VERSION",
    "Stage5RouterSummaryError",
    "aggregate_records",
    "exact_sign_flip_pvalue",
    "load_cv_records",
    "main",
    "paired_ablation_deltas",
    "parse_args",
    "render_paper_tables",
]
