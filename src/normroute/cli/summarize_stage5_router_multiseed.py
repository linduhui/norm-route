"""Aggregate complete Stage 5 Router CV runs across deterministic seeds.

Each seed is first validated with the ordinary five-fold summarizer contract.
Descriptive metrics use all seed/fold blocks. Paired inference averages seeds
within a fold before the exact sign-flip test, so optimizer repeats are not
misrepresented as independent held-out categories.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Mapping, Sequence

from .stage5_artifacts import atomic_write_json, environment_record, file_sha256, git_commit
from .summarize_stage5_router import (
    DEFAULT_FOLDS,
    METRICS,
    load_cv_records,
    write_csv,
)


MULTISEED_SUMMARY_PROTOCOL_VERSION = "stage5.router_multiseed_summary.v1"
_ARTIFACT_CONFIG_FIELDS = frozenset(
    {
        "router_features",
        "fold_manifest",
        "routing_matrix",
        "teacher_data",
        "capability_bank",
    }
)
_RUN_IDENTITY_CONFIG_FIELDS = frozenset({"fold", "variant", "seed"})


class Stage5RouterMultiseedSummaryError(ValueError):
    """Raised when a multi-seed comparison is incomplete or inconsistent."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--folds", nargs="+", default=DEFAULT_FOLDS)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument(
        "--paired-comparison",
        nargs=3,
        action="append",
        required=True,
        metavar=("REFERENCE", "CANDIDATE", "NAME"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    outputs: dict[str, str | None] = {
        "per_seed_fold_metrics": None,
        "summary_metrics": None,
        "paired_deltas": None,
        "selected_policies": None,
        "route_changes": None,
    }
    input_hashes: dict[str, str] = {}
    prediction_paths: list[str] = []
    try:
        seeds = _unique_seeds(args.seeds)
        _verify_multiseed_provenance(
            Path(args.input_root),
            seeds=seeds,
            folds=tuple(args.folds),
            variants=tuple(args.variants),
        )
        records: list[dict[str, Any]] = []
        for seed in seeds:
            seed_root = Path(args.input_root) / f"seed{seed}"
            seed_records, hashes, predictions = load_cv_records(
                seed_root,
                folds=tuple(args.folds),
                variants=tuple(args.variants),
                experiment_family="teacher_ecpb",
            )
            records.extend({**row, "seed": seed} for row in seed_records)
            input_hashes.update(hashes)
            prediction_paths.extend(predictions)
        summaries = aggregate_multiseed_records(
            records,
            seeds=seeds,
            folds=tuple(args.folds),
            variants=tuple(args.variants),
        )
        paired = paired_multiseed_deltas(
            records,
            seeds=seeds,
            folds=tuple(args.folds),
            variants=tuple(args.variants),
            comparisons=tuple(tuple(item) for item in args.paired_comparison),
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        selected_policies, route_changes = load_mechanism_diagnostics(
            Path(args.input_root),
            seeds=seeds,
            folds=tuple(args.folds),
            variants=tuple(args.variants),
        )
        per_seed_fields = (
            "seed",
            "fold",
            "variant",
            "method",
            "feature_dimension",
            "num_samples",
            *METRICS,
        )
        summary_fields = (
            "variant",
            "method",
            "metric",
            "seed_count",
            "fold_count",
            "block_count",
            "mean",
            "std_across_seed_fold_blocks",
            "std_across_seed_means",
            "minimum",
            "maximum",
        )
        paired_fields = (
            "comparison",
            "reference_variant",
            "candidate_variant",
            "method",
            "metric",
            "seed_count",
            "fold_count",
            "pairing_unit",
            "mean_paired_delta",
            "std_paired_delta",
            "ci95_lower",
            "ci95_upper",
            "positive_fold_fraction",
            "two_sided_sign_flip_p_value",
        )
        outputs = {
            "per_seed_fold_metrics": str(
                write_csv(
                    output_dir / "per_seed_fold_metrics.csv",
                    sorted(
                        records,
                        key=lambda row: (
                            row["seed"], row["fold"], row["variant"], row["method"]
                        ),
                    ),
                    fieldnames=per_seed_fields,
                )
            ),
            "summary_metrics": str(
                write_csv(
                    output_dir / "multiseed_summary_metrics.csv",
                    summaries,
                    fieldnames=summary_fields,
                )
            ),
            "paired_deltas": str(
                write_csv(
                    output_dir / "multiseed_paired_deltas.csv",
                    paired,
                    fieldnames=paired_fields,
                )
            ),
            "selected_policies": str(
                write_csv(
                    output_dir / "selected_capability_policies.csv",
                    selected_policies,
                    fieldnames=(
                        "seed",
                        "fold",
                        "variant",
                        "policy_present",
                        "capability_weight",
                        "uncertainty_weight",
                        "cost_weight",
                        "validation_empirical_loss",
                        "validation_selection_accuracy",
                        "validation_oracle_regret",
                        "validation_runtime_tradeoff",
                        "uncertainty_mode",
                    ),
                )
            ),
            "route_changes": str(
                write_csv(
                    output_dir / "route_change_diagnostics.csv",
                    route_changes,
                    fieldnames=(
                        "seed",
                        "fold",
                        "reference_variant",
                        "ablation_variant",
                        "task_count",
                        "changed_count",
                        "route_change_rate",
                    ),
                )
            ),
        }
    except Exception as exc:
        failures.append({"code": type(exc).__name__, "message": str(exc)})

    failures_path = atomic_write_json(output_dir / "failures.json", failures)
    output_hashes = {
        name: file_sha256(path)
        for name, path in outputs.items()
        if path is not None and Path(path).is_file()
    }
    output_hashes["failures"] = file_sha256(failures_path)
    atomic_write_json(
        output_dir / "run.json",
        {
            "protocol_version": MULTISEED_SUMMARY_PROTOCOL_VERSION,
            "run_kind": "stage5_router_multiseed_summary",
            "ok": not failures,
            "config": {
                "input_root": args.input_root,
                "seeds": list(args.seeds),
                "folds": list(args.folds),
                "variants": list(args.variants),
                "paired_comparisons": list(args.paired_comparison),
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_seed": args.bootstrap_seed,
                "paired_inference_unit": "fold_after_seed_mean",
            },
            "seed": args.bootstrap_seed,
            "git_commit": git_commit(),
            "environment": environment_record(),
            "input_hashes": input_hashes,
            "output_hashes": output_hashes,
            "outputs": {**outputs, "failures": str(failures_path)},
            "predictions": prediction_paths,
            "failures": failures,
        },
    )
    if failures:
        print(f"Multi-seed Router summary failed: {failures[0]['message']}", file=sys.stderr)
        return 1
    print(f"Multi-seed Router summary PASS: output={output_dir}")
    return 0


def aggregate_multiseed_records(
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    folds: Sequence[str],
    variants: Sequence[str],
) -> list[dict[str, Any]]:
    _validate_record_grid(records, seeds=seeds, folds=folds, variants=variants)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["variant"]), str(row["method"]))].append(row)
    result: list[dict[str, Any]] = []
    for (variant, method), rows in sorted(grouped.items()):
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            seed_means = [
                statistics.fmean(
                    float(row[metric]) for row in rows if int(row["seed"]) == seed
                )
                for seed in seeds
            ]
            result.append(
                {
                    "variant": variant,
                    "method": method,
                    "metric": metric,
                    "seed_count": len(seeds),
                    "fold_count": len(folds),
                    "block_count": len(values),
                    "mean": statistics.fmean(values),
                    "std_across_seed_fold_blocks": statistics.stdev(values),
                    "std_across_seed_means": statistics.stdev(seed_means),
                    "minimum": min(values),
                    "maximum": max(values),
                }
            )
    return result


def load_mechanism_diagnostics(
    input_root: Path,
    *,
    seeds: Sequence[int],
    folds: Sequence[str],
    variants: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Expose selected weights and whether mechanism ablations change routes."""

    policies: list[dict[str, Any]] = []
    selections: dict[tuple[int, str, str], dict[str, str]] = {}
    for seed in seeds:
        for fold in folds:
            for variant in variants:
                run_dir = input_root / f"seed{seed}" / fold / variant
                metrics = json.loads(
                    (run_dir / "evaluator_only" / "metrics.json").read_text(
                        encoding="utf-8"
                    )
                )
                policy = metrics.get("capability_policy")
                if policy is not None and not isinstance(policy, Mapping):
                    raise Stage5RouterMultiseedSummaryError(
                        f"capability policy is invalid: {run_dir}"
                    )
                row: dict[str, Any] = {
                    "seed": seed,
                    "fold": fold,
                    "variant": variant,
                    "policy_present": policy is not None,
                    "capability_weight": "",
                    "uncertainty_weight": "",
                    "cost_weight": "",
                    "validation_empirical_loss": "",
                    "validation_selection_accuracy": "",
                    "validation_oracle_regret": "",
                    "validation_runtime_tradeoff": metrics.get(
                        "validation_runtime_tradeoff", ""
                    ),
                    "uncertainty_mode": metrics.get("uncertainty_mode", ""),
                }
                if policy is not None:
                    for name in (
                        "capability_weight",
                        "uncertainty_weight",
                        "cost_weight",
                        "validation_empirical_loss",
                        "validation_selection_accuracy",
                        "validation_oracle_regret",
                    ):
                        value = float(policy[name])
                        if not math.isfinite(value):
                            raise Stage5RouterMultiseedSummaryError(
                                f"capability policy {name} is non-finite: {run_dir}"
                            )
                        row[name] = value
                policies.append(row)

                selected: dict[str, str] = {}
                with (run_dir / "predictions.jsonl").open(
                    "r", encoding="utf-8"
                ) as handle:
                    for line_number, line in enumerate(handle, start=1):
                        prediction = json.loads(line)
                        task_id = str(prediction.get("task_id", ""))
                        expert = str(prediction.get("selected_expert", ""))
                        if not task_id or not expert or task_id in selected:
                            raise Stage5RouterMultiseedSummaryError(
                                f"invalid prediction at {run_dir}:{line_number}"
                            )
                        selected[task_id] = expert
                if not selected:
                    raise Stage5RouterMultiseedSummaryError(
                        f"empty predictions: {run_dir}"
                    )
                selections[(seed, fold, variant)] = selected

    changes: list[dict[str, Any]] = []
    for ablation in (
        "soft_no_capability",
        "soft_no_uncertainty",
        "soft_no_cost",
    ):
        if "soft_teacher_full" not in variants or ablation not in variants:
            continue
        for seed in seeds:
            for fold in folds:
                reference = selections[(seed, fold, "soft_teacher_full")]
                candidate = selections[(seed, fold, ablation)]
                if set(reference) != set(candidate):
                    raise Stage5RouterMultiseedSummaryError(
                        "route-change prediction task coverage disagrees"
                    )
                changed = sum(
                    reference[task_id] != candidate[task_id]
                    for task_id in reference
                )
                changes.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "reference_variant": "soft_teacher_full",
                        "ablation_variant": ablation,
                        "task_count": len(reference),
                        "changed_count": changed,
                        "route_change_rate": changed / len(reference),
                    }
                )
    return policies, changes


def paired_multiseed_deltas(
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    folds: Sequence[str],
    variants: Sequence[str],
    comparisons: Sequence[Sequence[str]],
    bootstrap_replicates: int = 20000,
    bootstrap_seed: int = 0,
) -> list[dict[str, Any]]:
    _validate_record_grid(records, seeds=seeds, folds=folds, variants=variants)
    if bootstrap_replicates <= 0:
        raise Stage5RouterMultiseedSummaryError("bootstrap_replicates must be positive")
    index = {
        (int(row["seed"]), str(row["fold"]), str(row["variant"]), str(row["method"])): row
        for row in records
    }
    result: list[dict[str, Any]] = []
    rng = random.Random(bootstrap_seed)
    for comparison in comparisons:
        if len(comparison) != 3:
            raise Stage5RouterMultiseedSummaryError(
                "paired comparisons require reference, candidate, and name"
            )
        reference, candidate, name = map(str, comparison)
        if reference not in variants or candidate not in variants:
            raise Stage5RouterMultiseedSummaryError(
                f"paired comparison {name!r} references an absent variant"
            )
        method = "learned_router"
        for metric in METRICS:
            fold_deltas = []
            for fold in folds:
                seed_deltas = [
                    float(index[(seed, fold, candidate, method)][metric])
                    - float(index[(seed, fold, reference, method)][metric])
                    for seed in seeds
                ]
                fold_deltas.append(statistics.fmean(seed_deltas))
            lower, upper = _bootstrap_mean_interval(
                fold_deltas, bootstrap_replicates, rng
            )
            result.append(
                {
                    "comparison": name,
                    "reference_variant": reference,
                    "candidate_variant": candidate,
                    "method": method,
                    "metric": metric,
                    "seed_count": len(seeds),
                    "fold_count": len(folds),
                    "pairing_unit": "fold_after_seed_mean",
                    "mean_paired_delta": statistics.fmean(fold_deltas),
                    "std_paired_delta": statistics.stdev(fold_deltas),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "positive_fold_fraction": sum(
                        value > 0.0 for value in fold_deltas
                    )
                    / len(fold_deltas),
                    "two_sided_sign_flip_p_value": _exact_sign_flip_p_value(
                        fold_deltas
                    ),
                }
            )
    return result


def _validate_record_grid(
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    folds: Sequence[str],
    variants: Sequence[str],
) -> None:
    expected_methods = {"learned_router", "global_best", "sample_oracle"}
    expected = {
        (seed, fold, variant, method)
        for seed in seeds
        for fold in folds
        for variant in variants
        for method in expected_methods
    }
    observed = {
        (int(row["seed"]), str(row["fold"]), str(row["variant"]), str(row["method"]))
        for row in records
    }
    if observed != expected or len(records) != len(expected):
        missing = sorted(expected - observed)[:5]
        extra = sorted(observed - expected)[:5]
        raise Stage5RouterMultiseedSummaryError(
            f"incomplete seed/fold/variant/method grid; missing={missing}, extra={extra}"
        )
    for row in records:
        for metric in METRICS:
            value = float(row[metric])
            if not math.isfinite(value):
                raise Stage5RouterMultiseedSummaryError(
                    f"non-finite {metric} in multi-seed record"
                )


def _verify_seed_provenance(
    seed_root: Path,
    seed: int,
    *,
    folds: Sequence[str],
    variants: Sequence[str],
) -> dict[str, str]:
    shared: dict[str, str] = {}
    for fold in folds:
        for variant in variants:
            path = seed_root / fold / variant / "run.json"
            try:
                run = json.loads(path.read_text(encoding="utf-8"))
                config = run["config"]
            except (OSError, KeyError, json.JSONDecodeError, TypeError) as exc:
                raise Stage5RouterMultiseedSummaryError(
                    f"could not read seed provenance: {path}"
                ) from exc
            if not isinstance(config, Mapping):
                raise Stage5RouterMultiseedSummaryError(
                    f"run config provenance is invalid: {path}"
                )
            if int(config.get("seed", -1)) != seed or int(run.get("seed", -1)) != seed:
                raise Stage5RouterMultiseedSummaryError(
                    f"seed provenance mismatch: {path}"
                )
            commit = run.get("git_commit")
            if not isinstance(commit, str) or not commit.strip():
                raise Stage5RouterMultiseedSummaryError(
                    f"run is missing git commit provenance: {path}"
                )
            _set_consistent_provenance(
                shared,
                "git_commit",
                commit,
                conflict="seed comparison mixes git commits",
            )
            if config.get("uncertainty_mode") != "entropy_scaled_lcb":
                raise Stage5RouterMultiseedSummaryError(
                    f"run does not use redesigned uncertainty: {path}"
                )
            if (
                not math.isclose(
                    float(config.get("validation_runtime_tradeoff", -1.0)),
                    0.05,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or config.get("validation_selection_metric")
                != "train_calibrated_zero_one_error_plus_normalized_runtime"
                or config.get("repeat_weighting")
                != "equal_category_equal_query_inverse_variant_frequency"
            ):
                raise Stage5RouterMultiseedSummaryError(
                    f"run does not use the frozen validation/weighting contract: {path}"
                )
            config_signature = json.dumps(
                {
                    str(name): value
                    for name, value in config.items()
                    if name not in _ARTIFACT_CONFIG_FIELDS
                    and name not in _RUN_IDENTITY_CONFIG_FIELDS
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            _set_consistent_provenance(
                shared,
                f"training_config:{variant}",
                config_signature,
                conflict=(
                    "seed comparison mixes non-seed training config for "
                    f"variant={variant}"
                ),
            )
            run_hashes = run.get("input_hashes")
            if not isinstance(run_hashes, Mapping):
                raise Stage5RouterMultiseedSummaryError(
                    f"run input hashes are invalid: {path}"
                )
            for name in ("fold_manifest", "routing_matrix"):
                value = str(run_hashes.get(name, ""))
                if not value:
                    raise Stage5RouterMultiseedSummaryError(
                        f"run is missing {name} hash: {path}"
                    )
                previous = shared.setdefault(name, value)
                if value != previous:
                    raise Stage5RouterMultiseedSummaryError(
                        f"seed comparison mixes {name} artifacts"
                    )
            router_features_hash = str(run_hashes.get("router_features", ""))
            if not router_features_hash:
                raise Stage5RouterMultiseedSummaryError(
                    f"run is missing router_features hash: {path}"
                )
            _set_consistent_provenance(
                shared,
                f"router_features:{fold}",
                router_features_hash,
                conflict=(
                    "seed comparison mixes router_features artifacts for "
                    f"fold={fold}"
                ),
            )
            for hash_name, config_name in (
                ("teacher_data", "teacher_data"),
                ("capability_bank", "capability_bank"),
            ):
                configured = bool(config.get(config_name))
                value = str(run_hashes.get(hash_name, ""))
                if configured != bool(value):
                    raise Stage5RouterMultiseedSummaryError(
                        f"{hash_name} path/hash presence mismatch: {path}"
                    )
                _set_consistent_provenance(
                    shared,
                    f"{hash_name}:{fold}:{variant}",
                    value if configured else "<absent>",
                    conflict=(
                        f"seed comparison mixes {hash_name} artifacts for "
                        f"fold={fold}, variant={variant}"
                    ),
                )
    return shared


def _verify_multiseed_provenance(
    input_root: Path,
    *,
    seeds: Sequence[int],
    folds: Sequence[str],
    variants: Sequence[str],
) -> dict[str, str]:
    """Require every non-seed training and input provenance value to match."""

    shared: dict[str, str] = {}
    for seed in seeds:
        observed = _verify_seed_provenance(
            input_root / f"seed{seed}",
            seed,
            folds=folds,
            variants=variants,
        )
        for name, value in observed.items():
            if name == "git_commit":
                conflict = "multi-seed comparison mixes git commits"
            elif name.startswith("training_config:"):
                conflict = (
                    "multi-seed comparison mixes non-seed training config for "
                    f"variant={name.split(':', 1)[1]}"
                )
            else:
                conflict = f"multi-seed comparison mixes {name} artifacts"
            _set_consistent_provenance(
                shared,
                name,
                value,
                conflict=conflict,
            )
    return shared


def _set_consistent_provenance(
    values: dict[str, str],
    name: str,
    value: str,
    *,
    conflict: str,
) -> None:
    previous = values.setdefault(name, value)
    if value != previous:
        raise Stage5RouterMultiseedSummaryError(conflict)


def _bootstrap_mean_interval(
    values: Sequence[float], replicates: int, rng: random.Random
) -> tuple[float, float]:
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(replicates)
    )
    lower_index = int(0.025 * (len(means) - 1))
    upper_index = int(0.975 * (len(means) - 1))
    return means[lower_index], means[upper_index]


def _exact_sign_flip_p_value(values: Sequence[float]) -> float:
    observed = abs(statistics.fmean(values))
    extreme = 0
    count = 1 << len(values)
    for mask in range(count):
        mean = statistics.fmean(
            value if mask & (1 << index) else -value
            for index, value in enumerate(values)
        )
        if abs(mean) >= observed - 1e-15:
            extreme += 1
    return extreme / count


def _unique_seeds(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or tuple(sorted(set(result))) != result or any(value < 0 for value in result):
        raise Stage5RouterMultiseedSummaryError(
            "seeds must be sorted, unique, and non-negative"
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
