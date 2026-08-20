"""Build category-cross-fitted soft teacher targets for one or all folds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from ..router.teacher import (
    TEACHER_PARQUET_NAME,
    build_and_write_teacher,
    ensure_evaluator_only_routing_matrix,
)
from .stage5_artifacts import (
    atomic_write_json,
    environment_record,
    file_sha256,
    git_commit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing-matrix", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument(
        "--fold",
        required=True,
        choices=("all", *(f"fold{index}" for index in range(5))),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=(0.25, 0.5, 1.0, 2.0, 4.0)
    )
    parser.add_argument(
        "--minimum-probabilities",
        type=float,
        nargs="+",
        default=(0.005, 0.01, 0.025),
        help=(
            "Per-expert probability floors calibrated jointly with temperature "
            "on validation categories."
        ),
    )
    parser.add_argument("--cost-weight", type=float, default=0.05)
    parser.add_argument("--failure-penalty", type=float, default=2.0)
    parser.add_argument(
        "--calibration-strategy",
        choices=("leave_one_train_category_out", "full_train"),
        default="leave_one_train_category_out",
    )
    parser.add_argument(
        "--repeat-weighting",
        choices=(
            "equal_category_equal_query_inverse_variant_frequency",
            "uniform_task",
        ),
        default="equal_category_equal_query_inverse_variant_frequency",
    )
    parser.add_argument(
        "--sharpness-strategy",
        choices=(
            "train_robust_gap_validation_oracle_nll",
            "legacy_raw_objective_soft_ce",
        ),
        default="train_robust_gap_validation_oracle_nll",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_dir)
    folds = tuple(f"fold{index}" for index in range(5)) if args.fold == "all" else (args.fold,)
    failures: list[dict[str, str]] = []
    outputs: dict[str, str] = {}
    metadata: dict[str, Any] = {}
    input_hashes: dict[str, str] = {}
    try:
        # Resolve and enforce the evaluator-only boundary before hashing this
        # or any other input.  The returned canonical path is also the only
        # routing-matrix path subsequently read by the builder.
        routing_matrix = ensure_evaluator_only_routing_matrix(args.routing_matrix)
        input_hashes = {
            "routing_matrix": file_sha256(routing_matrix),
            "fold_manifest": file_sha256(args.fold_manifest),
        }
        for fold in folds:
            output_dir = root / fold if args.fold == "all" else root
            teacher_path = output_dir / TEACHER_PARQUET_NAME
            artifact = build_and_write_teacher(
                routing_matrix_path=routing_matrix,
                fold_manifest_path=args.fold_manifest,
                fold=fold,
                output_path=teacher_path,
                temperatures=args.temperatures,
                minimum_probabilities=args.minimum_probabilities,
                cost_weight=args.cost_weight,
                failure_penalty=args.failure_penalty,
                calibration_strategy=args.calibration_strategy,
                repeat_weighting=args.repeat_weighting,
                sharpness_strategy=args.sharpness_strategy,
            )
            metadata_path = atomic_write_json(output_dir / "teacher_metadata.json", artifact.metadata())
            outputs[f"{fold}.teacher"] = str(teacher_path)
            outputs[f"{fold}.metadata"] = str(metadata_path)
            metadata[fold] = artifact.metadata()
    except Exception as exc:
        failures.append({"code": type(exc).__name__, "message": str(exc)})

    root.mkdir(parents=True, exist_ok=True)
    failures_path = atomic_write_json(root / "failures.json", failures)
    output_hashes = {
        name: file_sha256(path) for name, path in outputs.items() if Path(path).is_file()
    }
    output_hashes["failures"] = file_sha256(failures_path)
    run = {
        "protocol_version": "stage5.teacher_build_run.v1",
        "run_kind": "build_category_cross_fitted_soft_teacher",
        "ok": not failures,
        "config": {
            "routing_matrix": args.routing_matrix,
            "fold_manifest": args.fold_manifest,
            "fold": args.fold,
            "temperatures": list(args.temperatures),
            "minimum_probabilities": list(args.minimum_probabilities),
            "cost_weight": args.cost_weight,
            "failure_penalty": args.failure_penalty,
            "calibration_strategy": args.calibration_strategy,
            "repeat_weighting": args.repeat_weighting,
            "sharpness_strategy": args.sharpness_strategy,
            "seed": args.seed,
        },
        "seed": args.seed,
        "git_commit": git_commit(),
        "environment": environment_record(),
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "outputs": {**outputs, "failures": str(failures_path)},
        "predictions": [path for name, path in outputs.items() if name.endswith(".teacher")],
        "failures": failures,
        "fold_metadata": metadata,
    }
    atomic_write_json(root / "run.json", run)
    if failures:
        print(f"Teacher build failed: {failures[0]['message']}", file=sys.stderr)
        return 1
    print(f"Teacher build PASS: folds={','.join(folds)} output={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
