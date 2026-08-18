"""Build one fold-scoped Expert Capability Profile Bank from teacher targets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ..router.expert_bank import (
    CAPABILITY_BANK_NAME,
    build_expert_capability_bank,
    iter_feature_records_jsonl,
    read_stage2_runtime_summary,
    write_capability_bank,
)
from ..router.teacher import read_teacher_parquet
from .stage5_artifacts import atomic_write_json, environment_record, file_sha256, git_commit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-data", required=True)
    parser.add_argument("--router-features", required=True)
    parser.add_argument(
        "--stage2-summary",
        required=True,
        help=(
            "Stage 2 summary.csv used only to cross-check train-category "
            "per-task runtimes and failure counts."
        ),
    )
    parser.add_argument("--fold", required=True, choices=tuple(f"fold{i}" for i in range(5)))
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output")
    destination.add_argument("--output-dir")
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output) if args.output else Path(args.output_dir) / CAPABILITY_BANK_NAME
    run_dir = output_path.parent
    failures: list[dict[str, str]] = []
    outputs: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    try:
        input_hashes = {
            "teacher_data": file_sha256(args.teacher_data),
            "router_features": file_sha256(args.router_features),
            "stage2_summary": file_sha256(args.stage2_summary),
        }
        rows = read_teacher_parquet(args.teacher_data)
        categories = tuple(sorted({str(row["category"]) for row in rows}))
        runtime_summary_records = read_stage2_runtime_summary(
            args.stage2_summary,
            allowed_categories=set(categories),
        )
        bank = build_expert_capability_bank(
            rows,
            fold=args.fold,
            train_categories=categories,
            feature_records=iter_feature_records_jsonl(args.router_features),
            runtime_summary_records=runtime_summary_records,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.seed,
        )
        write_capability_bank(bank, output_path)
        outputs["capability_bank"] = str(output_path)
    except Exception as exc:
        failures.append({"code": type(exc).__name__, "message": str(exc)})
    run_dir.mkdir(parents=True, exist_ok=True)
    failures_path = atomic_write_json(run_dir / "failures.json", failures)
    output_hashes = {
        name: file_sha256(path) for name, path in outputs.items() if Path(path).is_file()
    }
    output_hashes["failures"] = file_sha256(failures_path)
    run = {
        "protocol_version": "stage5.capability_bank_build_run.v2",
        "run_kind": "build_expert_capability_profile_bank",
        "ok": not failures,
        "config": {
            "teacher_data": args.teacher_data,
            "router_features": args.router_features,
            "stage2_summary": args.stage2_summary,
            "fold": args.fold,
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
        },
        "seed": args.seed,
        "git_commit": git_commit(),
        "environment": environment_record(),
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "outputs": {**outputs, "failures": str(failures_path)},
        "predictions": outputs.get("capability_bank"),
        "failures": failures,
    }
    atomic_write_json(run_dir / "run.json", run)
    if failures:
        print(f"Capability bank build failed: {failures[0]['message']}", file=sys.stderr)
        return 1
    print(f"Capability bank build PASS: fold={args.fold} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
