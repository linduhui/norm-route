"""Run one Stage 4 policy fold/split against existing Stage 2 predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.normroute.agent.policy_registry import create_policy, list_policies
from src.normroute.agent.replay_executor import (
    ReplayExecutionError,
    execute_replay,
    read_fold_manifest,
)
from src.normroute.agent.task_builder import TaskBuildError, read_pre_route_tasks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one Stage 4 routing policy from immutable Stage 2 predictions."
    )
    parser.add_argument("--policy", default="always_anomalydino", choices=list_policies())
    parser.add_argument(
        "--tasks",
        "--pre-route-tasks",
        dest="tasks_path",
        default="outputs/stage4/tasks/pre_route_tasks.jsonl",
    )
    parser.add_argument(
        "--fold-manifest", default="outputs/stage4/splits/fold_manifest.csv"
    )
    parser.add_argument("--stage2-root", default="outputs/stage2")
    parser.add_argument("--output-dir", default="outputs/stage4/runs/always_anomalydino/fold0/test")
    parser.add_argument("--fold", default="fold0")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--tool-budget", "--budget", dest="tool_budget", type=int, default=1)
    parser.add_argument("--max-estimated-cost-ms", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        task_rows = read_pre_route_tasks(args.tasks_path)
        manifest_rows = read_fold_manifest(args.fold_manifest)
        result = execute_replay(
            tasks=task_rows,
            manifest_rows=manifest_rows,
            policy=create_policy(args.policy),
            stage2_root=args.stage2_root,
            output_dir=args.output_dir,
            fold=args.fold,
            split=args.split,
            tool_budget=args.tool_budget,
            max_estimated_cost_ms=args.max_estimated_cost_ms,
            tasks_path=args.tasks_path,
            fold_manifest_path=args.fold_manifest,
        )
    except (ReplayExecutionError, TaskBuildError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    for path in (
        result.route_decisions_path,
        result.selected_predictions_path,
        result.failures_path,
        result.run_metadata_path,
        result.budget_summary_path,
    ):
        print(f"Wrote {path}")
    if result.num_failures:
        print(
            f"ERROR: replay recorded {result.num_failures} failed task(s); see "
            f"{result.failures_path}",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
