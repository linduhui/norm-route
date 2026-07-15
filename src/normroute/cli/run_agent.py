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
from src.normroute.agent.protocol import AgentTask
from src.normroute.cli.run_stage4_grid import build_training_runtime_records


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
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Policy RNG seed; used by random_seeded and saved in policy state.",
    )
    parser.add_argument(
        "--expert-cost-card",
        help="Predeclared JSON runtime cost card for fastest_expert.",
    )
    parser.add_argument(
        "--policy-artifact",
        help="Final fold-specific policy_artifact.json for calibrated policies.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        task_rows = read_pre_route_tasks(args.tasks_path)
        manifest_rows = read_fold_manifest(args.fold_manifest)
        policy_kwargs = {}
        if args.policy == "random_seeded":
            policy_kwargs["seed"] = args.seed
        if args.policy == "fastest_expert" and args.expert_cost_card:
            policy_kwargs["cost_card"] = args.expert_cost_card
        if args.policy in {"category_prior", "category_shot_prior", "cost_aware"}:
            if not args.policy_artifact:
                raise ValueError(
                    f"Policy {args.policy!r} requires --policy-artifact"
                )
            policy_kwargs["artifact"] = args.policy_artifact
        policy = create_policy(args.policy, **policy_kwargs)
        train_records = None
        if args.policy == "fastest_expert" and not args.expert_cost_card:
            train_records = build_training_runtime_records(
                tasks=[AgentTask.from_mapping(row) for row in task_rows],
                manifest_rows=manifest_rows,
                stage2_root=args.stage2_root,
                fold=args.fold,
            )
        result = execute_replay(
            tasks=task_rows,
            manifest_rows=manifest_rows,
            policy=policy,
            stage2_root=args.stage2_root,
            output_dir=args.output_dir,
            fold=args.fold,
            split=args.split,
            tool_budget=args.tool_budget,
            max_estimated_cost_ms=args.max_estimated_cost_ms,
            train_records=train_records,
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
