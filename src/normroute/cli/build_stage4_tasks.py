"""CLI for converting Stage 3 Agent tasks into Stage 4 pre-route tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.normroute.agent.task_builder import TaskBuildError, build_pre_route_tasks


DEFAULT_SOURCE = "outputs/stage3/routing_matrix/agent_routing_tasks.jsonl"
DEFAULT_OUTPUT = "outputs/stage4/tasks/pre_route_tasks.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe Stage 4 pre-route tasks from Stage 3 Agent tasks."
    )
    parser.add_argument(
        "--input",
        "--agent-tasks",
        "--stage3-agent-tasks",
        dest="source_path",
        default=DEFAULT_SOURCE,
        help="Stage 3 agent_routing_tasks.jsonl path.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default=DEFAULT_OUTPUT,
        help="Stage 4 pre_route_tasks.jsonl path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        output_path = build_pre_route_tasks(args.source_path, args.output_path)
    except TaskBuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    task_count = _count_nonempty_lines(output_path)
    print(f"Wrote {task_count} pre-route tasks to {output_path}")


def _count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


if __name__ == "__main__":
    main()
