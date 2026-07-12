"""Export Stage 3 agent-visible routing tasks from evaluator-only matrices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FORBIDDEN_AGENT_FIELDS = {
    "label",
    "mask_path",
    "defect_type",
    "anomaly_type",
    "oracle_best_expert",
}
KEY_COLUMNS = ("image_id", "dataset", "category", "support_set_id", "k_shot", "seed")
EXPERT_SCORE_COLUMNS = {
    "PatchCore": "patchcore_score",
    "WinCLIP": "winclip_score",
    "AnomalyDINO": "anomalydino_score",
}
REQUIRED_WIDE_COLUMNS = (*KEY_COLUMNS, *EXPERT_SCORE_COLUMNS.values())


class RoutingTaskExportError(ValueError):
    """Raised when Stage 3 routing tasks cannot be exported without leakage."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Stage 3 agent-visible routing tasks from routing_matrix_wide.csv."
    )
    parser.add_argument(
        "--routing-matrix-wide",
        default="outputs/stage3/evaluator_only/routing_matrix_wide.csv",
    )
    parser.add_argument("--output-dir", default="outputs/stage3/agent_visible")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        task_path, card_path = export_routing_tasks(
            routing_matrix_wide=args.routing_matrix_wide,
            output_dir=args.output_dir,
        )
    except RoutingTaskExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Wrote {task_path}")
    print(f"Wrote {card_path}")


def export_routing_tasks(
    *,
    routing_matrix_wide: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write agent-visible routing tasks and expert cards."""

    output_path = Path(output_dir)
    ensure_agent_visible_output(output_path)
    rows = read_routing_matrix_wide(routing_matrix_wide)
    tasks = [routing_task_from_wide_row(row) for row in rows]

    output_path.mkdir(parents=True, exist_ok=True)
    task_path = output_path / "agent_routing_tasks.jsonl"
    card_path = output_path / "expert_cards.json"
    write_jsonl(task_path, tasks)
    write_json(card_path, expert_cards())
    return task_path, card_path


def read_routing_matrix_wide(path: str | Path) -> list[dict[str, str]]:
    """Read a wide routing matrix and reject evaluator-only columns for export."""

    matrix_path = Path(path)
    with matrix_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in REQUIRED_WIDE_COLUMNS if column not in fieldnames]
        if missing:
            raise RoutingTaskExportError(f"{matrix_path} is missing columns: {missing}")
        forbidden = sorted(FORBIDDEN_AGENT_FIELDS.intersection(fieldnames))
        allowed_forbidden = {"label", "mask_path"}
        unexpected = [field for field in forbidden if field not in allowed_forbidden]
        if unexpected:
            raise RoutingTaskExportError(
                f"{matrix_path} contains fields that cannot be exported: {unexpected}"
            )

        rows: list[dict[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            clean = {key: (value or "").strip() for key, value in row.items()}
            missing_values = [column for column in REQUIRED_WIDE_COLUMNS if not clean.get(column)]
            if missing_values:
                raise RoutingTaskExportError(
                    f"{matrix_path}:{line_number} is missing required values: {missing_values}"
                )
            rows.append(clean)
    return rows


def routing_task_from_wide_row(row: dict[str, str]) -> dict[str, Any]:
    """Convert one evaluator-only wide row into an agent-visible routing task."""

    task = {
        "task_id": "|".join(row[column] for column in KEY_COLUMNS),
        "sample_id": row["image_id"],
        "dataset": row["dataset"],
        "category": row["category"],
        "support_set_id": row["support_set_id"],
        "k_shot": int(row["k_shot"]),
        "seed": int(row["seed"]),
        "expert_scores": {
            expert: _parse_float(row[column], column)
            for expert, column in EXPERT_SCORE_COLUMNS.items()
        },
    }
    validate_no_forbidden_agent_fields(task, "routing task")
    return task


def expert_cards() -> dict[str, dict[str, str]]:
    """Return stable agent-visible expert descriptions."""

    cards = {
        "PatchCore": {
            "method_description": (
                "Memory-bank nearest-neighbor anomaly detector using local patch features from "
                "normal support images."
            ),
            "input_requirements": (
                "Query image and K normal support images from the official train/good split."
            ),
            "compute_cost": "Medium to high: feature extraction plus nearest-neighbor search.",
        },
        "WinCLIP": {
            "method_description": (
                "CLIP-based zero/few-shot visual-language anomaly detector with prompt-driven "
                "normal and abnormal scoring."
            ),
            "input_requirements": (
                "Query image, target class name, and optional normal support images from the "
                "official train/good split."
            ),
            "compute_cost": "Medium: image/text encoding and prompt ensemble scoring.",
        },
        "AnomalyDINO": {
            "method_description": (
                "DINO feature based anomaly detector comparing query representations against "
                "normal support references."
            ),
            "input_requirements": (
                "Query image and K normal support images from the official train/good split."
            ),
            "compute_cost": "Medium: dense feature extraction and support comparison.",
        },
    }
    validate_no_forbidden_agent_fields(cards, "expert cards")
    return cards


def ensure_agent_visible_output(output_dir: Path) -> None:
    parts = {part.lower() for part in output_dir.parts}
    if "evaluator_only" in parts or "oracle" in parts:
        raise RoutingTaskExportError(
            "Agent-visible routing tasks cannot be written under evaluator_only or oracle paths."
        )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            validate_no_forbidden_agent_fields(row, str(path))
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    validate_no_forbidden_agent_fields(payload, str(path))
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def validate_no_forbidden_agent_fields(value: Any, context: str) -> None:
    """Reject forbidden Agent-visible keys anywhere in a JSON-like payload."""

    if isinstance(value, dict):
        forbidden = sorted(FORBIDDEN_AGENT_FIELDS.intersection(value))
        if forbidden:
            raise RoutingTaskExportError(f"{context} contains forbidden fields: {forbidden}")
        for child in value.values():
            validate_no_forbidden_agent_fields(child, context)
    elif isinstance(value, list):
        for child in value:
            validate_no_forbidden_agent_fields(child, context)


def _parse_float(value: str, column: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise RoutingTaskExportError(f"{column} has invalid numeric value: {value!r}") from exc


if __name__ == "__main__":
    main()
