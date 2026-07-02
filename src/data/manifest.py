"""Writers for split agent/evaluator manifests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


AGENT_INPUT_COLUMNS = ["image_id", "dataset", "category", "split", "image_path"]
EVALUATOR_COLUMNS = ["image_id", "label", "mask_path", "defect_type"]
FORBIDDEN_AGENT_COLUMNS = {"label", "mask_path", "defect_type"}


def write_manifest_outputs(
    *,
    agent_rows: list[dict[str, str]],
    evaluator_rows: list[dict[str, str | int]],
    audit: dict[str, Any],
    manifest_dir: str | Path,
    audit_path: str | Path,
    dataset: str = "mvtec",
) -> tuple[Path, Path, Path]:
    manifest_dir = Path(manifest_dir)
    audit_path = Path(audit_path)
    agent_path = manifest_dir / f"{dataset}_agent_input.csv"
    evaluator_path = manifest_dir / f"{dataset}_evaluator.csv"

    _validate_agent_rows(agent_rows)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    _write_csv(agent_path, AGENT_INPUT_COLUMNS, agent_rows)
    _write_csv(evaluator_path, EVALUATOR_COLUMNS, evaluator_rows)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    return agent_path, evaluator_path, audit_path


def _validate_agent_rows(rows: list[dict[str, str]]) -> None:
    for index, row in enumerate(rows):
        fields = set(row)
        forbidden = sorted(fields & FORBIDDEN_AGENT_COLUMNS)
        if forbidden:
            raise ValueError(f"Agent row {index} contains forbidden fields: {forbidden}")
        missing = [column for column in AGENT_INPUT_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"Agent row {index} is missing fields: {missing}")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
