"""Validate Stage 3 evaluator-only and agent-visible artifacts."""

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

from src.normroute.cli.export_routing_tasks import FORBIDDEN_AGENT_FIELDS


EXPECTED_EXPERTS = ("patchcore", "winclip", "anomalydino")
WIDE_SCORE_COLUMNS = ("patchcore_score", "winclip_score", "anomalydino_score")
DEFAULT_AGENT_VISIBLE_FILES = ("agent_routing_tasks.jsonl", "expert_cards.json")
DEFAULT_ORACLE_FILES = ("oracle_summary.csv", "oracle_selection_counts.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 3 outputs.")
    parser.add_argument("--stage3-root", default="outputs/stage3")
    parser.add_argument(
        "--evaluator-only-dir",
        default="outputs/stage3/evaluator_only",
    )
    parser.add_argument(
        "--agent-visible-dir",
        default="outputs/stage3/agent_visible",
    )
    parser.add_argument("--oracle-dir", default="outputs/stage3/oracle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = validate_stage3_outputs(
        stage3_root=Path(args.stage3_root),
        evaluator_only_dir=Path(args.evaluator_only_dir),
        agent_visible_dir=Path(args.agent_visible_dir),
        oracle_dir=Path(args.oracle_dir),
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Validated Stage 3 outputs under {args.stage3_root}")


def validate_stage3_outputs(
    *,
    stage3_root: Path,
    evaluator_only_dir: Path,
    agent_visible_dir: Path,
    oracle_dir: Path,
) -> list[str]:
    """Return all Stage 3 validation errors."""

    errors: list[str] = []
    errors.extend(validate_evaluator_only_files(evaluator_only_dir))
    errors.extend(validate_agent_visible_files(agent_visible_dir))
    errors.extend(validate_oracle_files(oracle_dir))
    errors.extend(validate_routing_matrix_alignment(evaluator_only_dir))
    if stage3_root.exists():
        errors.extend(validate_no_agent_visible_leakage_outside_evaluator_only(stage3_root))
    return errors


def validate_evaluator_only_files(evaluator_only_dir: Path) -> list[str]:
    errors: list[str] = []
    required = ("routing_matrix_long.csv", "routing_matrix_wide.csv")
    for file_name in required:
        path = evaluator_only_dir / file_name
        if not path.is_file():
            errors.append(f"Missing evaluator_only file: {path}")
        elif "evaluator_only" not in {part.lower() for part in path.parts}:
            errors.append(f"Evaluator-only file is not under evaluator_only path: {path}")
    return errors


def validate_agent_visible_files(agent_visible_dir: Path) -> list[str]:
    errors: list[str] = []
    for file_name in DEFAULT_AGENT_VISIBLE_FILES:
        path = agent_visible_dir / file_name
        if not path.is_file():
            errors.append(f"Missing agent-visible file: {path}")
            continue
        errors.extend(validate_json_no_forbidden_fields(path))
    return errors


def validate_oracle_files(oracle_dir: Path) -> list[str]:
    errors: list[str] = []
    for file_name in DEFAULT_ORACLE_FILES:
        path = oracle_dir / file_name
        if not path.is_file():
            errors.append(f"Missing oracle file: {path}")
            continue
        errors.extend(validate_csv_evaluator_only_flag(path))
    return errors


def validate_routing_matrix_alignment(evaluator_only_dir: Path) -> list[str]:
    long_path = evaluator_only_dir / "routing_matrix_long.csv"
    wide_path = evaluator_only_dir / "routing_matrix_wide.csv"
    if long_path.is_file():
        return validate_long_matrix_expert_alignment(long_path)
    if wide_path.is_file():
        return validate_wide_matrix_expert_alignment(wide_path)
    return []


def validate_long_matrix_expert_alignment(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            missing = [column for column in ("image_id", "expert_name") if column not in fieldnames]
            if missing:
                return [f"{path} is missing columns needed for expert alignment: {missing}"]
            sample_ids_by_expert: dict[str, set[str]] = {expert: set() for expert in EXPECTED_EXPERTS}
            for row in reader:
                expert = normalize_expert_name(row.get("expert_name", ""))
                sample_id = (row.get("sample_id") or row.get("image_id") or "").strip()
                if expert in sample_ids_by_expert:
                    sample_ids_by_expert[expert].add(sample_id)
    except Exception as exc:
        return [f"{path} could not be read as routing matrix: {exc}"]

    missing_experts = [expert for expert, sample_ids in sample_ids_by_expert.items() if not sample_ids]
    if missing_experts:
        errors.append(f"{path} is missing expert rows for: {missing_experts}")
        return errors

    reference = sample_ids_by_expert[EXPECTED_EXPERTS[0]]
    for expert in EXPECTED_EXPERTS[1:]:
        if sample_ids_by_expert[expert] != reference:
            errors.append(
                f"{path} sample_id/image_id values are not aligned between "
                f"{EXPECTED_EXPERTS[0]} and {expert}"
            )
    return errors


def validate_wide_matrix_expert_alignment(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            missing = [column for column in ("image_id", *WIDE_SCORE_COLUMNS) if column not in fieldnames]
            if missing:
                return [f"{path} is missing columns needed for expert alignment: {missing}"]
            for line_number, row in enumerate(reader, start=2):
                missing_scores = [column for column in WIDE_SCORE_COLUMNS if not (row.get(column) or "").strip()]
                if missing_scores:
                    errors.append(
                        f"{path}:{line_number} has unaligned/missing expert scores: {missing_scores}"
                    )
    except Exception as exc:
        return [f"{path} could not be read as routing matrix: {exc}"]
    return errors


def validate_no_agent_visible_leakage_outside_evaluator_only(stage3_root: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted(stage3_root.rglob("*")):
        if not path.is_file():
            continue
        parts = {part.lower() for part in path.parts}
        if "evaluator_only" in parts or "oracle" in parts:
            continue
        if path.suffix.lower() in {".json", ".jsonl"}:
            errors.extend(validate_json_no_forbidden_fields(path))
        elif path.suffix.lower() == ".csv":
            errors.extend(validate_csv_no_forbidden_columns(path))
    return errors


def validate_json_no_forbidden_fields(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    errors.extend(
                        _find_forbidden_json_fields(
                            json.loads(line),
                            context=f"{path}:{line_number}",
                        )
                    )
        else:
            errors.extend(
                _find_forbidden_json_fields(
                    json.loads(path.read_text(encoding="utf-8")),
                    context=str(path),
                )
            )
    except Exception as exc:
        errors.append(f"{path} could not be read as JSON/JSONL: {exc}")
    return errors


def validate_csv_no_forbidden_columns(path: Path) -> list[str]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            fieldnames = csv.DictReader(handle).fieldnames or []
    except Exception as exc:
        return [f"{path} could not be read as CSV: {exc}"]
    forbidden = sorted(FORBIDDEN_AGENT_FIELDS.intersection(fieldnames))
    if forbidden:
        return [f"{path} contains forbidden agent-visible columns: {forbidden}"]
    return []


def validate_csv_evaluator_only_flag(path: Path) -> list[str]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if "evaluator_only" not in fieldnames:
                return [f"{path} is missing evaluator_only column"]
            bad_rows = [
                line_number
                for line_number, row in enumerate(reader, start=2)
                if (row.get("evaluator_only") or "").strip().lower() != "true"
            ]
    except Exception as exc:
        return [f"{path} could not be read as oracle CSV: {exc}"]
    if bad_rows:
        return [f"{path} has oracle rows not marked evaluator_only=True: {bad_rows}"]
    return []


def _find_forbidden_json_fields(value: Any, *, context: str) -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        forbidden = sorted(FORBIDDEN_AGENT_FIELDS.intersection(value))
        if forbidden:
            errors.append(f"{context} contains forbidden agent-visible fields: {forbidden}")
        for child in value.values():
            errors.extend(_find_forbidden_json_fields(child, context=context))
    elif isinstance(value, list):
        for child in value:
            errors.extend(_find_forbidden_json_fields(child, context=context))
    return errors


def normalize_expert_name(expert_name: str) -> str:
    return "".join(char for char in expert_name.lower() if char.isalnum())


if __name__ == "__main__":
    main()
