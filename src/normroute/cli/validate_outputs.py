"""Validate Stage 2 output artifacts for schema and no-leakage invariants."""

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

from ..evaluation.export import PREDICTION_COLUMNS


RESULT_FILES = ("predictions.csv", "metrics.json", "failures.json")
FORBIDDEN_PREDICTION_COLUMNS = {"label", "mask_path", "defect_type", "anomaly_type"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 2 output artifacts.")
    parser.add_argument(
        "output_root",
        nargs="?",
        default="outputs/stage2",
        help="Stage 2 output root or one run output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = validate_output_tree(Path(args.output_root))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Validated Stage 2 outputs under {args.output_root}")


def validate_output_tree(output_root: Path) -> list[str]:
    """Return validation errors for every Stage 2 run directory under output_root."""

    errors: list[str] = []
    if not output_root.exists():
        return [f"Output path does not exist: {output_root}"]

    run_dirs = _discover_run_dirs(output_root)
    if not run_dirs:
        return [f"No Stage 2 output artifacts found under {output_root}"]

    support_ids_by_combo: dict[tuple[str, str, str, str], dict[str, set[str]]] = {}
    for run_dir in run_dirs:
        errors.extend(_validate_required_files(run_dir))
        predictions_path = run_dir / "predictions.csv"
        metrics_path = run_dir / "metrics.json"
        failures_path = run_dir / "failures.json"
        if not predictions_path.exists():
            continue

        rows = _read_predictions(predictions_path, errors)
        if rows is not None:
            _collect_support_set_ids(
                rows=rows,
                path=predictions_path,
                support_ids_by_combo=support_ids_by_combo,
                errors=errors,
            )

        if metrics_path.exists() and failures_path.exists():
            errors.extend(_validate_failure_count(metrics_path, failures_path))

    errors.extend(_validate_support_set_consistency(support_ids_by_combo))
    return errors


def _discover_run_dirs(output_root: Path) -> list[Path]:
    if any((output_root / file_name).exists() for file_name in RESULT_FILES):
        return [output_root]
    return sorted(
        {
            path.parent
            for file_name in RESULT_FILES
            for path in output_root.rglob(file_name)
        }
    )


def _validate_required_files(run_dir: Path) -> list[str]:
    return [
        f"{run_dir} is missing required artifact {file_name}"
        for file_name in RESULT_FILES
        if not (run_dir / file_name).is_file()
    ]


def _read_predictions(path: Path, errors: list[str]) -> list[dict[str, str]] | None:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            missing = [column for column in PREDICTION_COLUMNS if column not in fieldnames]
            if missing:
                errors.append(f"{path} is missing prediction columns: {missing}")
            extra_forbidden = sorted(FORBIDDEN_PREDICTION_COLUMNS.intersection(fieldnames))
            if extra_forbidden:
                errors.append(f"{path} contains forbidden prediction columns: {extra_forbidden}")
            return [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    except Exception as exc:
        errors.append(f"{path} could not be read as predictions.csv: {exc}")
        return None


def _collect_support_set_ids(
    *,
    rows: list[dict[str, str]],
    path: Path,
    support_ids_by_combo: dict[tuple[str, str, str, str], dict[str, set[str]]],
    errors: list[str],
) -> None:
    for index, row in enumerate(rows, start=2):
        missing = [
            column
            for column in ("expert_name", "dataset", "category", "k_shot", "seed", "support_set_id")
            if not row.get(column)
        ]
        if missing:
            errors.append(f"{path}:{index} is missing required support consistency values: {missing}")
            continue
        combo = (row["dataset"], row["category"], row["k_shot"], row["seed"])
        expert = row["expert_name"]
        support_ids_by_combo.setdefault(combo, {}).setdefault(expert, set()).add(
            row["support_set_id"]
        )


def _validate_failure_count(metrics_path: Path, failures_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{metrics_path} could not be read as metrics.json: {exc}"]
    try:
        failures = json.loads(failures_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{failures_path} could not be read as failures.json: {exc}"]

    try:
        expected = _metric_failed_count(metrics)
    except (TypeError, ValueError) as exc:
        errors.append(f"{metrics_path} has an invalid failed-sample count: {exc}")
        return errors
    if expected is None:
        errors.append(
            f"{metrics_path} is missing num_failed_samples or compatible num_failed metric"
        )
        return errors

    failed_predictions = failures.get("failed_predictions")
    if not isinstance(failed_predictions, list):
        errors.append(f"{failures_path} must contain a failed_predictions list")
        return errors

    actual = len(failed_predictions)
    if actual != expected:
        errors.append(
            f"{failures_path} has {actual} failed_predictions, but {metrics_path} reports {expected}"
        )
    return errors


def _metric_failed_count(metrics: dict[str, Any]) -> int | None:
    for key in ("num_failed_samples", "num_failed"):
        if key in metrics:
            return int(metrics[key])
    return None


def _validate_support_set_consistency(
    support_ids_by_combo: dict[tuple[str, str, str, str], dict[str, set[str]]],
) -> list[str]:
    errors: list[str] = []
    for combo, ids_by_expert in sorted(support_ids_by_combo.items()):
        flattened = {
            support_set_id
            for support_set_ids in ids_by_expert.values()
            for support_set_id in support_set_ids
        }
        multi_id_experts = {
            expert: sorted(support_set_ids)
            for expert, support_set_ids in ids_by_expert.items()
            if len(support_set_ids) > 1
        }
        if multi_id_experts:
            errors.append(
                "Expert has multiple support_set_id values for "
                f"dataset/category/k_shot/seed={combo}: {multi_id_experts}"
            )
        if len(flattened) > 1:
            errors.append(
                "Inconsistent support_set_id across experts for "
                f"dataset/category/k_shot/seed={combo}: {ids_by_expert}"
            )
    return errors


if __name__ == "__main__":
    main()
