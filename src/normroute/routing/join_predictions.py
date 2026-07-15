"""Build Stage 3 evaluator-only routing matrices from Stage 2 predictions."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.normroute.evaluation.export import PREDICTION_COLUMNS


REQUIRED_RUN_FILES = (
    "predictions.csv",
    "metrics.json",
    "failures.json",
    "run_metadata.json",
)
FORBIDDEN_PREDICTION_COLUMNS = {"label", "mask_path", "defect_type", "anomaly_type"}
EVALUATOR_COLUMNS = ("image_id", "label", "mask_path")
KEY_COLUMNS = ("image_id", "dataset", "category", "support_set_id", "k_shot", "seed")
REQUIRED_STAGE3_PREDICTION_COLUMNS = (
    *KEY_COLUMNS,
    "expert_name",
    "final_score",
    "final_decision",
    "runtime_ms",
    "anomaly_map_path",
    "status",
    "error_message",
)
LONG_COLUMNS = (
    *KEY_COLUMNS,
    "expert_name",
    "final_score",
    "final_decision",
    "status",
    "error_message",
    "anomaly_map_path",
    "pixel_score_path",
    "runtime_ms",
    "label",
    "mask_path",
    "run_dir",
)
EXPERT_SCORE_COLUMNS = {
    "patchcore": "patchcore_score",
    "winclip": "winclip_score",
    "anomalydino": "anomalydino_score",
}
WIDE_COLUMNS = (
    *KEY_COLUMNS,
    "label",
    "mask_path",
    "patchcore_score",
    "winclip_score",
    "anomalydino_score",
)


class RoutingMatrixError(ValueError):
    """Raised when Stage 3 routing-matrix construction would be invalid."""


@dataclass(frozen=True)
class Stage2Audit:
    """Result of auditing Stage 2 run directories."""

    run_dirs: list[Path]
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class RoutingMatrices:
    """In-memory long and wide evaluator-only routing matrices."""

    long_rows: list[dict[str, str]]
    wide_rows: list[dict[str, str]]


def audit_stage2_output_tree(output_root: str | Path) -> Stage2Audit:
    """Check every discovered Stage 2 run directory for required artifacts and leakage."""

    root = Path(output_root)
    if not root.exists():
        return Stage2Audit(run_dirs=[], errors=[f"Stage 2 output root does not exist: {root}"])

    run_dirs = discover_stage2_run_dirs(root)
    if not run_dirs:
        return Stage2Audit(
            run_dirs=[],
            errors=[f"No predictions.csv files found under Stage 2 output root: {root}"],
        )

    errors: list[str] = []
    for run_dir in run_dirs:
        for file_name in REQUIRED_RUN_FILES:
            if not (run_dir / file_name).is_file():
                errors.append(f"{run_dir} is missing required artifact {file_name}")
        predictions_path = run_dir / "predictions.csv"
        if predictions_path.is_file():
            errors.extend(validate_prediction_csv(predictions_path))
        for json_name in ("metrics.json", "failures.json", "run_metadata.json"):
            json_path = run_dir / json_name
            if json_path.is_file():
                try:
                    json.loads(json_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    errors.append(f"{json_path} could not be read as JSON: {exc}")

    return Stage2Audit(run_dirs=run_dirs, errors=errors)


def discover_stage2_run_dirs(output_root: str | Path) -> list[Path]:
    """Return directories containing recursively discovered predictions.csv files."""

    root = Path(output_root)
    if root.name == "predictions.csv" and root.is_file():
        return [root.parent]
    if (root / "predictions.csv").is_file():
        return [root]
    return sorted({path.parent for path in root.rglob("predictions.csv")})


def validate_prediction_csv(path: str | Path) -> list[str]:
    """Validate prediction CSV columns without reading evaluator-only labels."""

    predictions_path = Path(path)
    errors: list[str] = []
    try:
        with predictions_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            forbidden = sorted(FORBIDDEN_PREDICTION_COLUMNS.intersection(fieldnames))
            if forbidden:
                errors.append(
                    f"{predictions_path} contains forbidden prediction columns: {forbidden}"
                )
            missing = [
                column for column in REQUIRED_STAGE3_PREDICTION_COLUMNS if column not in fieldnames
            ]
            if missing:
                errors.append(f"{predictions_path} is missing prediction columns: {missing}")
    except Exception as exc:
        errors.append(f"{predictions_path} could not be read as predictions.csv: {exc}")
    return errors


def build_routing_matrices(
    *,
    stage2_root: str | Path,
    evaluator_csv: str | Path,
) -> RoutingMatrices:
    """Build long and wide evaluator-only matrices from Stage 2 predictions."""

    audit = audit_stage2_output_tree(stage2_root)
    if audit.errors:
        raise RoutingMatrixError("Stage 2 output audit failed:\n" + "\n".join(audit.errors))

    evaluator_by_id = read_evaluator_manifest(evaluator_csv)
    long_rows: list[dict[str, str]] = []
    for run_dir in audit.run_dirs:
        for row in read_prediction_rows(run_dir / "predictions.csv"):
            image_id = row["image_id"]
            evaluator = evaluator_by_id.get(image_id)
            if evaluator is None:
                raise RoutingMatrixError(
                    f"{run_dir / 'predictions.csv'} has image_id not present in evaluator CSV: "
                    f"{image_id}"
                )
            long_rows.append(
                {
                    **{column: row[column] for column in KEY_COLUMNS},
                    "expert_name": row["expert_name"],
                    "final_score": row["final_score"],
                    "final_decision": row["final_decision"],
                    "status": row["status"],
                    "error_message": row["error_message"],
                    "anomaly_map_path": row["anomaly_map_path"],
                    "pixel_score_path": row.get("pixel_score_path", ""),
                    "runtime_ms": row["runtime_ms"],
                    "label": evaluator["label"],
                    "mask_path": evaluator["mask_path"],
                    "run_dir": str(run_dir),
                }
            )

    long_rows.sort(key=lambda row: tuple(row[column] for column in (*KEY_COLUMNS, "expert_name")))
    return RoutingMatrices(long_rows=long_rows, wide_rows=build_wide_rows(long_rows))


def read_evaluator_manifest(path: str | Path) -> dict[str, dict[str, str]]:
    """Read evaluator-only fields for the Stage 3 join."""

    evaluator_path = Path(path)
    with evaluator_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in EVALUATOR_COLUMNS if column not in fieldnames]
        if missing:
            raise RoutingMatrixError(f"{evaluator_path} is missing evaluator columns: {missing}")
        rows: dict[str, dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            image_id = (row.get("image_id") or "").strip()
            if not image_id:
                raise RoutingMatrixError(f"{evaluator_path}:{line_number} has empty image_id")
            if image_id in rows:
                raise RoutingMatrixError(f"{evaluator_path} has duplicate image_id: {image_id}")
            rows[image_id] = {
                "label": (row.get("label") or "").strip(),
                "mask_path": (row.get("mask_path") or "").strip(),
            }
    return rows


def read_prediction_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a Stage 2 predictions.csv after enforcing no-leakage columns."""

    predictions_path = Path(path)
    errors = validate_prediction_csv(predictions_path)
    if errors:
        raise RoutingMatrixError("\n".join(errors))
    with predictions_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            clean = {key: (value or "").strip() for key, value in row.items()}
            missing = [column for column in KEY_COLUMNS if not clean.get(column)]
            missing.extend(
                column
                for column in ("expert_name", "final_score", "runtime_ms")
                if not clean.get(column)
            )
            if missing:
                raise RoutingMatrixError(
                    f"{predictions_path}:{line_number} is missing required values: {missing}"
                )
            rows.append(clean)
    return rows


def build_wide_rows(long_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Pivot long sample-expert predictions into one row per sample/support combo."""

    wide_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    seen_experts: set[tuple[str, ...]] = set()
    for row in long_rows:
        key = tuple(row[column] for column in KEY_COLUMNS)
        wide_row = wide_by_key.setdefault(
            key,
            {
                **{column: row[column] for column in KEY_COLUMNS},
                "label": row["label"],
                "mask_path": row["mask_path"],
                "patchcore_score": "",
                "winclip_score": "",
                "anomalydino_score": "",
            },
        )
        expert_key = normalize_expert_name(row["expert_name"])
        duplicate_key = (*key, expert_key)
        if duplicate_key in seen_experts:
            raise RoutingMatrixError(
                "Duplicate prediction for sample/support/expert: "
                f"{dict(zip(KEY_COLUMNS, key))}, expert={row['expert_name']}"
            )
        seen_experts.add(duplicate_key)
        score_column = EXPERT_SCORE_COLUMNS.get(expert_key)
        if score_column:
            wide_row[score_column] = row["final_score"]

    return [wide_by_key[key] for key in sorted(wide_by_key)]


def write_routing_matrices(matrices: RoutingMatrices, output_dir: str | Path) -> tuple[Path, Path]:
    """Write evaluator-only long and wide routing matrices."""

    output_path = Path(output_dir)
    ensure_evaluator_only_output(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    long_path = output_path / "routing_matrix_long.csv"
    wide_path = output_path / "routing_matrix_wide.csv"
    write_csv(long_path, list(LONG_COLUMNS), matrices.long_rows)
    write_csv(wide_path, list(WIDE_COLUMNS), matrices.wide_rows)
    return long_path, wide_path


def ensure_evaluator_only_output(output_dir: Path) -> None:
    """Keep evaluator-only labels and masks out of agent-visible paths."""

    if "evaluator_only" not in output_dir.parts:
        raise RoutingMatrixError(
            "Stage 3 routing matrices contain evaluator-only label/mask fields and must be "
            "written under an evaluator_only directory."
        )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def normalize_expert_name(expert_name: str) -> str:
    """Normalize display names such as AnomalyDINO into stable column keys."""

    return re.sub(r"[^a-z0-9]+", "", expert_name.lower())
