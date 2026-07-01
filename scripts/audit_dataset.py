"""Audit dataset manifests from split agent/evaluator CSV files.

This script only reads manifest CSVs and image/mask files to produce dataset
counts. It does not read model outputs, run models, tune thresholds, or modify
manifest definitions.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.common import inspect_image
from src.data.manifest import AGENT_INPUT_COLUMNS, EVALUATOR_COLUMNS


AUDIT_COLUMNS = [
    "dataset",
    "category",
    "train_images",
    "test_good_images",
    "test_anomaly_images",
    "anomaly_types",
    "missing_masks",
    "broken_files",
]


@dataclass
class CategoryAudit:
    dataset: str
    category: str
    train_images: int = 0
    test_good_images: int = 0
    test_anomaly_images: int = 0
    anomaly_types: set[str] | None = None
    missing_masks: int = 0
    broken_files: int = 0

    def __post_init__(self) -> None:
        if self.anomaly_types is None:
            self.anomaly_types = set()

    def to_row(self) -> dict[str, str | int]:
        return {
            "dataset": self.dataset,
            "category": self.category,
            "train_images": self.train_images,
            "test_good_images": self.test_good_images,
            "test_anomaly_images": self.test_anomaly_images,
            "anomaly_types": ";".join(sorted(self.anomaly_types or set())),
            "missing_masks": self.missing_masks,
            "broken_files": self.broken_files,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-input-csv", required=True, help="Path to agent_input CSV.")
    parser.add_argument("--evaluator-csv", required=True, help="Path to evaluator CSV.")
    parser.add_argument("--output-csv", required=True, help="Path for the audit summary CSV.")
    parser.add_argument(
        "--path-root",
        default=".",
        help="Root used to resolve relative image and mask paths from the manifests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = audit_dataset(
        agent_input_csv=Path(args.agent_input_csv),
        evaluator_csv=Path(args.evaluator_csv),
        path_root=Path(args.path_root),
    )
    write_audit_csv(Path(args.output_csv), rows)
    print(f"Wrote {args.output_csv}")


def audit_dataset(
    *,
    agent_input_csv: Path,
    evaluator_csv: Path,
    path_root: Path,
) -> list[dict[str, str | int]]:
    agent_rows = _read_csv(agent_input_csv, AGENT_INPUT_COLUMNS)
    evaluator_rows = _read_csv(evaluator_csv, EVALUATOR_COLUMNS)
    evaluator_by_id = _index_evaluator_rows(evaluator_rows)

    agent_ids = [row["image_id"] for row in agent_rows]
    evaluator_ids = set(evaluator_by_id)
    missing_evaluator = sorted(set(agent_ids) - evaluator_ids)
    extra_evaluator = sorted(evaluator_ids - set(agent_ids))
    if missing_evaluator or extra_evaluator:
        raise ValueError(
            "agent_input/evaluator image_id mismatch: "
            f"missing_evaluator={missing_evaluator}, extra_evaluator={extra_evaluator}"
        )
    if len(agent_ids) != len(set(agent_ids)):
        raise ValueError("agent_input CSV contains duplicate image_id values")

    audits: dict[tuple[str, str], CategoryAudit] = {}
    resolved_root = path_root.resolve()
    checked_files: set[Path] = set()

    for agent_row in agent_rows:
        evaluator_row = evaluator_by_id[agent_row["image_id"]]
        dataset = agent_row["dataset"]
        category = agent_row["category"]
        split = agent_row["split"]
        label = _parse_label(evaluator_row["label"], agent_row["image_id"])
        defect_type = evaluator_row["defect_type"]
        mask_path = evaluator_row["mask_path"]

        key = (dataset, category)
        if key not in audits:
            audits[key] = CategoryAudit(dataset=dataset, category=category)
        audit = audits[key]

        if split == "train":
            audit.train_images += 1
        elif split == "test" and label == 0:
            audit.test_good_images += 1
        elif split == "test" and label == 1:
            audit.test_anomaly_images += 1
            if defect_type and defect_type != "good":
                assert audit.anomaly_types is not None
                audit.anomaly_types.add(defect_type)
            if not mask_path:
                audit.missing_masks += 1
        else:
            raise ValueError(
                f"Unsupported split/label combination for image_id "
                f"{agent_row['image_id']}: split={split!r}, label={label!r}"
            )

        image_path = _resolve_manifest_path(agent_row["image_path"], resolved_root)
        audit.broken_files += _broken_file_count(image_path, checked_files)
        if mask_path:
            resolved_mask = _resolve_manifest_path(mask_path, resolved_root)
            audit.broken_files += _broken_file_count(resolved_mask, checked_files)

    return [audits[key].to_row() for key in sorted(audits)]


def write_audit_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"{path} has unexpected columns: {reader.fieldnames}; "
                f"expected {expected_columns}"
            )
        return [dict(row) for row in reader]


def _index_evaluator_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        image_id = row["image_id"]
        if image_id in by_id:
            duplicates.append(image_id)
        by_id[image_id] = row
    if duplicates:
        raise ValueError(f"evaluator CSV contains duplicate image_id values: {sorted(duplicates)}")
    return by_id


def _parse_label(value: Any, image_id: str) -> int:
    try:
        label = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid evaluator label for image_id {image_id}: {value!r}") from exc
    if label not in {0, 1}:
        raise ValueError(f"Invalid evaluator label for image_id {image_id}: {value!r}")
    return label


def _resolve_manifest_path(path_value: str, path_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return path_root / path


def _broken_file_count(path: Path, checked_files: set[Path]) -> int:
    resolved = path.resolve()
    if resolved in checked_files:
        return 0
    checked_files.add(resolved)
    return 0 if inspect_image(resolved).ok else 1


if __name__ == "__main__":
    main()
