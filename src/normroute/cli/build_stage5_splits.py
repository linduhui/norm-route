"""Build the frozen Stage 5 category-held-out five-fold manifest."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

from .audit_stage5_inputs import (
    find_forbidden_field_paths,
    forbidden_path_reason,
)
from .build_stage4_splits import Stage4SplitError, _parse_simple_yaml


CATEGORY_CV_PROTOCOL_VERSION = "stage5.category_cv.v2"
SPLIT_NAMES = ("train", "val", "test")
EXPECTED_GROUPS: dict[str, tuple[str, ...]] = {
    "G0": ("carpet", "bottle", "cable"),
    "G1": ("grid", "capsule", "hazelnut"),
    "G2": ("leather", "metal_nut", "pill"),
    "G3": ("tile", "screw", "toothbrush"),
    "G4": ("wood", "transistor", "zipper"),
}
CATEGORY_GROUPS = EXPECTED_GROUPS
"""Public alias for the frozen Stage 5 category groups."""
EXPECTED_FOLD_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {
    "fold0": {"train": ("G2", "G3", "G4"), "val": ("G1",), "test": ("G0",)},
    "fold1": {"train": ("G0", "G3", "G4"), "val": ("G2",), "test": ("G1",)},
    "fold2": {"train": ("G0", "G1", "G4"), "val": ("G3",), "test": ("G2",)},
    "fold3": {"train": ("G0", "G1", "G2"), "val": ("G4",), "test": ("G3",)},
    "fold4": {"train": ("G1", "G2", "G3"), "val": ("G0",), "test": ("G4",)},
}
EXPECTED_FOLDS: dict[str, dict[str, tuple[str, ...]]] = {
    fold: {
        split: tuple(
            category
            for group in groups
            for category in EXPECTED_GROUPS[group]
        )
        for split, groups in split_groups.items()
    }
    for fold, split_groups in EXPECTED_FOLD_GROUPS.items()
}
EXPECTED_CATEGORIES = tuple(
    category for categories in EXPECTED_GROUPS.values() for category in categories
)
FOLD_MANIFEST_COLUMNS = (
    "fold",
    "split",
    "task_id",
    "sample_id",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
)
QUERY_ID_FIELDS = ("sample_id", "image_id", "query_id", "query_path")


class Stage5SplitError(ValueError):
    """Raised when category-held-out splits cannot be built safely."""


@dataclass(frozen=True)
class CategoryFoldSpec:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def categories_for(self, split: str) -> tuple[str, ...]:
        if split not in SPLIT_NAMES:
            raise Stage5SplitError(f"Unknown split name: {split!r}")
        return getattr(self, split)

    def split_for_category(self, category: str) -> str:
        matches = [
            split for split in SPLIT_NAMES if category in self.categories_for(split)
        ]
        if len(matches) != 1:
            raise Stage5SplitError(
                f"category={category!r} must belong to exactly one split; matched {matches}"
            )
        return matches[0]


@dataclass(frozen=True)
class CategoryCVConfig:
    protocol_version: str
    split_unit: str
    groups: dict[str, tuple[str, ...]]
    folds: dict[str, CategoryFoldSpec]

    @property
    def expected_categories(self) -> tuple[str, ...]:
        return tuple(category for group in self.groups.values() for category in group)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "split_unit": self.split_unit,
            "groups": {name: list(categories) for name, categories in self.groups.items()},
            "folds": {
                fold: {
                    split: list(spec.categories_for(split)) for split in SPLIT_NAMES
                }
                for fold, spec in self.folds.items()
            },
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "--tasks",
        dest="tasks_path",
        default="outputs/stage4/tasks/pre_route_tasks.jsonl",
        help="Complete leakage-safe task table in JSONL or CSV format.",
    )
    parser.add_argument("--config", default="configs/stage5/category_cv_v2.yaml")
    parser.add_argument("--output-dir", default="outputs/stage5/splits")
    parser.add_argument("--fold-manifest", default=None)
    parser.add_argument("--split-audit", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    manifest_path = (
        Path(args.fold_manifest)
        if args.fold_manifest is not None
        else output_dir / "fold_manifest.csv"
    )
    audit_path = (
        Path(args.split_audit)
        if args.split_audit is not None
        else output_dir / "split_audit.json"
    )
    try:
        written_manifest, written_audit = build_stage5_splits(
            tasks_path=args.tasks_path,
            config_path=args.config,
            manifest_path=manifest_path,
            audit_path=audit_path,
        )
    except Stage5SplitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {written_manifest}")
    print(f"Wrote {written_audit}")
    return 0


def load_category_cv_config(path: str | Path) -> CategoryCVConfig:
    """Load and validate the dependency-free Stage 5 YAML config."""

    config_path = Path(path)
    if not config_path.is_file():
        raise Stage5SplitError(f"Category CV config does not exist: {config_path}")
    try:
        raw = _parse_simple_yaml(config_path.read_text(encoding="utf-8-sig"))
    except (Stage4SplitError, OSError, UnicodeDecodeError) as exc:
        raise Stage5SplitError(f"Could not parse {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise Stage5SplitError(f"{config_path} must contain a YAML mapping")
    required = {"protocol_version", "split_unit", "groups", "folds"}
    missing = sorted(required - set(raw))
    extra = sorted(set(raw) - required)
    if missing or extra:
        raise Stage5SplitError(
            f"{config_path} has invalid top-level keys; missing={missing}, extra={extra}"
        )

    groups_raw = raw["groups"]
    if not isinstance(groups_raw, dict):
        raise Stage5SplitError(f"{config_path} groups must be a mapping")
    groups = {
        str(name): _string_tuple(categories, f"groups.{name}")
        for name, categories in groups_raw.items()
    }

    folds_raw = raw["folds"]
    if not isinstance(folds_raw, dict):
        raise Stage5SplitError(f"{config_path} folds must be a mapping")
    folds: dict[str, CategoryFoldSpec] = {}
    for fold, split_mapping in folds_raw.items():
        if not isinstance(split_mapping, dict) or set(split_mapping) != set(SPLIT_NAMES):
            raise Stage5SplitError(
                f"{config_path} {fold} must define exactly train, val, and test"
            )
        folds[str(fold)] = CategoryFoldSpec(
            train=_string_tuple(split_mapping["train"], f"{fold}.train"),
            val=_string_tuple(split_mapping["val"], f"{fold}.val"),
            test=_string_tuple(split_mapping["test"], f"{fold}.test"),
        )

    config = CategoryCVConfig(
        protocol_version=_nonempty_string(raw["protocol_version"], "protocol_version"),
        split_unit=_nonempty_string(raw["split_unit"], "split_unit"),
        groups=groups,
        folds=folds,
    )
    validate_category_cv_config(config, context=str(config_path))
    return config


def validate_category_cv_config(
    config: CategoryCVConfig, *, context: str = "category CV config"
) -> None:
    """Freeze the five groups and cyclic train/validation/test assignment."""

    if config.protocol_version != CATEGORY_CV_PROTOCOL_VERSION:
        raise Stage5SplitError(
            f"{context} protocol_version must be {CATEGORY_CV_PROTOCOL_VERSION!r}"
        )
    if config.split_unit != "category":
        raise Stage5SplitError(f"{context} split_unit must be 'category'")
    if config.groups != EXPECTED_GROUPS:
        raise Stage5SplitError(
            f"{context} groups must exactly match the frozen G0--G4 assignment"
        )
    if len(config.expected_categories) != len(set(config.expected_categories)):
        raise Stage5SplitError(f"{context} assigns a category to multiple groups")
    if tuple(config.folds) != tuple(EXPECTED_FOLDS):
        raise Stage5SplitError(
            f"{context} folds must be ordered as {list(EXPECTED_FOLDS)}"
        )

    expected_set = set(EXPECTED_CATEGORIES)
    for fold, expected in EXPECTED_FOLDS.items():
        spec = config.folds[fold]
        actual = {split: spec.categories_for(split) for split in SPLIT_NAMES}
        if actual != expected:
            raise Stage5SplitError(f"{context} {fold} must be {expected}; got {actual}")
        category_sets = {split: set(actual[split]) for split in SPLIT_NAMES}
        if any(
            category_sets[left].intersection(category_sets[right])
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            raise Stage5SplitError(f"{context} {fold} has overlapping categories")
        if set().union(*category_sets.values()) != expected_set:
            raise Stage5SplitError(f"{context} {fold} does not cover all categories")

    for category in EXPECTED_CATEGORIES:
        test_count = sum(category in spec.test for spec in config.folds.values())
        val_count = sum(category in spec.val for spec in config.folds.values())
        if test_count != 1 or val_count != 1:
            raise Stage5SplitError(
                f"{context} category={category!r} must be test and val exactly once"
            )


def build_stage5_splits(
    *,
    tasks_path: str | Path,
    config_path: str | Path,
    manifest_path: str | Path,
    audit_path: str | Path,
) -> tuple[Path, Path]:
    """Write one category-derived assignment per task and fold plus its audit."""

    tasks_file = Path(tasks_path)
    config_file = Path(config_path)
    manifest_file = Path(manifest_path)
    audit_file = Path(audit_path)
    if manifest_file.resolve() == audit_file.resolve():
        raise Stage5SplitError("fold_manifest.csv and split_audit.json paths must differ")
    for path in (tasks_file, manifest_file, audit_file):
        reason = forbidden_path_reason(path)
        if reason is not None:
            raise Stage5SplitError(f"Stage 5 inference-visible path is forbidden: {path}: {reason}")

    config = load_category_cv_config(config_file)
    tasks = read_stage5_tasks(tasks_file)
    observed_categories = tuple(
        category for category in EXPECTED_CATEGORIES if any(task["category"] == category for task in tasks)
    )
    missing_categories = sorted(set(config.expected_categories) - {task["category"] for task in tasks})
    unexpected_categories = sorted({task["category"] for task in tasks} - set(config.expected_categories))
    if missing_categories or unexpected_categories:
        raise Stage5SplitError(
            "Stage 5 tasks must contain every frozen category and no others; "
            f"missing={missing_categories}, unexpected={unexpected_categories}"
        )

    _validate_task_identity_consistency(tasks)
    fold_audits = _build_fold_audits(tasks, config)
    manifest_row_count = len(tasks) * len(config.folds)
    all_folds_valid = all(item["isolation_ok"] for item in fold_audits.values())
    audit: dict[str, Any] = {
        "protocol_version": config.protocol_version,
        "split_unit": config.split_unit,
        "source_tasks": str(tasks_file),
        "source_tasks_sha256": _sha256(tasks_file),
        "config_path": str(config_file),
        "config_sha256": _sha256(config_file),
        "config": config.to_dict(),
        "source_task_count": len(tasks),
        "observed_categories": list(observed_categories),
        "manifest_columns": list(FOLD_MANIFEST_COLUMNS),
        "manifest_row_count": manifest_row_count,
        "expected_manifest_row_count": manifest_row_count,
        "folds": fold_audits,
        "all_folds_valid": all_folds_valid,
        "failures": [],
        "provenance": {
            "git_commit": _git_commit(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    if not all_folds_valid:
        raise Stage5SplitError("Category split isolation audit failed")

    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = manifest_file.with_name(f".{manifest_file.name}.tmp")
    audit_tmp = audit_file.with_name(f".{audit_file.name}.tmp")
    try:
        _write_manifest(manifest_tmp, tasks, config)
        audit_tmp.write_text(
            json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        manifest_tmp.replace(manifest_file)
        audit_tmp.replace(audit_file)
    finally:
        for temporary in (manifest_tmp, audit_tmp):
            if temporary.exists():
                temporary.unlink()
    return manifest_file, audit_file


def read_stage5_tasks(path: str | Path) -> list[dict[str, Any]]:
    """Read every JSONL/CSV task, rejecting malformed, leaked, or skipped rows."""

    tasks_path = Path(path)
    if not tasks_path.is_file():
        raise Stage5SplitError(f"Stage 5 task file does not exist: {tasks_path}")
    suffix = tasks_path.suffix.lower()
    if suffix not in {".jsonl", ".csv"}:
        raise Stage5SplitError(f"Stage 5 task file must be JSONL or CSV: {tasks_path}")

    source_rows: list[Mapping[str, Any]] = []
    contexts: list[str] = []
    try:
        if suffix == ".jsonl":
            with tasks_path.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    context = f"{tasks_path}:{line_number}"
                    if not line.strip():
                        raise Stage5SplitError(
                            f"{context} is blank; failed tasks must not be skipped"
                        )
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise Stage5SplitError(f"{context} must contain a JSON object")
                    source_rows.append(payload)
                    contexts.append(context)
        else:
            with tasks_path.open("r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise Stage5SplitError(f"{tasks_path} has no CSV header")
                forbidden = find_forbidden_field_paths({field: None for field in reader.fieldnames})
                if forbidden:
                    raise Stage5SplitError(
                        f"{tasks_path} contains forbidden inference-visible columns: "
                        f"{[item['field_path'] for item in forbidden]}"
                    )
                for line_number, row in enumerate(reader, start=2):
                    source_rows.append({key: (value or "").strip() for key, value in row.items()})
                    contexts.append(f"{tasks_path}:{line_number}")
    except json.JSONDecodeError as exc:
        raise Stage5SplitError(
            f"Could not parse {tasks_path} as JSONL at line {exc.lineno}: {exc.msg}"
        ) from exc
    except (csv.Error, OSError, UnicodeDecodeError) as exc:
        raise Stage5SplitError(f"Could not read {tasks_path}: {exc}") from exc

    if not source_rows:
        raise Stage5SplitError(f"Stage 5 task file is empty: {tasks_path}")

    tasks: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    for source, context in zip(source_rows, contexts):
        task = _normalize_task(source, context=context)
        if task["task_id"] in seen_task_ids:
            raise Stage5SplitError(f"{context} has duplicate task_id={task['task_id']!r}")
        seen_task_ids.add(task["task_id"])
        tasks.append(task)
    return tasks


def _normalize_task(source: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    findings = find_forbidden_field_paths(source)
    if findings:
        raise Stage5SplitError(
            f"{context} contains forbidden inference-visible fields: "
            f"{[item['field_path'] for item in findings]}"
        )
    required = ("task_id", "dataset", "category", "k_shot", "seed", "support_set_id")
    missing = [field for field in required if field not in source]
    if missing:
        raise Stage5SplitError(f"{context} is missing required fields: {missing}")
    query_fields = [field for field in QUERY_ID_FIELDS if _has_value(source.get(field))]
    if not query_fields:
        raise Stage5SplitError(
            f"{context} must provide one query identity field from {list(QUERY_ID_FIELDS)}"
        )

    task = {
        "task_id": _task_string(source["task_id"], "task_id", context),
        "sample_id": _task_string(source[query_fields[0]], query_fields[0], context),
        "dataset": _task_string(source["dataset"], "dataset", context),
        "category": _task_string(source["category"], "category", context),
        "k_shot": _task_integer(source["k_shot"], "k_shot", context, minimum=1),
        "seed": _task_integer(source["seed"], "seed", context, minimum=0),
        "support_set_id": _task_string(source["support_set_id"], "support_set_id", context),
    }
    return task


def _validate_task_identity_consistency(tasks: Sequence[Mapping[str, Any]]) -> None:
    query_categories: dict[tuple[str, str], set[str]] = {}
    support_signatures: dict[str, set[tuple[str, str, int, int]]] = {}
    signature_support_ids: dict[tuple[str, str, int, int], set[str]] = {}
    for task in tasks:
        query_key = (str(task["dataset"]), str(task["sample_id"]))
        query_categories.setdefault(query_key, set()).add(str(task["category"]))
        signature = (
            str(task["dataset"]),
            str(task["category"]),
            int(task["k_shot"]),
            int(task["seed"]),
        )
        support_set_id = str(task["support_set_id"])
        support_signatures.setdefault(support_set_id, set()).add(signature)
        signature_support_ids.setdefault(signature, set()).add(support_set_id)

    bad_queries = {key: sorted(values) for key, values in query_categories.items() if len(values) != 1}
    if bad_queries:
        raise Stage5SplitError(
            f"Query identities cross category boundaries: {dict(list(bad_queries.items())[:5])}"
        )
    bad_support_ids = {
        support_id: sorted(values)
        for support_id, values in support_signatures.items()
        if len(values) != 1
    }
    if bad_support_ids:
        raise Stage5SplitError(
            "support_set_id values map to multiple dataset/category/K/seed signatures: "
            f"{dict(list(bad_support_ids.items())[:5])}"
        )
    bad_signatures = {
        signature: sorted(values)
        for signature, values in signature_support_ids.items()
        if len(values) != 1
    }
    if bad_signatures:
        raise Stage5SplitError(
            "A dataset/category/K/seed signature maps to multiple support_set_id values: "
            f"{dict(list(bad_signatures.items())[:5])}"
        )


def _build_fold_audits(
    tasks: Sequence[Mapping[str, Any]], config: CategoryCVConfig
) -> dict[str, dict[str, Any]]:
    audits: dict[str, dict[str, Any]] = {}
    for fold, spec in config.folds.items():
        task_counts = {split: 0 for split in SPLIT_NAMES}
        assigned_task_ids: set[str] = set()
        duplicate_assignments = 0
        category_splits: dict[str, set[str]] = {}
        for task in tasks:
            category = str(task["category"])
            split = spec.split_for_category(category)
            task_counts[split] += 1
            task_id = str(task["task_id"])
            if task_id in assigned_task_ids:
                duplicate_assignments += 1
            assigned_task_ids.add(task_id)
            category_splits.setdefault(category, set()).add(split)

        category_sets = {
            split: set(spec.categories_for(split)) for split in SPLIT_NAMES
        }
        overlaps = {
            "train_val": sorted(category_sets["train"].intersection(category_sets["val"])),
            "train_test": sorted(category_sets["train"].intersection(category_sets["test"])),
            "val_test": sorted(category_sets["val"].intersection(category_sets["test"])),
        }
        inconsistent_categories = {
            category: sorted(splits)
            for category, splits in category_splits.items()
            if len(splits) != 1
        }
        unassigned = len(tasks) - sum(task_counts.values())
        audits[fold] = {
            "train_categories": list(spec.train),
            "val_categories": list(spec.val),
            "test_categories": list(spec.test),
            "task_counts": task_counts,
            "manifest_row_count": sum(task_counts.values()),
            "unique_task_count": len(assigned_task_ids),
            "unassigned_task_count": unassigned,
            "duplicate_task_assignment_count": duplicate_assignments,
            "category_overlaps": overlaps,
            "category_bundle_violations": inconsistent_categories,
            "isolation_ok": (
                not any(overlaps.values())
                and not inconsistent_categories
                and unassigned == 0
                and duplicate_assignments == 0
                and len(assigned_task_ids) == len(tasks)
            ),
        }
    return audits


def _write_manifest(
    path: Path,
    tasks: Sequence[Mapping[str, Any]],
    config: CategoryCVConfig,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FOLD_MANIFEST_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for fold, spec in config.folds.items():
            for task in tasks:
                writer.writerow(
                    {
                        "fold": fold,
                        "split": spec.split_for_category(str(task["category"])),
                        **{column: task[column] for column in FOLD_MANIFEST_COLUMNS[2:]},
                    }
                )


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise Stage5SplitError(f"{field} must be a YAML list of non-empty strings")
    if len(value) != len(set(value)):
        raise Stage5SplitError(f"{field} contains duplicate categories")
    return tuple(value)


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage5SplitError(f"{field} must be a non-empty string")
    return value


def _task_string(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage5SplitError(f"{context} field {field!r} must be a non-empty string")
    return value.strip()


def _task_integer(value: Any, field: str, context: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise Stage5SplitError(f"{context} field {field!r} must be an integer >= {minimum}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise Stage5SplitError(
            f"{context} field {field!r} must be an integer >= {minimum}"
        ) from exc
    if isinstance(value, float) and not value.is_integer():
        raise Stage5SplitError(f"{context} field {field!r} must be an integer >= {minimum}")
    if isinstance(value, str) and str(parsed) != value.strip():
        raise Stage5SplitError(f"{context} field {field!r} must be an integer >= {minimum}")
    if parsed < minimum:
        raise Stage5SplitError(f"{context} field {field!r} must be an integer >= {minimum}")
    return parsed


def _has_value(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    project_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
