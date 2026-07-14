"""Build the fixed Stage 4 five-fold seed manifest and isolation audit."""

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
from typing import Any, Mapping

from src.normroute.agent.protocol import POLICY_FEATURE_ALLOWLIST, PROVENANCE_FIELDS
from src.normroute.agent.task_builder import TaskBuildError, read_pre_route_tasks


SEED_CV_PROTOCOL_VERSION = "stage4.seed_cv.v1"
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_FOLDS: dict[str, dict[str, tuple[int, ...]]] = {
    "fold0": {"train": (2, 3, 4), "val": (1,), "test": (0,)},
    "fold1": {"train": (0, 3, 4), "val": (2,), "test": (1,)},
    "fold2": {"train": (0, 1, 4), "val": (3,), "test": (2,)},
    "fold3": {"train": (0, 1, 2), "val": (4,), "test": (3,)},
    "fold4": {"train": (1, 2, 3), "val": (0,), "test": (4,)},
}
SPLIT_NAMES = ("train", "val", "test")
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


class Stage4SplitError(ValueError):
    """Raised when the fixed seed split cannot be built or audited safely."""


@dataclass(frozen=True)
class FoldSpec:
    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]

    def seeds_for(self, split: str) -> tuple[int, ...]:
        if split not in SPLIT_NAMES:
            raise Stage4SplitError(f"Unknown split name: {split!r}")
        return getattr(self, split)

    def split_for_seed(self, seed: int) -> str:
        matches = [split for split in SPLIT_NAMES if seed in self.seeds_for(split)]
        if len(matches) != 1:
            raise Stage4SplitError(
                f"seed={seed} must belong to exactly one split; matched {matches}"
            )
        return matches[0]


@dataclass(frozen=True)
class SeedCVConfig:
    protocol_version: str
    split_unit: str
    expected_seeds: tuple[int, ...]
    policy_feature_allowlist: tuple[str, ...]
    provenance_fields: tuple[str, ...]
    folds: dict[str, FoldSpec]

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "split_unit": self.split_unit,
            "expected_seeds": list(self.expected_seeds),
            "policy_feature_allowlist": list(self.policy_feature_allowlist),
            "provenance_fields": list(self.provenance_fields),
            "folds": {
                fold: {
                    split: list(spec.seeds_for(split))
                    for split in SPLIT_NAMES
                }
                for fold, spec in self.folds.items()
            },
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fixed Stage 4 five-fold seed manifest and isolation audit."
    )
    parser.add_argument(
        "--tasks",
        "--pre-route-tasks",
        dest="tasks_path",
        default="outputs/stage4/tasks/pre_route_tasks.jsonl",
    )
    parser.add_argument("--config", default="configs/stage4/seed_cv.yaml")
    parser.add_argument("--output-dir", default="outputs/stage4/splits")
    parser.add_argument("--fold-manifest", default=None)
    parser.add_argument("--split-audit", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
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
        written_manifest, written_audit = build_stage4_splits(
            tasks_path=args.tasks_path,
            config_path=args.config,
            manifest_path=manifest_path,
            audit_path=audit_path,
        )
    except (Stage4SplitError, TaskBuildError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Wrote {written_manifest}")
    print(f"Wrote {written_audit}")


def load_seed_cv_config(path: str | Path) -> SeedCVConfig:
    """Load the dependency-free YAML subset used by the checked-in split config."""

    config_path = Path(path)
    if not config_path.is_file():
        raise Stage4SplitError(f"Seed CV config does not exist: {config_path}")
    try:
        raw = _parse_simple_yaml(config_path.read_text(encoding="utf-8-sig"))
    except Stage4SplitError:
        raise
    except Exception as exc:
        raise Stage4SplitError(f"Could not parse {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise Stage4SplitError(f"{config_path} must contain a YAML mapping")

    required = {
        "protocol_version",
        "split_unit",
        "expected_seeds",
        "policy_feature_allowlist",
        "provenance_fields",
        "folds",
    }
    missing = sorted(required - set(raw))
    extra = sorted(set(raw) - required)
    if missing or extra:
        raise Stage4SplitError(
            f"{config_path} has invalid top-level keys; missing={missing}, extra={extra}"
        )

    folds_raw = raw["folds"]
    if not isinstance(folds_raw, dict):
        raise Stage4SplitError(f"{config_path} folds must be a mapping")
    folds: dict[str, FoldSpec] = {}
    for fold, split_mapping in folds_raw.items():
        if not isinstance(split_mapping, dict) or set(split_mapping) != set(SPLIT_NAMES):
            raise Stage4SplitError(
                f"{config_path} {fold} must define exactly train, val, and test"
            )
        folds[str(fold)] = FoldSpec(
            train=_integer_tuple(split_mapping["train"], f"{fold}.train"),
            val=_integer_tuple(split_mapping["val"], f"{fold}.val"),
            test=_integer_tuple(split_mapping["test"], f"{fold}.test"),
        )

    config = SeedCVConfig(
        protocol_version=_string_value(raw["protocol_version"], "protocol_version"),
        split_unit=_string_value(raw["split_unit"], "split_unit"),
        expected_seeds=_integer_tuple(raw["expected_seeds"], "expected_seeds"),
        policy_feature_allowlist=_string_tuple(
            raw["policy_feature_allowlist"], "policy_feature_allowlist"
        ),
        provenance_fields=_string_tuple(raw["provenance_fields"], "provenance_fields"),
        folds=folds,
    )
    validate_seed_cv_config(config, context=str(config_path))
    return config


def validate_seed_cv_config(config: SeedCVConfig, *, context: str = "seed CV config") -> None:
    """Freeze the requested five folds and the policy/provenance boundary."""

    if config.protocol_version != SEED_CV_PROTOCOL_VERSION:
        raise Stage4SplitError(
            f"{context} protocol_version must be {SEED_CV_PROTOCOL_VERSION!r}"
        )
    if config.split_unit != "seed":
        raise Stage4SplitError(f"{context} split_unit must be 'seed'")
    if config.expected_seeds != EXPECTED_SEEDS:
        raise Stage4SplitError(f"{context} expected_seeds must be {list(EXPECTED_SEEDS)}")
    if config.policy_feature_allowlist != POLICY_FEATURE_ALLOWLIST:
        raise Stage4SplitError(
            f"{context} policy_feature_allowlist must be {list(POLICY_FEATURE_ALLOWLIST)}"
        )
    if config.provenance_fields != PROVENANCE_FIELDS:
        raise Stage4SplitError(
            f"{context} provenance_fields must be {list(PROVENANCE_FIELDS)}"
        )
    if set(config.policy_feature_allowlist).intersection(config.provenance_fields):
        raise Stage4SplitError(f"{context} mixes provenance fields into policy features")
    if tuple(config.folds) != tuple(EXPECTED_FOLDS):
        raise Stage4SplitError(
            f"{context} folds must be ordered as {list(EXPECTED_FOLDS)}"
        )

    for fold, expected in EXPECTED_FOLDS.items():
        spec = config.folds[fold]
        actual = {split: spec.seeds_for(split) for split in SPLIT_NAMES}
        if actual != expected:
            raise Stage4SplitError(f"{context} {fold} must be {expected}; got {actual}")
        seed_sets = {split: set(actual[split]) for split in SPLIT_NAMES}
        if any(
            seed_sets[left].intersection(seed_sets[right])
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            raise Stage4SplitError(f"{context} {fold} has overlapping seed partitions")
        if set().union(*seed_sets.values()) != set(EXPECTED_SEEDS):
            raise Stage4SplitError(f"{context} {fold} does not cover every expected seed")


def build_stage4_splits(
    *,
    tasks_path: str | Path,
    config_path: str | Path,
    manifest_path: str | Path,
    audit_path: str | Path,
) -> tuple[Path, Path]:
    """Write one manifest row per task/fold and a machine-readable isolation audit."""

    tasks_file = Path(tasks_path)
    config_file = Path(config_path)
    manifest_file = Path(manifest_path)
    audit_file = Path(audit_path)
    if manifest_file.resolve() == audit_file.resolve():
        raise Stage4SplitError("fold_manifest.csv and split_audit.json paths must differ")
    _ensure_non_evaluator_output(manifest_file)
    _ensure_non_evaluator_output(audit_file)

    config = load_seed_cv_config(config_file)
    tasks = read_pre_route_tasks(tasks_file)
    observed_seeds = tuple(sorted({task["seed"] for task in tasks}))
    unexpected_seeds = sorted(set(observed_seeds) - set(config.expected_seeds))
    missing_seeds = sorted(set(config.expected_seeds) - set(observed_seeds))
    if unexpected_seeds or missing_seeds:
        raise Stage4SplitError(
            "Pre-route tasks must contain every configured seed and no others; "
            f"missing={missing_seeds}, unexpected={unexpected_seeds}"
        )

    fold_audits = _build_fold_audits(tasks, config)
    manifest_row_count = len(tasks) * len(config.folds)
    audit = {
        "protocol_version": config.protocol_version,
        "split_unit": config.split_unit,
        "source_tasks": str(tasks_file),
        "source_tasks_sha256": _sha256(tasks_file),
        "config_path": str(config_file),
        "config_sha256": _sha256(config_file),
        "config": config.to_dict(),
        "source_task_count": len(tasks),
        "observed_seeds": list(observed_seeds),
        "manifest_columns": list(FOLD_MANIFEST_COLUMNS),
        "manifest_row_count": manifest_row_count,
        "expected_manifest_row_count": manifest_row_count,
        "folds": fold_audits,
        "all_folds_valid": all(item["isolation_ok"] for item in fold_audits.values()),
        "failures": [],
        "provenance": {
            "git_commit": _git_commit(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    if not audit["all_folds_valid"]:
        raise Stage4SplitError("Seed split isolation audit failed")

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


def _build_fold_audits(
    tasks: list[dict[str, Any]],
    config: SeedCVConfig,
) -> dict[str, dict[str, Any]]:
    audits: dict[str, dict[str, Any]] = {}
    for fold, spec in config.folds.items():
        task_counts = {split: 0 for split in SPLIT_NAMES}
        assigned_task_ids: set[str] = set()
        duplicate_assignments = 0
        for task in tasks:
            split = spec.split_for_seed(task["seed"])
            task_counts[split] += 1
            if task["task_id"] in assigned_task_ids:
                duplicate_assignments += 1
            assigned_task_ids.add(task["task_id"])

        train_set, val_set, test_set = set(spec.train), set(spec.val), set(spec.test)
        overlaps = {
            "train_val": sorted(train_set.intersection(val_set)),
            "train_test": sorted(train_set.intersection(test_set)),
            "val_test": sorted(val_set.intersection(test_set)),
        }
        unassigned = len(tasks) - sum(task_counts.values())
        audits[fold] = {
            "train_seeds": list(spec.train),
            "val_seeds": list(spec.val),
            "test_seeds": list(spec.test),
            "task_counts": task_counts,
            "manifest_row_count": sum(task_counts.values()),
            "unique_task_count": len(assigned_task_ids),
            "unassigned_task_count": unassigned,
            "duplicate_task_assignment_count": duplicate_assignments,
            "seed_overlaps": overlaps,
            "isolation_ok": (
                not any(overlaps.values())
                and unassigned == 0
                and duplicate_assignments == 0
                and len(assigned_task_ids) == len(tasks)
            ),
        }
    return audits


def _write_manifest(
    path: Path,
    tasks: list[dict[str, Any]],
    config: SeedCVConfig,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FOLD_MANIFEST_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for fold, spec in config.folds.items():
            for task in tasks:
                writer.writerow(
                    {
                        "fold": fold,
                        "split": spec.split_for_seed(task["seed"]),
                        "task_id": task["task_id"],
                        "sample_id": task["sample_id"],
                        "dataset": task["dataset"],
                        "category": task["category"],
                        "k_shot": task["k_shot"],
                        "seed": task["seed"],
                        "support_set_id": task["support_set_id"],
                    }
                )


def _ensure_non_evaluator_output(path: Path) -> None:
    parts = {part.lower() for part in path.parts}
    if "evaluator_only" in parts or "oracle" in parts:
        raise Stage4SplitError("Stage 4 split artifacts cannot be written under evaluator_only/oracle")


def _integer_tuple(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise Stage4SplitError(f"{field} must be a YAML list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise Stage4SplitError(f"{field} must contain only integer seeds")
    if len(value) != len(set(value)):
        raise Stage4SplitError(f"{field} contains duplicate seeds")
    return tuple(value)


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise Stage4SplitError(f"{field} must be a YAML list of non-empty strings")
    if len(value) != len(set(value)):
        raise Stage4SplitError(f"{field} contains duplicate values")
    return tuple(value)


def _string_value(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage4SplitError(f"{field} must be a non-empty string")
    return value


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


def _parse_simple_yaml(text: str) -> Any:
    """Parse mappings and scalar lists without adding a runtime YAML dependency."""

    entries: list[tuple[int, str, int]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if "\t" in raw_line:
            raise Stage4SplitError(f"YAML line {line_number} contains a tab")
        content_without_comment = raw_line.split("#", 1)[0].rstrip()
        if not content_without_comment.strip():
            continue
        indent = len(content_without_comment) - len(content_without_comment.lstrip(" "))
        if indent % 2:
            raise Stage4SplitError(f"YAML line {line_number} indentation must use two spaces")
        entries.append((indent, content_without_comment.strip(), line_number))
    if not entries:
        raise Stage4SplitError("YAML config is empty")
    if entries[0][0] != 0:
        raise Stage4SplitError("YAML root must start at indentation zero")

    value, next_index = _parse_yaml_node(entries, 0, 0)
    if next_index != len(entries):
        _, _, line_number = entries[next_index]
        raise Stage4SplitError(f"Unexpected YAML content at line {line_number}")
    return value


def _parse_yaml_node(
    entries: list[tuple[int, str, int]],
    index: int,
    indent: int,
) -> tuple[Any, int]:
    first_indent, first_content, first_line = entries[index]
    if first_indent != indent:
        raise Stage4SplitError(f"Unexpected YAML indentation at line {first_line}")

    if first_content.startswith("- "):
        values: list[Any] = []
        while index < len(entries):
            current_indent, content, line_number = entries[index]
            if current_indent < indent:
                break
            if current_indent != indent or not content.startswith("- "):
                raise Stage4SplitError(f"Invalid YAML list item at line {line_number}")
            scalar = content[2:].strip()
            if not scalar:
                raise Stage4SplitError(f"Nested YAML list items are unsupported at line {line_number}")
            values.append(_parse_yaml_scalar(scalar))
            index += 1
        return values, index

    mapping: dict[str, Any] = {}
    while index < len(entries):
        current_indent, content, line_number = entries[index]
        if current_indent < indent:
            break
        if current_indent != indent or content.startswith("- "):
            raise Stage4SplitError(f"Invalid YAML mapping entry at line {line_number}")
        if ":" not in content:
            raise Stage4SplitError(f"YAML mapping entry lacks ':' at line {line_number}")
        key, scalar = content.split(":", 1)
        key = key.strip()
        scalar = scalar.strip()
        if not key or key in mapping:
            raise Stage4SplitError(f"Invalid or duplicate YAML key at line {line_number}: {key!r}")
        index += 1
        if scalar:
            mapping[key] = _parse_yaml_scalar(scalar)
            continue
        if index >= len(entries) or entries[index][0] <= current_indent:
            raise Stage4SplitError(f"YAML key {key!r} has no nested value at line {line_number}")
        child_indent = entries[index][0]
        if child_indent != current_indent + 2:
            raise Stage4SplitError(f"Unexpected YAML nesting below line {line_number}")
        mapping[key], index = _parse_yaml_node(entries, index, child_indent)
    return mapping, index


def _parse_yaml_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_parse_yaml_scalar(item.strip()) for item in inner.split(",")]
    if value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise Stage4SplitError(f"Invalid quoted YAML scalar: {value}") from exc
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "Null", "NULL", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
