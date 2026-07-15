"""Calibrate fold-specific Stage 4 category rule policies.

Only rows assigned to ``train`` in the requested fold are allowed to supply a
quality metric or runtime.  Validation and test metric cells are deliberately
left unparsed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

from src.normroute.agent.protocol import CANDIDATE_EXPERTS
from src.normroute.policies.rule_based import (
    CATEGORY_PRIOR,
    CATEGORY_SHOT_PRIOR,
    RULE_POLICY_NAMES,
    RULE_POLICY_PROTOCOL_VERSION,
    RUNTIME_TIE_BREAK,
    SUPPORTED_METRICS,
    load_rule_policy_artifact,
)


MANIFEST_REQUIRED_COLUMNS = (
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
QUALITY_IDENTITY_COLUMNS = (
    "expert",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
)
RUNTIME_COLUMN_CANDIDATES = (
    "average_runtime_ms",
    "runtime_ms",
    "mean_runtime_ms",
)
ALLOWED_SPLITS = frozenset({"train", "val", "test"})
FLOAT_TOLERANCE = 1e-12

RunKey = tuple[str, str, int, int, str]


class PolicyCalibrationError(ValueError):
    """Raised when safe fold-specific calibration cannot be completed."""


@dataclass(frozen=True)
class TrainingQualityRow:
    expert: str
    dataset: str
    category: str
    k_shot: int
    seed: int
    support_set_id: str
    metric_value: float
    runtime_ms: float | None

    @property
    def run_key(self) -> RunKey:
        return (
            self.dataset,
            self.category,
            self.k_shot,
            self.seed,
            self.support_set_id,
        )

    def hash_record(self) -> dict[str, Any]:
        return {
            "expert": self.expert,
            "dataset": self.dataset,
            "category": self.category,
            "k_shot": self.k_shot,
            "seed": self.seed,
            "support_set_id": self.support_set_id,
            "metric_value": self.metric_value,
            "runtime_ms": self.runtime_ms,
        }


@dataclass(frozen=True)
class ManifestFold:
    fold: str
    run_splits: dict[RunKey, str]
    train_seeds: tuple[int, ...]
    training_manifest_runs_sha256: str

    @property
    def training_run_keys(self) -> set[RunKey]:
        return {key for key, split in self.run_splits.items() if split == "train"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate fold-specific Stage 4 category priors from train rows only."
        )
    )
    parser.add_argument(
        "--expert-quality-by-run",
        default="outputs/stage3/quality/expert_quality_by_run.csv",
    )
    parser.add_argument(
        "--fold-manifest",
        default="outputs/stage4/splits/fold_manifest.csv",
    )
    parser.add_argument(
        "--policy",
        choices=RULE_POLICY_NAMES,
        default=CATEGORY_SHOT_PRIOR,
    )
    parser.add_argument(
        "--metric",
        choices=SUPPORTED_METRICS,
        default="image_auroc",
    )
    parser.add_argument(
        "--fold",
        dest="folds",
        action="append",
        help="Fold to calibrate; repeat for multiple folds. Defaults to every manifest fold.",
    )
    parser.add_argument(
        "--runtime-column",
        help=(
            "Training runtime column used for metric ties. By default, auto-detects "
            "average_runtime_ms, runtime_ms, or mean_runtime_ms."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="outputs/stage4/policies",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        paths = calibrate_policy_artifacts(
            expert_quality_by_run=args.expert_quality_by_run,
            fold_manifest=args.fold_manifest,
            output_root=args.output_root,
            policy_name=args.policy,
            metric=args.metric,
            folds=args.folds,
            runtime_column=args.runtime_column,
        )
    except (OSError, PolicyCalibrationError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    for path in paths:
        print(f"Wrote {path}")


def calibrate_policy_artifacts(
    *,
    expert_quality_by_run: str | Path,
    fold_manifest: str | Path,
    output_root: str | Path,
    policy_name: str = CATEGORY_SHOT_PRIOR,
    metric: str = "image_auroc",
    folds: Sequence[str] | None = None,
    runtime_column: str | None = None,
) -> list[Path]:
    """Calibrate and write one ``policy_artifact.json`` per requested fold."""

    _validate_policy_and_metric(policy_name, metric)
    requested_folds = _normalize_requested_folds(folds)
    manifest_folds = read_manifest_folds(fold_manifest, folds=requested_folds)
    training_rows = read_fold_training_quality(
        expert_quality_by_run,
        manifest_folds=manifest_folds,
        metric=metric,
        runtime_column=runtime_column,
    )

    root = Path(output_root)
    paths: list[Path] = []
    for fold, manifest in manifest_folds.items():
        artifact = build_policy_artifact(
            policy_name=policy_name,
            metric=metric,
            manifest=manifest,
            training_rows=training_rows[fold],
        )
        destination = root / policy_name / fold / "policy_artifact.json"
        _ensure_safe_output(destination)
        paths.append(write_policy_artifact(artifact, destination))
    return paths


def calibrate_policy(
    *,
    expert_quality_by_run: str | Path,
    fold_manifest: str | Path,
    fold: str,
    output_path: str | Path,
    policy_name: str = CATEGORY_SHOT_PRIOR,
    metric: str = "image_auroc",
    runtime_column: str | None = None,
) -> Path:
    """Convenience API for calibrating exactly one fold to an explicit path."""

    _validate_policy_and_metric(policy_name, metric)
    manifests = read_manifest_folds(fold_manifest, folds=(fold,))
    rows = read_fold_training_quality(
        expert_quality_by_run,
        manifest_folds=manifests,
        metric=metric,
        runtime_column=runtime_column,
    )
    artifact = build_policy_artifact(
        policy_name=policy_name,
        metric=metric,
        manifest=manifests[fold],
        training_rows=rows[fold],
    )
    destination = Path(output_path)
    if destination.is_dir():
        destination = destination / "policy_artifact.json"
    _ensure_safe_output(destination)
    return write_policy_artifact(artifact, destination)


def read_manifest_folds(
    path: str | Path,
    *,
    folds: Sequence[str] | None = None,
) -> dict[str, ManifestFold]:
    """Collapse task-level manifest rows into run-level split assignments."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise PolicyCalibrationError(f"Fold manifest does not exist: {manifest_path}")
    requested = set(_normalize_requested_folds(folds) or ())
    run_splits: dict[str, dict[RunKey, str]] = {}
    task_ids: dict[str, set[str]] = {}
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in MANIFEST_REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise PolicyCalibrationError(
                f"{manifest_path} is missing manifest columns: {missing}"
            )
        for line_number, raw in enumerate(reader, start=2):
            fold = (raw.get("fold") or "").strip()
            if requested and fold not in requested:
                continue
            if not fold:
                raise PolicyCalibrationError(
                    f"{manifest_path}:{line_number} has an empty fold"
                )
            clean = {key: (value or "").strip() for key, value in raw.items()}
            missing_values = [
                column for column in MANIFEST_REQUIRED_COLUMNS if not clean.get(column)
            ]
            if missing_values:
                raise PolicyCalibrationError(
                    f"{manifest_path}:{line_number} has empty values: {missing_values}"
                )
            split = clean["split"]
            if split not in ALLOWED_SPLITS:
                raise PolicyCalibrationError(
                    f"{manifest_path}:{line_number} has invalid split={split!r}"
                )
            key = _run_key(clean, manifest_path, line_number)
            fold_splits = run_splits.setdefault(fold, {})
            existing = fold_splits.get(key)
            if existing is not None and existing != split:
                raise PolicyCalibrationError(
                    f"{manifest_path} assigns run {key!r} to both {existing!r} and {split!r} "
                    f"in {fold}"
                )
            fold_splits[key] = split
            seen_ids = task_ids.setdefault(fold, set())
            if clean["task_id"] in seen_ids:
                raise PolicyCalibrationError(
                    f"{manifest_path} assigns task_id={clean['task_id']!r} more than once in {fold}"
                )
            seen_ids.add(clean["task_id"])

    if requested:
        missing_folds = sorted(requested - set(run_splits))
        if missing_folds:
            raise PolicyCalibrationError(
                f"{manifest_path} has no rows for requested folds: {missing_folds}"
            )
    if not run_splits:
        raise PolicyCalibrationError(f"{manifest_path} has no usable fold rows")

    result: dict[str, ManifestFold] = {}
    for fold in sorted(run_splits, key=_natural_fold_key):
        splits = run_splits[fold]
        training_keys = sorted(key for key, split in splits.items() if split == "train")
        if not training_keys:
            raise PolicyCalibrationError(f"{manifest_path} {fold} has no train runs")
        train_seeds = tuple(sorted({key[3] for key in training_keys}))
        result[fold] = ManifestFold(
            fold=fold,
            run_splits=splits,
            train_seeds=train_seeds,
            training_manifest_runs_sha256=_canonical_sha256(
                [_run_key_record(key) for key in training_keys]
            ),
        )
    return result


def read_fold_training_quality(
    path: str | Path,
    *,
    manifest_folds: Mapping[str, ManifestFold],
    metric: str = "image_auroc",
    runtime_column: str | None = None,
) -> dict[str, list[TrainingQualityRow]]:
    """Read metric/runtime cells only when that run is train in the given fold."""

    if metric not in SUPPORTED_METRICS:
        raise PolicyCalibrationError(
            f"metric must be one of {list(SUPPORTED_METRICS)!r}"
        )
    quality_path = Path(path)
    if not quality_path.is_file():
        raise PolicyCalibrationError(
            f"expert_quality_by_run.csv does not exist: {quality_path}"
        )
    rows_by_fold: dict[str, list[TrainingQualityRow]] = {
        fold: [] for fold in manifest_folds
    }
    seen_by_fold: dict[str, set[tuple[RunKey, str]]] = {
        fold: set() for fold in manifest_folds
    }
    with quality_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = (*QUALITY_IDENTITY_COLUMNS, metric)
        missing = [column for column in required if column not in fieldnames]
        if missing:
            raise PolicyCalibrationError(
                f"{quality_path} is missing quality columns: {missing}"
            )
        selected_runtime_column = _resolve_runtime_column(
            fieldnames,
            requested=runtime_column,
            path=quality_path,
        )
        for line_number, raw in enumerate(reader, start=2):
            clean_identity = {
                column: (raw.get(column) or "").strip()
                for column in QUALITY_IDENTITY_COLUMNS
            }
            missing_values = [
                column for column, value in clean_identity.items() if not value
            ]
            if missing_values:
                raise PolicyCalibrationError(
                    f"{quality_path}:{line_number} has empty identity values: {missing_values}"
                )
            key = _run_key(clean_identity, quality_path, line_number)
            expert = _canonical_expert(clean_identity["expert"], quality_path, line_number)
            matched_any_fold = False
            for fold, manifest in manifest_folds.items():
                split = manifest.run_splits.get(key)
                if split is None:
                    continue
                matched_any_fold = True
                if split != "train":
                    # This is the isolation boundary: no val/test quality or
                    # runtime cell is parsed, validated, hashed, or persisted.
                    continue
                row_key = (key, expert)
                if row_key in seen_by_fold[fold]:
                    raise PolicyCalibrationError(
                        f"{quality_path} has duplicate train row for fold={fold}, "
                        f"run={key!r}, expert={expert}"
                    )
                seen_by_fold[fold].add(row_key)
                metric_value = _finite_number(
                    raw.get(metric),
                    field=metric,
                    path=quality_path,
                    line_number=line_number,
                    nonnegative=False,
                )
                if not 0.0 <= metric_value <= 1.0:
                    raise PolicyCalibrationError(
                        f"{quality_path}:{line_number} training {metric} must be in [0, 1]"
                    )
                runtime_ms = None
                if selected_runtime_column is not None:
                    raw_runtime = (raw.get(selected_runtime_column) or "").strip()
                    if raw_runtime:
                        runtime_ms = _finite_number(
                            raw_runtime,
                            field=selected_runtime_column,
                            path=quality_path,
                            line_number=line_number,
                            nonnegative=True,
                        )
                rows_by_fold[fold].append(
                    TrainingQualityRow(
                        expert=expert,
                        dataset=key[0],
                        category=key[1],
                        k_shot=key[2],
                        seed=key[3],
                        support_set_id=key[4],
                        metric_value=metric_value,
                        runtime_ms=runtime_ms,
                    )
                )
            if not matched_any_fold:
                raise PolicyCalibrationError(
                    f"{quality_path}:{line_number} run {key!r} is absent from the requested "
                    "fold manifest"
                )

    for fold, manifest in manifest_folds.items():
        expected = {
            (run_key, expert)
            for run_key in manifest.training_run_keys
            for expert in CANDIDATE_EXPERTS
        }
        actual = seen_by_fold[fold]
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise PolicyCalibrationError(
                f"Training quality coverage mismatch for {fold}; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        if not rows_by_fold[fold]:
            raise PolicyCalibrationError(f"No training quality rows found for {fold}")
    return rows_by_fold


def build_policy_artifact(
    *,
    policy_name: str,
    metric: str,
    manifest: ManifestFold,
    training_rows: Sequence[TrainingQualityRow],
) -> dict[str, Any]:
    """Select global/category/category+shot experts from one fold's train rows."""

    _validate_policy_and_metric(policy_name, metric)
    if not training_rows:
        raise PolicyCalibrationError(f"No training rows supplied for {manifest.fold}")
    groups_category: dict[tuple[str, str], list[TrainingQualityRow]] = {}
    groups_shot: dict[tuple[str, str, int], list[TrainingQualityRow]] = {}
    for row in training_rows:
        if row.run_key not in manifest.training_run_keys:
            raise PolicyCalibrationError(
                f"Non-training row reached artifact builder for {manifest.fold}: {row.run_key!r}"
            )
        groups_category.setdefault((row.dataset, row.category), []).append(row)
        groups_shot.setdefault((row.dataset, row.category, row.k_shot), []).append(row)

    global_best = _select_best(training_rows, context=f"{manifest.fold} global")
    category_rules = [
        {
            "dataset": key[0],
            "category": key[1],
            "selected_expert": _select_best(
                rows, context=f"{manifest.fold} category={key!r}"
            ),
        }
        for key, rows in sorted(groups_category.items())
    ]
    category_shot_rules: list[dict[str, Any]] = []
    if policy_name == CATEGORY_SHOT_PRIOR:
        category_shot_rules = [
            {
                "dataset": key[0],
                "category": key[1],
                "k_shot": key[2],
                "selected_expert": _select_best(
                    rows, context=f"{manifest.fold} category+shot={key!r}"
                ),
            }
            for key, rows in sorted(groups_shot.items())
        ]

    canonical_training_rows = [
        row.hash_record()
        for row in sorted(
            training_rows,
            key=lambda item: (*item.run_key, CANDIDATE_EXPERTS.index(item.expert)),
        )
    ]
    artifact = {
        "protocol_version": RULE_POLICY_PROTOCOL_VERSION,
        "policy_name": policy_name,
        "fold": manifest.fold,
        "train_seeds": list(manifest.train_seeds),
        "metric": metric,
        "tie_break": RUNTIME_TIE_BREAK,
        "tie_break_details": {
            "primary": "maximize_mean_training_metric",
            "secondary": RUNTIME_TIE_BREAK,
            "tertiary": "candidate_expert_order",
            "candidate_expert_order": list(CANDIDATE_EXPERTS),
            "float_tolerance": FLOAT_TOLERANCE,
        },
        "git_commit": _git_commit(),
        "global_best": global_best,
        "category_rules": category_rules,
        "category_shot_rules": category_shot_rules,
        "fallback_order": (
            ["category", "global_best"]
            if policy_name == CATEGORY_PRIOR
            else ["category+shot", "category", "global_best"]
        ),
        "num_training_runs": len(manifest.training_run_keys),
        "num_training_rows": len(training_rows),
        "training_rows_sha256": _canonical_sha256(canonical_training_rows),
        "training_manifest_runs_sha256": manifest.training_manifest_runs_sha256,
    }
    return load_rule_policy_artifact(artifact)


def write_policy_artifact(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Validate and atomically persist a leakage-safe policy artifact."""

    artifact = load_rule_policy_artifact(payload)
    destination = Path(path)
    if destination.name != "policy_artifact.json":
        raise PolicyCalibrationError(
            f"Policy artifact filename must be 'policy_artifact.json'; got {destination.name!r}"
        )
    _ensure_safe_output(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _select_best(rows: Sequence[TrainingQualityRow], *, context: str) -> str:
    by_expert: dict[str, list[TrainingQualityRow]] = {
        expert: [] for expert in CANDIDATE_EXPERTS
    }
    for row in rows:
        by_expert[row.expert].append(row)
    missing = [expert for expert, expert_rows in by_expert.items() if not expert_rows]
    if missing:
        raise PolicyCalibrationError(f"{context} is missing experts: {missing}")

    mean_metrics = {
        expert: _mean(row.metric_value for row in expert_rows)
        for expert, expert_rows in by_expert.items()
    }
    best_metric = max(mean_metrics.values())
    tied = [
        expert
        for expert in CANDIDATE_EXPERTS
        if math.isclose(
            mean_metrics[expert],
            best_metric,
            rel_tol=FLOAT_TOLERANCE,
            abs_tol=FLOAT_TOLERANCE,
        )
    ]
    if len(tied) == 1:
        return tied[0]

    mean_runtimes: dict[str, float] = {}
    for expert in tied:
        runtimes = [row.runtime_ms for row in by_expert[expert]]
        if any(runtime is None for runtime in runtimes):
            raise PolicyCalibrationError(
                f"{context} ties {tied} on training metric, but {expert} lacks complete "
                "training runtime values required for the lower-runtime tie-break"
            )
        mean_runtimes[expert] = _mean(float(runtime) for runtime in runtimes if runtime is not None)
    fastest = min(mean_runtimes.values())
    runtime_tied = [
        expert
        for expert in tied
        if math.isclose(
            mean_runtimes[expert],
            fastest,
            rel_tol=FLOAT_TOLERANCE,
            abs_tol=FLOAT_TOLERANCE,
        )
    ]
    return min(runtime_tied, key=CANDIDATE_EXPERTS.index)


def _resolve_runtime_column(
    fieldnames: Sequence[str],
    *,
    requested: str | None,
    path: Path,
) -> str | None:
    if requested is not None:
        requested = requested.strip()
        if not requested or requested not in fieldnames:
            raise PolicyCalibrationError(
                f"{path} does not contain requested runtime column {requested!r}"
            )
        return requested
    return next((column for column in RUNTIME_COLUMN_CANDIDATES if column in fieldnames), None)


def _run_key(row: Mapping[str, Any], path: Path, line_number: int) -> RunKey:
    dataset = str(row.get("dataset", "")).strip()
    category = str(row.get("category", "")).strip()
    support_set_id = str(row.get("support_set_id", "")).strip()
    if not dataset or not category or not support_set_id:
        raise PolicyCalibrationError(
            f"{path}:{line_number} has empty run identity fields"
        )
    return (
        dataset,
        category,
        _positive_csv_int(row.get("k_shot"), "k_shot", path, line_number),
        _nonnegative_csv_int(row.get("seed"), "seed", path, line_number),
        support_set_id,
    )


def _run_key_record(key: RunKey) -> dict[str, Any]:
    return {
        "dataset": key[0],
        "category": key[1],
        "k_shot": key[2],
        "seed": key[3],
        "support_set_id": key[4],
    }


def _canonical_expert(value: str, path: Path, line_number: int) -> str:
    normalized = "".join(character for character in value.lower() if character.isalnum())
    mapping = {
        "".join(character for character in expert.lower() if character.isalnum()): expert
        for expert in CANDIDATE_EXPERTS
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise PolicyCalibrationError(
            f"{path}:{line_number} has unknown expert={value!r}"
        ) from exc


def _finite_number(
    value: Any,
    *,
    field: str,
    path: Path,
    line_number: int,
    nonnegative: bool,
) -> float:
    text = str(value or "").strip()
    try:
        parsed = float(text)
    except ValueError as exc:
        raise PolicyCalibrationError(
            f"{path}:{line_number} has invalid training {field}={text!r}"
        ) from exc
    if not math.isfinite(parsed) or (nonnegative and parsed < 0):
        qualifier = "finite and >= 0" if nonnegative else "finite"
        raise PolicyCalibrationError(
            f"{path}:{line_number} training {field} must be {qualifier}"
        )
    return parsed


def _positive_csv_int(value: Any, field: str, path: Path, line_number: int) -> int:
    parsed = _nonnegative_csv_int(value, field, path, line_number)
    if parsed < 1:
        raise PolicyCalibrationError(f"{path}:{line_number} {field} must be >= 1")
    return parsed


def _nonnegative_csv_int(value: Any, field: str, path: Path, line_number: int) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise PolicyCalibrationError(
            f"{path}:{line_number} {field} must be an integer"
        ) from exc
    if parsed < 0:
        raise PolicyCalibrationError(f"{path}:{line_number} {field} must be >= 0")
    return parsed


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise PolicyCalibrationError("Cannot average an empty training group")
    return math.fsum(materialized) / len(materialized)


def _canonical_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    serialized = json.dumps(
        list(records),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


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


def _validate_policy_and_metric(policy_name: str, metric: str) -> None:
    if policy_name not in RULE_POLICY_NAMES:
        raise PolicyCalibrationError(
            f"policy_name must be one of {list(RULE_POLICY_NAMES)!r}"
        )
    if metric not in SUPPORTED_METRICS:
        raise PolicyCalibrationError(
            f"metric must be one of {list(SUPPORTED_METRICS)!r}"
        )


def _normalize_requested_folds(folds: Sequence[str] | None) -> tuple[str, ...] | None:
    if folds is None:
        return None
    cleaned = tuple(str(fold).strip() for fold in folds)
    if not cleaned or any(not fold for fold in cleaned) or len(cleaned) != len(set(cleaned)):
        raise PolicyCalibrationError("folds must contain unique non-empty names")
    return cleaned


def _natural_fold_key(value: str) -> tuple[str, int, str]:
    prefix = value.rstrip("0123456789")
    suffix = value[len(prefix) :]
    return prefix, int(suffix) if suffix else -1, value


def _ensure_safe_output(path: Path) -> None:
    lowered = {part.lower() for part in path.parts}
    if "evaluator_only" in lowered or "oracle" in lowered:
        raise PolicyCalibrationError(
            "Rule policy artifacts must not be written under evaluator_only/oracle"
        )


if __name__ == "__main__":
    main()


__all__ = [
    "ManifestFold",
    "PolicyCalibrationError",
    "TrainingQualityRow",
    "build_policy_artifact",
    "calibrate_policy",
    "calibrate_policy_artifacts",
    "main",
    "read_fold_training_quality",
    "read_manifest_folds",
    "write_policy_artifact",
]
