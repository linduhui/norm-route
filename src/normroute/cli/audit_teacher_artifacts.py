"""Audit teacher/ECPB schemas, fold isolation, weighting, and soft targets."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from ..router.expert_bank import read_capability_bank
from ..router.teacher import read_teacher_parquet
from .stage5_artifacts import atomic_write_json, file_sha256


_FORBIDDEN_BANK_KEYS = frozenset(
    {"task_id", "image_id", "sample_id", "label", "raw_expert_score", "teacher"}
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-data", required=True)
    parser.add_argument("--capability-bank", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--fold", required=True, choices=tuple(f"fold{i}" for i in range(5)))
    parser.add_argument("--output")
    return parser.parse_args(argv)


def audit_teacher_artifacts(
    *,
    teacher_data: str | Path,
    capability_bank: str | Path,
    fold_manifest: str | Path,
    fold: str,
) -> dict[str, Any]:
    rows = read_teacher_parquet(teacher_data)
    bank = read_capability_bank(capability_bank)
    split_categories = _manifest_categories(fold_manifest, fold)
    teacher_categories = {str(row["category"]) for row in rows}
    checks: dict[str, bool] = {
        "teacher_rows_are_train_only": all(
            row.get("split") == "train" and row.get("evaluator_only") is True for row in rows
        ),
        "teacher_fold_matches": all(str(row.get("fold")) == fold for row in rows),
        "teacher_categories_match_manifest_train": teacher_categories == split_categories["train"],
        "teacher_excludes_val_categories": not teacher_categories.intersection(split_categories["val"]),
        "teacher_excludes_test_categories": not teacher_categories.intersection(split_categories["test"]),
        "teacher_runtime_complete": all(
            row.get("runtime_ms") is not None
            and math.isfinite(float(row["runtime_ms"]))
            and float(row["runtime_ms"]) >= 0.0
            for row in rows
        ),
        "crossfit_excludes_own_category": _crossfit_excludes_own_category(rows),
        "soft_distributions_sum_to_one": _soft_distributions_valid(rows),
        "one_hard_oracle_per_task": _hard_oracles_valid(rows),
        "sample_weights_consistent_across_experts": _weights_consistent(rows),
        "category_weight_mass_balanced": _category_weight_mass_balanced(rows),
        "query_weight_mass_balanced_within_category": _query_weight_mass_balanced(rows),
        "bank_fold_matches": bank.get("fold") == fold,
        "bank_categories_match_teacher": set(bank.get("train_categories", ())) == teacher_categories,
        "bank_contains_no_query_outcomes": not _forbidden_paths(bank),
        "bank_profiles_cover_teacher_experts": set(bank.get("profiles", {}))
        == {str(row["expert_name"]) for row in rows},
        "bank_latency_profiles_complete": _bank_latency_profiles_complete(bank),
        "bank_runtime_provenance_is_train_only": _bank_runtime_provenance_is_train_only(
            bank, teacher_categories
        ),
        "bank_stage2_runtime_cross_check": bool(
            bank.get("runtime_provenance", {}).get("stage2_summary_cross_check")
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "protocol_version": "stage5.teacher_artifact_audit.v2",
        "ok": not failures,
        "fold": fold,
        "checks": checks,
        "failures": failures,
        "counts": {
            "teacher_rows": len(rows),
            "teacher_tasks": len({str(row["task_id"]) for row in rows}),
            "train_categories": len(teacher_categories),
            "experts": len({str(row["expert_name"]) for row in rows}),
        },
        "input_hashes": {
            "teacher_data": file_sha256(teacher_data),
            "capability_bank": file_sha256(capability_bank),
            "fold_manifest": file_sha256(fold_manifest),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit_teacher_artifacts(
            teacher_data=args.teacher_data,
            capability_bank=args.capability_bank,
            fold_manifest=args.fold_manifest,
            fold=args.fold,
        )
    except Exception as exc:
        report = {
            "protocol_version": "stage5.teacher_artifact_audit.v2",
            "ok": False,
            "fold": args.fold,
            "checks": {},
            "failures": [f"{type(exc).__name__}: {exc}"],
        }
    output = Path(args.output) if args.output else Path(args.teacher_data).parent / "dataset_audit.json"
    atomic_write_json(output, report)
    if not report["ok"]:
        print(f"Teacher artifact audit failed: {report['failures']}", file=sys.stderr)
        return 1
    print(f"Teacher artifact audit PASS: fold={args.fold} output={output}")
    return 0


def _manifest_categories(path: str | Path, fold: str) -> dict[str, set[str]]:
    result = {split: set() for split in ("train", "val", "test")}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"fold", "split", "category"}.issubset(reader.fieldnames or ()):
            raise ValueError("fold manifest lacks fold/split/category")
        for row in reader:
            if str(row.get("fold", "")).strip() != fold:
                continue
            split = str(row.get("split", "")).strip()
            category = str(row.get("category", "")).strip()
            if split not in result or not category:
                raise ValueError("fold manifest contains invalid split/category")
            result[split].add(category)
    if any(not values for values in result.values()):
        raise ValueError(f"fold manifest does not contain all splits for {fold}")
    return result


def _by_task(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row["task_id"])].append(row)
    return result


def _crossfit_excludes_own_category(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        if row.get("calibration_scope") != "leave_one_train_category_out":
            return False
        try:
            categories = json.loads(str(row.get("calibration_fit_categories", "")))
        except json.JSONDecodeError:
            return False
        if not isinstance(categories, list) or str(row["category"]) in categories:
            return False
    return True


def _soft_distributions_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        math.isclose(
            sum(float(row["soft_utility_probability"]) for row in task_rows),
            1.0,
            rel_tol=1e-7,
            abs_tol=1e-7,
        )
        for task_rows in _by_task(rows).values()
    )


def _hard_oracles_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(sum(bool(row["hard_oracle"]) for row in task_rows) == 1 for task_rows in _by_task(rows).values())


def _weights_consistent(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        len({float(row["sample_weight"]) for row in task_rows}) == 1
        for task_rows in _by_task(rows).values()
    )


def _unique_task_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [task_rows[0] for task_rows in _by_task(rows).values()]


def _category_weight_mass_balanced(rows: Sequence[Mapping[str, Any]]) -> bool:
    totals: dict[str, float] = defaultdict(float)
    for row in _unique_task_rows(rows):
        totals[str(row["category"])] += float(row["sample_weight"])
    values = list(totals.values())
    return bool(values) and max(values) - min(values) <= 1e-7 * max(1.0, max(values))


def _query_weight_mass_balanced(rows: Sequence[Mapping[str, Any]]) -> bool:
    totals: dict[tuple[str, str], float] = defaultdict(float)
    for row in _unique_task_rows(rows):
        totals[(str(row["category"]), str(row["query_group_id"]))] += float(row["sample_weight"])
    by_category: dict[str, list[float]] = defaultdict(list)
    for (category, _), total in totals.items():
        by_category[category].append(total)
    return all(max(values) - min(values) <= 1e-7 * max(1.0, max(values)) for values in by_category.values())


def _forbidden_paths(value: Any, path: str = "root") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN_BANK_KEYS:
                findings.append(f"{path}.{key}")
            findings.extend(_forbidden_paths(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_forbidden_paths(child, f"{path}[{index}]"))
    return findings


def _bank_latency_profiles_complete(bank: Mapping[str, Any]) -> bool:
    profiles = bank.get("profiles", {})
    if not isinstance(profiles, Mapping) or not profiles:
        return False
    for profile in profiles.values():
        if not isinstance(profile, Mapping):
            return False
        values = (profile.get("latency_p50"), profile.get("latency_p95"))
        try:
            parsed = tuple(float(value) for value in values)
        except (TypeError, ValueError):
            return False
        if any(not math.isfinite(value) or value <= 0.0 for value in parsed):
            return False
        if parsed[1] < parsed[0]:
            return False
    return True


def _bank_runtime_provenance_is_train_only(
    bank: Mapping[str, Any], teacher_categories: set[str]
) -> bool:
    provenance = bank.get("runtime_provenance", {})
    return (
        isinstance(provenance, Mapping)
        and provenance.get("category_scope") == "train_only"
        and set(provenance.get("train_categories", ())) == teacher_categories
        and float(provenance.get("teacher_runtime_coverage", 0.0)) == 1.0
    )


if __name__ == "__main__":
    raise SystemExit(main())
