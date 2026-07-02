"""Stage 1 leakage checks for NORM-Route manifest and support-set CSVs."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.data.manifest import FORBIDDEN_AGENT_COLUMNS


REQUIRED_AGENT_COLUMNS = ["image_id", "dataset", "category", "split", "image_path"]
REQUIRED_EVALUATOR_COLUMNS = ["image_id", "label"]
REQUIRED_SUPPORT_COLUMNS = [
    "support_set_id",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "image_id",
]
NORMAL_DEFECT_TYPES = {"", "good", "normal", "ok"}


@dataclass(frozen=True)
class LeakageIssue:
    code: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
        }


def check_leakage(
    *,
    agent_input_csv: str | Path,
    evaluator_csv: str | Path,
    support: str | Path,
) -> dict[str, Any]:
    """Check Stage 1 leakage invariants and return a machine-readable report."""

    agent_path = Path(agent_input_csv)
    evaluator_path = Path(evaluator_csv)
    support_path = Path(support)
    issues: list[LeakageIssue] = []

    agent_rows, agent_header = _read_csv(agent_path)
    evaluator_rows, evaluator_header = _read_csv(evaluator_path)
    support_rows, support_files = _read_support_rows(support_path)

    _check_agent_header(agent_path, agent_header, issues)
    _check_evaluator_header(evaluator_path, evaluator_header, issues)

    agent_ids = _column_values(agent_rows, "image_id")
    evaluator_ids = _column_values(evaluator_rows, "image_id")
    _check_unique(agent_ids, "agent_input", "AGENT_IMAGE_ID_NOT_UNIQUE", issues)
    _check_unique(evaluator_ids, "evaluator", "EVALUATOR_IMAGE_ID_NOT_UNIQUE", issues)
    _check_join_row_count(agent_ids, evaluator_ids, issues)

    agent_by_id = _last_by_id(agent_rows)
    evaluator_by_id = _last_by_id(evaluator_rows)
    _check_support_header(support_files, support_rows, issues)
    _check_support_rows(support_rows, agent_by_id, evaluator_by_id, issues)

    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    report = {
        "ok": error_count == 0,
        "error_count": error_count,
        "warning_count": warning_count,
        "inputs": {
            "agent_input_csv": str(agent_path),
            "evaluator_csv": str(evaluator_path),
            "support": str(support_path),
            "support_files": [str(path) for path in support_files],
        },
        "counts": {
            "agent_input_rows": len(agent_rows),
            "evaluator_rows": len(evaluator_rows),
            "support_rows": len(support_rows),
            "support_sets": len({row.get("support_set_id", "") for row in support_rows if row.get("support_set_id", "")}),
        },
        "issues": [issue.to_dict() for issue in issues],
    }
    report["summary"] = build_summary(report)
    return report


def build_summary(report: dict[str, Any]) -> str:
    status = "PASS" if report["ok"] else "FAIL"
    lines = [
        f"NORM-Route leakage check: {status}",
        (
            f"agent_input={report['counts']['agent_input_rows']} rows, "
            f"evaluator={report['counts']['evaluator_rows']} rows, "
            f"support={report['counts']['support_rows']} rows, "
            f"support_sets={report['counts']['support_sets']}"
        ),
        f"errors={report['error_count']}, warnings={report['warning_count']}",
    ]
    if report["issues"]:
        lines.append("issues:")
        for issue in report["issues"]:
            lines.append(f"- [{issue['severity']}] {issue['code']}: {issue['message']}")
    return "\n".join(lines)


def write_json_report(path: str | Path, report: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def write_summary(path: str | Path, report: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report["summary"] + "\n", encoding="utf-8")
    return output_path


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
        return rows, list(reader.fieldnames)


def _read_support_rows(path: Path) -> tuple[list[dict[str, str]], list[Path]]:
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No support CSV files found: {path}")

    rows: list[dict[str, str]] = []
    for file_path in files:
        file_rows, _ = _read_csv(file_path)
        for row in file_rows:
            row["_source_csv"] = str(file_path)
        rows.extend(file_rows)
    return rows, files


def _check_agent_header(path: Path, header: list[str], issues: list[LeakageIssue]) -> None:
    fields = set(header)
    forbidden = sorted(fields & FORBIDDEN_AGENT_COLUMNS)
    if forbidden:
        issues.append(
            LeakageIssue(
                "AGENT_INPUT_FORBIDDEN_COLUMNS",
                "error",
                f"{path} contains evaluator-only columns: {forbidden}",
                {"columns": forbidden},
            )
        )
    missing = [column for column in REQUIRED_AGENT_COLUMNS if column not in fields]
    if missing:
        issues.append(
            LeakageIssue(
                "AGENT_INPUT_MISSING_COLUMNS",
                "error",
                f"{path} is missing required agent-input columns: {missing}",
                {"columns": missing},
            )
        )


def _check_evaluator_header(path: Path, header: list[str], issues: list[LeakageIssue]) -> None:
    fields = set(header)
    missing = [column for column in REQUIRED_EVALUATOR_COLUMNS if column not in fields]
    if missing:
        issues.append(
            LeakageIssue(
                "EVALUATOR_MISSING_COLUMNS",
                "error",
                f"{path} is missing required evaluator columns: {missing}",
                {"columns": missing},
            )
        )


def _check_support_header(
    support_files: list[Path],
    support_rows: list[dict[str, str]],
    issues: list[LeakageIssue],
) -> None:
    for file_path in support_files:
        _, header = _read_csv(file_path)
        fields = set(header)
        missing = [column for column in REQUIRED_SUPPORT_COLUMNS if column not in fields]
        if missing:
            issues.append(
                LeakageIssue(
                    "SUPPORT_MISSING_COLUMNS",
                    "error",
                    f"{file_path} is missing required support columns: {missing}",
                    {"file": str(file_path), "columns": missing},
                )
            )
    if not support_rows:
        issues.append(LeakageIssue("SUPPORT_EMPTY", "error", "support set CSV has no rows"))


def _column_values(rows: Iterable[dict[str, str]], column: str) -> list[str]:
    return [row.get(column, "") for row in rows]


def _check_unique(
    values: list[str],
    source: str,
    code: str,
    issues: list[LeakageIssue],
) -> None:
    duplicates = sorted(value for value, count in Counter(values).items() if value and count > 1)
    if duplicates:
        issues.append(
            LeakageIssue(
                code,
                "error",
                f"{source} contains duplicate image_id values: {duplicates[:10]}",
                {"duplicates": duplicates, "duplicate_count": len(duplicates)},
            )
        )


def _check_join_row_count(
    agent_ids: list[str],
    evaluator_ids: list[str],
    issues: list[LeakageIssue],
) -> None:
    evaluator_counts = Counter(evaluator_ids)
    joined_count = sum(evaluator_counts[image_id] for image_id in agent_ids)
    if joined_count != len(agent_ids):
        missing = sorted(set(agent_ids) - set(evaluator_ids))
        extra = sorted(set(evaluator_ids) - set(agent_ids))
        issues.append(
            LeakageIssue(
                "AGENT_EVALUATOR_JOIN_ROW_COUNT_CHANGED",
                "error",
                "agent_input joined to evaluator by image_id would change row count",
                {
                    "agent_input_rows": len(agent_ids),
                    "joined_rows": joined_count,
                    "missing_in_evaluator": missing,
                    "extra_in_evaluator": extra,
                },
            )
        )


def _last_by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("image_id", ""): row for row in rows if row.get("image_id", "")}


def _check_support_rows(
    support_rows: list[dict[str, str]],
    agent_by_id: dict[str, dict[str, str]],
    evaluator_by_id: dict[str, dict[str, str]],
    issues: list[LeakageIssue],
) -> None:
    rows_by_set: dict[str, list[dict[str, str]]] = defaultdict(list)
    ids_by_signature: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)

    for row in support_rows:
        support_set_id = row.get("support_set_id", "")
        rows_by_set[support_set_id].append(row)
        ids_by_signature[
            (
                row.get("dataset", ""),
                row.get("category", ""),
                row.get("k_shot", ""),
                row.get("seed", ""),
            )
        ].add(support_set_id)

        image_id = row.get("image_id", "")
        agent_row = agent_by_id.get(image_id)
        evaluator_row = evaluator_by_id.get(image_id)
        if agent_row is None:
            issues.append(
                LeakageIssue(
                    "SUPPORT_IMAGE_ID_NOT_IN_AGENT_INPUT",
                    "error",
                    f"support image_id {image_id!r} is not present in agent_input",
                    _support_detail(row),
                )
            )
            continue

        if agent_row.get("split") != "train":
            issues.append(
                LeakageIssue(
                    "SUPPORT_NOT_TRAIN_SPLIT",
                    "error",
                    f"support image_id {image_id!r} has split={agent_row.get('split')!r}, expected train",
                    _support_detail(row, agent_row),
                )
            )
        if agent_row.get("split") == "test":
            issues.append(
                LeakageIssue(
                    "SUPPORT_USES_TEST_IMAGE",
                    "error",
                    f"support image_id {image_id!r} uses a test image",
                    _support_detail(row, agent_row),
                )
            )

        if not _is_normal_train_support(row, agent_row, evaluator_row):
            issues.append(
                LeakageIssue(
                    "SUPPORT_NOT_NORMAL_TRAIN_GOOD",
                    "error",
                    f"support image_id {image_id!r} is not proven to be a normal train/good image",
                    _support_detail(row, agent_row, evaluator_row),
                )
            )

    for support_set_id, rows in sorted(rows_by_set.items()):
        image_ids = [row.get("image_id", "") for row in rows]
        duplicates = sorted(value for value, count in Counter(image_ids).items() if value and count > 1)
        if duplicates:
            issues.append(
                LeakageIssue(
                    "SUPPORT_SET_DUPLICATE_IMAGE_ID",
                    "error",
                    f"support_set_id {support_set_id!r} contains duplicate image_id values",
                    {"support_set_id": support_set_id, "duplicates": duplicates},
                )
            )

        k_values = {row.get("k_shot", "") for row in rows}
        for k_value in sorted(k_values):
            k_rows = [row for row in rows if row.get("k_shot", "") == k_value]
            expected = _parse_positive_int(k_value)
            if expected is None or len(k_rows) != expected:
                issues.append(
                    LeakageIssue(
                        "SUPPORT_K_MISMATCH",
                        "error",
                        f"support_set_id {support_set_id!r} declares K={k_value!r} but has {len(k_rows)} rows",
                        {
                            "support_set_id": support_set_id,
                            "k_shot": k_value,
                            "actual_count": len(k_rows),
                        },
                    )
                )

    for signature, support_set_ids in sorted(ids_by_signature.items()):
        ids = sorted(support_set_ids)
        if len(ids) > 1:
            dataset, category, k_shot, seed = signature
            issues.append(
                LeakageIssue(
                    "SUPPORT_SET_ID_NOT_UNIQUE_FOR_SIGNATURE",
                    "error",
                    "same dataset/category/K/seed maps to multiple support_set_id values",
                    {
                        "dataset": dataset,
                        "category": category,
                        "k_shot": k_shot,
                        "seed": seed,
                        "support_set_ids": ids,
                    },
                )
            )


def _is_normal_train_support(
    support_row: dict[str, str],
    agent_row: dict[str, str],
    evaluator_row: dict[str, str] | None,
) -> bool:
    if agent_row.get("split") != "train":
        return False

    support_path = support_row.get("image_path", "")
    agent_path = agent_row.get("image_path", "")
    if _path_looks_train_good(support_path) or _path_looks_train_good(agent_path):
        return True

    if evaluator_row is None:
        return False
    return (
        evaluator_row.get("label") in {"0", 0}
        and evaluator_row.get("defect_type", "").strip().lower() in NORMAL_DEFECT_TYPES
    )


def _path_looks_train_good(path_value: str) -> bool:
    parts = tuple(part.lower() for part in re.split(r"[\\/]+", path_value) if part)
    return (
        _has_adjacent_parts(parts, "train", "good")
        or _has_adjacent_parts(parts, "train", "normal")
        or _has_subsequence(parts, ("data", "images", "normal"))
    )


def _has_adjacent_parts(parts: tuple[str, ...], first: str, second: str) -> bool:
    return any(left == first and right == second for left, right in zip(parts, parts[1:]))


def _has_subsequence(parts: tuple[str, ...], subsequence: tuple[str, ...]) -> bool:
    return any(
        parts[start : start + len(subsequence)] == subsequence
        for start in range(0, len(parts) - len(subsequence) + 1)
    )


def _parse_positive_int(value: str) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _support_detail(
    support_row: dict[str, str],
    agent_row: dict[str, str] | None = None,
    evaluator_row: dict[str, str] | None = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "support_set_id": support_row.get("support_set_id", ""),
        "image_id": support_row.get("image_id", ""),
        "source_csv": support_row.get("_source_csv", ""),
    }
    if agent_row is not None:
        details["agent_split"] = agent_row.get("split", "")
        details["agent_image_path"] = agent_row.get("image_path", "")
    if evaluator_row is not None:
        details["evaluator_label"] = evaluator_row.get("label", "")
        details["evaluator_defect_type"] = evaluator_row.get("defect_type", "")
    return details
