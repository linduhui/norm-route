"""Audit Stage 5 inference-visible files for schema-level leakage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

from .build_stage4_splits import Stage4SplitError, _parse_simple_yaml


AUDIT_PROTOCOL_VERSION = "stage5.input_audit.v1"
SUPPORTED_SUFFIXES = frozenset({".csv", ".json", ".jsonl", ".yaml", ".yml"})
KNOWN_EXPERT_NAMES = frozenset({"patchcore", "winclip", "anomalydino"})
EXACT_FORBIDDEN_FIELDS = frozenset(
    {
        "label",
        "labels",
        "mask",
        "masks",
        "mask_path",
        "mask_paths",
        "defect_type",
        "anomaly_type",
        "test_statistics",
        "ground_truth",
        "ground_truth_label",
        "ground_truth_mask",
        "gt_label",
        "expert_score",
        "expert_scores",
        "expert_utility",
        "expert_utilities",
        "realized_expert_score",
        "realized_expert_scores",
        "patchcore_score",
        "winclip_score",
        "anomalydino_score",
        "oracle",
        "oracle_answer",
        "oracle_best_expert",
        "oracle_score",
        "oracle_utility",
        "best_expert",
        "teacher_score",
        "teacher_scores",
        "teacher_target",
        "teacher_targets",
        "teacher_utility",
        "teacher_utilities",
    }
)
FORBIDDEN_PATH_PARTS = frozenset(
    {
        "evaluator_only",
        "oracle",
        "oracles",
        "teacher_utility",
        "teacher_utilities",
        "expert_scores",
    }
)


class Stage5InputAuditError(ValueError):
    """Raised when an inference-visible artifact cannot be audited safely."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Inference-visible file(s) or directories to audit.",
    )
    parser.add_argument(
        "--report",
        default="outputs/stage5/audits/input_audit.json",
        help="Machine-readable audit report path.",
    )
    parser.add_argument(
        "--config",
        default="configs/stage5/category_cv_v2.yaml",
        help="Frozen Stage 5 config recorded as audit provenance.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_stage5_inputs(
        input_paths=[Path(value) for value in args.inputs],
        report_path=Path(args.report),
        config_path=Path(args.config),
    )
    if not report["ok"]:
        for failure in report["failures"]:
            print(f"ERROR: {failure['message']}", file=sys.stderr)
        return 1
    print(
        f"Stage 5 input audit PASS: {report['files_scanned']} file(s), "
        f"report={args.report}"
    )
    return 0


def audit_stage5_inputs(
    input_paths: Sequence[str | Path],
    *,
    report_path: str | Path | None = None,
    config_path: str | Path = "configs/stage5/category_cv_v2.yaml",
) -> dict[str, Any]:
    """Audit every supported inference-visible file and optionally write a report."""

    requested = [Path(path) for path in input_paths]
    config = Path(config_path)
    report_file = Path(report_path) if report_path is not None else None
    failures: list[dict[str, Any]] = []
    ignored_files: list[str] = []
    files = _resolve_input_files(
        requested,
        report_path=report_file,
        failures=failures,
        ignored_files=ignored_files,
    )

    file_records: list[dict[str, Any]] = []
    for path in files:
        record = {
            "path": str(path),
            "sha256": _sha256(path),
            "status": "PASS",
            "failures": [],
        }
        file_failures = audit_inference_visible_file(path)
        if file_failures:
            record["status"] = "FAIL"
            record["failures"] = file_failures
            failures.extend(file_failures)
        file_records.append(record)

    if not files:
        failures.append(
            _failure(
                code="NO_AUDITABLE_INPUTS",
                message="No supported inference-visible files were found; nothing was audited.",
            )
        )

    config_record: dict[str, Any] = {"path": str(config), "sha256": None}
    if config.is_file():
        config_record["sha256"] = _sha256(config)
    else:
        failures.append(
            _failure(
                code="CONFIG_NOT_FOUND",
                message=f"Stage 5 config does not exist: {config}",
                path=config,
            )
        )

    report: dict[str, Any] = {
        "protocol_version": AUDIT_PROTOCOL_VERSION,
        "ok": not failures,
        "input_paths": [str(path) for path in requested],
        "config": config_record,
        "files_scanned": len(file_records),
        "files": file_records,
        "ignored_unsupported_files": ignored_files,
        "forbidden_field_families": [
            "label_or_ground_truth",
            "mask",
            "defect_or_anomaly_type",
            "target_test_statistics",
            "expert_score_or_utility",
            "oracle",
            "teacher_score_target_or_utility",
        ],
        "failure_count": len(failures),
        "failures": failures,
        "provenance": {
            "git_commit": _git_commit(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    if report_file is not None:
        _write_report(report_file, report)
    return report


def audit_inference_visible_file(path: str | Path) -> list[dict[str, Any]]:
    """Return all path, parse, and forbidden-field failures for one file."""

    input_path = Path(path)
    failures: list[dict[str, Any]] = []
    path_reason = forbidden_path_reason(input_path)
    if path_reason is not None:
        failures.append(
            _failure(
                code="FORBIDDEN_INFERENCE_PATH",
                message=f"{input_path} is not an inference-visible location: {path_reason}",
                path=input_path,
            )
        )
    if not input_path.is_file():
        failures.append(
            _failure(
                code="INPUT_NOT_FOUND",
                message=f"Inference-visible input does not exist: {input_path}",
                path=input_path,
            )
        )
        return failures
    if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        failures.append(
            _failure(
                code="UNSUPPORTED_INPUT_FORMAT",
                message=f"Unsupported inference-visible input format: {input_path}",
                path=input_path,
            )
        )
        return failures

    try:
        failures.extend(_audit_file_contents(input_path))
    except (csv.Error, json.JSONDecodeError, OSError, UnicodeDecodeError, Stage4SplitError) as exc:
        failures.append(
            _failure(
                code="INPUT_PARSE_ERROR",
                message=f"Could not parse {input_path}: {type(exc).__name__}: {exc}",
                path=input_path,
            )
        )
    return failures


def find_forbidden_field_paths(value: Any, *, context: str = "$") -> list[dict[str, str]]:
    """Find semantically forbidden mapping keys at every nesting depth."""

    findings: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_context = f"{context}.{key}"
            reason = forbidden_field_reason(key)
            if reason is not None:
                findings.append({"field_path": child_context, "reason": reason})
            findings.extend(find_forbidden_field_paths(child, context=child_context))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(
                find_forbidden_field_paths(child, context=f"{context}[{index}]")
            )
    return findings


def forbidden_field_reason(field_name: str) -> str | None:
    """Classify a field name using a case-insensitive semantic denylist."""

    normalized = normalize_field_name(field_name)
    if normalized in EXACT_FORBIDDEN_FIELDS:
        return _exact_forbidden_reason(normalized)
    tokens = set(part for part in normalized.split("_") if part)
    if tokens.intersection({"label", "labels"}):
        return "label_or_ground_truth"
    if tokens.intersection({"mask", "masks"}):
        return "mask"
    if "oracle" in tokens or normalized.startswith("oracle"):
        return "oracle"
    if "teacher" in tokens and tokens.intersection(
        {"score", "scores", "target", "targets", "utility", "utilities"}
    ):
        return "teacher_score_target_or_utility"
    if "expert" in tokens and tokens.intersection(
        {"score", "scores", "outcome", "outcomes", "utility", "utilities"}
    ):
        return "expert_score_or_utility"
    if normalized.endswith("_score") and normalized.removesuffix("_score") in KNOWN_EXPERT_NAMES:
        return "expert_score_or_utility"
    if normalized.endswith("_scores") and normalized.removesuffix("_scores") in KNOWN_EXPERT_NAMES:
        return "expert_score_or_utility"
    return None


def normalize_field_name(field_name: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(field_name).strip())
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def forbidden_path_reason(path: str | Path) -> str | None:
    """Reject evaluator/Oracle/teacher outcome storage as an inference source."""

    normalized_parts: set[str] = set()
    for part in Path(path).parts:
        normalized_parts.add(normalize_field_name(part))
        normalized_parts.add(normalize_field_name(Path(part).stem))
    matched = set(normalized_parts.intersection(FORBIDDEN_PATH_PARTS))
    for part in normalized_parts:
        tokens = set(part.split("_"))
        if "oracle" in tokens:
            matched.add(part)
        if {"evaluator", "only"}.issubset(tokens):
            matched.add(part)
        if "teacher" in tokens and tokens.intersection({"score", "scores", "utility", "utilities"}):
            matched.add(part)
        if "expert" in tokens and tokens.intersection({"score", "scores", "utility", "utilities"}):
            matched.add(part)
    matched = sorted(matched)
    if matched:
        return f"forbidden path component(s) {matched}"
    return None


def _resolve_input_files(
    paths: Sequence[Path],
    *,
    report_path: Path | None,
    failures: list[dict[str, Any]],
    ignored_files: list[str],
) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    report_resolved = report_path.resolve() if report_path is not None else None
    for path in paths:
        if not path.exists():
            failures.append(
                _failure(
                    code="INPUT_NOT_FOUND",
                    message=f"Inference-visible input path does not exist: {path}",
                    path=path,
                )
            )
            continue
        candidates = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
        for candidate in candidates:
            resolved = candidate.resolve()
            if report_resolved is not None and resolved == report_resolved:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if candidate.suffix.lower() in SUPPORTED_SUFFIXES:
                files.append(candidate)
            elif path.is_file():
                failures.append(
                    _failure(
                        code="UNSUPPORTED_INPUT_FORMAT",
                        message=f"Unsupported inference-visible input format: {candidate}",
                        path=candidate,
                    )
                )
            else:
                ignored_files.append(str(candidate))
    return files


def _audit_file_contents(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise csv.Error("CSV is empty") from exc
        if not header or any(not field.strip() for field in header):
            raise csv.Error("CSV header contains an empty field name")
        if len(header) != len({normalize_field_name(field) for field in header}):
            raise csv.Error("CSV header contains duplicate normalized field names")
        return _field_failures(path, header)

    if suffix == ".jsonl":
        failures: list[dict[str, Any]] = []
        row_count = 0
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    failures.append(
                        _failure(
                            code="BLANK_JSONL_RECORD",
                            message=f"{path}:{line_number} is blank; records must not be skipped.",
                            path=path,
                            location=f"line {line_number}",
                        )
                    )
                    continue
                row_count += 1
                payload = json.loads(line)
                findings = find_forbidden_field_paths(payload)
                failures.extend(_finding_failures(path, findings, line_number=line_number))
        if row_count == 0:
            failures.append(
                _failure(
                    code="EMPTY_JSONL",
                    message=f"Inference-visible JSONL has no records: {path}",
                    path=path,
                )
            )
        return failures

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    else:
        payload = _parse_simple_yaml(path.read_text(encoding="utf-8-sig"))
    return _finding_failures(path, find_forbidden_field_paths(payload))


def _field_failures(path: Path, fields: Sequence[str]) -> list[dict[str, Any]]:
    findings = []
    for field in fields:
        reason = forbidden_field_reason(field)
        if reason is not None:
            findings.append({"field_path": f"$.{field}", "reason": reason})
    return _finding_failures(path, findings)


def _finding_failures(
    path: Path,
    findings: Sequence[Mapping[str, str]],
    *,
    line_number: int | None = None,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for finding in findings:
        location = finding["field_path"]
        if line_number is not None:
            location = f"line {line_number} {location}"
        failures.append(
            _failure(
                code="FORBIDDEN_INFERENCE_FIELD",
                message=(
                    f"{path} contains forbidden inference-visible field "
                    f"{finding['field_path']!r} ({finding['reason']})."
                ),
                path=path,
                location=location,
                field_family=finding["reason"],
            )
        )
    return failures


def _exact_forbidden_reason(normalized: str) -> str:
    if "mask" in normalized:
        return "mask"
    if normalized in {"defect_type", "anomaly_type"}:
        return "defect_or_anomaly_type"
    if normalized == "test_statistics":
        return "target_test_statistics"
    if "oracle" in normalized or normalized == "best_expert":
        return "oracle"
    if normalized.startswith("teacher_"):
        return "teacher_score_target_or_utility"
    if "expert" in normalized or normalized.startswith(tuple(f"{name}_" for name in KNOWN_EXPERT_NAMES)):
        return "expert_score_or_utility"
    return "label_or_ground_truth"


def _failure(
    *,
    code: str,
    message: str,
    path: str | Path | None = None,
    location: str | None = None,
    field_family: str | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "path": str(path) if path is not None else None,
        "location": location,
        "field_family": field_family,
    }


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


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
