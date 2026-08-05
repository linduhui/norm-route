"""Audit FBDP-AD artifacts without opening the evaluator channel.

This command is intentionally label-free.  It validates task coverage,
artifact provenance and numerical invariants, then reports descriptive gate,
prototype, fallback and residual statistics.  Warnings are diagnostic only
unless ``--fail-on-warning`` is requested.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from ..router.fbdp_ad import FBDP_AD_PROTOCOL_VERSION
from ..router.fbdp_ad_pipeline import (
    FBDP_AD_SIGNATURE_COLUMNS,
    FBDP_AD_SIGNATURE_PROTOCOL_VERSION,
    read_tasks_jsonl,
)


FBDP_AD_AUDIT_PROTOCOL_VERSION = "stage5.fbdp_ad_audit.v1"
AUDIT_REPORT_NAME = "fbdp_ad_audit.json"
GROUP_STATISTICS_NAME = "fbdp_ad_audit_groups.csv"
RUN_RECORD_NAME = "fbdp_ad_audit_run.json"

_FORBIDDEN_FIELDS = frozenset(
    {
        "label",
        "labels",
        "mask",
        "masks",
        "defect_type",
        "defecttype",
        "anomaly_type",
        "anomalytype",
        "target_label",
        "ground_truth",
        "groundtruth",
    }
)
_UNIT_INTERVAL_FIELDS = (
    "foreground_candidate_ratio",
    "background_candidate_ratio",
    "foreground_background_confusion",
    "objectness_gate",
    "objectness_contrast",
    "support_reliability",
    "support_assignment_confidence",
    "prototype_compactness",
    "foreground_support_coverage",
)
_NONNEGATIVE_FIELDS = (
    "residual_q50",
    "residual_q90",
    "residual_q95",
    "residual_q99",
    "gated_residual_q50",
    "gated_residual_q90",
    "gated_residual_q95",
    "gated_residual_q99",
    "residual_mean",
    "residual_max",
    "gated_residual_mean",
    "gated_residual_max",
    "assignment_entropy_mean",
    "assignment_entropy_max",
)
_GROUP_COLUMNS = (
    "dataset",
    "category",
    "k_shot",
    "task_count",
    "gate_mean",
    "gate_min",
    "gate_max",
    "fbc_mean",
    "reliability_mean",
    "fallback_rate",
    "consistency_valid_rate",
    "loo_valid_rate",
    "foreground_prototype_mean",
    "background_prototype_mean",
    "residual_q95_mean",
    "gated_residual_q95_mean",
    "assignment_entropy_mean",
)


class FBDPADAuditError(ValueError):
    """Raised when FBDP artifacts violate a hard acceptance invariant."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="Stage 5 task JSONL.")
    parser.add_argument("--signatures", required=True, help="FBDP signature JSONL.")
    parser.add_argument("--failures", required=True, help="FBDP failure JSON array.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gate-low", type=float, default=0.02)
    parser.add_argument("--gate-high", type=float, default=0.98)
    parser.add_argument("--max-fallback-rate", type=float, default=0.25)
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / AUDIT_REPORT_NAME
    groups_path = output_dir / GROUP_STATISTICS_NAME
    failures: list[dict[str, str]] = []
    report: dict[str, Any] | None = None
    try:
        _validate_thresholds(args)
        tasks = read_tasks_jsonl(args.tasks)
        signatures = _read_jsonl(args.signatures)
        recorded_failures = _read_failure_array(args.failures)
        report, group_rows = audit_fbdp_ad_artifacts(
            tasks,
            signatures,
            recorded_failures,
            gate_low=args.gate_low,
            gate_high=args.gate_high,
            max_fallback_rate=args.max_fallback_rate,
        )
        _atomic_write_json(report_path, report)
        _atomic_write_csv(groups_path, _GROUP_COLUMNS, group_rows)
        if args.fail_on_warning and report["warnings"]:
            raise FBDPADAuditError(
                "FBDP audit produced warnings and --fail-on-warning was set"
            )
    except Exception as exc:
        failures.append({"code": type(exc).__name__, "message": str(exc)})

    run_record = {
        "protocol_version": FBDP_AD_AUDIT_PROTOCOL_VERSION,
        "run_kind": "label_free_artifact_audit",
        "ok": not failures,
        "config": dict(vars(args)),
        "seed": args.seed,
        "git_commit": _git_commit(),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "input_hashes": {
            name: _file_sha256(getattr(args, name))
            for name in ("tasks", "signatures", "failures")
            if Path(getattr(args, name)).is_file()
        },
        "predictions": str(report_path) if report_path.is_file() else None,
        "outputs": {
            "audit": str(report_path) if report_path.is_file() else None,
            "group_statistics": str(groups_path) if groups_path.is_file() else None,
        },
        "failures": failures,
    }
    _atomic_write_json(output_dir / RUN_RECORD_NAME, run_record)
    if failures:
        print(f"FBDP-AD audit failed: {failures[0]['message']}", file=sys.stderr)
        return 1
    assert report is not None
    print(
        "FBDP-AD audit PASS: "
        f"tasks={report['coverage']['task_count']}, "
        f"warnings={len(report['warnings'])}, report={report_path}"
    )
    return 0


def audit_fbdp_ad_artifacts(
    tasks: Sequence[Mapping[str, Any]],
    signatures: Sequence[Mapping[str, Any]],
    recorded_failures: Sequence[Mapping[str, Any]],
    *,
    gate_low: float = 0.02,
    gate_high: float = 0.98,
    max_fallback_rate: float = 0.25,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a hard-invariant audit plus label-free descriptive warnings."""

    if not tasks:
        raise FBDPADAuditError("tasks must be non-empty")
    task_ids = [_text(row.get("task_id"), "task_id") for row in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise FBDPADAuditError("task manifest contains duplicate task_id")
    task_by_id = dict(zip(task_ids, tasks))
    signature_by_id = _unique_records(signatures, "signature")
    failure_by_id = _unique_records(recorded_failures, "failure")
    overlap = set(signature_by_id).intersection(failure_by_id)
    if overlap:
        raise FBDPADAuditError(
            f"tasks cannot be both predictions and failures: {sorted(overlap)[:5]}"
        )
    unexpected = (set(signature_by_id) | set(failure_by_id)) - set(task_by_id)
    missing = set(task_by_id) - (set(signature_by_id) | set(failure_by_id))
    if unexpected or missing:
        raise FBDPADAuditError(
            "signature/failure coverage disagrees with tasks; "
            f"missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]}"
        )
    if failure_by_id:
        raise FBDPADAuditError(
            f"{len(failure_by_id)} tasks have explicit FBDP failures"
        )

    ordered = [signature_by_id[task_id] for task_id in task_ids]
    for record in ordered:
        _validate_signature(record, task_by_id[str(record["task_id"])])

    gate_values = _numbers(ordered, "objectness_gate")
    fallback_rate = _boolean_rate(ordered, "foreground_candidate_fallback")
    warnings: list[dict[str, Any]] = []
    low_rate = sum(value <= gate_low for value in gate_values) / len(gate_values)
    high_rate = sum(value >= gate_high for value in gate_values) / len(gate_values)
    if low_rate >= 0.5:
        warnings.append(
            {
                "code": "gate_low_saturation",
                "rate": low_rate,
                "threshold": gate_low,
            }
        )
    if high_rate >= 0.5:
        warnings.append(
            {
                "code": "gate_high_saturation",
                "rate": high_rate,
                "threshold": gate_high,
            }
        )
    if fallback_rate > max_fallback_rate:
        warnings.append(
            {
                "code": "foreground_fallback_rate_high",
                "rate": fallback_rate,
                "threshold": max_fallback_rate,
            }
        )

    group_rows = _group_statistics(ordered)
    report = {
        "protocol_version": FBDP_AD_AUDIT_PROTOCOL_VERSION,
        "label_free": True,
        "forbidden_inputs": sorted(_FORBIDDEN_FIELDS),
        "coverage": {
            "task_count": len(task_ids),
            "signature_count": len(ordered),
            "failure_count": 0,
            "complete": True,
        },
        "numerical_invariants": {
            "finite": True,
            "candidate_counts_positive": True,
            "prototype_counts_positive": True,
            "gated_quantiles_match_gate_scaling": True,
            "quantiles_monotone": True,
        },
        "gate": {
            **_describe(gate_values),
            "low_saturation_rate": low_rate,
            "high_saturation_rate": high_rate,
        },
        "fallback_rate": fallback_rate,
        "support_consistency_valid_rate": _boolean_rate(
            ordered, "support_consistency_valid"
        ),
        "leave_one_out_valid_rate": _boolean_rate(
            ordered, "leave_one_out_reconstruction_valid"
        ),
        "correlations": {
            "gate_vs_fbc": _pearson(
                gate_values,
                _numbers(ordered, "foreground_background_confusion"),
            ),
            "gate_vs_support_reliability": _pearson(
                gate_values, _numbers(ordered, "support_reliability")
            ),
            "gate_vs_assignment_entropy": _pearson(
                gate_values, _numbers(ordered, "assignment_entropy_mean")
            ),
            "raw_vs_gated_q95": _pearson(
                _numbers(ordered, "residual_q95"),
                _numbers(ordered, "gated_residual_q95"),
            ),
        },
        "group_count": len(group_rows),
        "warnings": warnings,
    }
    return report, group_rows


def _validate_signature(
    record: Mapping[str, Any], task: Mapping[str, Any]
) -> None:
    keys = set(record)
    expected = set(FBDP_AD_SIGNATURE_COLUMNS)
    if keys != expected:
        raise FBDPADAuditError(
            "FBDP signature schema mismatch; "
            f"missing={sorted(expected - keys)}, extra={sorted(keys - expected)}"
        )
    forbidden = {_canonical_key(key) for key in keys}.intersection(_FORBIDDEN_FIELDS)
    if forbidden:
        raise FBDPADAuditError(
            f"FBDP signature contains forbidden fields: {sorted(forbidden)}"
        )
    if record.get("protocol_version") != FBDP_AD_SIGNATURE_PROTOCOL_VERSION:
        raise FBDPADAuditError("incompatible FBDP signature protocol")
    if record.get("fbdp_ad_protocol_version") != FBDP_AD_PROTOCOL_VERSION:
        raise FBDPADAuditError("incompatible FBDP numerical protocol")
    _assert_numeric_values_finite(record)
    for field in ("task_id", "dataset", "category", "k_shot", "seed", "support_set_id"):
        if str(record.get(field)) != str(task.get(field)):
            raise FBDPADAuditError(
                f"FBDP signature provenance disagrees on {field}"
            )
    k_shot = _positive_int(record.get("k_shot"), "k_shot")
    support_count = _positive_int(record.get("support_count"), "support_count")
    hashes = record.get("support_image_sha256s")
    if not isinstance(hashes, list) or len(hashes) != k_shot:
        raise FBDPADAuditError("support hashes must align with k_shot")
    if support_count != k_shot:
        raise FBDPADAuditError("support_count must equal k_shot")
    if len(set(str(value) for value in hashes)) != len(hashes):
        raise FBDPADAuditError("support hashes must be unique")
    for field in ("query_patch_count", "support_patch_count", "foreground_candidate_count", "background_candidate_count", "foreground_prototype_count", "background_prototype_count"):
        _positive_int(record.get(field), field)
    grid = record.get("patch_grid_shape")
    if (
        not isinstance(grid, list)
        or len(grid) != 2
        or any(_positive_int(value, "patch_grid_shape") <= 0 for value in grid)
    ):
        raise FBDPADAuditError("patch_grid_shape must contain two positive integers")
    patch_count = int(grid[0]) * int(grid[1])
    if int(record["query_patch_count"]) != patch_count:
        raise FBDPADAuditError("query_patch_count disagrees with patch_grid_shape")
    if int(record["support_patch_count"]) != support_count * patch_count:
        raise FBDPADAuditError("support_patch_count disagrees with supports/grid")
    for field in _UNIT_INTERVAL_FIELDS:
        value = _finite(record.get(field), field)
        if value < 0.0 or value > 1.0:
            raise FBDPADAuditError(f"{field} must be in [0, 1]")
    for field in _NONNEGATIVE_FIELDS:
        if _finite(record.get(field), field) < 0.0:
            raise FBDPADAuditError(f"{field} must be non-negative")
    raw = _finite_vector(record.get("residual_quantiles"), "residual_quantiles", 4)
    gated = _finite_vector(
        record.get("gated_residual_quantiles"), "gated_residual_quantiles", 4
    )
    levels = _finite_vector(
        record.get("residual_quantile_levels"), "residual_quantile_levels", 4
    )
    if levels != sorted(levels) or raw != sorted(raw) or gated != sorted(gated):
        raise FBDPADAuditError("residual levels and quantiles must be monotone")
    gate = float(record["objectness_gate"])
    if any(abs(gated_value - raw_value * gate) > 2e-5 for raw_value, gated_value in zip(raw, gated)):
        raise FBDPADAuditError("gated residual quantiles disagree with gate scaling")
    _finite_vector(
        record.get("foreground_background_margin_quantiles"),
        "foreground_background_margin_quantiles",
        4,
    )
    _finite_vector(
        record.get("assignment_entropy_quantiles"),
        "assignment_entropy_quantiles",
        4,
    )


def _group_statistics(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (
            str(record["dataset"]),
            str(record["category"]),
            int(record["k_shot"]),
        )
        groups.setdefault(key, []).append(record)
    rows = []
    for (dataset, category, k_shot), values in sorted(groups.items()):
        gate = _numbers(values, "objectness_gate")
        rows.append(
            {
                "dataset": dataset,
                "category": category,
                "k_shot": k_shot,
                "task_count": len(values),
                "gate_mean": _mean(gate),
                "gate_min": min(gate),
                "gate_max": max(gate),
                "fbc_mean": _mean(_numbers(values, "foreground_background_confusion")),
                "reliability_mean": _mean(_numbers(values, "support_reliability")),
                "fallback_rate": _boolean_rate(values, "foreground_candidate_fallback"),
                "consistency_valid_rate": _boolean_rate(values, "support_consistency_valid"),
                "loo_valid_rate": _boolean_rate(values, "leave_one_out_reconstruction_valid"),
                "foreground_prototype_mean": _mean(_numbers(values, "foreground_prototype_count")),
                "background_prototype_mean": _mean(_numbers(values, "background_prototype_count")),
                "residual_q95_mean": _mean(_numbers(values, "residual_q95")),
                "gated_residual_q95_mean": _mean(_numbers(values, "gated_residual_q95")),
                "assignment_entropy_mean": _mean(_numbers(values, "assignment_entropy_mean")),
            }
        )
    return rows


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise FBDPADAuditError(f"{source}:{line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise FBDPADAuditError(f"{source}:{line_number} is not an object")
            rows.append(value)
    return rows


def _read_failure_array(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise FBDPADAuditError("failure artifact must be a JSON object array")
    return value


def _unique_records(
    records: Sequence[Mapping[str, Any]], name: str
) -> dict[str, Mapping[str, Any]]:
    result = {}
    for record in records:
        task_id = _text(record.get("task_id"), f"{name}.task_id")
        if task_id in result:
            raise FBDPADAuditError(f"duplicate {name} task_id={task_id!r}")
        result[task_id] = record
    return result


def _numbers(records: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    return [_finite(record.get(field), field) for record in records]


def _boolean_rate(records: Sequence[Mapping[str, Any]], field: str) -> float:
    return sum(bool(record.get(field)) for record in records) / len(records)


def _describe(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": _mean(values),
        "min": ordered[0],
        "q25": _linear_quantile(ordered, 0.25),
        "q50": _linear_quantile(ordered, 0.50),
        "q75": _linear_quantile(ordered, 0.75),
        "max": ordered[-1],
    }


def _linear_quantile(ordered: Sequence[float], level: float) -> float:
    index = level * (len(ordered) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def _finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FBDPADAuditError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise FBDPADAuditError(f"{field} must be finite")
    return result


def _finite_vector(value: Any, field: str, length: int) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FBDPADAuditError(f"{field} must be a sequence")
    if len(value) != length:
        raise FBDPADAuditError(f"{field} must have length {length}")
    return [_finite(item, field) for item in value]


def _assert_numeric_values_finite(value: Any, path: str = "signature") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_numeric_values_finite(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _assert_numeric_values_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FBDPADAuditError(f"{path} must be finite")


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise FBDPADAuditError(f"{field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise FBDPADAuditError(f"{field} must be a positive integer") from exc
    if result <= 0 or str(result) != str(value):
        raise FBDPADAuditError(f"{field} must be a positive integer")
    return result


def _text(value: Any, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise FBDPADAuditError(f"{field} must be non-empty")
    return result


def _canonical_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _validate_thresholds(args: argparse.Namespace) -> None:
    if not 0.0 <= args.gate_low < args.gate_high <= 1.0:
        raise FBDPADAuditError("gate thresholds must satisfy 0 <= low < high <= 1")
    if not 0.0 <= args.max_fallback_rate <= 1.0:
        raise FBDPADAuditError("--max-fallback-rate must be in [0, 1]")


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _atomic_write_json(path: str | Path, value: Any) -> Path:
    return _atomic_write_text(
        Path(path), json.dumps(value, indent=2, sort_keys=True) + "\n"
    )


def _atomic_write_csv(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


__all__ = [
    "AUDIT_REPORT_NAME",
    "FBDP_AD_AUDIT_PROTOCOL_VERSION",
    "FBDPADAuditError",
    "GROUP_STATISTICS_NAME",
    "RUN_RECORD_NAME",
    "audit_fbdp_ad_artifacts",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
