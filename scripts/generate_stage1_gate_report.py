"""Generate the Stage 1 Gate acceptance report from existing artifacts.

This script is reporting-only. It reads existing audit, support, leakage,
FakeExpert evaluation, pytest, docs, and config files, then writes a Markdown
report. It does not run models, modify experiment results, or invent missing
numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1_DIR = PROJECT_ROOT / "outputs" / "stage1_gate"
DEFAULT_OUTPUT = DEFAULT_STAGE1_DIR / "STAGE1_GATE_REPORT.md"


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    title: str
    relative_path: Path


AUDIT_ARTIFACTS = [
    ArtifactSpec("mvtec_audit", "MVTec full audit", Path("mvtec_full_audit.json")),
    ArtifactSpec("visa_audit", "VisA full audit", Path("visa_full_audit.json")),
]

REFERENCE_FILES = [
    ArtifactSpec("task_spec", "Task spec", Path("docs") / "task_spec.md"),
    ArtifactSpec("data_leakage_policy", "Data leakage policy", Path("docs") / "data_leakage_policy.md"),
    ArtifactSpec("folds", "Fold config", Path("configs") / "folds.yaml"),
]

FAILURE_REPORT_NAME = "evaluation_failures.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-dir",
        default=str(DEFAULT_STAGE1_DIR),
        help="Directory containing Stage 1 output artifacts.",
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Repository root used to find docs, configs, support sets, and evaluation artifacts.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Markdown report output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage1_dir = Path(args.stage1_dir)
    project_root = Path(args.project_root)
    output_path = Path(args.output)
    report = build_report(stage1_dir=stage1_dir, project_root=project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


def build_report(*, stage1_dir: Path, project_root: Path) -> str:
    artifacts = _load_artifacts(stage1_dir=stage1_dir, project_root=project_root)
    references = {
        spec.key: _load_reference(project_root / spec.relative_path, spec)
        for spec in REFERENCE_FILES
    }
    gate = _gate_status(artifacts, references)

    lines: list[str] = [
        "# Stage 1 Gate Acceptance Report",
        "",
        "> This report is generated from existing Stage 1 artifacts only. FakeExpert metrics are a protocol smoke/evaluation isolation check, not real anomaly detection results.",
        "",
        "## 1. Stage 1目标",
        "",
        "- 固化 Agent 可见输入与 evaluator-only ground truth 的边界。",
        "- 完成 MVTec 与 VisA manifest/audit/support/leakage 基础检查。",
        "- 验证 FakeExpert 闭环只用于协议与评估链路 smoke test，不作为真实检测性能。",
        "- 为 Stage 2 的 PatchCore / WinCLIP / AnomalyDINO baseline 接入准备可复现输入与 Gate 条件。",
        "",
        "## 2. 已完成文件列表",
        "",
    ]
    lines.extend(_artifact_table(artifacts, references))
    lines.extend(
        [
            "",
            "## 3. MVTec数据统计",
            "",
            *_audit_section(artifacts["mvtec_audit"]),
            "",
            "## 4. VisA数据统计",
            "",
            *_audit_section(artifacts["visa_audit"]),
            "",
            "## 5. support set统计",
            "",
            "### MVTec",
            "",
            *_support_section(artifacts["mvtec_support"]),
            "",
            "### VisA",
            "",
            *_support_section(artifacts["visa_support"]),
            "",
            "## 6. 泄漏检查结果",
            "",
            "### MVTec",
            "",
            *_leakage_section(artifacts["mvtec_leakage"]),
            "",
            "### VisA",
            "",
            *_leakage_section(artifacts["visa_leakage"]),
            "",
            "## 7. fake expert闭环结果",
            "",
            *_fake_expert_section(artifacts["fake_expert_metrics"]),
            "",
            "## 8. Evaluation与pytest结果",
            "",
            *_evaluation_section(artifacts["fake_expert_metrics"]),
            "",
            *_pytest_section(artifacts["pytest_server"]),
            "",
            "## 9. 当前已知问题",
            "",
            *_known_issues(artifacts, references),
            "",
            "## 10. 是否满足进入Stage 2的Gate条件",
            "",
            f"- Gate结论: **{gate['label']}**",
            f"- 判定依据: {gate['reason']}",
            "",
            "## 11. 下一阶段PatchCore/WinCLIP/AnomalyDINO准备事项",
            "",
            "- PatchCore: 接入前固定 support_set_id、k_shot、seed，并记录 config/git commit/environment/predictions/failures。",
            "- WinCLIP: 确认 prompt 与阈值选择不读取目标 test label/mask/defect_type，不使用目标异常样本调参。",
            "- AnomalyDINO: 明确权重路径与版本记录；不得自动下载权重，缺失权重应显式失败。",
            "- 三个 baseline: 在同一比较中使用相同 support_set_id，保存完整失败样本，不静默跳过。",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_artifacts(*, stage1_dir: Path, project_root: Path) -> dict[str, dict[str, Any]]:
    artifacts = {
        spec.key: _load_artifact(stage1_dir / spec.relative_path, spec)
        for spec in AUDIT_ARTIFACTS
    }
    artifacts["mvtec_support"] = _load_support(project_root=project_root, stage1_dir=stage1_dir, dataset="mvtec")
    artifacts["visa_support"] = _load_support(project_root=project_root, stage1_dir=stage1_dir, dataset="visa")
    artifacts["mvtec_leakage"] = _load_leakage(stage1_dir=stage1_dir, dataset="mvtec")
    artifacts["visa_leakage"] = _load_leakage(stage1_dir=stage1_dir, dataset="visa")
    artifacts["fake_expert_metrics"] = _load_fake_expert(project_root=project_root, stage1_dir=stage1_dir)
    artifacts["pytest_server"] = _load_pytest_log(stage1_dir=stage1_dir)
    return artifacts


def _base_record(*, key: str, title: str, path: Path, relative_path: str) -> dict[str, Any]:
    return {
        "key": key,
        "title": title,
        "path": path,
        "relative_path": relative_path,
        "exists": False,
        "parse_error": "",
        "data": None,
    }


def _load_artifact(path: Path, spec: ArtifactSpec) -> dict[str, Any]:
    record = _base_record(
        key=spec.key,
        title=spec.title,
        path=path,
        relative_path=spec.relative_path.as_posix(),
    )
    record["exists"] = path.is_file()
    if not path.is_file():
        return record
    try:
        if path.suffix.lower() == ".json":
            record["data"] = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".csv":
            record["data"] = _read_csv_rows(path)
        else:
            record["data"] = path.read_text(encoding="utf-8")
    except (csv.Error, json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        record["parse_error"] = f"{type(exc).__name__}: {exc}"
    return record


def _load_support(*, project_root: Path, stage1_dir: Path, dataset: str) -> dict[str, Any]:
    summary_path = stage1_dir / f"{dataset}_support_summary.csv"
    title = f"{_dataset_title(dataset)} support summary"
    record = _base_record(
        key=f"{dataset}_support",
        title=title,
        path=summary_path,
        relative_path=f"{dataset}_support_summary.csv",
    )
    if summary_path.is_file():
        loaded = _load_artifact(summary_path, ArtifactSpec(f"{dataset}_support", title, Path(summary_path.name)))
        if isinstance(loaded.get("data"), dict):
            loaded["data"]["layout"] = "summary_csv"
            loaded["data"]["source_file_count"] = 1
        return loaded

    support_dir = project_root / "data" / "support_sets"
    files = sorted(support_dir.glob(f"{dataset}_k*_seed*.csv"))
    record["path"] = support_dir
    record["relative_path"] = f"data/support_sets/{dataset}_k*_seed*.csv"
    record["exists"] = bool(files)
    if not files:
        return record

    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    parse_errors: list[str] = []
    for path in files:
        try:
            data = _read_csv_rows(path)
            if not fieldnames:
                fieldnames = list(data["fieldnames"])
            rows.extend(data["rows"])
        except (csv.Error, UnicodeDecodeError, OSError) as exc:
            parse_errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    record["parse_error"] = "; ".join(parse_errors)
    record["data"] = {
        "layout": "support_set_directory",
        "fieldnames": fieldnames,
        "rows": rows,
        "source_files": [path.as_posix() for path in files],
        "source_file_count": len(files),
    }
    return record


def _load_leakage(*, stage1_dir: Path, dataset: str) -> dict[str, Any]:
    report_path = stage1_dir / f"{dataset}_leakage_report.json"
    title = f"{_dataset_title(dataset)} leakage report"
    record = _base_record(
        key=f"{dataset}_leakage",
        title=title,
        path=report_path,
        relative_path=f"{dataset}_leakage_report.json",
    )
    if report_path.is_file():
        return _load_artifact(report_path, ArtifactSpec(f"{dataset}_leakage", title, Path(report_path.name)))

    leakage_dir = stage1_dir / "leakage"
    files = sorted(leakage_dir.glob(f"{dataset}_*_leakage.json"))
    record["path"] = leakage_dir
    record["relative_path"] = f"outputs/stage1_gate/leakage/{dataset}_*_leakage.json"
    record["exists"] = bool(files)
    if not files:
        return record

    reports: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for path in files:
        try:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            parse_errors.append(f"{path.name}: {type(exc).__name__}: {exc}")

    error_count = sum(_int_value(report.get("error_count")) for report in reports)
    warning_count = sum(_int_value(report.get("warning_count")) for report in reports)
    issues = [issue for report in reports for issue in report.get("issues", []) if isinstance(report.get("issues", []), list)]
    ok = bool(reports) and all(report.get("ok") is True for report in reports)
    support_rows = sum(_int_value(report.get("counts", {}).get("support_rows")) for report in reports if isinstance(report.get("counts"), dict))
    support_sets = sum(_int_value(report.get("counts", {}).get("support_sets")) for report in reports if isinstance(report.get("counts"), dict))
    record["parse_error"] = "; ".join(parse_errors)
    record["data"] = {
        "layout": "leakage_directory",
        "ok": ok,
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": issues,
        "counts": {
            "report_files": len(files),
            "support_rows_total": support_rows,
            "support_sets_total": support_sets,
        },
        "source_files": [path.as_posix() for path in files],
    }
    return record


def _load_fake_expert(*, project_root: Path, stage1_dir: Path) -> dict[str, Any]:
    candidates = [
        stage1_dir / "fake_expert_eval" / "metrics_stub.json",
        project_root / "outputs" / "evaluation_fake_expert" / "metrics_stub.json",
    ]
    chosen = next((path for path in candidates if path.is_file()), candidates[0])
    record = _load_artifact(
        chosen,
        ArtifactSpec("fake_expert_metrics", "FakeExpert metrics stub", _relative_to_project(chosen, project_root, stage1_dir)),
    )
    if record["exists"] and isinstance(record["data"], dict):
        failure_path = chosen.parent / FAILURE_REPORT_NAME
        record["data"]["failure_report_path"] = _relative_to_project(failure_path, project_root, stage1_dir).as_posix()
        if failure_path.is_file():
            try:
                record["data"]["failure_report"] = json.loads(failure_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
                record["parse_error"] = f"{record['parse_error']}; {FAILURE_REPORT_NAME}: {type(exc).__name__}: {exc}".strip("; ")
        joined_path = chosen.parent / "per_sample_joined.csv"
        if joined_path.is_file():
            record["data"]["joined_csv_path"] = _relative_to_project(joined_path, project_root, stage1_dir).as_posix()
    return record


def _load_pytest_log(*, stage1_dir: Path) -> dict[str, Any]:
    path = stage1_dir / "pytest_server.log"
    record = _base_record(
        key="pytest_server",
        title="Server pytest log",
        path=path,
        relative_path="outputs/stage1_gate/pytest_server.log",
    )
    record["exists"] = path.is_file()
    if not path.is_file():
        return record
    try:
        text = path.read_text(encoding="utf-8")
        record["data"] = {
            "text": text,
            "passed": " passed" in text and " failed" not in text,
            "summary": _last_nonempty_line(text),
        }
    except (UnicodeDecodeError, OSError) as exc:
        record["parse_error"] = f"{type(exc).__name__}: {exc}"
    return record


def _load_reference(path: Path, spec: ArtifactSpec) -> dict[str, Any]:
    record = _base_record(
        key=spec.key,
        title=spec.title,
        path=path,
        relative_path=spec.relative_path.as_posix(),
    )
    record["exists"] = path.is_file()
    if path.is_file():
        try:
            record["data"] = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            record["parse_error"] = f"{type(exc).__name__}: {exc}"
    return record


def _read_csv_rows(path: Path) -> dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            "fieldnames": reader.fieldnames or [],
            "rows": [dict(row) for row in reader],
        }


def _artifact_table(artifacts: dict[str, dict[str, Any]], references: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["| 文件 | 状态 | 说明 |", "| --- | --- | --- |"]
    for record in [*artifacts.values(), *references.values()]:
        status = "present" if record["exists"] else "missing"
        note = "parse_error: " + record["parse_error"] if record["parse_error"] else record["title"]
        lines.append(f"| `{_md_escape(record['relative_path'])}` | {status} | {_md_escape(note)} |")
    return lines


def _audit_section(record: dict[str, Any]) -> list[str]:
    if not record["exists"]:
        return ["- audit file: missing"]
    if record["parse_error"]:
        return [f"- audit file: parse_error: `{record['parse_error']}`"]
    data = record["data"] if isinstance(record["data"], dict) else {}
    counts = data.get("counts", {}) if isinstance(data.get("counts"), dict) else {}
    categories = data.get("categories", [])
    lines = [
        f"- dataset: {_value(data.get('dataset'))}",
        f"- categories: {_value(counts.get('categories', len(categories) if isinstance(categories, list) else None))}",
        f"- train_good_images: {_value(counts.get('train_good_images'))}",
        f"- test_good_images: {_value(counts.get('test_good_images'))}",
        f"- test_anomaly_images: {_value(counts.get('test_anomaly_images'))}",
        f"- missing_masks: {_value(counts.get('missing_masks'))}",
        f"- broken_image_files: {_value(counts.get('broken_image_files'))}",
        f"- broken_mask_files: {_value(counts.get('broken_mask_files'))}",
        f"- duplicate_image_ids: {_value(counts.get('duplicate_image_ids'))}",
        f"- mask_size_mismatches: {_value(counts.get('mask_size_mismatches'))}",
    ]
    run = data.get("run", {}) if isinstance(data.get("run"), dict) else {}
    if run:
        lines.extend(
            [
                f"- run.git_commit: {_value(run.get('git_commit'))}",
                f"- run.seed: {_value(run.get('seed'))}",
                f"- run.predictions: {_value(run.get('predictions'))}",
            ]
        )
    return lines


def _support_section(record: dict[str, Any]) -> list[str]:
    if not record["exists"]:
        return ["- support summary/source files: missing"]
    if record["parse_error"]:
        return [f"- support summary/source files: parse_error: `{record['parse_error']}`"]
    data = record["data"] if isinstance(record["data"], dict) else {}
    rows = data.get("rows", []) if isinstance(data.get("rows"), list) else []
    fieldnames = data.get("fieldnames", []) if isinstance(data.get("fieldnames"), list) else []
    lines = [
        f"- layout: {_value(data.get('layout'))}",
        f"- source_files: {_value(data.get('source_file_count'))}",
        f"- rows: {len(rows)}",
        f"- columns: {_value(', '.join(str(name) for name in fieldnames) if fieldnames else None)}",
    ]
    for column in ["support_set_id", "k_shot", "seed", "category", "dataset"]:
        if column in fieldnames:
            values = sorted({row.get(column, "") for row in rows})
            lines.append(f"- unique {column}: {len(values)} ({_compact_values(values)})")
    return lines


def _leakage_section(record: dict[str, Any]) -> list[str]:
    if not record["exists"]:
        return ["- leakage report/source files: missing"]
    if record["parse_error"]:
        return [f"- leakage report/source files: parse_error: `{record['parse_error']}`"]
    data = record["data"] if isinstance(record["data"], dict) else {}
    counts = data.get("counts", {}) if isinstance(data.get("counts"), dict) else {}
    lines = [
        f"- layout: {_value(data.get('layout'))}",
        f"- ok: {_value(data.get('ok'))}",
        f"- error_count: {_value(data.get('error_count'))}",
        f"- warning_count: {_value(data.get('warning_count'))}",
    ]
    for key in sorted(counts):
        lines.append(f"- {key}: {_value(counts[key])}")
    issues = data.get("issues", [])
    if isinstance(issues, list) and issues:
        lines.append("- issues:")
        for issue in issues[:10]:
            if isinstance(issue, dict):
                code = _value(issue.get("code"))
                message = _value(issue.get("message"))
                lines.append(f"  - {code}: {message}")
            else:
                lines.append(f"  - {_value(issue)}")
        if len(issues) > 10:
            lines.append(f"  - ... {len(issues) - 10} additional issues omitted from report display.")
    return lines


def _fake_expert_section(record: dict[str, Any]) -> list[str]:
    if not record["exists"]:
        return ["- fake expert metrics file: missing"]
    if record["parse_error"]:
        return [f"- fake expert metrics file: parse_error: `{record['parse_error']}`"]
    data = record["data"] if isinstance(record["data"], dict) else {}
    lines = [
        "- 说明: FakeExpert 只用于闭环 smoke test 与评估隔离检查，不代表真实异常检测结果。",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in sorted(data):
        if key == "failure_report":
            continue
        lines.append(f"| `{_md_escape(str(key))}` | {_md_escape(_value(data[key]))} |")
    return lines


def _evaluation_section(record: dict[str, Any]) -> list[str]:
    if not record["exists"] or record["parse_error"] or not isinstance(record["data"], dict):
        return ["- evaluation artifacts: missing or unreadable"]
    failure_report = record["data"].get("failure_report")
    lines = [
        f"- joined_csv: {_value(record['data'].get('joined_csv_path'))}",
        f"- failure_report: {_value(record['data'].get('failure_report_path'))}",
    ]
    if isinstance(failure_report, dict):
        missing = failure_report.get("missing_predictions", [])
        extra = failure_report.get("extra_predictions", [])
        failed = failure_report.get("failed_predictions", [])
        lines.extend(
            [
                f"- missing_predictions: {len(missing) if isinstance(missing, list) else 'unknown'}",
                f"- extra_predictions: {len(extra) if isinstance(extra, list) else 'unknown'}",
                f"- failed_predictions: {len(failed) if isinstance(failed, list) else 'unknown'}",
            ]
        )
    else:
        lines.append("- evaluation_failures.json: missing")
    return lines


def _pytest_section(record: dict[str, Any]) -> list[str]:
    if not record["exists"]:
        return ["- pytest_server.log: missing"]
    if record["parse_error"]:
        return [f"- pytest_server.log: parse_error: `{record['parse_error']}`"]
    data = record["data"] if isinstance(record["data"], dict) else {}
    return [
        f"- pytest passed: {_value(data.get('passed'))}",
        f"- pytest summary: {_value(data.get('summary'))}",
    ]


def _known_issues(artifacts: dict[str, dict[str, Any]], references: dict[str, dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for record in [*artifacts.values(), *references.values()]:
        if not record["exists"]:
            issues.append(f"- `{record['relative_path']}`: missing")
        elif record["parse_error"]:
            issues.append(f"- `{record['relative_path']}`: parse_error: `{record['parse_error']}`")

    for key in ["mvtec_audit", "visa_audit"]:
        record = artifacts[key]
        if record["exists"] and not record["parse_error"] and isinstance(record["data"], dict):
            counts = record["data"].get("counts", {})
            if isinstance(counts, dict):
                for count_key in [
                    "missing_masks",
                    "broken_image_files",
                    "broken_mask_files",
                    "duplicate_image_ids",
                    "mask_size_mismatches",
                ]:
                    value = counts.get(count_key)
                    if isinstance(value, int) and value > 0:
                        issues.append(f"- `{record['relative_path']}`: {count_key}={value}")

    for key in ["mvtec_leakage", "visa_leakage"]:
        record = artifacts[key]
        if record["exists"] and not record["parse_error"] and isinstance(record["data"], dict):
            if record["data"].get("ok") is not True:
                issues.append(f"- `{record['relative_path']}`: leakage ok is {_value(record['data'].get('ok'))}")
            for count_key in ["error_count", "warning_count"]:
                value = record["data"].get(count_key)
                if isinstance(value, int) and value > 0:
                    issues.append(f"- `{record['relative_path']}`: {count_key}={value}")

    fake_metrics = artifacts["fake_expert_metrics"]
    if fake_metrics["exists"] and not fake_metrics["parse_error"] and isinstance(fake_metrics["data"], dict):
        for count_key in ["num_failed_predictions", "num_missing_labels"]:
            value = fake_metrics["data"].get(count_key)
            if isinstance(value, int) and value > 0:
                issues.append(f"- `{fake_metrics['relative_path']}`: {count_key}={value}")
        num_predictions = fake_metrics["data"].get("num_predictions")
        num_joined = fake_metrics["data"].get("num_joined")
        if isinstance(num_predictions, int) and isinstance(num_joined, int) and num_joined != num_predictions:
            issues.append(f"- `{fake_metrics['relative_path']}`: num_joined={num_joined} differs from num_predictions={num_predictions}")
        failure_report = fake_metrics["data"].get("failure_report")
        if isinstance(failure_report, dict):
            for key, label in [
                ("missing_predictions", "missing_predictions"),
                ("extra_predictions", "extra_predictions"),
                ("failed_predictions", "failed_predictions"),
            ]:
                values = failure_report.get(key)
                if isinstance(values, list) and values:
                    issues.append(f"- `{fake_metrics['data'].get('failure_report_path')}`: {label}={len(values)}")

    pytest_record = artifacts["pytest_server"]
    if pytest_record["exists"] and not pytest_record["parse_error"] and isinstance(pytest_record["data"], dict):
        if pytest_record["data"].get("passed") is not True:
            issues.append(f"- `{pytest_record['relative_path']}`: pytest did not pass")

    return issues or ["- none recorded from the provided artifacts."]


def _gate_status(artifacts: dict[str, dict[str, Any]], references: dict[str, dict[str, Any]]) -> dict[str, str]:
    blockers = _known_issues(artifacts, references)
    has_blocker = blockers != ["- none recorded from the provided artifacts."]
    if has_blocker:
        return {
            "label": "NOT MET",
            "reason": "存在 missing/parse_error/非零失败计数/泄漏未通过/pytest未通过/evaluation未闭合等问题；见第9节。",
        }
    return {
        "label": "PASS",
        "reason": "指定与实际布局中的输入文件均存在且可解析，audit关键失败计数为0，support与leakage闭合，FakeExpert evaluation无缺失或额外预测，服务器pytest通过。",
    }


def _relative_to_project(path: Path, project_root: Path, stage1_dir: Path) -> Path:
    for root in [project_root, stage1_dir]:
        try:
            return path.relative_to(root)
        except ValueError:
            pass
    return path


def _dataset_title(dataset: str) -> str:
    return "MVTec" if dataset == "mvtec" else "VisA"


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) else 0


def _last_nonempty_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _value(value: Any) -> str:
    if value is None or value == "":
        return "missing"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _compact_values(values: list[Any]) -> str:
    text_values = [_value(value) for value in values]
    preview = text_values[:8]
    suffix = "" if len(text_values) <= 8 else f", ... +{len(text_values) - 8}"
    return ", ".join(preview) + suffix


def _md_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


if __name__ == "__main__":
    raise SystemExit(main())
