"""Generate the Stage 1 Gate acceptance report from existing artifacts.

This script is reporting-only. It reads existing audit, support summary,
leakage, fake expert evaluation, docs, and config files, then writes a Markdown
report. It does not run models, modify experiment results, or invent missing
numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1_DIR = PROJECT_ROOT / "outputs" / "stage1_gate"
DEFAULT_OUTPUT = DEFAULT_STAGE1_DIR / "STAGE1_GATE_REPORT.md"


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    title: str
    relative_path: Path


ARTIFACTS = [
    ArtifactSpec("mvtec_audit", "MVTec full audit", Path("mvtec_full_audit.json")),
    ArtifactSpec("visa_audit", "VisA full audit", Path("visa_full_audit.json")),
    ArtifactSpec("mvtec_support", "MVTec support summary", Path("mvtec_support_summary.csv")),
    ArtifactSpec("visa_support", "VisA support summary", Path("visa_support_summary.csv")),
    ArtifactSpec("mvtec_leakage", "MVTec leakage report", Path("mvtec_leakage_report.json")),
    ArtifactSpec("visa_leakage", "VisA leakage report", Path("visa_leakage_report.json")),
    ArtifactSpec(
        "fake_expert_metrics",
        "FakeExpert metrics stub",
        Path("fake_expert_eval") / "metrics_stub.json",
    ),
]

REFERENCE_FILES = [
    ArtifactSpec("task_spec", "Task spec", Path("docs") / "task_spec.md"),
    ArtifactSpec("data_leakage_policy", "Data leakage policy", Path("docs") / "data_leakage_policy.md"),
    ArtifactSpec("folds", "Fold config", Path("configs") / "folds.yaml"),
]


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
        help="Repository root used to find docs/task_spec.md, docs/data_leakage_policy.md, and configs/folds.yaml.",
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
    artifacts = {spec.key: _load_artifact(stage1_dir / spec.relative_path, spec) for spec in ARTIFACTS}
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
            "## 8. 当前已知问题",
            "",
            *_known_issues(artifacts, references),
            "",
            "## 9. 是否满足进入Stage 2的Gate条件",
            "",
            f"- Gate结论: **{gate['label']}**",
            f"- 判定依据: {gate['reason']}",
            "",
            "## 10. 下一阶段PatchCore/WinCLIP/AnomalyDINO准备事项",
            "",
            "- PatchCore: 接入前固定 support_set_id、k_shot、seed，并记录 config/git commit/environment/predictions/failures。",
            "- WinCLIP: 确认 prompt 与阈值选择不读取目标 test label/mask/defect_type，不使用目标异常样本调参。",
            "- AnomalyDINO: 明确权重路径与版本记录；不得自动下载权重，缺失权重应显式失败。",
            "- 三个 baseline: 在同一比较中使用相同 support_set_id，保存完整失败样本，不静默跳过。",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_artifact(path: Path, spec: ArtifactSpec) -> dict[str, Any]:
    record: dict[str, Any] = {
        "key": spec.key,
        "title": spec.title,
        "path": path,
        "relative_path": spec.relative_path.as_posix(),
        "exists": path.is_file(),
        "parse_error": "",
        "data": None,
    }
    if not path.is_file():
        return record
    try:
        if path.suffix.lower() == ".json":
            record["data"] = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".csv":
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                record["data"] = {
                    "fieldnames": reader.fieldnames or [],
                    "rows": [dict(row) for row in reader],
                }
        else:
            record["data"] = path.read_text(encoding="utf-8")
    except (csv.Error, json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        record["parse_error"] = f"{type(exc).__name__}: {exc}"
    return record


def _load_reference(path: Path, spec: ArtifactSpec) -> dict[str, Any]:
    record = {
        "key": spec.key,
        "title": spec.title,
        "path": path,
        "relative_path": spec.relative_path.as_posix(),
        "exists": path.is_file(),
        "parse_error": "",
        "data": None,
    }
    if path.is_file():
        try:
            record["data"] = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            record["parse_error"] = f"{type(exc).__name__}: {exc}"
    return record


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
        return ["- support summary file: missing"]
    if record["parse_error"]:
        return [f"- support summary file: parse_error: `{record['parse_error']}`"]
    data = record["data"] if isinstance(record["data"], dict) else {}
    rows = data.get("rows", []) if isinstance(data.get("rows"), list) else []
    fieldnames = data.get("fieldnames", []) if isinstance(data.get("fieldnames"), list) else []
    lines = [
        f"- rows: {len(rows)}",
        f"- columns: {_value(', '.join(str(name) for name in fieldnames) if fieldnames else None)}",
    ]
    for column in ["support_set_id", "k_shot", "seed", "category", "dataset"]:
        if column in fieldnames:
            values = sorted({row.get(column, "") for row in rows})
            lines.append(f"- unique {column}: {len(values)} ({_compact_values(values)})")
    if rows:
        lines.extend(["", "| " + " | ".join(_md_escape(str(name)) for name in fieldnames) + " |"])
        lines.append("| " + " | ".join("---" for _ in fieldnames) + " |")
        for row in rows[:10]:
            lines.append("| " + " | ".join(_md_escape(str(row.get(name, ""))) for name in fieldnames) + " |")
        if len(rows) > 10:
            lines.append(f"- ... {len(rows) - 10} additional rows omitted from report display.")
    return lines


def _leakage_section(record: dict[str, Any]) -> list[str]:
    if not record["exists"]:
        return ["- leakage report file: missing"]
    if record["parse_error"]:
        return [f"- leakage report file: parse_error: `{record['parse_error']}`"]
    data = record["data"] if isinstance(record["data"], dict) else {}
    counts = data.get("counts", {}) if isinstance(data.get("counts"), dict) else {}
    lines = [
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
        lines.append(f"| `{_md_escape(str(key))}` | {_md_escape(_value(data[key]))} |")
    return lines


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

    fake_metrics = artifacts["fake_expert_metrics"]
    if fake_metrics["exists"] and not fake_metrics["parse_error"] and isinstance(fake_metrics["data"], dict):
        failed = fake_metrics["data"].get("num_failed_predictions")
        missing = fake_metrics["data"].get("num_missing_labels")
        if isinstance(failed, int) and failed > 0:
            issues.append(f"- `{fake_metrics['relative_path']}`: num_failed_predictions={failed}")
        if isinstance(missing, int) and missing > 0:
            issues.append(f"- `{fake_metrics['relative_path']}`: num_missing_labels={missing}")

    return issues or ["- none recorded from the provided artifacts."]


def _gate_status(artifacts: dict[str, dict[str, Any]], references: dict[str, dict[str, Any]]) -> dict[str, str]:
    blockers = _known_issues(artifacts, references)
    has_blocker = blockers != ["- none recorded from the provided artifacts."]
    label = "PASS" if not has_blocker else "NOT MET"
    if has_blocker:
        return {
            "label": label,
            "reason": "存在 missing/parse_error/非零失败计数/泄漏未通过等问题；见第8节。",
        }
    return {
        "label": label,
        "reason": "所有指定输入文件存在且可解析，audit 关键失败计数为0，leakage ok=true，FakeExpert smoke metrics 无失败或缺失标签。",
    }


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
