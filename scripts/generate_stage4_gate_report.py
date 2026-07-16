"""Generate the Stage 4 gate JSON and Markdown report from frozen artifacts.

This is a reporting-only command.  It invokes the read-only Stage 4 validator,
summarizes evaluator-only CSVs, and records limitations.  It never reruns an
expert, changes a threshold, or presents the local wrapper results as SOTA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normroute.cli.validate_stage4_outputs import inspect_stage4_outputs  # noqa: E402


FIXED_BASELINES = frozenset(
    {
        "always_anomalydino",
        "always_patchcore",
        "always_winclip",
        "random_seeded",
        "fastest_expert",
    }
)
RULE_AGENTS = frozenset({"category_prior", "category_shot_prior", "cost_aware"})
LEARNED_METADATA_BASELINES = frozenset(
    {"decision_tree_metadata", "multinomial_logistic_metadata"}
)
PRIMARY_AGGREGATION = "micro"
PRIMARY_METRIC = "auroc_mean"
IMPLEMENTATION_MODE = (
    "Local no-download, support-guided PatchCore-style, WinCLIP-style, and "
    "AnomalyDINO-style wrappers used to validate the protocol and routing stack; "
    "these are not the official external baseline implementations."
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4-root", default="outputs/stage4")
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--evaluation-root", default=None)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--fold-manifest", default=None)
    parser.add_argument("--policy-root", default=None)
    parser.add_argument("--live-root", default=None)
    parser.add_argument("--report-dir", default="reports/stage4")
    parser.add_argument("--gate-output", default=None)
    parser.add_argument("--report-output", default=None)
    parser.add_argument("--no-live-smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report_dir = Path(args.report_dir)
    gate_path = Path(args.gate_output) if args.gate_output else report_dir / "stage4_gate.json"
    report_path = (
        Path(args.report_output) if args.report_output else report_dir / "stage4_report.md"
    )
    gate, report = generate_stage4_gate_report(
        stage4_root=Path(args.stage4_root),
        runs_root=Path(args.runs_root) if args.runs_root else None,
        evaluation_root=Path(args.evaluation_root) if args.evaluation_root else None,
        tasks_path=Path(args.tasks) if args.tasks else None,
        fold_manifest_path=Path(args.fold_manifest) if args.fold_manifest else None,
        policy_root=Path(args.policy_root) if args.policy_root else None,
        live_root=Path(args.live_root) if args.live_root else None,
        require_live=not args.no_live_smoke,
    )
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {gate_path}")
    print(f"Wrote {report_path}")
    if gate["status"] != "PASS":
        raise SystemExit(1)


def generate_stage4_gate_report(
    *,
    stage4_root: Path = Path("outputs/stage4"),
    runs_root: Path | None = None,
    evaluation_root: Path | None = None,
    tasks_path: Path | None = None,
    fold_manifest_path: Path | None = None,
    policy_root: Path | None = None,
    live_root: Path | None = None,
    require_live: bool = True,
) -> tuple[dict[str, Any], str]:
    """Build the report-ready gate payload and Markdown without writing files."""

    gate = inspect_stage4_outputs(
        stage4_root=stage4_root,
        runs_root=runs_root,
        evaluation_root=evaluation_root,
        tasks_path=tasks_path,
        fold_manifest_path=fold_manifest_path,
        policy_root=policy_root,
        live_root=live_root,
        require_live=require_live,
    )
    resolved_evaluation = Path(gate["config"]["evaluation_root"])
    resolved_stage4 = Path(gate["config"]["stage4_root"])
    policy_summary_path = resolved_evaluation / "stage4_policy_summary.csv"
    cost_quality_path = resolved_evaluation / "stage4_cost_quality.csv"
    split_audit_path = resolved_stage4 / "splits" / "split_audit.json"

    report_errors: list[str] = []
    policy_rows = _read_csv(policy_summary_path, report_errors)
    cost_rows = _read_csv(cost_quality_path, report_errors)
    split_audit = _read_json(split_audit_path, report_errors)
    analysis = analyze_stage4_results(
        gate=gate,
        policy_rows=policy_rows,
        cost_rows=cost_rows,
        split_audit=split_audit,
    )
    _add_report_input_check(gate, report_errors, policy_rows, cost_rows, split_audit)
    gate["report_summary"] = analysis
    gate["report_input_sha256"] = {
        str(path): _sha256(path)
        for path in (policy_summary_path, cost_quality_path, split_audit_path)
        if path.is_file()
    }
    return gate, build_report(gate)


def analyze_stage4_results(
    *,
    gate: Mapping[str, Any],
    policy_rows: Sequence[Mapping[str, str]],
    cost_rows: Sequence[Mapping[str, str]],
    split_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract the fixed/rule/learned/Oracle and cost-quality conclusions."""

    primary_rows = [
        dict(row)
        for row in policy_rows
        if row.get("aggregation") == PRIMARY_AGGREGATION
        and row.get("policy_kind") == "realizable_policy"
    ]
    best_fixed = _best_policy(primary_rows, FIXED_BASELINES)
    best_rule = _best_policy(primary_rows, RULE_AGENTS)
    best_learned = _best_policy(primary_rows, LEARNED_METADATA_BASELINES)
    oracle = next(
        (
            dict(row)
            for row in policy_rows
            if row.get("aggregation") == PRIMARY_AGGREGATION
            and row.get("policy_name") == "run_level_oracle"
        ),
        {},
    )
    frontier = _pareto_frontier(primary_rows)
    best_overall = _best_policy(primary_rows, set(FIXED_BASELINES | RULE_AGENTS | LEARNED_METADATA_BASELINES))
    folds = _fold_protocol(split_audit)
    live_check = next(
        (dict(check) for check in gate.get("checks", []) if check.get("name") == "live_smoke"),
        {},
    )
    limitations = [
        "All quality numbers come from local no-download support-guided wrappers, not official external baseline implementations or a final NORM-Route model.",
        "The current Stage 4 full-CV report covers MVTec; it is not a multi-dataset SOTA comparison.",
        "Replay costs are historical estimated runtimes copied from immutable Stage 2 outputs; only the one-sample live smoke records actual subprocess runtime.",
        "F1 uses saved wrapper decisions and frozen thresholds; target abnormal test samples were not used for threshold or hyperparameter tuning.",
        "The live smoke contains one routed sample, so it validates wiring and ordering rather than production latency or quality.",
    ]
    route_check = next(
        (
            check
            for check in gate.get("checks", [])
            if check.get("name") == "route_decisions_one_expert_per_run"
        ),
        {},
    )
    if route_check.get("status") == "FAIL":
        limitations.append(
            "The saved random_seeded control selects experts per sample and therefore violates the frozen one-expert-per-complete-run gate; the existing results are retained and the gate remains FAIL."
        )
    stage5 = [
        "Integrate official, version-pinned baseline adapters and user-provided weights without automatic downloads; preserve external core algorithms.",
        "Freeze a leakage-safe learned router with query-visible features only, then repeat fold-isolated calibration and testing.",
        "Make every stochastic control choose once per complete run and add the invariant to replay/grid tests before regenerating comparisons.",
        "Expand live execution beyond a one-sample smoke and report actual latency, memory, failures, and quality with uncertainty.",
        "Add VisA and other held-out datasets under the same support_set_id and evaluator-isolation protocol.",
    ]
    return {
        "claim_scope": "Stage 4 protocol/wrapper comparison only; not final SOTA",
        "expert_implementation_mode": IMPLEMENTATION_MODE,
        "primary_selection": {
            "aggregation": PRIMARY_AGGREGATION,
            "metric": PRIMARY_METRIC,
            "tie_break": "policy_name",
        },
        "five_fold_protocol": folds,
        "best_fixed_baseline": _policy_record(best_fixed),
        "best_rule_agent": _policy_record(best_rule),
        "best_learned_metadata_baseline": _policy_record(best_learned),
        "best_overall_realizable": _policy_record(best_overall),
        "fold_matched_run_level_oracle": _policy_record(oracle),
        "oracle_gap": _oracle_gap(best_overall, oracle),
        "cost_quality_frontier": [_policy_record(row) for row in frontier],
        "cost_quality_tradeoff": _cost_quality_tradeoff(best_fixed, best_rule),
        "num_cost_quality_fold_rows": len(cost_rows),
        "live_smoke": {
            "status": live_check.get("status", "FAIL"),
            **dict(live_check.get("stats", {})),
            "errors": list(live_check.get("errors", [])),
        },
        "limitations": limitations,
        "stage5_directions": stage5,
    }


def build_report(gate: Mapping[str, Any]) -> str:
    """Render the final Markdown report from an augmented gate payload."""

    summary = gate.get("report_summary", {})
    fixed = summary.get("best_fixed_baseline", {})
    rule = summary.get("best_rule_agent", {})
    learned = summary.get("best_learned_metadata_baseline", {})
    oracle = summary.get("fold_matched_run_level_oracle", {})
    gap = summary.get("oracle_gap", {})
    tradeoff = summary.get("cost_quality_tradeoff", {})
    live = summary.get("live_smoke", {})
    folds = summary.get("five_fold_protocol", {})
    lines = [
        "# Stage 4 最终 Gate 报告",
        "",
        f"> Gate 状态：**{gate.get('status', 'FAIL')}**。本报告仅总结当前 Stage 4 协议和本地 wrapper 实验，**不构成最终 NORM-Route SOTA 声明**。",
        "",
        "## 1. 结论与范围",
        "",
        f"- Expert implementation mode：{summary.get('expert_implementation_mode', 'unavailable')}",
        "- 当前数值用于验证输入隔离、五折路由、预算、回放、评估和 live wiring；不得与官方实现结果或最终 SOTA 等同。",
        f"- 选择口径：五折 test 的 `{PRIMARY_AGGREGATION}` `{PRIMARY_METRIC}` 均值；Oracle 是每折独立重算的 evaluator-only run-level reference。",
        "",
        "## 2. Gate 校验",
        "",
        "| Check | 状态 | 摘要 |",
        "| --- | --- | --- |",
    ]
    for check in gate.get("checks", []):
        lines.append(
            f"| `{_md(check.get('name'))}` | {check.get('status')} | {_md(check.get('summary'))} |"
        )
    failed = [check for check in gate.get("checks", []) if check.get("status") == "FAIL"]
    if failed:
        lines.extend(["", "阻塞项：", ""])
        for check in failed:
            errors = check.get("errors", [])
            detail = errors[0] if errors else "unknown validation failure"
            lines.append(f"- `{check.get('name')}`：{_md(detail)}")

    lines.extend(
        [
            "",
            "## 3. 五折协议",
            "",
            f"- split unit：`{folds.get('split_unit', 'seed')}`；fold 数：{folds.get('num_folds', 'unavailable')}。",
            "- 同一 fold 的 train/val/test seed 两两不重叠；每个 seed 恰好一次作为 test、一次作为 validation。",
            "",
            "| Fold | Train seeds | Val seed | Test seed |",
            "| --- | --- | --- | --- |",
        ]
    )
    for fold, values in folds.get("folds", {}).items():
        lines.append(
            f"| `{_md(fold)}` | {_join(values.get('train', []))} | "
            f"{_join(values.get('val', []))} | {_join(values.get('test', []))} |"
        )

    lines.extend(
        [
            "",
            "## 4. 五折结果摘要",
            "",
            "| 组别 | 最佳方法 | AUROC | AP | F1 | 估计耗时/样本 (ms) | Oracle AUROC gap |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            _result_row("Fixed baseline", fixed),
            _result_row("规则 Agent", rule),
            _result_row("Learned metadata baseline", learned),
            _result_row("Fold-matched run-level Oracle", oracle),
            "",
            f"- 最佳 fixed baseline：`{fixed.get('policy_name', 'unavailable')}`。",
            f"- 最佳规则 Agent：`{rule.get('policy_name', 'unavailable')}`。",
            f"- 最佳 learned metadata baseline：`{learned.get('policy_name', 'unavailable')}`。该方法是诊断 baseline，不是最终 NORM-Route 模型。",
            f"- 最佳可实现策略到 fold-matched Oracle 的 AUROC 差距：{_fmt(gap.get('auroc_gap'))}（`{gap.get('policy_name', 'unavailable')}` 对 `{gap.get('oracle_name', 'run_level_oracle')}`）。",
            "",
            "## 5. Cost-quality tradeoff",
            "",
            f"- 低成本端：`{tradeoff.get('low_cost_policy', 'unavailable')}`，AUROC={_fmt(tradeoff.get('low_cost_auroc'))}，估计 {_fmt(tradeoff.get('low_cost_runtime_ms'))} ms/样本。",
            f"- 高质量端：`{tradeoff.get('high_quality_policy', 'unavailable')}`，AUROC={_fmt(tradeoff.get('high_quality_auroc'))}，估计 {_fmt(tradeoff.get('high_quality_runtime_ms'))} ms/样本。",
            f"- 从低成本端切换到高质量端：AUROC {_signed(tradeoff.get('auroc_delta'))}，估计耗时 {_signed(tradeoff.get('runtime_delta_ms'))} ms/样本（{_signed(tradeoff.get('runtime_delta_percent'))}%）。回放耗时是 Stage 2 历史估计值，不是本次回放 wall-clock。",
            "- 可实现策略的均值 Pareto frontier："
            + ", ".join(
                f"`{row.get('policy_name')}` ({_fmt(row.get('auroc_mean'))}, {_fmt(row.get('average_estimated_runtime_ms_mean'))} ms)"
                for row in summary.get("cost_quality_frontier", [])
            )
            + "。",
            "",
            "## 6. Live smoke",
            "",
            f"- 状态：**{live.get('status', 'FAIL')}**；policy=`{live.get('policy_name', 'unavailable')}`；selected expert=`{live.get('selected_expert', 'unavailable')}`。",
            f"- 输入：dataset=`{live.get('dataset', 'unavailable')}`，category=`{live.get('category', 'unavailable')}`，K={live.get('k_shot', 'unavailable')}，seed={live.get('seed', 'unavailable')}。",
            f"- 决策/预测：{live.get('num_decisions', 'unavailable')}/{live.get('num_selected_predictions', 'unavailable')}；actual subprocess runtime={_fmt(live.get('actual_subprocess_runtime_ms'))} ms。",
            "- evaluator 输入哈希与 routed selected_predictions 一致，并在 live run 执行窗口内完成；单样本 smoke 只证明链路闭环。",
            "",
            "## 7. 当前限制",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary.get("limitations", []))
    lines.extend(["", "## 8. Stage 5 方向", ""])
    lines.extend(f"- {item}" for item in summary.get("stage5_directions", []))
    lines.extend(
        [
            "",
            "## 9. 可复现性",
            "",
            f"- Gate protocol：`{gate.get('protocol_version', 'unavailable')}`",
            f"- Generated at UTC：`{gate.get('generated_at_utc', 'unavailable')}`",
            f"- Git commit：`{gate.get('git_commit', 'unavailable')}`",
            f"- Runs root：`{gate.get('config', {}).get('runs_root', 'unavailable')}`",
            f"- Evaluation root：`{gate.get('config', {}).get('evaluation_root', 'unavailable')}`",
            f"- 记录的输入 SHA-256：{len(gate.get('input_sha256', {})) + len(gate.get('report_input_sha256', {}))} 个。",
            "",
        ]
    )
    return "\n".join(lines)


def _add_report_input_check(
    gate: dict[str, Any],
    errors: Sequence[str],
    policy_rows: Sequence[Mapping[str, str]],
    cost_rows: Sequence[Mapping[str, str]],
    split_audit: Mapping[str, Any],
) -> None:
    required_policies = FIXED_BASELINES | RULE_AGENTS | LEARNED_METADATA_BASELINES | {
        "run_level_oracle"
    }
    present = {row.get("policy_name", "") for row in policy_rows}
    check_errors = list(errors)
    missing = sorted(required_policies - present)
    if missing:
        check_errors.append(f"Policy summary is missing required rows for: {missing}")
    if not cost_rows:
        check_errors.append("Cost-quality CSV contains no rows")
    folds = split_audit.get("config", {}).get("folds", {}) if split_audit else {}
    if not isinstance(folds, Mapping) or len(folds) != 5:
        check_errors.append("Split audit does not define exactly five folds")
    check = {
        "name": "stage4_report_inputs",
        "status": "PASS" if not check_errors else "FAIL",
        "summary": (
            f"validated {len(policy_rows)} policy-summary rows, {len(cost_rows)} "
            "cost-quality rows, and five-fold config"
        ),
        "stats": {
            "num_policy_summary_rows": len(policy_rows),
            "num_cost_quality_rows": len(cost_rows),
            "num_folds": len(folds) if isinstance(folds, Mapping) else 0,
        },
        "errors": check_errors,
        "num_errors": len(check_errors),
    }
    gate["checks"].append(check)
    gate["num_checks"] = len(gate["checks"])
    gate["num_passed"] = sum(item["status"] == "PASS" for item in gate["checks"])
    gate["num_failed"] = sum(item["status"] == "FAIL" for item in gate["checks"])
    gate["num_skipped"] = sum(item["status"] == "SKIP" for item in gate["checks"])
    gate["errors"] = [
        f"{item['name']}: {error}"
        for item in gate["checks"]
        if item["status"] == "FAIL"
        for error in item.get("errors", [])
    ]
    gate["status"] = "PASS" if not gate["errors"] else "FAIL"


def _best_policy(
    rows: Sequence[Mapping[str, str]], names: Iterable[str]
) -> dict[str, str]:
    allowed = set(names)
    candidates = [row for row in rows if row.get("policy_name") in allowed]
    return dict(
        max(
            candidates,
            key=lambda row: (_number(row.get(PRIMARY_METRIC)), str(row.get("policy_name", ""))),
            default={},
        )
    )


def _pareto_frontier(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    candidates = [
        dict(row)
        for row in rows
        if math.isfinite(_number(row.get("auroc_mean")))
        and math.isfinite(_number(row.get("average_estimated_runtime_ms_mean")))
    ]
    frontier = []
    for row in candidates:
        quality = _number(row["auroc_mean"])
        cost = _number(row["average_estimated_runtime_ms_mean"])
        dominated = any(
            _number(other["auroc_mean"]) >= quality
            and _number(other["average_estimated_runtime_ms_mean"]) <= cost
            and (
                _number(other["auroc_mean"]) > quality
                or _number(other["average_estimated_runtime_ms_mean"]) < cost
            )
            for other in candidates
            if other is not row
        )
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: _number(row["average_estimated_runtime_ms_mean"]))


def _policy_record(row: Mapping[str, str]) -> dict[str, Any]:
    if not row:
        return {}
    numeric_fields = (
        "auroc_mean",
        "auroc_std",
        "ap_mean",
        "ap_std",
        "f1_mean",
        "f1_std",
        "oracle_regret_mean",
        "average_estimated_runtime_ms_mean",
        "average_tool_calls_mean",
    )
    return {
        "policy_name": row.get("policy_name", ""),
        "policy_kind": row.get("policy_kind", ""),
        "aggregation": row.get("aggregation", ""),
        **{field: _finite_or_none(row.get(field)) for field in numeric_fields},
        "num_folds": _int_or_none(row.get("num_folds")),
    }


def _oracle_gap(policy: Mapping[str, str], oracle: Mapping[str, str]) -> dict[str, Any]:
    policy_value = _finite_or_none(policy.get("auroc_mean"))
    oracle_value = _finite_or_none(oracle.get("auroc_mean"))
    return {
        "policy_name": policy.get("policy_name", ""),
        "oracle_name": oracle.get("policy_name", "run_level_oracle"),
        "policy_auroc": policy_value,
        "oracle_auroc": oracle_value,
        "auroc_gap": (
            oracle_value - policy_value
            if oracle_value is not None and policy_value is not None
            else None
        ),
        "reported_oracle_regret_mean": _finite_or_none(policy.get("oracle_regret_mean")),
    }


def _cost_quality_tradeoff(
    low_cost: Mapping[str, str], high_quality: Mapping[str, str]
) -> dict[str, Any]:
    low_runtime = _finite_or_none(low_cost.get("average_estimated_runtime_ms_mean"))
    high_runtime = _finite_or_none(high_quality.get("average_estimated_runtime_ms_mean"))
    low_quality = _finite_or_none(low_cost.get("auroc_mean"))
    high_quality_value = _finite_or_none(high_quality.get("auroc_mean"))
    runtime_delta = (
        high_runtime - low_runtime
        if high_runtime is not None and low_runtime is not None
        else None
    )
    return {
        "low_cost_policy": low_cost.get("policy_name", ""),
        "low_cost_runtime_ms": low_runtime,
        "low_cost_auroc": low_quality,
        "high_quality_policy": high_quality.get("policy_name", ""),
        "high_quality_runtime_ms": high_runtime,
        "high_quality_auroc": high_quality_value,
        "runtime_delta_ms": runtime_delta,
        "runtime_delta_percent": (
            runtime_delta / low_runtime * 100.0
            if runtime_delta is not None and low_runtime not in (None, 0.0)
            else None
        ),
        "auroc_delta": (
            high_quality_value - low_quality
            if high_quality_value is not None and low_quality is not None
            else None
        ),
    }


def _fold_protocol(split_audit: Mapping[str, Any]) -> dict[str, Any]:
    config = split_audit.get("config", {}) if isinstance(split_audit, Mapping) else {}
    folds = config.get("folds", {}) if isinstance(config, Mapping) else {}
    return {
        "protocol_version": config.get("protocol_version", ""),
        "split_unit": config.get("split_unit", ""),
        "expected_seeds": list(config.get("expected_seeds", [])),
        "num_folds": len(folds) if isinstance(folds, Mapping) else 0,
        "folds": {
            str(name): {
                split: list(values.get(split, []))
                for split in ("train", "val", "test")
            }
            for name, values in folds.items()
            if isinstance(values, Mapping)
        }
        if isinstance(folds, Mapping)
        else {},
    }


def _read_csv(path: Path, errors: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        errors.append(f"Missing {path}")
        return []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        errors.append(f"Could not read {path}: {exc}")
        return []


def _read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"Missing {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        errors.append(f"Could not read {path}: {exc}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"{path} root must be a JSON object")
        return {}
    return payload


def _number(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return parsed if math.isfinite(parsed) else float("-inf")


def _finite_or_none(value: Any) -> float | None:
    parsed = _number(value)
    return parsed if math.isfinite(parsed) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _result_row(label: str, row: Mapping[str, Any]) -> str:
    return (
        f"| {label} | `{row.get('policy_name', 'unavailable')}` | "
        f"{_fmt(row.get('auroc_mean'))} | {_fmt(row.get('ap_mean'))} | "
        f"{_fmt(row.get('f1_mean'))} | "
        f"{_fmt(row.get('average_estimated_runtime_ms_mean'))} | "
        f"{_fmt(row.get('oracle_regret_mean'))} |"
    )


def _fmt(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(parsed):
        return "N/A"
    return f"{parsed:.6f}"


def _signed(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(parsed):
        return "N/A"
    return f"{parsed:+.6f}"


def _join(values: Sequence[Any]) -> str:
    return ", ".join(str(value) for value in values)


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
