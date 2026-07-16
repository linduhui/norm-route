from scripts.generate_stage4_gate_report import analyze_stage4_results, build_report


def _row(name: str, auroc: float, runtime: float, *, kind: str = "realizable_policy") -> dict[str, str]:
    return {
        "policy_name": name,
        "policy_kind": kind,
        "aggregation": "micro",
        "num_folds": "5",
        "auroc_mean": str(auroc),
        "auroc_std": "0.01",
        "ap_mean": "0.7",
        "ap_std": "0.01",
        "f1_mean": "0.5",
        "f1_std": "0.01",
        "oracle_regret_mean": str(0.9 - auroc),
        "average_estimated_runtime_ms_mean": str(runtime),
        "average_tool_calls_mean": "1",
    }


def test_stage4_report_names_best_groups_and_disclaims_sota() -> None:
    rows = [
        _row("always_anomalydino", 0.5, 70.0),
        _row("always_patchcore", 0.55, 50.0),
        _row("always_winclip", 0.45, 80.0),
        _row("random_seeded", 0.4, 65.0),
        _row("fastest_expert", 0.6, 45.0),
        _row("category_prior", 0.61, 60.0),
        _row("category_shot_prior", 0.62, 62.0),
        _row("cost_aware", 0.65, 58.0),
        _row("decision_tree_metadata", 0.64, 63.0),
        _row("multinomial_logistic_metadata", 0.63, 61.0),
        _row("run_level_oracle", 0.9, 60.0, kind="evaluator_only_reference"),
    ]
    split_audit = {
        "config": {
            "protocol_version": "stage4.seed_cv.v1",
            "split_unit": "seed",
            "expected_seeds": [0, 1, 2, 3, 4],
            "folds": {
                f"fold{index}": {"train": [2, 3, 4], "val": [1], "test": [0]}
                for index in range(5)
            },
        }
    }
    gate = {
        "status": "PASS",
        "protocol_version": "stage4.gate.v1",
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "git_commit": "abc",
        "config": {"runs_root": "runs", "evaluation_root": "eval"},
        "checks": [
            {
                "name": "live_smoke",
                "status": "PASS",
                "summary": "ok",
                "stats": {"num_decisions": 1, "num_selected_predictions": 1},
                "errors": [],
            }
        ],
        "input_sha256": {},
    }
    analysis = analyze_stage4_results(
        gate=gate,
        policy_rows=rows,
        cost_rows=rows,
        split_audit=split_audit,
    )
    gate["report_summary"] = analysis
    gate["report_input_sha256"] = {}
    report = build_report(gate)

    assert analysis["best_fixed_baseline"]["policy_name"] == "fastest_expert"
    assert analysis["best_rule_agent"]["policy_name"] == "cost_aware"
    assert analysis["best_learned_metadata_baseline"]["policy_name"] == "decision_tree_metadata"
    assert analysis["oracle_gap"]["auroc_gap"] == 0.25
    assert "no-download" in report
    assert "五折协议" in report
    assert "不构成最终 NORM-Route SOTA 声明" in report
