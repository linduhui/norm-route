from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.normroute.cli.audit_teacher_ecpb_v3_closure import (
    _audit_clean_run,
    _audit_completion_state,
    _audit_teacher_rows,
)
from src.normroute.router.teacher import TEACHER_PROTOCOL_VERSION


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_clean_run_accepts_empty_failures_array(tmp_path: Path) -> None:
    failures = tmp_path / "failures.json"
    _write_json(failures, [])
    run = tmp_path / "run.json"
    _write_json(
        run,
        {
            "ok": True,
            "failures": [],
            "outputs": {"failures": str(failures)},
            "output_hashes": {"failures": _sha256(failures)},
        },
    )

    assert _audit_clean_run(run)["ok"] is True


def test_clean_run_rejects_non_array_failures_artifact(tmp_path: Path) -> None:
    failures = tmp_path / "failures.json"
    _write_json(failures, {"failures": []})
    run = tmp_path / "run.json"
    _write_json(
        run,
        {
            "ok": True,
            "failures": [],
            "outputs": {"failures": str(failures)},
            "output_hashes": {"failures": _sha256(failures)},
        },
    )

    with pytest.raises(ValueError, match="must be an array"):
        _audit_clean_run(run)


def test_completion_state_can_be_reaudited_after_success(tmp_path: Path) -> None:
    commit = "a" * 40
    (tmp_path / "STATUS").write_text(
        f"status=complete\ngit_commit={commit}\n", encoding="utf-8"
    )
    audit = tmp_path / "final_acceptance_v3.json"
    audit.write_text('{"ok":true}\n', encoding="utf-8")
    paired = (
        tmp_path
        / "router"
        / "summary_multiseed_v3"
        / "multiseed_paired_deltas.csv"
    )
    paired.parent.mkdir(parents=True)
    paired.write_text("metric,delta\nutility,0.1\n", encoding="utf-8")
    (tmp_path / "COMPLETE.txt").write_text(
        "\n".join(
            (
                f"run_tag={tmp_path.name}",
                f"run_root={tmp_path}",
                f"git_commit={commit}",
                "router_runs=120",
                f"audit_sha256={_sha256(audit)}",
                f"multiseed_paired_deltas_sha256={_sha256(paired)}",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    _audit_completion_state(tmp_path, commit)

    audit.write_text('{"ok":false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="audit_sha256 mismatch"):
        _audit_completion_state(tmp_path, commit)


def test_completion_state_accepts_in_flight_audit(tmp_path: Path) -> None:
    commit = "b" * 40
    (tmp_path / "STATUS").write_text(
        f"status=auditing\ngit_commit={commit}\n", encoding="utf-8"
    )

    _audit_completion_state(tmp_path, commit)


def test_teacher_diagnostics_are_invariant_to_duplicate_query_variants() -> None:
    selected = {
        "validation_empirical_nll": 0.2,
        "validation_multiclass_brier": 0.1,
        "validation_ece": 0.05,
        "validation_distribution_entropy": 0.4,
        "validation_effective_class_count": 1.5,
        "validation_selection_accuracy": 0.8,
    }
    metadata = {
        "sharpness_strategy": "train_robust_gap_validation_oracle_nll",
        "objective_scale_scope": "train_categories_only",
        "objective_scale": 1.0,
        "minimum_probability": 0.01,
        "selected_sharpness_diagnostics": selected,
        "teacher_entropy_by_category": {"bottle": {"weighted_entropy": 0.4}},
        "temperature": 0.5,
    }

    def rows(specification: tuple[tuple[str, float, float], ...]) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for task_id, probability, weight in specification:
            for expert, value in (("a", probability), ("b", 1.0 - probability)):
                result.append(
                    {
                        "protocol_version": TEACHER_PROTOCOL_VERSION,
                        "evaluator_only": True,
                        "split": "train",
                        "fold": "fold0",
                        "task_id": task_id,
                        "expert_name": expert,
                        "soft_utility_probability": value,
                        "sample_weight": weight,
                    }
                )
        return result

    base = _audit_teacher_rows(
        rows((("normal-k1", 0.9, 0.5), ("anomaly-k1", 0.6, 0.5))),
        metadata,
        "fold0",
        legacy=False,
    )
    duplicated = _audit_teacher_rows(
        rows(
            (
                ("normal-k1", 0.9, 0.25),
                ("normal-k4", 0.9, 0.25),
                ("anomaly-k1", 0.6, 0.5),
            )
        ),
        metadata,
        "fold0",
        legacy=False,
    )

    assert duplicated["train_teacher_entropy"] == pytest.approx(
        base["train_teacher_entropy"]
    )
    assert duplicated["train_teacher_top1_probability"] == pytest.approx(
        base["train_teacher_top1_probability"]
    )
