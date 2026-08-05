from __future__ import annotations

import json
from pathlib import Path

import pytest


np = pytest.importorskip("numpy")

from src.normroute.cli.audit_fbdp_ad import (  # noqa: E402
    AUDIT_REPORT_NAME,
    FBDPADAuditError,
    GROUP_STATISTICS_NAME,
    RUN_RECORD_NAME,
    audit_fbdp_ad_artifacts,
    main,
)
from src.normroute.router.fbdp_ad import compute_fbdp_ad  # noqa: E402
from src.normroute.router.fbdp_ad_pipeline import FBDPADSignature  # noqa: E402


def _record(*, task_id: str = "task-0", k_shot: int = 2):
    support = []
    for offset in range(k_shot):
        patches = np.tile(np.asarray([1.0, 0.0, 0.0]), (25, 1))
        patches[6:9] = np.asarray([0.0, 1.0, 0.05 * offset])
        patches[11:14] = np.asarray([0.0, 1.0, 0.05 * offset])
        patches[16:19] = np.asarray([0.0, 1.0, 0.05 * offset])
        support.append(patches)
    query = support[0].copy()
    result = compute_fbdp_ad(query, support, patch_grid_shape=(5, 5))
    return FBDPADSignature(
        task_id=task_id,
        dataset="mvtec",
        category="bottle",
        k_shot=k_shot,
        seed=0,
        support_set_id="support-0",
        encoder_fingerprint="encoder",
        query_image_sha256=f"query-{task_id}",
        support_image_sha256s=tuple(
            f"support-hash-{index}" for index in range(k_shot)
        ),
        result=result,
    ).to_record()


def _task(task_id: str = "task-0", k_shot: int = 2):
    return {
        "task_id": task_id,
        "dataset": "mvtec",
        "category": "bottle",
        "k_shot": k_shot,
        "seed": 0,
        "support_set_id": "support-0",
        "query_path": "unused.png",
    }


def test_label_free_audit_checks_coverage_and_numerical_invariants() -> None:
    record = _record()
    report, groups = audit_fbdp_ad_artifacts([_task()], [record], [])

    assert report["label_free"] is True
    assert report["coverage"] == {
        "task_count": 1,
        "signature_count": 1,
        "failure_count": 0,
        "complete": True,
    }
    assert report["numerical_invariants"]["finite"] is True
    assert len(groups) == 1
    assert groups[0]["task_count"] == 1

    bad_gate = dict(record)
    bad_gate["gated_residual_quantiles"] = [1.0, 1.0, 1.0, 1.0]
    with pytest.raises(FBDPADAuditError, match="gate scaling"):
        audit_fbdp_ad_artifacts([_task()], [bad_gate], [])

    with pytest.raises(FBDPADAuditError, match="explicit FBDP failures"):
        audit_fbdp_ad_artifacts(
            [_task()], [], [{"task_id": "task-0", "message": "failed"}]
        )


def test_audit_cli_writes_reproducible_outputs(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    signatures_path = tmp_path / "signatures.jsonl"
    failures_path = tmp_path / "failures.json"
    output_dir = tmp_path / "audit"
    tasks_path.write_text(json.dumps(_task()) + "\n", encoding="utf-8")
    signatures_path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    failures_path.write_text("[]\n", encoding="utf-8")

    assert main(
        [
            "--tasks",
            str(tasks_path),
            "--signatures",
            str(signatures_path),
            "--failures",
            str(failures_path),
            "--output-dir",
            str(output_dir),
        ]
    ) == 0
    assert (output_dir / AUDIT_REPORT_NAME).is_file()
    assert (output_dir / GROUP_STATISTICS_NAME).is_file()
    run = json.loads((output_dir / RUN_RECORD_NAME).read_text(encoding="utf-8"))
    assert run["ok"] is True
    assert run["config"]["seed"] == 0
    assert run["failures"] == []
