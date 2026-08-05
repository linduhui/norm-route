from __future__ import annotations

import json
from pathlib import Path

from src.normroute.cli import materialize_fbdp_ad_features as module


def _provenance() -> dict:
    return {
        "task_id": "task-0",
        "dataset": "mvtec",
        "category": "bottle",
        "k_shot": 2,
        "seed": 0,
        "support_set_id": "support-0",
        "encoder_fingerprint": "encoder",
        "query_image_sha256": "a" * 64,
        "support_image_sha256s": ["b" * 64, "c" * 64],
    }


def _normal() -> dict:
    return {
        **_provenance(),
        "niv": [0.1, 0.2, 0.3, 1.0, 1.0, 1.0],
        "query_global_residual": [0.1, -0.2],
        "query_global_residual_l2": 0.3,
        "patch_nn_q50": 0.1,
        "patch_nn_q90": 0.2,
        "patch_nn_q95": 0.3,
        "patch_nn_q99": 0.4,
        "patch_nn_mean": 0.2,
        "patch_nn_max": 0.5,
    }


def _fbdp() -> dict:
    return {
        **_provenance(),
        "query_bank_mode": "decoupled",
        "use_fbc_in_gate": True,
        "use_objectness_gate": True,
        "background_candidate_mode": "combined",
        "use_cross_support_consistency": True,
        "prototype_method": "kmeans",
        "consistency_mode": "local_window",
        "foreground_background_confusion": 0.2,
        "objectness_gate": 0.8,
        "objectness_contrast": 0.7,
        "support_reliability": 0.7,
        "support_assignment_confidence": 0.8,
        "prototype_compactness": 0.9,
        "foreground_support_coverage": 1.0,
        "foreground_candidate_ratio": 0.2,
        "background_candidate_ratio": 0.6,
        "leave_one_out_reconstruction_margin": 0.4,
        "leave_one_out_reconstruction_valid": True,
        "support_consistency_valid": True,
        "foreground_candidate_fallback": False,
        "residual_q50": 0.1,
        "residual_q90": 0.2,
        "residual_q95": 0.3,
        "residual_q99": 0.4,
        "residual_mean": 0.2,
        "residual_max": 0.5,
        "gated_residual_q50": 0.08,
        "gated_residual_q90": 0.16,
        "gated_residual_q95": 0.24,
        "gated_residual_q99": 0.32,
        "gated_residual_mean": 0.16,
        "gated_residual_max": 0.4,
        "foreground_background_margin_quantiles": [-0.4, 0.1, 0.2, 0.4],
        "foreground_background_margin_mean": 0.0,
        "foreground_background_margin_min": -0.8,
        "foreground_background_margin_max": 0.8,
        "assignment_entropy_quantiles": [0.1, 0.2, 0.3, 0.4],
        "assignment_entropy_mean": 0.2,
        "assignment_entropy_max": 0.5,
    }


def test_materializer_requires_bir_for_joint_view(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    status = module.main(
        [
            "--normal-signatures",
            str(tmp_path / "normal.parquet"),
            "--fbdp-signatures",
            str(tmp_path / "fbdp.jsonl"),
            "--fbdp-ablation",
            "full",
            "--feature-view",
            "normal_bir_fbdp",
            "--output-dir",
            str(output_dir),
            "--seed",
            "23",
        ]
    )

    assert status == 1
    failures = json.loads(
        (output_dir / module.FAILURES_NAME).read_text(encoding="utf-8")
    )
    run = json.loads(
        (output_dir / module.RUN_RECORD_NAME).read_text(encoding="utf-8")
    )
    assert "--bir-signatures" in failures[0]["message"]
    assert run["ok"] is False
    assert run["seed"] == 23
    assert run["failures"] == failures


def test_materializer_writes_a_provenance_checked_fbdp_view(
    tmp_path: Path, monkeypatch
) -> None:
    normal_path = tmp_path / "normal.parquet"
    fbdp_path = tmp_path / "fbdp.jsonl"
    normal_path.write_bytes(b"immutable-normal")
    fbdp_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "_read_normal_signatures", lambda _: [_normal()])
    monkeypatch.setattr(module, "_read_jsonl", lambda _: [_fbdp()])
    output_dir = tmp_path / "output"

    status = module.main(
        [
            "--normal-signatures",
            str(normal_path),
            "--fbdp-signatures",
            str(fbdp_path),
            "--fbdp-ablation",
            "full",
            "--feature-view",
            "normal_fbdp",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    rows = [
        json.loads(line)
        for line in (output_dir / module.ROUTER_FEATURES_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["feature_view"] == "normal_fbdp"
    assert rows[0]["fbdp_ablation_name"] == "full"
    assert any(name.startswith("fbdp_") for name in rows[0]["feature_names"])
