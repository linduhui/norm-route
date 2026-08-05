import json
from pathlib import Path

import pytest

from src.normroute.router.fbdp_ad_ablation import (
    FBDP_AD_ABLATIONS,
    FBDP_AD_ABLATION_PROTOCOL_VERSION,
    fbdp_ad_ablation_records,
    validate_fbdp_ad_ablation_record,
)
from src.normroute.router.feature_bundle import (
    ROUTER_FEATURE_VIEWS,
    RouterFeatureBundleError,
    build_router_feature_bundle,
    build_router_feature_bundles,
)


def _provenance() -> dict:
    return {
        "task_id": "task-0",
        "dataset": "mvtec",
        "category": "bottle",
        "k_shot": 2,
        "seed": 7,
        "support_set_id": "support-0",
        "encoder_fingerprint": "encoder-v1",
        "query_image_sha256": "a" * 64,
        "support_image_sha256s": ["b" * 64, "c" * 64],
    }


def _normal() -> dict:
    return {
        **_provenance(),
        "niv": [0.1, 0.2, 0.3, 1.0, 1.0, 1.0],
        "query_global_residual": [0.1, -0.2, 0.3],
        "query_global_residual_l2": 0.4,
        "patch_nn_q50": 0.1,
        "patch_nn_q90": 0.2,
        "patch_nn_q95": 0.3,
        "patch_nn_q99": 0.4,
        "patch_nn_mean": 0.15,
        "patch_nn_max": 0.5,
    }


def _bir() -> dict:
    return {
        **_provenance(),
        "normalization_sha256": "d" * 64,
        "alignment_fingerprint": "align-v1",
        "support_bai": 0.1,
        "support_bai_std": 0.02,
        "query_bai": 0.2,
        "query_support_boundary_shift": 0.1,
        "absolute_boundary_shift": 0.1,
    }


def _fbdp() -> dict:
    return {
        **_provenance(),
        "foreground_background_confusion": 0.2,
        "objectness_gate": 0.8,
        "objectness_contrast": 0.7,
        "support_reliability": 0.75,
        "support_assignment_confidence": 0.9,
        "prototype_compactness": 0.85,
        "foreground_support_coverage": 1.0,
        "foreground_candidate_ratio": 0.2,
        "background_candidate_ratio": 0.6,
        "leave_one_out_reconstruction_margin": 0.5,
        "leave_one_out_reconstruction_valid": True,
        "support_consistency_valid": True,
        "foreground_candidate_fallback": False,
        "residual_q50": 0.1,
        "residual_q90": 0.2,
        "residual_q95": 0.3,
        "residual_q99": 0.4,
        "residual_mean": 0.15,
        "residual_max": 0.5,
        "gated_residual_q50": 0.08,
        "gated_residual_q90": 0.16,
        "gated_residual_q95": 0.24,
        "gated_residual_q99": 0.32,
        "gated_residual_mean": 0.12,
        "gated_residual_max": 0.4,
        "foreground_background_margin_quantiles": [-0.4, 0.1, 0.2, 0.4],
        "foreground_background_margin_mean": 0.0,
        "foreground_background_margin_min": -0.8,
        "foreground_background_margin_max": 0.8,
        "assignment_entropy_quantiles": [0.1, 0.2, 0.3, 0.4],
        "assignment_entropy_mean": 0.15,
        "assignment_entropy_max": 0.5,
        "query_bank_mode": "decoupled",
        "use_fbc_in_gate": True,
        "use_objectness_gate": True,
        "background_candidate_mode": "combined",
        "use_cross_support_consistency": True,
        "prototype_method": "kmeans",
        "consistency_mode": "local_window",
    }


def test_all_four_router_feature_views_are_strict_and_executable() -> None:
    assert ROUTER_FEATURE_VIEWS == (
        "normal_only",
        "normal_bir",
        "normal_fbdp",
        "normal_bir_fbdp",
    )
    normal = build_router_feature_bundle(_normal(), feature_view="normal_only")
    bir = build_router_feature_bundle(
        _normal(), _bir(), ablation="sigma_l2", feature_view="normal_bir"
    )
    fbdp = build_router_feature_bundle(
        _normal(), fbdp_ad_signature=_fbdp(), feature_view="normal_fbdp"
    )
    joint = build_router_feature_bundle(
        _normal(),
        _bir(),
        fbdp_ad_signature=_fbdp(),
        ablation="sigma_l2",
        feature_view="normal_bir_fbdp",
    )

    assert all(name.startswith("normal_") for name in normal.feature_names)
    assert any(name.startswith("bir_") for name in bir.feature_names)
    assert not any(name.startswith("fbdp_") for name in bir.feature_names)
    assert any(name.startswith("fbdp_") for name in fbdp.feature_names)
    assert not any(name.startswith("bir_") for name in fbdp.feature_names)
    assert any(name.startswith("bir_") for name in joint.feature_names)
    assert any(name.startswith("fbdp_") for name in joint.feature_names)
    assert len(joint.values) > len(bir.values)
    assert len(joint.values) > len(fbdp.values)


def test_fbdp_ablation_registry_controls_router_evidence() -> None:
    full = build_router_feature_bundle(
        _normal(), fbdp_ad_signature=_fbdp(), feature_view="normal_fbdp"
    )
    raw = build_router_feature_bundle(
        _normal(),
        fbdp_ad_signature=_fbdp(),
        feature_view="normal_fbdp",
        fbdp_ablation="raw_residual_only",
    )

    assert any(name.startswith("fbdp_gated_") for name in full.feature_names)
    assert not any(name.startswith("fbdp_gated_") for name in raw.feature_names)
    assert [record["name"] for record in fbdp_ad_ablation_records()] == list(
        FBDP_AD_ABLATIONS
    )
    config = json.loads(
        Path("configs/stage5/fbdp_ad_ablations.json").read_text(encoding="utf-8")
    )
    assert config["protocol_version"] == FBDP_AD_ABLATION_PROTOCOL_VERSION
    assert config["ablation_order"] == list(FBDP_AD_ABLATIONS)


def test_fbdp_join_rejects_provenance_or_task_coverage_drift() -> None:
    bad = {**_fbdp(), "support_set_id": "different"}
    with pytest.raises(RouterFeatureBundleError, match="support_set_id"):
        build_router_feature_bundle(
            _normal(), fbdp_ad_signature=bad, feature_view="normal_fbdp"
        )

    other = {**_fbdp(), "task_id": "task-1"}
    with pytest.raises(RouterFeatureBundleError, match="task sets disagree"):
        build_router_feature_bundles(
            [_normal()],
            fbdp_ad_signatures=[other],
            feature_view="normal_fbdp",
        )


def test_fbdp_ablation_materialization_rejects_compute_mismatch() -> None:
    validate_fbdp_ad_ablation_record(_fbdp(), "full")
    validate_fbdp_ad_ablation_record(
        {**_fbdp(), "query_bank_mode": "single"}, "single_bank"
    )
    with pytest.raises(ValueError, match="compute settings disagree"):
        validate_fbdp_ad_ablation_record(_fbdp(), "single_bank")
