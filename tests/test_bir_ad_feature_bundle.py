import json
import math
from pathlib import Path

import pytest

from src.normroute.router.bir_ad_ablation import (
    BIR_AD_ABLATIONS,
    bir_ad_ablation_records,
)
from src.normroute.router.feature_bundle import (
    RouterFeatureBundleError,
    build_router_feature_bundle,
    build_router_feature_bundles,
    write_router_feature_bundles_jsonl,
)


def _normal_record() -> dict:
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
        "niv": [0.1, 0.2, 0.3, 1.0, 1.0, 1.0],
        "query_global_residual": [0.25, -0.5, 0.75],
        "query_global_residual_l2": 0.935,
        "patch_nn_q50": 0.1,
        "patch_nn_q90": 0.2,
        "patch_nn_q95": 0.3,
        "patch_nn_q99": 0.4,
        "patch_nn_mean": 0.15,
        "patch_nn_max": 0.5,
    }


def _bir_record() -> dict:
    return {
        "task_id": "task-0",
        "dataset": "mvtec",
        "category": "bottle",
        "k_shot": 2,
        "seed": 7,
        "support_set_id": "support-0",
        "encoder_fingerprint": "encoder-v1",
        "normalization_sha256": "d" * 64,
        "alignment_fingerprint": "spatial-v1",
        "query_image_sha256": "a" * 64,
        "support_image_sha256s": ["c" * 64, "b" * 64],
        "support_bai": 0.3,
        "support_bai_std": 0.05,
        "query_bai": 0.4,
        "query_support_boundary_shift": 0.1,
        "absolute_boundary_shift": 0.1,
        "support_bai_reliability": 0.8,
        "query_bai_reliability": 0.7,
        "support_boundary_consistency": 0.9,
        "support_boundary_consistency_valid": True,
        "query_support_boundary_consistency": 0.6,
        "support_pixel_feature_disagreement": 0.2,
        "query_pixel_feature_disagreement": 0.25,
        "clear_representation": [1.0, 2.0, 3.0],
        "ambiguous_representation": [-1.0, -2.0, -3.0],
    }


def test_full_router_bundle_keeps_provenance_out_of_learned_vector(
    tmp_path: Path,
) -> None:
    bundle = build_router_feature_bundle(_normal_record(), _bir_record())
    output = write_router_feature_bundles_jsonl([bundle], tmp_path / "router.jsonl")
    record = json.loads(output.read_text(encoding="utf-8"))

    assert bundle.ablation_name == "full"
    assert "normal_niv_0000" in bundle.feature_names
    assert "bir_query_bai" in bundle.feature_names
    assert "bir_query_support_boundary_consistency" in bundle.feature_names
    assert "bir_clear_representation_0000" in bundle.feature_names
    assert not {
        "task_id",
        "dataset",
        "category",
        "seed",
        "support_set_id",
    }.intersection(bundle.feature_names)
    assert all(math.isfinite(value) for value in bundle.values)
    assert record["task_id"] == "task-0"
    assert len(record["feature_names"]) == len(record["values"])


def test_router_bundle_rejects_cross_task_or_cross_support_join() -> None:
    bir = _bir_record()
    bir["support_set_id"] = "wrong-support"
    with pytest.raises(RouterFeatureBundleError, match="support_set_id"):
        build_router_feature_bundle(_normal_record(), bir)

    bir = _bir_record()
    bir["support_image_sha256s"] = ["e" * 64, "f" * 64]
    with pytest.raises(RouterFeatureBundleError, match="support image hashes"):
        build_router_feature_bundle(_normal_record(), bir)


def test_router_ablation_registry_is_executable_and_nested() -> None:
    lengths = {}
    for name, spec in BIR_AD_ABLATIONS.items():
        assert spec.compute_kwargs()["clarity_weights"] == spec.clarity_weights
        bundle = build_router_feature_bundle(
            _normal_record(), _bir_record(), ablation=name
        )
        lengths[name] = len(bundle.values)
        assert bundle.ablation_name == name

    assert lengths["full"] > lengths["plus_support_consistency"]
    assert lengths["plus_support_consistency"] > lengths["sigma_l2"]
    assert lengths["representations_only"] < lengths["full"]
    assert [record["name"] for record in bir_ad_ablation_records()] == list(
        BIR_AD_ABLATIONS
    )


def test_ablation_config_matches_frozen_registry() -> None:
    config = json.loads(
        Path("configs/stage5/bir_ad_ablations.json").read_text(encoding="utf-8")
    )
    assert config["ablation_order"] == list(BIR_AD_ABLATIONS)
    assert "support_set_id" in config["comparison_contract"]


def test_router_bundle_imputes_invalid_support_consistency_with_mask() -> None:
    bir = _bir_record()
    bir["support_boundary_consistency"] = None
    bir["support_boundary_consistency_valid"] = False
    bundle = build_router_feature_bundle(
        _normal_record(), bir, ablation="plus_support_consistency"
    )
    values = dict(zip(bundle.feature_names, bundle.values))
    assert values["bir_support_boundary_consistency"] == 0.0
    assert values["bir_support_boundary_consistency_valid"] == 0.0


def test_batch_router_join_rejects_missing_task_and_mixed_fold_artifacts(
    tmp_path: Path,
) -> None:
    with pytest.raises(RouterFeatureBundleError, match="task sets disagree"):
        build_router_feature_bundles(
            [_normal_record()],
            [{**_bir_record(), "task_id": "different"}],
        )

    first = build_router_feature_bundle(_normal_record(), _bir_record())
    normal = {**_normal_record(), "task_id": "task-1"}
    bir = {
        **_bir_record(),
        "task_id": "task-1",
        "normalization_sha256": "e" * 64,
    }
    second = build_router_feature_bundle(normal, bir)
    with pytest.raises(RouterFeatureBundleError, match="fold normalizations"):
        write_router_feature_bundles_jsonl(
            [first, second], tmp_path / "mixed.jsonl"
        )
