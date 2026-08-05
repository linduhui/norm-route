import json
from pathlib import Path

import pytest


np = pytest.importorskip("numpy")

from src.normroute.router.fbdp_ad_pipeline import (  # noqa: E402
    FBDP_AD_FAILURES_NAME,
    FBDP_AD_SIGNATURE_COLUMNS,
    FBDP_AD_SIGNATURES_NAME,
    FBDPADPipelineError,
    FBDPADTaskEncoder,
    build_fbdp_ad_signatures,
)
from src.normroute.router.feature_cache import FeatureCache  # noqa: E402


class _FakeEncoder:
    fingerprint = "fbdp-pipeline-test-encoder"

    def __init__(self) -> None:
        self.encoded_paths: list[Path] = []

    def encode_global_and_patches(self, paths):
        globals_ = []
        patch_sets = []
        for raw_path in paths:
            path = Path(raw_path)
            self.encoded_paths.append(path)
            token = float(sum(path.read_bytes()) % 11) / 100.0
            patches = np.tile(np.asarray([1.0, token, 0.0]), (25, 1))
            patches[6:9] = np.asarray([0.0, 1.0, token])
            patches[11:14] = np.asarray([0.0, 1.0, token])
            patches[16:19] = np.asarray([0.0, 1.0, token])
            globals_.append([1.0, token, 0.5])
            patch_sets.append(patches)
        return {
            "global_features": np.asarray(globals_, dtype=np.float32),
            "patch_features": np.asarray(patch_sets, dtype=np.float32),
        }


def _inputs(tmp_path: Path):
    paths = []
    for index in range(5):
        path = tmp_path / f"image-{index}.bin"
        path.write_bytes(bytes([index + 1]) * (index + 2))
        paths.append(path)
    tasks = [
        {
            "task_id": "task-a",
            "dataset": "mvtec",
            "category": "bottle",
            "k_shot": 2,
            "seed": 0,
            "support_set_id": "support-a",
            "query_path": str(paths[0]),
        },
        {
            "task_id": "task-b",
            "dataset": "mvtec",
            "category": "bottle",
            "k_shot": 2,
            "seed": 0,
            "support_set_id": "support-a",
            "query_path": str(paths[1]),
        },
        {
            "task_id": "task-c",
            "dataset": "mvtec",
            "category": "cable",
            "k_shot": 1,
            "seed": 1,
            "support_set_id": "support-b",
            "query_path": str(paths[2]),
        },
    ]
    supports = [
        {
            "support_set_id": "support-a",
            "dataset": "mvtec",
            "category": "bottle",
            "k_shot": 2,
            "seed": 0,
            "split": "train/good",
            "image_id": "support-0",
            "image_path": str(paths[3]),
        },
        {
            "support_set_id": "support-a",
            "dataset": "mvtec",
            "category": "bottle",
            "k_shot": 2,
            "seed": 0,
            "split": "train/good",
            "image_id": "support-1",
            "image_path": str(paths[4]),
        },
        {
            "support_set_id": "support-b",
            "dataset": "mvtec",
            "category": "cable",
            "k_shot": 1,
            "seed": 1,
            "split": "train/good",
            "image_id": "support-2",
            "image_path": str(paths[3]),
        },
    ]
    return tasks, supports, paths


def test_batch_encoder_reuses_features_and_support_contexts(tmp_path: Path) -> None:
    tasks, supports, paths = _inputs(tmp_path)
    backbone = _FakeEncoder()
    cache = FeatureCache(
        tmp_path / "cache", encoder_fingerprint=backbone.fingerprint
    )
    encoder = FBDPADTaskEncoder(
        cache,
        backbone,
        patch_grid_shape=(5, 5),
    )

    signatures = encoder.encode_tasks(tasks, supports)
    encoded_once = list(backbone.encoded_paths)
    repeated = encoder.encode_tasks(list(reversed(tasks)), list(reversed(supports)))

    assert [item.task_id for item in signatures] == ["task-a", "task-b", "task-c"]
    assert [item.to_record() for item in repeated] == [
        item.to_record() for item in signatures
    ]
    assert len(set(encoded_once)) == len(paths)
    assert backbone.encoded_paths == encoded_once
    assert encoder.last_run_statistics == {
        "task_count": 3,
        "unique_image_count": 5,
        "unique_support_context_count": 2,
        "support_context_cache_hits": 1,
    }
    records = [item.to_record() for item in signatures]
    assert all(tuple(record) == FBDP_AD_SIGNATURE_COLUMNS for record in records)
    assert records[0]["leave_one_out_reconstruction_valid"] is True
    assert records[2]["leave_one_out_reconstruction_valid"] is False
    assert records[2]["support_consistency_valid"] is False
    assert all(len(record["residual_quantiles"]) == 4 for record in records)


def test_batch_writer_has_complete_success_or_explicit_failures(
    tmp_path: Path,
) -> None:
    tasks, supports, _ = _inputs(tmp_path)
    backbone = _FakeEncoder()
    encoder = FBDPADTaskEncoder(
        FeatureCache(tmp_path / "cache", encoder_fingerprint=backbone.fingerprint),
        backbone,
        patch_grid_shape=(5, 5),
    )
    signatures_path, failures_path = build_fbdp_ad_signatures(
        tasks, supports, encoder=encoder, output_dir=tmp_path / "success"
    )

    assert signatures_path.name == FBDP_AD_SIGNATURES_NAME
    assert failures_path.name == FBDP_AD_FAILURES_NAME
    assert len(signatures_path.read_text(encoding="utf-8").splitlines()) == 3
    assert json.loads(failures_path.read_text(encoding="utf-8")) == []

    supports[0]["split"] = "test/bad"
    with pytest.raises(FBDPADPipelineError):
        build_fbdp_ad_signatures(
            tasks,
            supports,
            encoder=encoder,
            output_dir=tmp_path / "failed",
        )
    failures = json.loads(
        (tmp_path / "failed" / FBDP_AD_FAILURES_NAME).read_text(encoding="utf-8")
    )
    assert len(failures) == len(tasks)
    assert {failure["task_id"] for failure in failures} == {
        task["task_id"] for task in tasks
    }
    assert (tmp_path / "failed" / FBDP_AD_SIGNATURES_NAME).read_text(
        encoding="utf-8"
    ) == ""
