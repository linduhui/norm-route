import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image


np = pytest.importorskip("numpy")

from src.normroute.router.bir_ad import PatchAlignedImage  # noqa: E402
from src.normroute.router.bir_ad_pipeline import (  # noqa: E402
    BIR_AD_FAILURES_NAME,
    BIR_AD_SIGNATURES_NAME,
    BIRADArtifactError,
    BIRADPipelineError,
    BIRADTaskEncoder,
    build_bir_ad_signatures,
    fit_fold_bir_ad_normalization,
    load_fold_normalization_artifact,
)
from src.normroute.router.feature_cache import FeatureCache, file_sha256  # noqa: E402


class _FakeEncoder:
    fingerprint = "bir-ad-pipeline-test-encoder"

    def __init__(self) -> None:
        self.encoded_paths = []

    def encode_global_and_patches(self, paths):
        globals_ = []
        patches = []
        for path in paths:
            self.encoded_paths.append(Path(path))
            value = float(sum(Path(path).read_bytes()) % 17) / 17.0
            globals_.append([value + 1.0, value + 2.0, value + 3.0])
            patches.append(
                [
                    [value + 1.0, 0.0, 0.0],
                    [0.0, value + 1.5, 0.0],
                    [0.0, 0.0, value + 2.0],
                    [value + 0.5, value + 1.0, value + 1.5],
                ]
            )
        return {
            "global_features": np.asarray(globals_, dtype=np.float32),
            "patch_features": np.asarray(patches, dtype=np.float32),
        }


class _FakeAligner:
    fingerprint = "fake-spatial-transform-v1"

    def __init__(self) -> None:
        self.aligned_paths = []

    def align_images_for_patches(self, paths, *, patch_grid_shape):
        aligned = []
        for path in paths:
            image_path = Path(path)
            self.aligned_paths.append(image_path)
            with Image.open(image_path) as image:
                pixels = np.asarray(
                    image.convert("RGB").resize((8, 8)), dtype=np.float32
                ) / 255.0
            aligned.append(
                PatchAlignedImage(
                    pixels=pixels,
                    patch_grid_shape=patch_grid_shape,
                    source_image_sha256=file_sha256(image_path),
                    transform_fingerprint=self.fingerprint,
                )
            )
        return aligned


def _image(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (10, 6), color=color).save(path)
    return path


def _inputs(tmp_path: Path):
    query_train = _image(tmp_path / "query-train.png", (10, 20, 30))
    query_val = _image(tmp_path / "query-val.png", (30, 40, 50))
    support0 = _image(tmp_path / "support0.png", (60, 10, 10))
    support1 = _image(tmp_path / "support1.png", (10, 60, 10))
    support2 = _image(tmp_path / "support2.png", (10, 10, 60))
    tasks = [
        {
            "task_id": "task-train",
            "dataset": "mvtec",
            "category": "bottle",
            "k_shot": 2,
            "seed": 0,
            "support_set_id": "support-train",
            "query_path": str(query_train),
        },
        {
            "task_id": "task-val",
            "dataset": "mvtec",
            "category": "cable",
            "k_shot": 1,
            "seed": 0,
            "support_set_id": "support-val",
            "query_path": str(query_val),
        },
    ]
    supports = [
        {
            "support_set_id": "support-train",
            "dataset": "mvtec",
            "category": "bottle",
            "k_shot": 2,
            "seed": 0,
            "split": "train/good",
            "image_id": "support0",
            "image_path": str(support0),
        },
        {
            "support_set_id": "support-train",
            "dataset": "mvtec",
            "category": "bottle",
            "k_shot": 2,
            "seed": 0,
            "split": "train/good",
            "image_id": "support1",
            "image_path": str(support1),
        },
        {
            "support_set_id": "support-val",
            "dataset": "mvtec",
            "category": "cable",
            "k_shot": 1,
            "seed": 0,
            "split": "train/good",
            "image_id": "support2",
            "image_path": str(support2),
        },
    ]
    folds = [
        {"fold": "fold0", "task_id": "task-train", "split": "train"},
        {"fold": "fold0", "task_id": "task-val", "split": "val"},
    ]
    return tasks, supports, folds, (query_train, query_val, support0, support1, support2)


def test_fold_normalization_uses_only_unique_train_support_images(
    tmp_path: Path,
) -> None:
    tasks, supports, folds, paths = _inputs(tmp_path)
    encoder = _FakeEncoder()
    aligner = _FakeAligner()
    cache = FeatureCache(
        tmp_path / "cache",
        encoder_fingerprint=encoder.fingerprint,
    )
    artifact_path = tmp_path / "fold0-normalization.json"

    artifact = fit_fold_bir_ad_normalization(
        tasks,
        supports,
        folds,
        fold="fold0",
        feature_cache=cache,
        feature_encoder=encoder,
        pixel_aligner=aligner,
        output_path=artifact_path,
    )
    loaded = load_fold_normalization_artifact(artifact_path)

    assert artifact.statistics.is_fitted
    assert artifact.train_categories == ("bottle",)
    assert artifact.train_support_set_ids == ("support-train",)
    assert artifact.training_image_sha256s == tuple(
        sorted((file_sha256(paths[2]), file_sha256(paths[3])))
    )
    assert file_sha256(paths[4]) not in artifact.training_image_sha256s
    assert loaded.to_dict() == artifact.to_dict()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["split"] == "train"
    assert payload["fold"] == "fold0"

    payload["training_image_sha256s"][0] = "g" * 64
    payload["training_image_sha256s"].sort()
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BIRADArtifactError, match="SHA-256"):
        load_fold_normalization_artifact(artifact_path)


def test_batch_encoder_reuses_cache_and_writes_router_ready_signatures(
    tmp_path: Path,
) -> None:
    tasks, supports, folds, paths = _inputs(tmp_path)
    encoder = _FakeEncoder()
    aligner = _FakeAligner()
    cache = FeatureCache(
        tmp_path / "cache",
        encoder_fingerprint=encoder.fingerprint,
    )
    fold_artifact = fit_fold_bir_ad_normalization(
        tasks,
        supports,
        folds,
        fold="fold0",
        feature_cache=cache,
        feature_encoder=encoder,
        pixel_aligner=aligner,
    )
    task_encoder = BIRADTaskEncoder(
        cache,
        encoder,
        normalization_artifact=fold_artifact,
        pixel_aligner=aligner,
    )

    signatures = task_encoder.encode_tasks(tasks, supports)
    encoded_after_first = list(encoder.encoded_paths)
    repeated = task_encoder.encode_tasks(list(reversed(tasks)), list(reversed(supports)))
    signature_path, failures_path = build_bir_ad_signatures(
        tasks,
        supports,
        encoder=task_encoder,
        output_dir=tmp_path / "outputs",
    )

    assert [item.task_id for item in signatures] == ["task-train", "task-val"]
    assert [item.to_record() for item in repeated] == [
        item.to_record() for item in signatures
    ]
    assert encoder.encoded_paths == encoded_after_first
    assert len(set(encoded_after_first)) == len(paths)
    assert all(item.result.query.alignment_verified for item in signatures)
    assert signatures[0].result.support_boundary_consistency_valid is True
    assert signatures[1].result.support_boundary_consistency_valid is False
    assert signature_path.name == BIR_AD_SIGNATURES_NAME
    assert failures_path.name == BIR_AD_FAILURES_NAME
    assert json.loads(failures_path.read_text(encoding="utf-8")) == []
    records = [
        json.loads(line)
        for line in signature_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["task_id"] for record in records] == [
        "task-train",
        "task-val",
    ]
    assert all(len(record["bai_vector"]) == 5 for record in records)
    assert all(len(record["clear_representation"]) == 3 for record in records)
    assert all(record["normalization_sha256"] for record in records)


def test_batch_encoder_rejects_non_train_good_support(tmp_path: Path) -> None:
    tasks, supports, folds, _ = _inputs(tmp_path)
    supports[0]["split"] = "test/bad"
    encoder = _FakeEncoder()
    cache = FeatureCache(
        tmp_path / "cache",
        encoder_fingerprint=encoder.fingerprint,
    )

    with pytest.raises(BIRADPipelineError, match="train/good"):
        fit_fold_bir_ad_normalization(
            tasks,
            supports,
            folds,
            fold="fold0",
            feature_cache=cache,
            feature_encoder=encoder,
            pixel_aligner=_FakeAligner(),
        )


def test_batch_encoder_rejects_fold_alignment_fingerprint_drift(
    tmp_path: Path,
) -> None:
    tasks, supports, folds, _ = _inputs(tmp_path)
    encoder = _FakeEncoder()
    aligner = _FakeAligner()
    cache = FeatureCache(
        tmp_path / "cache",
        encoder_fingerprint=encoder.fingerprint,
    )
    artifact = fit_fold_bir_ad_normalization(
        tasks,
        supports,
        folds,
        fold="fold0",
        feature_cache=cache,
        feature_encoder=encoder,
        pixel_aligner=aligner,
    )
    task_encoder = BIRADTaskEncoder(
        cache,
        encoder,
        normalization_artifact=replace(
            artifact, alignment_fingerprint="different-spatial-transform"
        ),
        pixel_aligner=aligner,
    )

    with pytest.raises(BIRADPipelineError, match="fold normalization"):
        task_encoder.encode_tasks(tasks, supports)


def test_batch_encoder_requires_verifiable_support_split(tmp_path: Path) -> None:
    tasks, supports, folds, _ = _inputs(tmp_path)
    for row in supports:
        row.pop("split")
    encoder = _FakeEncoder()
    cache = FeatureCache(
        tmp_path / "cache",
        encoder_fingerprint=encoder.fingerprint,
    )

    with pytest.raises(BIRADPipelineError, match="verifiable official"):
        fit_fold_bir_ad_normalization(
            tasks,
            supports,
            folds,
            fold="fold0",
            feature_cache=cache,
            feature_encoder=encoder,
            pixel_aligner=_FakeAligner(),
        )


def test_batch_encoder_rejects_byte_identical_supports(tmp_path: Path) -> None:
    tasks, supports, folds, paths = _inputs(tmp_path)
    paths[3].write_bytes(paths[2].read_bytes())
    encoder = _FakeEncoder()
    aligner = _FakeAligner()
    cache = FeatureCache(
        tmp_path / "cache",
        encoder_fingerprint=encoder.fingerprint,
    )
    artifact = fit_fold_bir_ad_normalization(
        tasks,
        supports,
        folds,
        fold="fold0",
        feature_cache=cache,
        feature_encoder=encoder,
        pixel_aligner=aligner,
    )
    task_encoder = BIRADTaskEncoder(
        cache,
        encoder,
        normalization_artifact=artifact,
        pixel_aligner=aligner,
    )

    with pytest.raises(BIRADPipelineError, match="byte-identical support"):
        task_encoder.encode_tasks(tasks, supports)
