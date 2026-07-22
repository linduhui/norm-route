import csv
import hashlib
from pathlib import Path

import pytest


np = pytest.importorskip("numpy")

from src.normroute.router.feature_cache import (  # noqa: E402
    FeatureCache,
    FeatureCacheIntegrityError,
)
from src.normroute.router.normal_domain import (  # noqa: E402
    NormalDomainSignatureEncoder,
    build_normal_signatures,
    compute_normal_domain_statistics,
)


class _CountingEncoder:
    fingerprint = "counting-encoder-v1"

    def __init__(self) -> None:
        self.encoded_paths: list[Path] = []

    def encode_global_and_patches(self, paths):
        globals_ = []
        patches = []
        for raw_path in paths:
            path = Path(raw_path)
            self.encoded_paths.append(path)
            value = float(sum(path.read_bytes()) % 100)
            globals_.append([value, value + 1.0])
            patches.append([[value, value + 1.0], [value + 2.0, value + 3.0]])
        return np.asarray(globals_, dtype=np.float32), np.asarray(
            patches, dtype=np.float32
        )


def _write_image_bytes(path: Path, value: bytes) -> Path:
    path.write_bytes(value)
    return path


def test_cache_is_fp16_reproducible_and_resume_deduplicates_by_image_hash(
    tmp_path: Path,
) -> None:
    query = _write_image_bytes(tmp_path / "query.bin", b"query")
    query_copy = _write_image_bytes(tmp_path / "query-copy.bin", b"query")
    support = _write_image_bytes(tmp_path / "support.bin", b"support")
    manifest = tmp_path / "feature_cache_manifest.csv"
    encoder = _CountingEncoder()
    cache = FeatureCache(
        tmp_path / "cache",
        encoder_fingerprint=encoder.fingerprint,
        manifest_path=manifest,
    )

    first = cache.get_or_encode([query, support, query, query_copy], encoder)
    first_manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    first_feature_hashes = [
        (item.image_sha256, item.global_feature.tobytes(), item.patch_features.tobytes())
        for item in first
    ]

    assert len(encoder.encoded_paths) == 2
    assert first[0].image_sha256 == first[2].image_sha256 == first[3].image_sha256
    assert all(item.global_feature.dtype == np.float16 for item in first)
    assert all(item.patch_features.dtype == np.float16 for item in first)

    resumed = FeatureCache(
        tmp_path / "cache",
        encoder_fingerprint=encoder.fingerprint,
        manifest_path=manifest,
    )
    second = resumed.get_or_encode([support, query_copy, query], encoder)

    assert len(encoder.encoded_paths) == 2
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == first_manifest_hash
    assert cache.validate() == resumed.validate()
    assert sorted(set(
        (item.image_sha256, item.global_feature.tobytes(), item.patch_features.tobytes())
        for item in second
    )) == sorted(set(first_feature_hashes))


def test_resume_fails_on_cached_feature_hash_mismatch(tmp_path: Path) -> None:
    image = _write_image_bytes(tmp_path / "image.bin", b"image")
    encoder = _CountingEncoder()
    cache = FeatureCache(tmp_path / "cache", encoder_fingerprint=encoder.fingerprint)
    cache.get_or_encode([image], encoder)

    with cache.manifest_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    patch_path = cache.cache_dir / row["patch_path"]
    patch_path.write_bytes(patch_path.read_bytes() + b"corruption")

    with pytest.raises(FeatureCacheIntegrityError, match="patch_sha256 mismatch"):
        cache.get_or_encode([image], encoder)
    assert len(encoder.encoded_paths) == 1


def test_resume_fails_when_manifest_integrity_anchor_is_modified(tmp_path: Path) -> None:
    image = _write_image_bytes(tmp_path / "image.bin", b"image")
    encoder = _CountingEncoder()
    cache = FeatureCache(tmp_path / "cache", encoder_fingerprint=encoder.fingerprint)
    cache.get_or_encode([image], encoder)
    original = cache.manifest_path.read_text(encoding="utf-8")
    cache.manifest_path.write_text(
        original.replace("float16", "float32"), encoding="utf-8"
    )

    with pytest.raises(FeatureCacheIntegrityError, match="manifest disagrees"):
        cache.get_or_encode([image], encoder)
    assert len(encoder.encoded_paths) == 1


def test_duplicate_query_across_k_seed_tasks_is_encoded_once(tmp_path: Path) -> None:
    query = _write_image_bytes(tmp_path / "query.bin", b"same-query")
    support0 = _write_image_bytes(tmp_path / "support0.bin", b"support-0")
    support1 = _write_image_bytes(tmp_path / "support1.bin", b"support-1")
    encoder = _CountingEncoder()
    cache = FeatureCache(tmp_path / "cache", encoder_fingerprint=encoder.fingerprint)
    signature_encoder = NormalDomainSignatureEncoder(cache, encoder)
    tasks = [
        {
            "task_id": "task-seed0",
            "dataset": "mvtec",
            "category": "bottle",
            "k_shot": 1,
            "seed": 0,
            "support_set_id": "support-0",
            "query_path": str(query),
        },
        {
            "task_id": "task-seed1",
            "dataset": "mvtec",
            "category": "bottle",
            "k_shot": 1,
            "seed": 1,
            "support_set_id": "support-1",
            "query_path": str(query),
        },
    ]

    signatures = signature_encoder.encode_tasks(
        tasks,
        {"support-0": [support0], "support-1": [support1]},
    )

    assert len(signatures) == 2
    assert len(encoder.encoded_paths) == 3
    assert encoder.encoded_paths.count(query) == 1
    assert signatures[0].query_image_sha256 == signatures[1].query_image_sha256


def test_normal_domain_statistics_are_support_permutation_invariant() -> None:
    query_global = np.asarray([2.0, 3.0], dtype=np.float16)
    query_patches = np.asarray([[1.0, 0.0], [3.0, 0.0]], dtype=np.float16)
    support_globals = np.asarray([[0.0, 0.0], [2.0, 2.0]], dtype=np.float16)
    support_patches = [
        np.asarray([[0.0, 0.0]], dtype=np.float16),
        np.asarray([[2.0, 0.0]], dtype=np.float16),
    ]

    first = compute_normal_domain_statistics(
        query_global, query_patches, support_globals, support_patches
    )
    second = compute_normal_domain_statistics(
        query_global,
        query_patches,
        support_globals[::-1],
        list(reversed(support_patches)),
    )

    assert np.array_equal(first.normal_global_prototype, second.normal_global_prototype)
    assert np.array_equal(
        first.normal_diagonal_variance, second.normal_diagonal_variance
    )
    assert np.array_equal(first.query_global_residual, second.query_global_residual)
    assert np.array_equal(
        first.patch_nearest_distance_quantiles,
        second.patch_nearest_distance_quantiles,
    )
    assert first.niv == second.niv
    assert first.query_global_residual_l2 == second.query_global_residual_l2
    assert np.array_equal(first.normal_global_prototype, np.asarray([1.0, 1.0]))
    assert np.array_equal(first.normal_diagonal_variance, np.asarray([1.0, 1.0]))
    assert first.niv == pytest.approx(np.sqrt(2.0))
    assert np.array_equal(first.query_global_residual, np.asarray([1.0, 2.0]))
    assert np.array_equal(
        first.patch_nearest_distance_quantiles, np.ones(4, dtype=np.float32)
    )


def test_build_writes_required_manifest_and_parquet_artifacts(tmp_path: Path) -> None:
    parquet = pytest.importorskip("pyarrow.parquet")
    query = _write_image_bytes(tmp_path / "query.bin", b"query")
    support = _write_image_bytes(tmp_path / "support.bin", b"support")
    output_dir = tmp_path / "outputs"
    encoder = _CountingEncoder()
    tasks = [
        {
            "task_id": "task-0",
            "dataset": "mvtec",
            "category": "bottle",
            "k_shot": 1,
            "seed": 0,
            "support_set_id": "support-0",
            "query_path": str(query),
        }
    ]

    manifest_path, parquet_path = build_normal_signatures(
        tasks,
        {"support-0": [support]},
        feature_encoder=encoder,
        output_dir=output_dir,
    )
    first_manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    first_parquet_hash = hashlib.sha256(parquet_path.read_bytes()).hexdigest()

    assert manifest_path.name == "feature_cache_manifest.csv"
    assert parquet_path.name == "normal_signatures.parquet"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    assert len(manifest_rows) == 2
    assert {row["dtype"] for row in manifest_rows} == {"float16"}
    table = parquet.read_table(parquet_path)
    row = table.to_pylist()[0]
    assert table.num_rows == 1
    assert row["task_id"] == "task-0"
    assert len(row["normal_global_prototype"]) == 2
    assert len(row["normal_diagonal_variance"]) == 2
    assert len(row["query_global_residual"]) == 2
    assert len(row["patch_nearest_distance_quantiles"]) == 4

    build_normal_signatures(
        tasks,
        {"support-0": [support]},
        feature_encoder=encoder,
        output_dir=output_dir,
    )
    assert len(encoder.encoded_paths) == 2
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == first_manifest_hash
    assert hashlib.sha256(parquet_path.read_bytes()).hexdigest() == first_parquet_hash
