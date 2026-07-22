"""Content-addressed global and patch feature cache for Stage 5.

The cache key is derived from the SHA-256 of the image bytes and an explicit
encoder fingerprint.  It intentionally does not include task, K-shot, seed,
or support-set identity, so the same query is encoded only once across Stage 5
comparisons.  Feature arrays are persisted as deterministic ``.npy`` files in
IEEE fp16 and accompanied by hashes that are checked on every resume read.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence


FEATURE_CACHE_PROTOCOL_VERSION = "stage5.feature_cache.v1"
FEATURE_CACHE_DTYPE = "float16"
FEATURE_CACHE_MANIFEST_NAME = "feature_cache_manifest.csv"

FEATURE_CACHE_MANIFEST_COLUMNS = (
    "protocol_version",
    "image_sha256",
    "encoder_fingerprint",
    "cache_key",
    "dtype",
    "global_shape",
    "patch_shape",
    "global_path",
    "global_sha256",
    "patch_path",
    "patch_sha256",
    "metadata_path",
    "metadata_sha256",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FeatureCacheError(RuntimeError):
    """Base error for Stage 5 feature-cache failures."""


class FeatureCacheIntegrityError(FeatureCacheError):
    """Raised when a resumed cache entry fails content or schema checks."""


class FeatureEncodingError(FeatureCacheError):
    """Raised when an encoder does not produce valid global and patch arrays."""


class FeatureCacheDependencyError(FeatureCacheError, ImportError):
    """Raised when the optional numeric Stage 5 environment is unavailable."""


@dataclass(frozen=True)
class EncodedFeatureBatch:
    """Canonical in-memory output accepted from a global/patch encoder."""

    global_features: Any
    patch_features: Any


@dataclass(frozen=True)
class CachedImageFeatures:
    """One verified image feature pair loaded from the cache."""

    image_sha256: str
    encoder_fingerprint: str
    cache_key: str
    global_feature: Any
    patch_features: Any


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of one local file or fail explicitly."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"feature-cache input does not exist: {file_path}")
    digest = hashlib.sha256()
    try:
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FeatureCacheError(f"could not hash {file_path}: {exc}") from exc
    return digest.hexdigest()


def encoder_fingerprint(encoder: Any) -> str:
    """Build a stable fingerprint from an encoder's frozen configuration.

    A provider can expose ``fingerprint`` directly.  Otherwise its complete
    ``config`` mapping (or ``config.to_dict()``) is hashed canonically.  This
    keeps caches from being reused across different checkpoints or transforms.
    """

    direct = getattr(encoder, "fingerprint", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    config = getattr(encoder, "config", None)
    if config is not None and hasattr(config, "to_dict"):
        config = config.to_dict()
    if isinstance(config, Mapping):
        payload = json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    raise FeatureCacheError(
        "encoder fingerprint is required when the encoder exposes no frozen config"
    )


class FeatureCache:
    """Resume-safe, content-addressed fp16 cache for image features."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        encoder_fingerprint: str | None = None,
        encoder_id: str | None = None,
        manifest_path: str | Path | None = None,
    ) -> None:
        fingerprint = encoder_fingerprint if encoder_fingerprint is not None else encoder_id
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError("encoder_fingerprint must be a non-empty string")
        if encoder_fingerprint is not None and encoder_id is not None:
            if encoder_fingerprint.strip() != encoder_id.strip():
                raise ValueError("encoder_fingerprint and encoder_id disagree")

        self.cache_dir = Path(cache_dir)
        self.encoder_fingerprint = fingerprint.strip()
        # A 128-bit directory token keeps Windows paths below MAX_PATH while
        # the full fingerprint and full cache key remain in verified metadata.
        self._encoder_key = hashlib.sha256(
            self.encoder_fingerprint.encode("utf-8")
        ).hexdigest()[:32]
        self.entries_dir = self.cache_dir / "entries" / self._encoder_key
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else self.cache_dir / FEATURE_CACHE_MANIFEST_NAME
        )
        self.entries_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_encoder(
        cls,
        cache_dir: str | Path,
        encoder: Any,
        *,
        manifest_path: str | Path | None = None,
    ) -> "FeatureCache":
        """Construct a cache bound to a provider's frozen identity."""

        return cls(
            cache_dir,
            encoder_fingerprint=encoder_fingerprint(encoder),
            manifest_path=manifest_path,
        )

    def get_or_encode(
        self,
        image_paths: Sequence[str | Path],
        encoder: Any,
        *,
        batch_size: int = 32,
    ) -> list[CachedImageFeatures]:
        """Load verified entries and encode each missing image hash once.

        The returned list preserves the caller's order and duplicates.  The
        encoder sees only one representative path for each missing image hash.
        Existing but corrupt entries are never silently replaced.
        """

        if isinstance(image_paths, (str, bytes, Path)) or not image_paths:
            raise ValueError("image_paths must be a non-empty sequence")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        # Treat an existing manifest as the resume integrity anchor.  This
        # catches altered metadata (including altered feature hashes) before a
        # newly generated manifest could overwrite the evidence.
        if self.manifest_path.is_file():
            self.validate()

        requested: list[tuple[Path, str]] = []
        representatives: dict[str, Path] = {}
        for raw_path in image_paths:
            path = Path(raw_path)
            image_hash = file_sha256(path)
            requested.append((path, image_hash))
            representatives.setdefault(image_hash, path)

        loaded: dict[str, CachedImageFeatures] = {}
        missing: list[tuple[str, Path]] = []
        for image_hash, path in sorted(representatives.items()):
            metadata_path = self._entry_paths(image_hash)[2]
            if metadata_path.is_file():
                loaded[image_hash] = self._load_verified(image_hash)
            else:
                global_path, patch_path, _ = self._entry_paths(image_hash)
                if global_path.exists() or patch_path.exists():
                    raise FeatureCacheIntegrityError(
                        f"incomplete feature-cache entry for image {image_hash}"
                    )
                missing.append((image_hash, path))

        for start in range(0, len(missing), batch_size):
            chunk = missing[start : start + batch_size]
            paths = [item[1] for item in chunk]
            output = self._call_encoder(encoder, paths)
            globals_, patches_ = _canonicalize_batch(output, len(chunk))
            for (image_hash, _), global_feature, patch_features in zip(
                chunk, globals_, patches_
            ):
                self._store(image_hash, global_feature, patch_features)
                loaded[image_hash] = self._load_verified(image_hash)

        self.write_manifest()
        return [loaded[image_hash] for _, image_hash in requested]

    def get(self, image_path: str | Path) -> CachedImageFeatures:
        """Load one previously cached image after verifying every hash."""

        image_hash = file_sha256(image_path)
        return self._load_verified(image_hash)

    def validate(self) -> list[dict[str, str]]:
        """Verify the current encoder namespace and return manifest rows."""

        rows = self._manifest_rows()
        if self.manifest_path.is_file():
            try:
                with self.manifest_path.open(newline="", encoding="utf-8-sig") as handle:
                    reader = csv.DictReader(handle)
                    if tuple(reader.fieldnames or ()) != FEATURE_CACHE_MANIFEST_COLUMNS:
                        raise FeatureCacheIntegrityError(
                            f"invalid feature-cache manifest columns: {reader.fieldnames}"
                        )
                    recorded = list(reader)
            except (OSError, UnicodeError, csv.Error) as exc:
                raise FeatureCacheIntegrityError(
                    f"could not read feature-cache manifest {self.manifest_path}: {exc}"
                ) from exc
            if recorded != rows:
                raise FeatureCacheIntegrityError(
                    "feature-cache manifest disagrees with verified cache entries"
                )
        return rows

    def write_manifest(self) -> Path:
        """Atomically write a deterministic manifest for all verified entries."""

        rows = self._manifest_rows()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                prefix=f".{self.manifest_path.name}.",
                suffix=".tmp",
                dir=self.manifest_path.parent,
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=FEATURE_CACHE_MANIFEST_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.manifest_path)
        except OSError as exc:
            if "temp_path" in locals():
                Path(temp_path).unlink(missing_ok=True)
            raise FeatureCacheError(
                f"could not write feature-cache manifest {self.manifest_path}: {exc}"
            ) from exc
        return self.manifest_path

    def _call_encoder(self, encoder: Any, paths: Sequence[Path]) -> Any:
        method = getattr(encoder, "encode_global_and_patches", None)
        if method is None:
            method = getattr(encoder, "encode_features", None)
        if method is None and callable(encoder):
            method = encoder
        if method is None:
            raise FeatureEncodingError(
                "encoder must implement encode_global_and_patches() or encode_features()"
            )
        try:
            return method(paths)
        except Exception as exc:
            if isinstance(exc, FeatureCacheError):
                raise
            raise FeatureEncodingError(
                f"global/patch feature encoding failed for {len(paths)} image(s): {exc}"
            ) from exc

    def _entry_paths(self, image_hash: str) -> tuple[Path, Path, Path]:
        if not _SHA256_RE.fullmatch(image_hash):
            raise ValueError(f"invalid image SHA-256: {image_hash!r}")
        entry_dir = self.entries_dir / image_hash[:2] / image_hash
        return (
            entry_dir / "global.npy",
            entry_dir / "patches.npy",
            entry_dir / "metadata.json",
        )

    def _store(self, image_hash: str, global_feature: Any, patch_features: Any) -> None:
        np = _numpy()
        global_path, patch_path, metadata_path = self._entry_paths(image_hash)
        entry_dir = metadata_path.parent
        entry_dir.mkdir(parents=True, exist_ok=True)

        global_array = np.ascontiguousarray(global_feature, dtype=np.float16)
        patch_array = np.ascontiguousarray(patch_features, dtype=np.float16)
        _atomic_save_npy(global_path, global_array, np)
        _atomic_save_npy(patch_path, patch_array, np)

        metadata = {
            "protocol_version": FEATURE_CACHE_PROTOCOL_VERSION,
            "image_sha256": image_hash,
            "encoder_fingerprint": self.encoder_fingerprint,
            "cache_key": self._cache_key(image_hash),
            "dtype": FEATURE_CACHE_DTYPE,
            "global_shape": list(global_array.shape),
            "patch_shape": list(patch_array.shape),
            "global_sha256": file_sha256(global_path),
            "patch_sha256": file_sha256(patch_path),
        }
        _atomic_write_json(metadata_path, metadata)

    def _load_verified(self, image_hash: str) -> CachedImageFeatures:
        np = _numpy()
        global_path, patch_path, metadata_path = self._entry_paths(image_hash)
        if not metadata_path.is_file() or not global_path.is_file() or not patch_path.is_file():
            raise FeatureCacheIntegrityError(
                f"feature-cache entry is missing files for image {image_hash}"
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FeatureCacheIntegrityError(
                f"invalid feature-cache metadata {metadata_path}: {exc}"
            ) from exc
        self._validate_metadata(metadata, image_hash, global_path, patch_path)
        try:
            global_feature = np.load(global_path, allow_pickle=False)
            patch_features = np.load(patch_path, allow_pickle=False)
        except Exception as exc:
            raise FeatureCacheIntegrityError(
                f"could not load cached arrays for image {image_hash}: {exc}"
            ) from exc

        expected_global_shape = tuple(metadata["global_shape"])
        expected_patch_shape = tuple(metadata["patch_shape"])
        if global_feature.dtype != np.dtype(np.float16) or patch_features.dtype != np.dtype(
            np.float16
        ):
            raise FeatureCacheIntegrityError(
                f"cached features for image {image_hash} are not fp16"
            )
        if global_feature.shape != expected_global_shape or patch_features.shape != expected_patch_shape:
            raise FeatureCacheIntegrityError(
                f"cached feature shape mismatch for image {image_hash}"
            )
        if global_feature.ndim != 1 or patch_features.ndim != 2:
            raise FeatureCacheIntegrityError(
                f"cached feature rank mismatch for image {image_hash}"
            )
        if global_feature.shape[0] != patch_features.shape[1]:
            raise FeatureCacheIntegrityError(
                f"global/patch feature dimensions disagree for image {image_hash}"
            )
        if not np.isfinite(global_feature).all() or not np.isfinite(patch_features).all():
            raise FeatureCacheIntegrityError(
                f"cached features contain non-finite values for image {image_hash}"
            )
        return CachedImageFeatures(
            image_sha256=image_hash,
            encoder_fingerprint=self.encoder_fingerprint,
            cache_key=self._cache_key(image_hash),
            global_feature=global_feature,
            patch_features=patch_features,
        )

    def _validate_metadata(
        self,
        metadata: Any,
        image_hash: str,
        global_path: Path,
        patch_path: Path,
    ) -> None:
        if not isinstance(metadata, Mapping):
            raise FeatureCacheIntegrityError("feature-cache metadata must be a mapping")
        expected = {
            "protocol_version": FEATURE_CACHE_PROTOCOL_VERSION,
            "image_sha256": image_hash,
            "encoder_fingerprint": self.encoder_fingerprint,
            "cache_key": self._cache_key(image_hash),
            "dtype": FEATURE_CACHE_DTYPE,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise FeatureCacheIntegrityError(
                    f"feature-cache metadata field {key!r} disagrees for image {image_hash}"
                )
        for key in ("global_shape", "patch_shape"):
            shape = metadata.get(key)
            if not isinstance(shape, list) or not shape or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in shape
            ):
                raise FeatureCacheIntegrityError(
                    f"feature-cache metadata has invalid {key} for image {image_hash}"
                )
        expected_hashes = {
            "global_sha256": file_sha256(global_path),
            "patch_sha256": file_sha256(patch_path),
        }
        for key, observed in expected_hashes.items():
            if metadata.get(key) != observed:
                raise FeatureCacheIntegrityError(
                    f"feature-cache {key} mismatch for image {image_hash}"
                )

    def _cache_key(self, image_hash: str) -> str:
        payload = json.dumps(
            {
                "encoder_fingerprint": self.encoder_fingerprint,
                "image_sha256": image_hash,
                "protocol_version": FEATURE_CACHE_PROTOCOL_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _manifest_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for metadata_path in sorted(self.entries_dir.glob("*/*/metadata.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise FeatureCacheIntegrityError(
                    f"invalid feature-cache metadata {metadata_path}: {exc}"
                ) from exc
            image_hash = metadata.get("image_sha256") if isinstance(metadata, Mapping) else None
            if not isinstance(image_hash, str) or not _SHA256_RE.fullmatch(image_hash):
                raise FeatureCacheIntegrityError(
                    f"invalid image hash in feature-cache metadata {metadata_path}"
                )
            global_path, patch_path, expected_metadata_path = self._entry_paths(image_hash)
            if metadata_path != expected_metadata_path:
                raise FeatureCacheIntegrityError(
                    f"misplaced feature-cache metadata: {metadata_path}"
                )
            self._load_verified(image_hash)
            relative = lambda path: path.relative_to(self.cache_dir).as_posix()
            rows.append(
                {
                    "protocol_version": FEATURE_CACHE_PROTOCOL_VERSION,
                    "image_sha256": image_hash,
                    "encoder_fingerprint": self.encoder_fingerprint,
                    "cache_key": self._cache_key(image_hash),
                    "dtype": FEATURE_CACHE_DTYPE,
                    "global_shape": _shape_text(metadata["global_shape"]),
                    "patch_shape": _shape_text(metadata["patch_shape"]),
                    "global_path": relative(global_path),
                    "global_sha256": metadata["global_sha256"],
                    "patch_path": relative(patch_path),
                    "patch_sha256": metadata["patch_sha256"],
                    "metadata_path": relative(metadata_path),
                    "metadata_sha256": file_sha256(metadata_path),
                }
            )
        rows.sort(key=lambda row: row["image_sha256"])
        return rows


def _canonicalize_batch(output: Any, expected_count: int) -> tuple[list[Any], list[Any]]:
    np = _numpy()
    if isinstance(output, EncodedFeatureBatch):
        globals_value = output.global_features
        patches_value = output.patch_features
    elif isinstance(output, Mapping):
        globals_value = output.get("global_features", output.get("global"))
        patches_value = output.get("patch_features", output.get("patches"))
    elif isinstance(output, (tuple, list)) and len(output) == 2:
        globals_value, patches_value = output
    else:
        globals_value = getattr(output, "global_features", None)
        patches_value = getattr(output, "patch_features", None)
    if globals_value is None or patches_value is None:
        raise FeatureEncodingError(
            "encoder output must contain both global_features and patch_features"
        )

    globals_array = _as_numpy(globals_value, np)
    if globals_array.ndim == 1 and expected_count == 1:
        globals_array = globals_array[None, :]
    if globals_array.ndim != 2 or globals_array.shape[0] != expected_count:
        raise FeatureEncodingError(
            f"global feature batch must have shape [B,D]; got {globals_array.shape}"
        )

    if isinstance(patches_value, (list, tuple)):
        if len(patches_value) != expected_count:
            raise FeatureEncodingError(
                "patch feature list length does not match the encoded image count"
            )
        patch_arrays = [_as_numpy(value, np) for value in patches_value]
    else:
        patch_batch = _as_numpy(patches_value, np)
        if patch_batch.ndim == 2 and expected_count == 1:
            patch_batch = patch_batch[None, :, :]
        if patch_batch.ndim != 3 or patch_batch.shape[0] != expected_count:
            raise FeatureEncodingError(
                f"patch feature batch must have shape [B,P,D]; got {patch_batch.shape}"
            )
        patch_arrays = [patch_batch[index] for index in range(expected_count)]

    globals_result: list[Any] = []
    patches_result: list[Any] = []
    for index, patch_array in enumerate(patch_arrays):
        global_array = np.asarray(globals_array[index])
        patch_array = np.asarray(patch_array)
        if patch_array.ndim == 1:
            patch_array = patch_array[None, :]
        if patch_array.ndim != 2 or patch_array.shape[0] == 0:
            raise FeatureEncodingError(
                f"patch features for batch item {index} must have shape [P,D]"
            )
        if global_array.ndim != 1 or global_array.shape[0] == 0:
            raise FeatureEncodingError(
                f"global features for batch item {index} must have shape [D]"
            )
        if global_array.shape[0] != patch_array.shape[1]:
            raise FeatureEncodingError(
                f"global and patch dimensions disagree for batch item {index}"
            )
        if not np.issubdtype(global_array.dtype, np.number) or not np.issubdtype(
            patch_array.dtype, np.number
        ):
            raise FeatureEncodingError("encoder features must be numeric")
        global_fp16 = np.ascontiguousarray(global_array, dtype=np.float16)
        patch_fp16 = np.ascontiguousarray(patch_array, dtype=np.float16)
        if not np.isfinite(global_fp16).all() or not np.isfinite(patch_fp16).all():
            raise FeatureEncodingError("encoder features must contain only finite values")
        globals_result.append(global_fp16)
        patches_result.append(patch_fp16)
    return globals_result, patches_result


def _as_numpy(value: Any, np: Any) -> Any:
    current = value
    for method_name in ("detach", "cpu"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    try:
        return np.asarray(current)
    except Exception as exc:
        raise FeatureEncodingError(f"could not convert encoder output to an array: {exc}") from exc


def _atomic_save_npy(path: Path, value: Any, np: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except OSError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise FeatureCacheError(f"could not write cached feature {path}: {exc}") from exc


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ) + "\n"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except OSError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise FeatureCacheError(f"could not write feature metadata {path}: {exc}") from exc


def _shape_text(shape: Sequence[int]) -> str:
    return "x".join(str(item) for item in shape)


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise FeatureCacheDependencyError(
            "Stage 5 feature caching requires numpy from the 'stage5' optional dependencies"
        ) from exc
    return np


__all__ = [
    "CachedImageFeatures",
    "EncodedFeatureBatch",
    "FEATURE_CACHE_DTYPE",
    "FEATURE_CACHE_MANIFEST_COLUMNS",
    "FEATURE_CACHE_MANIFEST_NAME",
    "FEATURE_CACHE_PROTOCOL_VERSION",
    "FeatureCache",
    "FeatureCacheDependencyError",
    "FeatureCacheError",
    "FeatureCacheIntegrityError",
    "FeatureEncodingError",
    "encoder_fingerprint",
    "file_sha256",
]
