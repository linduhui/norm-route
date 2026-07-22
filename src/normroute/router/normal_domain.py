"""Normal-Domain Signature Encoder for Stage 5 routing.

Only inference-visible image features are consumed here.  For each task, the
encoder summarizes its official normal support set and compares the query to
that normal domain at global and patch levels.  It never accepts or serializes
target labels, masks, defect types, expert outcomes, or Oracle information.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .feature_cache import (
    FEATURE_CACHE_MANIFEST_NAME,
    CachedImageFeatures,
    FeatureCache,
    encoder_fingerprint as fingerprint_encoder,
)


NORMAL_DOMAIN_PROTOCOL_VERSION = "stage5.normal_domain_signature.v1"
NORMAL_SIGNATURES_NAME = "normal_signatures.parquet"
PATCH_DISTANCE_QUANTILES = (0.50, 0.90, 0.95, 0.99)

NORMAL_SIGNATURE_COLUMNS = (
    "protocol_version",
    "task_id",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
    "encoder_fingerprint",
    "query_image_sha256",
    "support_image_sha256s",
    "normal_global_prototype",
    "normal_diagonal_variance",
    "niv",
    "query_global_residual",
    "query_global_residual_l2",
    "patch_nearest_distance_quantile_levels",
    "patch_nearest_distance_quantiles",
    "patch_nn_q50",
    "patch_nn_q90",
    "patch_nn_q95",
    "patch_nn_q99",
    "patch_nn_mean",
    "patch_nn_max",
    "query_patch_count",
    "support_patch_count",
)

_TASK_REQUIRED_FIELDS = (
    "task_id",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
    "query_path",
)
_FORBIDDEN_TOKENS = (
    "label",
    "ground_truth",
    "mask",
    "defect_type",
    "anomaly_type",
    "oracle",
    "teacher",
    "expert_score",
    "expert_outcome",
    "regret",
    "utility",
)


class NormalDomainError(RuntimeError):
    """Base error for normal-domain signature failures."""


class NormalDomainInputError(NormalDomainError, ValueError):
    """Raised when task/support provenance is incomplete or unsafe."""


class NormalDomainDependencyError(NormalDomainError, ImportError):
    """Raised when optional Parquet or numeric dependencies are absent."""


@dataclass(frozen=True)
class NormalDomainStatistics:
    """Numerical normal-domain and query-residual statistics."""

    normal_global_prototype: Any
    normal_diagonal_variance: Any
    niv: float
    query_global_residual: Any
    query_global_residual_l2: float
    patch_nearest_distances: Any
    patch_nearest_distance_quantiles: Any


@dataclass(frozen=True)
class NormalDomainSignature:
    """One leakage-safe Stage 5 signature row."""

    task_id: str
    dataset: str
    category: str
    k_shot: int
    seed: int
    support_set_id: str
    encoder_fingerprint: str
    query_image_sha256: str
    support_image_sha256s: tuple[str, ...]
    statistics: NormalDomainStatistics
    query_patch_count: int
    support_patch_count: int
    protocol_version: str = NORMAL_DOMAIN_PROTOCOL_VERSION

    def to_record(self) -> dict[str, Any]:
        """Return the exhaustive, deterministic Parquet record."""

        np = _numpy()
        quantiles = np.asarray(
            self.statistics.patch_nearest_distance_quantiles, dtype=np.float32
        )
        distances = np.asarray(self.statistics.patch_nearest_distances, dtype=np.float32)
        if quantiles.shape != (len(PATCH_DISTANCE_QUANTILES),):
            raise NormalDomainError("patch quantile count disagrees with the frozen schema")
        return {
            "protocol_version": self.protocol_version,
            "task_id": self.task_id,
            "dataset": self.dataset,
            "category": self.category,
            "k_shot": self.k_shot,
            "seed": self.seed,
            "support_set_id": self.support_set_id,
            "encoder_fingerprint": self.encoder_fingerprint,
            "query_image_sha256": self.query_image_sha256,
            "support_image_sha256s": list(self.support_image_sha256s),
            "normal_global_prototype": _float_list(
                self.statistics.normal_global_prototype, np
            ),
            "normal_diagonal_variance": _float_list(
                self.statistics.normal_diagonal_variance, np
            ),
            "niv": float(self.statistics.niv),
            "query_global_residual": _float_list(
                self.statistics.query_global_residual, np
            ),
            "query_global_residual_l2": float(
                self.statistics.query_global_residual_l2
            ),
            "patch_nearest_distance_quantile_levels": list(PATCH_DISTANCE_QUANTILES),
            "patch_nearest_distance_quantiles": _float_list(quantiles, np),
            "patch_nn_q50": float(quantiles[0]),
            "patch_nn_q90": float(quantiles[1]),
            "patch_nn_q95": float(quantiles[2]),
            "patch_nn_q99": float(quantiles[3]),
            "patch_nn_mean": float(distances.mean(dtype=np.float64)),
            "patch_nn_max": float(distances.max()),
            "query_patch_count": self.query_patch_count,
            "support_patch_count": self.support_patch_count,
        }


def compute_normal_domain_statistics(
    query_global: Any,
    query_patches: Any,
    support_globals: Any,
    support_patches: Any,
    *,
    distance_chunk_size: int = 1024,
) -> NormalDomainStatistics:
    """Compute the frozen Stage 5 normal-domain statistics.

    Definitions:

    * prototype: arithmetic mean of support global features;
    * diagonal variance: population variance (``ddof=0``), which is zero for
      one-shot support;
    * NIV: root mean squared support distance to the prototype, equivalently
      ``sqrt(sum(diagonal_variance))``;
    * query-global residual: signed ``query - prototype`` vector;
    * patch statistic: for each query patch, Euclidean distance to its nearest
      support patch, summarized at q50/q90/q95/q99.

    Support rows are lexicographically canonicalized before reductions so the
    result is exactly invariant to support permutation, not merely close up to
    floating-point summation order.
    """

    np = _numpy()
    if (
        isinstance(distance_chunk_size, bool)
        or not isinstance(distance_chunk_size, int)
        or distance_chunk_size <= 0
    ):
        raise ValueError("distance_chunk_size must be a positive integer")

    query_global_array = _vector(query_global, "query_global", np)
    query_patch_array = _matrix(query_patches, "query_patches", np)
    support_global_array = _matrix(support_globals, "support_globals", np)
    support_patch_array = _support_patch_matrix(support_patches, np)
    dimension = query_global_array.shape[0]
    for name, observed in (
        ("query_patches", query_patch_array.shape[1]),
        ("support_globals", support_global_array.shape[1]),
        ("support_patches", support_patch_array.shape[1]),
    ):
        if observed != dimension:
            raise NormalDomainInputError(
                f"{name} feature dimension {observed} disagrees with query dimension {dimension}"
            )

    canonical_globals = _canonical_rows(support_global_array, np).astype(
        np.float64, copy=False
    )
    prototype64 = canonical_globals.mean(axis=0, dtype=np.float64)
    centered = canonical_globals - prototype64
    variance64 = np.mean(centered * centered, axis=0, dtype=np.float64)
    niv = math.sqrt(float(variance64.sum(dtype=np.float64)))
    residual64 = query_global_array.astype(np.float64, copy=False) - prototype64

    canonical_support_patches = _canonical_rows(support_patch_array, np)
    distances = _nearest_patch_distances(
        query_patch_array,
        canonical_support_patches,
        chunk_size=distance_chunk_size,
        np=np,
    )
    quantiles = _linear_quantiles(distances, PATCH_DISTANCE_QUANTILES, np)

    return NormalDomainStatistics(
        normal_global_prototype=prototype64.astype(np.float32),
        normal_diagonal_variance=variance64.astype(np.float32),
        niv=niv,
        query_global_residual=residual64.astype(np.float32),
        query_global_residual_l2=math.sqrt(float(np.dot(residual64, residual64))),
        patch_nearest_distances=distances.astype(np.float32),
        patch_nearest_distance_quantiles=quantiles.astype(np.float32),
    )


class NormalDomainSignatureEncoder:
    """Compose verified cached features into per-task normal signatures."""

    def __init__(
        self,
        feature_cache: FeatureCache,
        feature_encoder: Any,
        *,
        batch_size: int = 32,
        distance_chunk_size: int = 1024,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.feature_cache = feature_cache
        self.feature_encoder = feature_encoder
        self.batch_size = batch_size
        self.distance_chunk_size = distance_chunk_size

    def encode_tasks(
        self,
        tasks: Sequence[Mapping[str, Any]],
        support_sets: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]],
    ) -> list[NormalDomainSignature]:
        """Encode all tasks with a single deduplicated cache pass."""

        normalized_tasks = _normalize_tasks(tasks)
        grouped_supports = _group_support_sets(support_sets)
        task_support_paths: dict[str, tuple[Path, ...]] = {}
        all_paths: list[Path] = []
        for task in normalized_tasks:
            support_id = task["support_set_id"]
            if support_id not in grouped_supports:
                raise NormalDomainInputError(
                    f"support_set_id {support_id!r} has no support rows"
                )
            support_paths = _validate_task_supports(task, grouped_supports[support_id])
            task_support_paths[task["task_id"]] = support_paths
            all_paths.append(Path(task["query_path"]))
            all_paths.extend(support_paths)

        cached = self.feature_cache.get_or_encode(
            all_paths,
            self.feature_encoder,
            batch_size=self.batch_size,
        )
        feature_by_path: dict[Path, CachedImageFeatures] = {}
        for path, features in zip(all_paths, cached):
            previous = feature_by_path.setdefault(path, features)
            if previous.image_sha256 != features.image_sha256:
                raise NormalDomainError(f"path changed during feature caching: {path}")

        signatures: list[NormalDomainSignature] = []
        for task in normalized_tasks:
            query_features = feature_by_path[Path(task["query_path"])]
            support_features = [
                feature_by_path[path] for path in task_support_paths[task["task_id"]]
            ]
            support_features.sort(key=lambda item: item.image_sha256)
            statistics = compute_normal_domain_statistics(
                query_features.global_feature,
                query_features.patch_features,
                [item.global_feature for item in support_features],
                [item.patch_features for item in support_features],
                distance_chunk_size=self.distance_chunk_size,
            )
            signatures.append(
                NormalDomainSignature(
                    task_id=task["task_id"],
                    dataset=task["dataset"],
                    category=task["category"],
                    k_shot=task["k_shot"],
                    seed=task["seed"],
                    support_set_id=task["support_set_id"],
                    encoder_fingerprint=self.feature_cache.encoder_fingerprint,
                    query_image_sha256=query_features.image_sha256,
                    support_image_sha256s=tuple(
                        item.image_sha256 for item in support_features
                    ),
                    statistics=statistics,
                    query_patch_count=int(query_features.patch_features.shape[0]),
                    support_patch_count=sum(
                        int(item.patch_features.shape[0]) for item in support_features
                    ),
                )
            )
        return signatures


def build_normal_signatures(
    tasks: Sequence[Mapping[str, Any]],
    support_sets: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]],
    *,
    feature_encoder: Any,
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    encoder_fingerprint: str | None = None,
    batch_size: int = 32,
    distance_chunk_size: int = 1024,
) -> tuple[Path, Path]:
    """Build the two required Stage 5 cache/signature artifacts.

    Returns ``(feature_cache_manifest.csv, normal_signatures.parquet)``.  A
    rerun reuses only cache entries whose metadata and feature hashes verify.
    """

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cache_dir) if cache_dir is not None else destination / "feature_cache"
    fingerprint = encoder_fingerprint or fingerprint_encoder(feature_encoder)
    cache = FeatureCache(
        cache_root,
        encoder_fingerprint=fingerprint,
        manifest_path=destination / FEATURE_CACHE_MANIFEST_NAME,
    )
    signature_encoder = NormalDomainSignatureEncoder(
        cache,
        feature_encoder,
        batch_size=batch_size,
        distance_chunk_size=distance_chunk_size,
    )
    signatures = signature_encoder.encode_tasks(tasks, support_sets)
    parquet_path = write_normal_signatures_parquet(
        signatures, destination / NORMAL_SIGNATURES_NAME
    )
    return cache.manifest_path, parquet_path


def write_normal_signatures_parquet(
    signatures: Sequence[NormalDomainSignature | Mapping[str, Any]],
    output_path: str | Path,
) -> Path:
    """Atomically write deterministic, typed normal signatures as Parquet."""

    if not signatures:
        raise NormalDomainInputError("cannot write an empty normal-signature artifact")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise NormalDomainDependencyError(
            "normal_signatures.parquet requires pyarrow from the 'stage5' optional dependencies"
        ) from exc

    records = [
        item.to_record() if isinstance(item, NormalDomainSignature) else dict(item)
        for item in signatures
    ]
    records.sort(key=lambda row: str(row.get("task_id", "")))
    for record in records:
        if tuple(record) != NORMAL_SIGNATURE_COLUMNS:
            raise NormalDomainInputError(
                "normal signature record does not match the frozen output schema"
            )
    schema = _parquet_schema(pa)
    try:
        table = pa.Table.from_pylist(records, schema=schema)
    except Exception as exc:
        raise NormalDomainInputError(f"could not construct normal-signature table: {exc}") from exc

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        pq.write_table(
            table,
            temp_path,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            version="2.6",
        )
        os.replace(temp_path, path)
    except Exception as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        if isinstance(exc, NormalDomainError):
            raise
        raise NormalDomainError(f"could not write normal signatures {path}: {exc}") from exc
    return path


def _normalize_tasks(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(tasks, (str, bytes)) or not tasks:
        raise NormalDomainInputError("tasks must be a non-empty sequence")
    normalized: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise NormalDomainInputError(f"task {index} is not a mapping")
        _reject_forbidden_fields(task, context=f"task {index}")
        missing = [field for field in _TASK_REQUIRED_FIELDS if field not in task]
        if missing:
            raise NormalDomainInputError(f"task {index} is missing fields: {missing}")
        row = {
            "task_id": _text(task["task_id"], "task_id"),
            "dataset": _text(task["dataset"], "dataset"),
            "category": _text(task["category"], "category"),
            "k_shot": _integer(task["k_shot"], "k_shot", minimum=1),
            "seed": _integer(task["seed"], "seed", minimum=0),
            "support_set_id": _text(task["support_set_id"], "support_set_id"),
            "query_path": _text(task["query_path"], "query_path"),
        }
        if row["task_id"] in seen_task_ids:
            raise NormalDomainInputError(f"duplicate task_id {row['task_id']!r}")
        seen_task_ids.add(row["task_id"])
        normalized.append(row)
    normalized.sort(key=lambda row: row["task_id"])
    return normalized


def _group_support_sets(
    support_sets: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]],
) -> dict[str, list[Any]]:
    if isinstance(support_sets, Mapping):
        grouped: dict[str, list[Any]] = {}
        for raw_support_id, values in support_sets.items():
            support_id = _text(raw_support_id, "support_set_id")
            if isinstance(values, (str, bytes, Path)) or not values:
                raise NormalDomainInputError(f"support set {support_id!r} is empty")
            grouped[support_id] = list(values)
        return grouped
    if isinstance(support_sets, (str, bytes)) or not support_sets:
        raise NormalDomainInputError("support_sets must be non-empty")
    grouped = {}
    for index, row in enumerate(support_sets):
        if not isinstance(row, Mapping):
            raise NormalDomainInputError(f"support row {index} is not a mapping")
        _reject_forbidden_fields(row, context=f"support row {index}")
        support_id = _text(row.get("support_set_id"), "support_set_id")
        grouped.setdefault(support_id, []).append(row)
    return grouped


def _validate_task_supports(task: Mapping[str, Any], supports: Sequence[Any]) -> tuple[Path, ...]:
    if len(supports) != task["k_shot"]:
        raise NormalDomainInputError(
            f"support_set_id {task['support_set_id']!r} declares K={task['k_shot']} "
            f"but contains {len(supports)} rows"
        )
    paths: list[Path] = []
    identities: set[str] = set()
    for index, value in enumerate(supports):
        if isinstance(value, Mapping):
            _reject_forbidden_fields(value, context=f"support {index}")
            for field in ("dataset", "category", "k_shot", "seed", "support_set_id"):
                if field not in value:
                    continue
                expected = task[field]
                observed: Any = value[field]
                if field in {"k_shot", "seed"}:
                    observed = _integer(observed, field, minimum=0)
                else:
                    observed = _text(observed, field)
                if observed != expected:
                    raise NormalDomainInputError(
                        f"support {field}={observed!r} disagrees with task value {expected!r}"
                    )
            if "split" in value:
                split = _text(value["split"], "split").casefold().replace("\\", "/")
                if split not in {"train", "train/good", "good"}:
                    raise NormalDomainInputError(
                        f"support {index} is not from an official train/good split"
                    )
            path = Path(_text(value.get("image_path"), "image_path"))
            identity = str(value.get("image_id", path))
        else:
            path = Path(_text(value, "support image path"))
            identity = str(path)
        if identity in identities:
            raise NormalDomainInputError(
                f"duplicate support image {identity!r} in {task['support_set_id']!r}"
            )
        identities.add(identity)
        paths.append(path)
    paths.sort(key=lambda item: str(item))
    return tuple(paths)


def _reject_forbidden_fields(value: Mapping[str, Any], *, context: str) -> None:
    for key, nested in value.items():
        normalized = str(key).strip().casefold().replace("-", "_").replace(" ", "_")
        if any(token in normalized for token in _FORBIDDEN_TOKENS):
            raise NormalDomainInputError(
                f"{context} contains forbidden inference field {key!r}"
            )
        if isinstance(nested, Mapping):
            _reject_forbidden_fields(nested, context=f"{context}.{key}")


def _support_patch_matrix(value: Any, np: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if not value:
            raise NormalDomainInputError("support_patches must be non-empty")
        matrices = [_matrix(item, "support_patches", np) for item in value]
        return np.concatenate(matrices, axis=0)
    array = np.asarray(value)
    if array.ndim == 3:
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise NormalDomainInputError("support_patches must be non-empty")
        return _matrix(array.reshape(-1, array.shape[-1]), "support_patches", np)
    return _matrix(array, "support_patches", np)


def _vector(value: Any, name: str, np: Any) -> Any:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] == 0:
        raise NormalDomainInputError(f"{name} must have shape [D]")
    return _finite_float_array(array, name, np)


def _matrix(value: Any, name: str, np: Any) -> Any:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise NormalDomainInputError(f"{name} must have shape [N,D]")
    return _finite_float_array(array, name, np)


def _finite_float_array(value: Any, name: str, np: Any) -> Any:
    if not np.issubdtype(value.dtype, np.number):
        raise NormalDomainInputError(f"{name} must be numeric")
    array = np.ascontiguousarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise NormalDomainInputError(f"{name} contains non-finite values")
    return array


def _canonical_rows(value: Any, np: Any) -> Any:
    contiguous = np.ascontiguousarray(value)
    order = sorted(range(contiguous.shape[0]), key=lambda index: contiguous[index].tobytes())
    return contiguous[order]


def _nearest_patch_distances(
    queries: Any,
    supports: Any,
    *,
    chunk_size: int,
    np: Any,
) -> Any:
    query64 = queries.astype(np.float64, copy=False)
    support64 = supports.astype(np.float64, copy=False)
    result = np.full(query64.shape[0], np.inf, dtype=np.float64)
    for query_start in range(0, query64.shape[0], chunk_size):
        query_chunk = query64[query_start : query_start + chunk_size]
        best_squared = np.full(query_chunk.shape[0], np.inf, dtype=np.float64)
        query_norm = np.sum(query_chunk * query_chunk, axis=1, dtype=np.float64)
        for support_start in range(0, support64.shape[0], chunk_size):
            support_chunk = support64[support_start : support_start + chunk_size]
            support_norm = np.sum(
                support_chunk * support_chunk, axis=1, dtype=np.float64
            )
            squared = (
                query_norm[:, None]
                + support_norm[None, :]
                - 2.0 * (query_chunk @ support_chunk.T)
            )
            np.maximum(squared, 0.0, out=squared)
            best_squared = np.minimum(best_squared, squared.min(axis=1))
        result[query_start : query_start + query_chunk.shape[0]] = np.sqrt(best_squared)
    return result


def _linear_quantiles(value: Any, quantiles: Iterable[float], np: Any) -> Any:
    try:
        return np.quantile(value, tuple(quantiles), method="linear")
    except TypeError:  # numpy<1.22 compatibility for constrained experiment hosts
        return np.quantile(value, tuple(quantiles), interpolation="linear")


def _parquet_schema(pa: Any) -> Any:
    string_list = pa.list_(pa.string())
    float_list = pa.list_(pa.float32())
    return pa.schema(
        [
            pa.field("protocol_version", pa.string(), nullable=False),
            pa.field("task_id", pa.string(), nullable=False),
            pa.field("dataset", pa.string(), nullable=False),
            pa.field("category", pa.string(), nullable=False),
            pa.field("k_shot", pa.int32(), nullable=False),
            pa.field("seed", pa.int64(), nullable=False),
            pa.field("support_set_id", pa.string(), nullable=False),
            pa.field("encoder_fingerprint", pa.string(), nullable=False),
            pa.field("query_image_sha256", pa.string(), nullable=False),
            pa.field("support_image_sha256s", string_list, nullable=False),
            pa.field("normal_global_prototype", float_list, nullable=False),
            pa.field("normal_diagonal_variance", float_list, nullable=False),
            pa.field("niv", pa.float32(), nullable=False),
            pa.field("query_global_residual", float_list, nullable=False),
            pa.field("query_global_residual_l2", pa.float32(), nullable=False),
            pa.field(
                "patch_nearest_distance_quantile_levels", float_list, nullable=False
            ),
            pa.field("patch_nearest_distance_quantiles", float_list, nullable=False),
            pa.field("patch_nn_q50", pa.float32(), nullable=False),
            pa.field("patch_nn_q90", pa.float32(), nullable=False),
            pa.field("patch_nn_q95", pa.float32(), nullable=False),
            pa.field("patch_nn_q99", pa.float32(), nullable=False),
            pa.field("patch_nn_mean", pa.float32(), nullable=False),
            pa.field("patch_nn_max", pa.float32(), nullable=False),
            pa.field("query_patch_count", pa.int32(), nullable=False),
            pa.field("support_patch_count", pa.int32(), nullable=False),
        ]
    )


def _float_list(value: Any, np: Any) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=np.float32).tolist()]


def _text(value: Any, field: str) -> str:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise NormalDomainInputError(f"{field} must be a non-empty string")
    return str(value).strip()


def _integer(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise NormalDomainInputError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise NormalDomainInputError(f"{field} must be an integer") from exc
    if str(value).strip() not in {str(result), f"+{result}"} and not isinstance(value, int):
        raise NormalDomainInputError(f"{field} must be an integer")
    if result < minimum:
        raise NormalDomainInputError(f"{field} must be >= {minimum}")
    return result


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise NormalDomainDependencyError(
            "normal-domain signatures require numpy from the 'stage5' optional dependencies"
        ) from exc
    return np


__all__ = [
    "NORMAL_DOMAIN_PROTOCOL_VERSION",
    "NORMAL_SIGNATURE_COLUMNS",
    "NORMAL_SIGNATURES_NAME",
    "PATCH_DISTANCE_QUANTILES",
    "NormalDomainDependencyError",
    "NormalDomainError",
    "NormalDomainInputError",
    "NormalDomainSignature",
    "NormalDomainSignatureEncoder",
    "NormalDomainStatistics",
    "build_normal_signatures",
    "compute_normal_domain_statistics",
    "write_normal_signatures_parquet",
]
