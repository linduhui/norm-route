"""Mask-free Boundary-Informed Reweighting for Stage 5 routing.

BIR-AD consumes only an image and its frozen patch tokens.  It does not use a
label, mask, defect type, expert output, or Oracle information, and it never
modifies an expert algorithm.

The implementation follows the formal M-BAI definition:

* feature evidence: channel standard deviation and RMS L2 norm;
* potential boundary location: mean cosine discontinuity to spatial neighbours;
* pixel evidence: Sobel edge energy and structure-tensor orientation coherence;
* two-sided feature contrast sampled along the image-gradient normal;
* clarity: a convex combination of normalized evidence passed through sigmoid;
* BAI: boundary-softmax-weighted unclear mass ``sum_p w_p * (1 - c_p)``.

Normalization can be fitted only through the explicitly training-scoped
``fit_bir_ad_normalization`` API.  Its statistics can then be frozen and reused
for support and query inference without fitting anything on the target query.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


BIR_AD_PROTOCOL_VERSION = "stage5.bir_ad.v3"
BIR_AD_ALIGNMENT_PROTOCOL_VERSION = "stage5.bir_ad_alignment.v1"
BIR_AD_NORMALIZATION_PROTOCOL_VERSION = "stage5.bir_ad_normalization.v1"
BIR_AD_CLARITY_COMPONENT_NAMES = (
    "patch_channel_std",
    "patch_l2_norm",
    "sobel_edge_energy",
    "boundary_orientation_consistency",
    "two_sided_feature_contrast",
)
BIR_AD_SIGNAL_NAMES = BIR_AD_CLARITY_COMPONENT_NAMES + ("structural_boundary",)
# Compatibility name for callers that treat clarity inputs as components.
BIR_AD_COMPONENT_NAMES = BIR_AD_CLARITY_COMPONENT_NAMES
DEFAULT_CLARITY_WEIGHTS = (0.2, 0.2, 0.2, 0.2, 0.2)
DEFAULT_COMPONENT_WEIGHTS = DEFAULT_CLARITY_WEIGHTS


class BIRADInputError(ValueError):
    """Raised when BIR-AD receives malformed, unsafe, or non-finite inputs."""


class BIRADDependencyError(ImportError):
    """Raised when the optional Stage 5 numerical dependency is unavailable."""


@dataclass(frozen=True)
class PatchAlignedImage:
    """Pixel view after the exact frozen-backbone spatial transform.

    The batch pipeline requires this wrapper so Sobel cells and patch tokens
    share one auditable geometry.  Raw images remain accepted only by the
    low-level diagnostic API when strict alignment is not requested.
    """

    pixels: Any
    patch_grid_shape: tuple[int, int]
    source_image_sha256: str
    transform_fingerprint: str
    protocol_version: str = BIR_AD_ALIGNMENT_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _resolve_grid_shape(
            self.patch_grid_shape[0] * self.patch_grid_shape[1],
            self.patch_grid_shape,
        )
        if (
            not isinstance(self.source_image_sha256, str)
            or len(self.source_image_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_image_sha256)
        ):
            raise BIRADInputError("source_image_sha256 must be lowercase SHA-256")
        if (
            not isinstance(self.transform_fingerprint, str)
            or not self.transform_fingerprint.strip()
        ):
            raise BIRADInputError("transform_fingerprint must be non-empty")
        if self.protocol_version != BIR_AD_ALIGNMENT_PROTOCOL_VERSION:
            raise BIRADInputError("invalid patch-alignment protocol version")


@dataclass(frozen=True)
class BIRADNormalizationStats:
    """Frozen training-category location/scale statistics for all BIR signals."""

    locations: tuple[float, ...]
    scales: tuple[float, ...]
    sample_count: int
    source: str = "training_categories"
    clip_value: float = 5.0
    protocol_version: str = BIR_AD_NORMALIZATION_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        expected = len(BIR_AD_SIGNAL_NAMES)
        if len(self.locations) != expected or len(self.scales) != expected:
            raise BIRADInputError(
                f"normalization statistics must contain {expected} signals"
            )
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 0
        ):
            raise BIRADInputError("normalization sample_count must be non-negative")
        if not isinstance(self.source, str) or not self.source.strip():
            raise BIRADInputError("normalization source must be non-empty")
        _positive_finite(self.clip_value, "normalization clip_value")
        for value in self.locations:
            if not math.isfinite(float(value)):
                raise BIRADInputError("normalization locations must be finite")
        for value in self.scales:
            _positive_finite(value, "normalization scale")

    @classmethod
    def identity(cls) -> "BIRADNormalizationStats":
        """Return a leakage-safe fixed transform for smoke tests/diagnostics.

        Production experiments should fit and freeze training-category
        statistics and may set ``require_fitted_normalization=True``.
        """

        count = len(BIR_AD_SIGNAL_NAMES)
        return cls(
            locations=(0.0,) * count,
            scales=(1.0,) * count,
            sample_count=0,
            source="fixed_identity",
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BIRADNormalizationStats":
        if not isinstance(value, Mapping):
            raise BIRADInputError("normalization statistics must be a mapping")
        names = tuple(value.get("signal_names", ()))
        if names and names != BIR_AD_SIGNAL_NAMES:
            raise BIRADInputError(
                "normalization signal_names disagree with the BIR-AD protocol"
            )
        try:
            locations = tuple(float(item) for item in value["locations"])
            scales = tuple(float(item) for item in value["scales"])
            sample_count = int(value["sample_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BIRADInputError("invalid BIR-AD normalization statistics") from exc
        return cls(
            locations=locations,
            scales=scales,
            sample_count=sample_count,
            source=str(value.get("source", "training_categories")),
            clip_value=float(value.get("clip_value", 5.0)),
            protocol_version=str(
                value.get(
                    "protocol_version", BIR_AD_NORMALIZATION_PROTOCOL_VERSION
                )
            ),
        )

    @property
    def is_fitted(self) -> bool:
        return self.sample_count > 0 and self.source != "fixed_identity"

    def transform(self, name: str, values: Any) -> Any:
        np = _numpy()
        try:
            index = BIR_AD_SIGNAL_NAMES.index(name)
        except ValueError as exc:
            raise BIRADInputError(f"unknown BIR-AD signal {name!r}") from exc
        array = np.asarray(values, dtype=np.float64)
        normalized = (array - self.locations[index]) / self.scales[index]
        return np.clip(normalized, -self.clip_value, self.clip_value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "signal_names": list(BIR_AD_SIGNAL_NAMES),
            "locations": list(self.locations),
            "scales": list(self.scales),
            "sample_count": self.sample_count,
            "source": self.source,
            "clip_value": self.clip_value,
        }


@dataclass(frozen=True)
class BIRADResult:
    """Patch evidence, dual representations, M-BAI, and reliability for one image."""

    patch_channel_std: Any
    patch_l2_norm: Any
    structural_boundary: Any
    sobel_edge_energy: Any
    boundary_orientation_consistency: Any
    two_sided_feature_contrast: Any
    normalized_patch_channel_std: Any
    normalized_patch_l2_norm: Any
    normalized_structural_boundary: Any
    normalized_sobel_edge_energy: Any
    normalized_boundary_orientation_consistency: Any
    normalized_two_sided_feature_contrast: Any
    pixel_feature_boundary_disagreement: Any
    pixel_feature_boundary_agreement: Any
    clarity: Any
    boundary_weights: Any
    clear_evidence: Any
    ambiguous_evidence: Any
    clear_weights: Any
    ambiguous_weights: Any
    clear_representation: Any
    ambiguous_representation: Any
    bai: float
    bai_reliability: float
    weighted_pixel_feature_disagreement: float
    normalized_patch_features: Any
    patch_grid_shape: tuple[int, int]
    clarity_weights: tuple[float, float, float, float, float]
    boundary_temperature: float
    use_structural_boundary_weighting: bool
    disagreement_penalty: float
    normalization_stats: BIRADNormalizationStats
    alignment_verified: bool
    alignment_fingerprint: str | None
    source_image_sha256: str | None
    protocol_version: str = BIR_AD_PROTOCOL_VERSION

    @property
    def patch_count(self) -> int:
        return int(self.clarity.shape[0])

    @property
    def boundary_ambiguity_index(self) -> float:
        return self.bai

    @property
    def reliability(self) -> float:
        return self.bai_reliability

    @property
    def component_weights(self) -> tuple[float, float, float, float, float]:
        """Compatibility alias for the five clarity coefficients."""

        return self.clarity_weights


@dataclass(frozen=True)
class BIRADTaskResult:
    """Support prior, query BAI, boundary shift, and Router-ready BAI vector."""

    query: BIRADResult
    supports: tuple[BIRADResult, ...]
    support_bai: float
    support_bai_variance: float
    support_bai_std: float
    query_bai: float
    query_support_boundary_shift: float
    support_bai_reliability: float
    query_bai_reliability: float
    support_boundary_consistency: float | None
    support_boundary_consistency_valid: bool
    query_support_boundary_consistency: float
    support_patch_consistency: tuple[Any, ...]
    query_patch_support_consistency: Any
    support_pixel_feature_disagreement: float
    query_pixel_feature_disagreement: float
    bai_vector: tuple[float, float, float, float, float]
    protocol_version: str = BIR_AD_PROTOCOL_VERSION

    @property
    def boundary_shift(self) -> float:
        return self.query_support_boundary_shift

    @property
    def absolute_boundary_shift(self) -> float:
        return abs(self.query_support_boundary_shift)

    @property
    def clear_representation(self) -> Any:
        return self.query.clear_representation

    @property
    def ambiguous_representation(self) -> Any:
        return self.query.ambiguous_representation


@dataclass(frozen=True)
class _RawBoundarySignals:
    patch_channel_std: Any
    patch_l2_norm: Any
    structural_boundary: Any
    sobel_edge_energy: Any
    boundary_orientation_consistency: Any
    two_sided_feature_contrast: Any
    normalized_patches: Any
    grid_shape: tuple[int, int]

    def by_name(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in BIR_AD_SIGNAL_NAMES
        }


def compute_bir_ad(
    patch_features: Any,
    image: Any,
    *,
    patch_grid_shape: Sequence[int] | None = None,
    normalization_stats: BIRADNormalizationStats | Mapping[str, Any] | None = None,
    clarity_weights: Sequence[float] = DEFAULT_CLARITY_WEIGHTS,
    boundary_temperature: float = 1.0,
    neighbor_connectivity: int = 4,
    two_sided_radius: int = 1,
    structure_tensor_sigma: float = 1.0,
    use_structural_boundary_weighting: bool = True,
    disagreement_penalty: float = 1.0,
    eps: float = 1e-8,
    require_fitted_normalization: bool = False,
    require_strict_alignment: bool = False,
) -> BIRADResult:
    """Compute mask-free BIR-AD evidence for one image.

    ``patch_features`` has shape ``[P,C]``.  ``image`` may be a local path, a
    PIL image, or a grayscale/RGB array.  A non-square token count requires an
    explicit ``patch_grid_shape``.  If no frozen normalization is supplied, a
    fixed identity transform is used and recorded; it never fits on the query.
    """

    np = _numpy()
    epsilon = _positive_finite(eps, "eps")
    temperature = _positive_finite(
        boundary_temperature, "boundary_temperature"
    )
    sigma = _positive_finite(structure_tensor_sigma, "structure_tensor_sigma")
    disagreement_scale = _nonnegative_finite(
        disagreement_penalty, "disagreement_penalty"
    )
    if not isinstance(use_structural_boundary_weighting, bool):
        raise BIRADInputError(
            "use_structural_boundary_weighting must be boolean"
        )
    alpha = _clarity_weights(clarity_weights)
    stats = _normalization_stats(normalization_stats)
    if require_fitted_normalization and not stats.is_fitted:
        raise BIRADInputError(
            "frozen training-category normalization statistics are required"
        )

    alignment_verified = isinstance(image, PatchAlignedImage)
    alignment_fingerprint: str | None = None
    source_image_sha256: str | None = None
    pixel_input = image
    effective_grid_shape = patch_grid_shape
    if alignment_verified:
        aligned = image
        if patch_grid_shape is not None and tuple(patch_grid_shape) != aligned.patch_grid_shape:
            raise BIRADInputError(
                "patch_grid_shape disagrees with PatchAlignedImage geometry"
            )
        effective_grid_shape = aligned.patch_grid_shape
        pixel_input = aligned.pixels
        alignment_fingerprint = aligned.transform_fingerprint
        source_image_sha256 = aligned.source_image_sha256
    elif require_strict_alignment:
        raise BIRADInputError(
            "strict BIR-AD requires PatchAlignedImage from the frozen backbone transform"
        )

    raw = _compute_raw_signals(
        patch_features,
        pixel_input,
        patch_grid_shape=effective_grid_shape,
        neighbor_connectivity=neighbor_connectivity,
        two_sided_radius=two_sided_radius,
        structure_tensor_sigma=sigma,
        eps=epsilon,
        np=np,
    )
    raw_by_name = raw.by_name()
    normalized = {
        name: stats.transform(name, values)
        for name, values in raw_by_name.items()
    }

    feature_boundary_probability = _stable_sigmoid(
        normalized["structural_boundary"], np
    )
    pixel_boundary_probability = _stable_sigmoid(
        normalized["sobel_edge_energy"], np
    )
    boundary_disagreement = np.abs(
        feature_boundary_probability - pixel_boundary_probability
    )
    boundary_agreement = 1.0 - boundary_disagreement

    clarity_logit = np.zeros(raw.normalized_patches.shape[0], dtype=np.float64)
    for weight, name in zip(alpha, BIR_AD_CLARITY_COMPONENT_NAMES):
        clarity_logit += weight * normalized[name]
    clarity_logit -= disagreement_scale * boundary_disagreement
    clarity = _stable_sigmoid(clarity_logit, np)
    if use_structural_boundary_weighting:
        boundary_weights = _stable_softmax(
            normalized["structural_boundary"] / temperature, np
        )
    else:
        boundary_weights = np.full(
            clarity.shape, 1.0 / clarity.shape[0], dtype=np.float64
        )
    clear_evidence = boundary_weights * clarity
    ambiguous_evidence = boundary_weights * (1.0 - clarity)
    clear_weights = _probability_weights(clear_evidence, epsilon, np)
    ambiguous_weights = _probability_weights(ambiguous_evidence, epsilon, np)
    clear_representation = _permutation_stable_weighted_pool(
        raw.normalized_patches, clear_weights, np
    )
    ambiguous_representation = _permutation_stable_weighted_pool(
        raw.normalized_patches, ambiguous_weights, np
    )
    bai = _stable_dot(boundary_weights, 1.0 - clarity)
    reliability = _stable_dot(
        boundary_weights, raw.boundary_orientation_consistency
    )
    weighted_disagreement = _stable_dot(
        boundary_weights, boundary_disagreement
    )
    bai = min(1.0, max(0.0, bai))
    reliability = min(1.0, max(0.0, reliability))

    arrays = (
        *raw_by_name.values(),
        *normalized.values(),
        boundary_disagreement,
        boundary_agreement,
        clarity,
        boundary_weights,
        clear_evidence,
        ambiguous_evidence,
        clear_weights,
        ambiguous_weights,
        clear_representation,
        ambiguous_representation,
    )
    if not all(bool(np.all(np.isfinite(value))) for value in arrays):
        raise BIRADInputError("BIR-AD produced a non-finite value")

    return BIRADResult(
        patch_channel_std=raw.patch_channel_std,
        patch_l2_norm=raw.patch_l2_norm,
        structural_boundary=raw.structural_boundary,
        sobel_edge_energy=raw.sobel_edge_energy,
        boundary_orientation_consistency=(
            raw.boundary_orientation_consistency
        ),
        two_sided_feature_contrast=raw.two_sided_feature_contrast,
        normalized_patch_channel_std=normalized["patch_channel_std"],
        normalized_patch_l2_norm=normalized["patch_l2_norm"],
        normalized_structural_boundary=normalized["structural_boundary"],
        normalized_sobel_edge_energy=normalized["sobel_edge_energy"],
        normalized_boundary_orientation_consistency=normalized[
            "boundary_orientation_consistency"
        ],
        normalized_two_sided_feature_contrast=normalized[
            "two_sided_feature_contrast"
        ],
        pixel_feature_boundary_disagreement=boundary_disagreement,
        pixel_feature_boundary_agreement=boundary_agreement,
        clarity=clarity,
        boundary_weights=boundary_weights,
        clear_evidence=clear_evidence,
        ambiguous_evidence=ambiguous_evidence,
        clear_weights=clear_weights,
        ambiguous_weights=ambiguous_weights,
        clear_representation=clear_representation,
        ambiguous_representation=ambiguous_representation,
        bai=bai,
        bai_reliability=reliability,
        weighted_pixel_feature_disagreement=weighted_disagreement,
        normalized_patch_features=raw.normalized_patches,
        patch_grid_shape=raw.grid_shape,
        clarity_weights=alpha,
        boundary_temperature=temperature,
        use_structural_boundary_weighting=use_structural_boundary_weighting,
        disagreement_penalty=disagreement_scale,
        normalization_stats=stats,
        alignment_verified=alignment_verified,
        alignment_fingerprint=alignment_fingerprint,
        source_image_sha256=source_image_sha256,
    )


def compute_patch_clarity(
    patch_features: Any,
    image: Any,
    **kwargs: Any,
) -> Any:
    """Return the five-evidence BIR-AD patch clarity vector."""

    return compute_bir_ad(patch_features, image, **kwargs).clarity


def compute_bir_ad_task(
    query_patch_features: Any,
    query_image: Any,
    support_patch_features: Sequence[Any],
    support_images: Sequence[Any],
    **kwargs: Any,
) -> BIRADTaskResult:
    """Compute BAI_S, sqrt(VarBAI_S), BAI_Q, delta BAI, and abs(delta)."""

    support_features = _support_sequence(
        support_patch_features, "support_patch_features"
    )
    support_pixels = _support_sequence(support_images, "support_images")
    if len(support_features) != len(support_pixels):
        raise BIRADInputError(
            "support_patch_features and support_images must have the same length"
        )
    if not support_features:
        raise BIRADInputError("at least one official normal support is required")

    consistency_temperature = kwargs.pop("consistency_temperature", 1.0)
    consistency_chunk_size = kwargs.pop("consistency_chunk_size", 1024)
    query_result = compute_bir_ad(query_patch_features, query_image, **kwargs)
    support_results = [
        compute_bir_ad(features, pixels, **kwargs)
        for features, pixels in zip(support_features, support_pixels)
    ]
    return compose_bir_ad_task(
        query_result,
        support_results,
        consistency_temperature=consistency_temperature,
        consistency_chunk_size=consistency_chunk_size,
    )


def compose_bir_ad_task(
    query_result: BIRADResult,
    support_results: Sequence[BIRADResult],
    *,
    consistency_temperature: float = 1.0,
    consistency_chunk_size: int = 1024,
) -> BIRADTaskResult:
    """Compose cached image-level results into support/query BIR evidence.

    Patch consistency uses feature-nearest matches rather than equal spatial
    coordinates, so normal pose changes do not masquerade as inconsistency.
    Candidate rows and supports are canonicalized before nearest-neighbour
    matching and reductions, making support permutation exactly invariant.
    """

    if not isinstance(query_result, BIRADResult):
        raise TypeError("query_result must be BIRADResult")
    supports = _support_sequence(support_results, "support_results")
    if not supports or not all(isinstance(item, BIRADResult) for item in supports):
        raise BIRADInputError("support_results must contain BIRADResult values")
    temperature = _positive_finite(
        consistency_temperature, "consistency_temperature"
    )
    if (
        isinstance(consistency_chunk_size, bool)
        or not isinstance(consistency_chunk_size, int)
        or consistency_chunk_size <= 0
    ):
        raise BIRADInputError("consistency_chunk_size must be a positive integer")
    np = _numpy()
    canonical_supports = tuple(sorted(supports, key=_result_sort_key))
    support_bais = sorted(result.bai for result in canonical_supports)
    support_bai = math.fsum(support_bais) / len(support_bais)
    squared_deviations = sorted(
        (value - support_bai) ** 2 for value in support_bais
    )
    support_variance = math.fsum(squared_deviations) / len(squared_deviations)
    support_std = math.sqrt(support_variance)
    query_bai = query_result.bai
    shift = query_bai - support_bai
    support_reliability = math.fsum(
        sorted(result.bai_reliability for result in canonical_supports)
    ) / len(canonical_supports)
    (
        support_patch_consistency,
        support_consistency,
        support_consistency_valid,
    ) = _cross_support_patch_consistency(
        canonical_supports,
        temperature=temperature,
        chunk_size=consistency_chunk_size,
        np=np,
    )
    support_bank_features, support_bank_responses = _canonical_patch_bank(
        canonical_supports, np
    )
    query_patch_consistency = _nearest_response_consistency(
        query_result.normalized_patch_features,
        _patch_response_matrix(query_result, np),
        support_bank_features,
        support_bank_responses,
        temperature=temperature,
        chunk_size=consistency_chunk_size,
        np=np,
    )
    query_support_consistency = _stable_dot(
        query_result.boundary_weights, query_patch_consistency
    )
    support_disagreement = math.fsum(
        sorted(
            result.weighted_pixel_feature_disagreement
            for result in canonical_supports
        )
    ) / len(canonical_supports)
    bai_vector = (
        support_bai,
        support_std,
        query_bai,
        shift,
        abs(shift),
    )
    return BIRADTaskResult(
        query=query_result,
        supports=canonical_supports,
        support_bai=support_bai,
        support_bai_variance=support_variance,
        support_bai_std=support_std,
        query_bai=query_bai,
        query_support_boundary_shift=shift,
        support_bai_reliability=support_reliability,
        query_bai_reliability=query_result.bai_reliability,
        support_boundary_consistency=support_consistency,
        support_boundary_consistency_valid=support_consistency_valid,
        query_support_boundary_consistency=query_support_consistency,
        support_patch_consistency=support_patch_consistency,
        query_patch_support_consistency=query_patch_consistency,
        support_pixel_feature_disagreement=support_disagreement,
        query_pixel_feature_disagreement=(
            query_result.weighted_pixel_feature_disagreement
        ),
        bai_vector=bai_vector,
    )


def compute_boundary_ambiguity(
    query_patch_features: Any,
    query_image: Any,
    support_patch_features: Sequence[Any],
    support_images: Sequence[Any],
    **kwargs: Any,
) -> BIRADTaskResult:
    return compute_bir_ad_task(
        query_patch_features,
        query_image,
        support_patch_features,
        support_images,
        **kwargs,
    )


def fit_bir_ad_normalization(
    training_patch_features: Sequence[Any],
    training_images: Sequence[Any],
    *,
    split: str,
    patch_grid_shape: Sequence[int] | None = None,
    neighbor_connectivity: int = 4,
    two_sided_radius: int = 1,
    structure_tensor_sigma: float = 1.0,
    scale_floor: float = 1e-6,
    clip_value: float = 5.0,
    eps: float = 1e-8,
) -> BIRADNormalizationStats:
    """Fit permutation-invariant signal statistics on training categories only.

    The API deliberately requires ``split="train"``.  It accepts no target
    label/mask/defect metadata, and callers must not pass target abnormal test
    samples.  Population mean and standard deviation are frozen per signal.
    """

    if not isinstance(split, str) or split.casefold() not in {
        "train",
        "training",
    }:
        raise BIRADInputError(
            "BIR-AD normalization may be fitted only with split='train'"
        )
    features = _support_sequence(
        training_patch_features, "training_patch_features"
    )
    images = _support_sequence(training_images, "training_images")
    if not features or len(features) != len(images):
        raise BIRADInputError(
            "training_patch_features and training_images must be equally non-empty"
        )
    np = _numpy()
    epsilon = _positive_finite(eps, "eps")
    sigma = _positive_finite(structure_tensor_sigma, "structure_tensor_sigma")
    floor = _positive_finite(scale_floor, "scale_floor")
    clip = _positive_finite(clip_value, "clip_value")

    buckets = {name: [] for name in BIR_AD_SIGNAL_NAMES}
    for patch_values, image in zip(features, images):
        image_grid_shape = patch_grid_shape
        pixel_input = image
        if isinstance(image, PatchAlignedImage):
            if (
                patch_grid_shape is not None
                and tuple(patch_grid_shape) != image.patch_grid_shape
            ):
                raise BIRADInputError(
                    "training patch_grid_shape disagrees with aligned pixels"
                )
            image_grid_shape = image.patch_grid_shape
            pixel_input = image.pixels
        raw = _compute_raw_signals(
            patch_values,
            pixel_input,
            patch_grid_shape=image_grid_shape,
            neighbor_connectivity=neighbor_connectivity,
            two_sided_radius=two_sided_radius,
            structure_tensor_sigma=sigma,
            eps=epsilon,
            np=np,
        )
        for name, values in raw.by_name().items():
            buckets[name].extend(float(value) for value in values)

    locations = []
    scales = []
    for name in BIR_AD_SIGNAL_NAMES:
        values = sorted(buckets[name])
        location = _stable_nonnegative_mean(values)
        scale = _stable_root_mean_square(
            sorted(abs(value - location) for value in values)
        )
        minimum_scale = floor * max(1.0, abs(location))
        locations.append(location)
        scales.append(max(scale, minimum_scale))
    patch_observation_count = len(buckets[BIR_AD_SIGNAL_NAMES[0]])
    if any(len(buckets[name]) != patch_observation_count for name in BIR_AD_SIGNAL_NAMES):
        raise BIRADInputError("BIR-AD training signals have inconsistent patch counts")
    return BIRADNormalizationStats(
        locations=tuple(locations),
        scales=tuple(scales),
        sample_count=patch_observation_count,
        source="training_categories",
        clip_value=clip,
    )


def load_bir_ad_normalization_stats(
    path: str | Path,
) -> BIRADNormalizationStats:
    stats_path = Path(path)
    if not stats_path.is_file():
        raise BIRADInputError(
            f"BIR-AD normalization statistics do not exist: {stats_path}"
        )
    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BIRADInputError(
            f"could not read BIR-AD normalization statistics {stats_path}: {exc}"
        ) from exc
    if isinstance(payload, Mapping) and "statistics" in payload:
        payload = payload["statistics"]
    return BIRADNormalizationStats.from_mapping(payload)


def _compute_raw_signals(
    patch_features: Any,
    image: Any,
    *,
    patch_grid_shape: Sequence[int] | None,
    neighbor_connectivity: int,
    two_sided_radius: int,
    structure_tensor_sigma: float,
    eps: float,
    np: Any,
) -> _RawBoundarySignals:
    patches = _patch_matrix(patch_features, np)
    grid_shape = _resolve_grid_shape(patches.shape[0], patch_grid_shape)
    if neighbor_connectivity not in (4, 8):
        raise BIRADInputError("neighbor_connectivity must be 4 or 8")
    if (
        isinstance(two_sided_radius, bool)
        or not isinstance(two_sided_radius, int)
        or two_sided_radius <= 0
    ):
        raise BIRADInputError("two_sided_radius must be a positive integer")
    normalized_patches = _normalize_patch_tokens(patches, eps, np)
    patch_channel_std = _stable_channel_std(patches, np)
    # Formal n_p = ||f_p||_2 / sqrt(d).
    patch_l2_norm = _stable_l2_norm_rows(patches, np) / math.sqrt(
        patches.shape[1]
    )
    structural_boundary = _structural_boundary(
        normalized_patches,
        patches,
        grid_shape,
        connectivity=neighbor_connectivity,
        eps=eps,
        np=np,
    )
    grayscale = _grayscale_image(image, np)
    (
        sobel_edge_energy,
        orientation_consistency,
        normal_directions,
    ) = _pixel_boundary_signals(
        grayscale,
        grid_shape,
        sigma=structure_tensor_sigma,
        eps=eps,
        np=np,
    )
    two_sided_contrast = _two_sided_feature_contrast(
        patches,
        normal_directions,
        grid_shape,
        radius=two_sided_radius,
        eps=eps,
        np=np,
    )
    return _RawBoundarySignals(
        patch_channel_std=patch_channel_std,
        patch_l2_norm=patch_l2_norm,
        structural_boundary=structural_boundary,
        sobel_edge_energy=sobel_edge_energy,
        boundary_orientation_consistency=orientation_consistency,
        two_sided_feature_contrast=two_sided_contrast,
        normalized_patches=normalized_patches,
        grid_shape=grid_shape,
    )


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise BIRADDependencyError(
            "BIR-AD requires NumPy from the optional Stage 5 environment"
        ) from exc
    return np


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise BIRADInputError(f"{name} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BIRADInputError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise BIRADInputError(f"{name} must be a positive finite number")
    return result


def _nonnegative_finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise BIRADInputError(f"{name} must be a non-negative finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BIRADInputError(
            f"{name} must be a non-negative finite number"
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise BIRADInputError(f"{name} must be a non-negative finite number")
    return result


def _normalization_stats(
    value: BIRADNormalizationStats | Mapping[str, Any] | None,
) -> BIRADNormalizationStats:
    if value is None:
        return BIRADNormalizationStats.identity()
    if isinstance(value, BIRADNormalizationStats):
        return value
    if isinstance(value, Mapping):
        return BIRADNormalizationStats.from_mapping(value)
    raise BIRADInputError(
        "normalization_stats must be BIRADNormalizationStats or a mapping"
    )


def _patch_matrix(value: Any, np: Any) -> Any:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BIRADInputError("patch_features must be a numeric [P,C] matrix") from exc
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise BIRADInputError("patch_features must be a non-empty [P,C] matrix")
    if not bool(np.all(np.isfinite(array))):
        raise BIRADInputError("patch_features contains NaN or infinity")
    return np.ascontiguousarray(array)


def _resolve_grid_shape(
    patch_count: int, value: Sequence[int] | None
) -> tuple[int, int]:
    if value is None:
        side = math.isqrt(patch_count)
        if side * side != patch_count:
            raise BIRADInputError(
                f"patch_grid_shape is required for non-square patch count {patch_count}"
            )
        return side, side
    if isinstance(value, (str, bytes)):
        raise BIRADInputError("patch_grid_shape must contain two positive integers")
    try:
        shape = tuple(value)
    except TypeError as exc:
        raise BIRADInputError(
            "patch_grid_shape must contain two positive integers"
        ) from exc
    if len(shape) != 2:
        raise BIRADInputError("patch_grid_shape must contain two positive integers")
    rows, columns = shape
    if (
        isinstance(rows, bool)
        or isinstance(columns, bool)
        or not isinstance(rows, int)
        or not isinstance(columns, int)
        or rows <= 0
        or columns <= 0
    ):
        raise BIRADInputError("patch_grid_shape must contain two positive integers")
    if rows * columns != patch_count:
        raise BIRADInputError(
            f"patch_grid_shape {(rows, columns)} does not match {patch_count} patches"
        )
    return rows, columns


def _clarity_weights(
    value: Sequence[float],
) -> tuple[float, float, float, float, float]:
    if isinstance(value, (str, bytes)):
        raise BIRADInputError("clarity_weights must contain five values")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise BIRADInputError("clarity_weights must contain five values") from exc
    if len(items) != len(BIR_AD_CLARITY_COMPONENT_NAMES):
        raise BIRADInputError("clarity_weights must contain five values")
    converted = []
    for item in items:
        if isinstance(item, bool):
            raise BIRADInputError(
                "clarity_weights must be finite, non-negative, and not all zero"
            )
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise BIRADInputError(
                "clarity_weights must be finite, non-negative, and not all zero"
            ) from exc
        if not math.isfinite(number) or number < 0.0:
            raise BIRADInputError(
                "clarity_weights must be finite, non-negative, and not all zero"
            )
        converted.append(number)
    total = math.fsum(converted)
    if total <= 0.0:
        raise BIRADInputError(
            "clarity_weights must be finite, non-negative, and not all zero"
        )
    return tuple(item / total for item in converted)  # type: ignore[return-value]


def _stable_channel_std(patches: Any, np: Any) -> Any:
    scale = np.max(np.abs(patches), axis=1)
    scaled = np.divide(
        patches,
        scale[:, None],
        out=np.zeros_like(patches),
        where=scale[:, None] > 0.0,
    )
    factors = np.std(scaled, axis=1, ddof=0, dtype=np.float64)
    return _bounded_product(scale, factors, np)


def _stable_l2_norm_rows(patches: Any, np: Any) -> Any:
    scale = np.max(np.abs(patches), axis=1)
    scaled = np.divide(
        patches,
        scale[:, None],
        out=np.zeros_like(patches),
        where=scale[:, None] > 0.0,
    )
    factors = np.sqrt(np.sum(scaled * scaled, axis=1, dtype=np.float64))
    return _bounded_product(scale, factors, np)


def _bounded_product(left: Any, right: Any, np: Any) -> Any:
    maximum = np.finfo(np.float64).max
    result = np.zeros_like(left, dtype=np.float64)
    nonzero = (left > 0.0) & (right > 0.0)
    safe = nonzero & (left <= maximum / np.maximum(right, 1.0))
    result[safe] = left[safe] * right[safe]
    result[nonzero & ~safe] = maximum
    return result


def _normalize_patch_tokens(patches: Any, eps: float, np: Any) -> Any:
    norms = _stable_l2_norm_rows(patches, np)
    return np.divide(
        patches,
        norms[:, None],
        out=np.zeros_like(patches, dtype=np.float64),
        where=norms[:, None] > eps,
    )


def _structural_boundary(
    normalized: Any,
    raw: Any,
    grid_shape: tuple[int, int],
    *,
    connectivity: int,
    eps: float,
    np: Any,
) -> Any:
    rows, columns = grid_shape
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    valid = _stable_l2_norm_rows(raw, np) > eps
    result = np.zeros(rows * columns, dtype=np.float64)
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            distances = []
            for dr, dc in offsets:
                neighbour_row = row + dr
                neighbour_column = column + dc
                if 0 <= neighbour_row < rows and 0 <= neighbour_column < columns:
                    neighbour = neighbour_row * columns + neighbour_column
                    if not valid[index] and not valid[neighbour]:
                        distance = 0.0
                    else:
                        cosine = float(
                            np.dot(normalized[index], normalized[neighbour])
                        )
                        distance = 1.0 - min(1.0, max(-1.0, cosine))
                    distances.append(distance)
            result[index] = (
                _stable_nonnegative_mean(sorted(distances)) if distances else 0.0
            )
    return result


def _grayscale_image(image: Any, np: Any) -> Any:
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise BIRADInputError(f"BIR-AD image does not exist: {path}")
        try:
            from PIL import Image

            with Image.open(path) as opened:
                array = np.asarray(opened.convert("RGB"), dtype=np.float64)
        except Exception as exc:
            raise BIRADInputError(f"could not read BIR-AD image {path}: {exc}") from exc
    elif hasattr(image, "convert") and hasattr(image, "size"):
        try:
            array = np.asarray(image.convert("RGB"), dtype=np.float64)
        except Exception as exc:
            raise BIRADInputError(f"could not convert BIR-AD image: {exc}") from exc
    else:
        try:
            array = np.asarray(image, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BIRADInputError("image must be a local path, PIL image, or array") from exc
    if array.ndim == 2:
        grayscale = array
    elif array.ndim == 3:
        if array.shape[-1] in (1, 3, 4):
            channels = array[..., :3]
        elif array.shape[0] in (1, 3, 4):
            channels = np.moveaxis(array[:3], 0, -1)
        else:
            raise BIRADInputError(
                "image array must have shape [H,W], [H,W,C], or [C,H,W]"
            )
        if channels.shape[-1] == 1:
            grayscale = channels[..., 0]
        else:
            grayscale = (
                0.2989 * channels[..., 0]
                + 0.5870 * channels[..., 1]
                + 0.1140 * channels[..., 2]
            )
    else:
        raise BIRADInputError(
            "image array must have shape [H,W], [H,W,C], or [C,H,W]"
        )
    if grayscale.shape[0] == 0 or grayscale.shape[1] == 0:
        raise BIRADInputError("image must have non-empty spatial dimensions")
    if not bool(np.all(np.isfinite(grayscale))):
        raise BIRADInputError("image contains NaN or infinity")
    scale = float(np.max(np.abs(grayscale)))
    if scale > 1.0:
        grayscale = grayscale / scale
    return np.ascontiguousarray(grayscale, dtype=np.float64)


def _sobel_gradients(grayscale: Any, np: Any) -> tuple[Any, Any]:
    padded = np.pad(grayscale, 1, mode="edge")
    gx = (
        -padded[:-2, :-2]
        + padded[:-2, 2:]
        - 2.0 * padded[1:-1, :-2]
        + 2.0 * padded[1:-1, 2:]
        - padded[2:, :-2]
        + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2]
        - 2.0 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return gx, gy


def _pixel_boundary_signals(
    grayscale: Any,
    grid_shape: tuple[int, int],
    *,
    sigma: float,
    eps: float,
    np: Any,
) -> tuple[Any, Any, Any]:
    rows, columns = grid_shape
    height, width = grayscale.shape
    if height < rows or width < columns:
        raise BIRADInputError(
            f"image shape {(height, width)} is smaller than patch grid {grid_shape}"
        )
    gx, gy = _sobel_gradients(grayscale, np)
    magnitude = np.hypot(gx, gy)
    tensor_xx = _gaussian_blur(gx * gx, sigma, np)
    tensor_xy = _gaussian_blur(gx * gy, sigma, np)
    tensor_yy = _gaussian_blur(gy * gy, sigma, np)
    bounds = _grid_bounds((height, width), grid_shape, np)
    edge = np.zeros(rows * columns, dtype=np.float64)
    coherence = np.zeros(rows * columns, dtype=np.float64)
    directions = np.zeros((rows * columns, 2), dtype=np.float64)
    index = 0
    for row in range(rows):
        for column in range(columns):
            slices = (
                slice(bounds[0][row], bounds[0][row + 1]),
                slice(bounds[1][column], bounds[1][column + 1]),
            )
            edge[index] = _stable_nonnegative_mean(
                sorted(float(value) for value in magnitude[slices].reshape(-1))
            )
            a = _stable_nonnegative_mean(
                sorted(float(value) for value in tensor_xx[slices].reshape(-1))
            )
            b = math.fsum(
                sorted(float(value) for value in tensor_xy[slices].reshape(-1))
            ) / tensor_xy[slices].size
            c = _stable_nonnegative_mean(
                sorted(float(value) for value in tensor_yy[slices].reshape(-1))
            )
            discriminant = math.hypot(a - c, 2.0 * b)
            trace = max(0.0, a + c)
            coherence[index] = min(1.0, max(0.0, discriminant / (trace + eps)))
            mean_gx = math.fsum(
                sorted(float(value) for value in gx[slices].reshape(-1))
            ) / gx[slices].size
            mean_gy = math.fsum(
                sorted(float(value) for value in gy[slices].reshape(-1))
            ) / gy[slices].size
            direction_norm = math.hypot(mean_gx, mean_gy)
            if direction_norm > eps:
                directions[index] = (mean_gx / direction_norm, mean_gy / direction_norm)
            elif discriminant > eps:
                eigenvalue = 0.5 * (trace + discriminant)
                if abs(b) > eps:
                    vx, vy = eigenvalue - c, b
                elif a >= c:
                    vx, vy = 1.0, 0.0
                else:
                    vx, vy = 0.0, 1.0
                norm = math.hypot(vx, vy)
                directions[index] = (vx / norm, vy / norm)
            index += 1
    return edge, coherence, directions


def _gaussian_blur(values: Any, sigma: float, np: Any) -> Any:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coordinates / sigma) ** 2)
    kernel /= kernel.sum(dtype=np.float64)
    blurred = _convolve_axis(values, kernel, axis=0, np=np)
    return _convolve_axis(blurred, kernel, axis=1, np=np)


def _convolve_axis(values: Any, kernel: Any, *, axis: int, np: Any) -> Any:
    radius = kernel.shape[0] // 2
    pad_width = [(0, 0)] * values.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(values, pad_width, mode="edge")
    output = np.zeros_like(values, dtype=np.float64)
    for offset, weight in enumerate(kernel):
        slices = [slice(None)] * values.ndim
        slices[axis] = slice(offset, offset + values.shape[axis])
        output += float(weight) * padded[tuple(slices)]
    return output


def _grid_bounds(
    image_shape: tuple[int, int], grid_shape: tuple[int, int], np: Any
) -> tuple[Any, Any]:
    return (
        np.linspace(0, image_shape[0], grid_shape[0] + 1, dtype=np.int64),
        np.linspace(0, image_shape[1], grid_shape[1] + 1, dtype=np.int64),
    )


def _two_sided_feature_contrast(
    patches: Any,
    directions: Any,
    grid_shape: tuple[int, int],
    *,
    radius: int,
    eps: float,
    np: Any,
) -> Any:
    rows, columns = grid_shape
    feature_grid = patches.reshape(rows, columns, patches.shape[1])
    result = np.zeros(rows * columns, dtype=np.float64)
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            nx, ny = directions[index]
            if math.hypot(float(nx), float(ny)) <= eps:
                continue
            plus = []
            minus = []
            for step in range(1, radius + 1):
                plus.append(
                    _bilinear_feature(
                        feature_grid,
                        row + step * ny,
                        column + step * nx,
                        np,
                    )
                )
                minus.append(
                    _bilinear_feature(
                        feature_grid,
                        row - step * ny,
                        column - step * nx,
                        np,
                    )
                )
            mu_plus = _stable_vector_mean(np.stack(plus), np)
            mu_minus = _stable_vector_mean(np.stack(minus), np)
            norm_plus = float(_stable_l2_norm_rows(mu_plus[None, :], np)[0])
            norm_minus = float(_stable_l2_norm_rows(mu_minus[None, :], np)[0])
            if norm_plus <= eps and norm_minus <= eps:
                contrast = 0.0
            elif norm_plus <= eps or norm_minus <= eps:
                contrast = 1.0
            else:
                cosine = float(
                    np.dot(mu_plus / norm_plus, mu_minus / norm_minus)
                )
                contrast = 1.0 - min(1.0, max(-1.0, cosine))
            result[index] = contrast
    return result


def _bilinear_feature(grid: Any, row: float, column: float, np: Any) -> Any:
    row = min(grid.shape[0] - 1.0, max(0.0, float(row)))
    column = min(grid.shape[1] - 1.0, max(0.0, float(column)))
    row0 = int(math.floor(row))
    column0 = int(math.floor(column))
    row1 = min(row0 + 1, grid.shape[0] - 1)
    column1 = min(column0 + 1, grid.shape[1] - 1)
    row_weight = row - row0
    column_weight = column - column0
    return (
        (1.0 - row_weight) * (1.0 - column_weight) * grid[row0, column0]
        + (1.0 - row_weight) * column_weight * grid[row0, column1]
        + row_weight * (1.0 - column_weight) * grid[row1, column0]
        + row_weight * column_weight * grid[row1, column1]
    )


def _stable_sigmoid(values: Any, np: Any) -> Any:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _stable_softmax(logits: Any, np: Any) -> Any:
    shifted = logits - float(np.max(logits))
    exponential = np.exp(shifted)
    total = math.fsum(sorted(float(value) for value in exponential))
    if not math.isfinite(total) or total <= 0.0:
        raise BIRADInputError("boundary weight softmax normalization failed")
    weights = exponential / total
    correction = math.fsum(sorted(float(value) for value in weights))
    return weights / correction


def _probability_weights(scores: Any, eps: float, np: Any) -> Any:
    if not bool(np.all(np.isfinite(scores))) or bool(np.any(scores < 0.0)):
        raise BIRADInputError("patch weights require finite non-negative scores")
    total = math.fsum(sorted(float(value) for value in scores))
    if total <= eps:
        return np.full(scores.shape, 1.0 / scores.shape[0], dtype=np.float64)
    weights = scores / total
    correction = math.fsum(sorted(float(value) for value in weights))
    return weights / correction


def _stable_dot(left: Any, right: Any) -> float:
    return math.fsum(
        sorted(float(a) * float(b) for a, b in zip(left, right))
    )


def _patch_response_matrix(result: BIRADResult, np: Any) -> Any:
    """Return bounded cross-modal responses used for support matching."""

    return np.stack(
        (
            _stable_sigmoid(result.normalized_structural_boundary, np),
            _stable_sigmoid(result.normalized_sobel_edge_energy, np),
            result.clarity,
            result.boundary_orientation_consistency,
            _stable_sigmoid(result.normalized_two_sided_feature_contrast, np),
        ),
        axis=1,
    )


def _canonical_patch_bank(
    results: Sequence[BIRADResult], np: Any
) -> tuple[Any, Any]:
    rows: list[tuple[bytes, Any, Any]] = []
    for result in results:
        responses = _patch_response_matrix(result, np)
        for feature, response in zip(result.normalized_patch_features, responses):
            feature_row = np.ascontiguousarray(feature, dtype=np.float64)
            response_row = np.ascontiguousarray(response, dtype=np.float64)
            key = feature_row.tobytes() + response_row.tobytes()
            rows.append((key, feature_row, response_row))
    if not rows:
        raise BIRADInputError("cannot build an empty support patch bank")
    rows.sort(key=lambda item: item[0])
    return (
        np.stack([item[1] for item in rows]),
        np.stack([item[2] for item in rows]),
    )


def _nearest_response_consistency(
    source_features: Any,
    source_responses: Any,
    candidate_features: Any,
    candidate_responses: Any,
    *,
    temperature: float,
    chunk_size: int,
    np: Any,
) -> Any:
    if source_features.shape[1] != candidate_features.shape[1]:
        raise BIRADInputError(
            "support/query feature dimensions disagree for boundary consistency"
        )
    consistency = np.empty(source_features.shape[0], dtype=np.float64)
    for start in range(0, source_features.shape[0], chunk_size):
        stop = min(start + chunk_size, source_features.shape[0])
        similarities = source_features[start:stop] @ candidate_features.T
        matches = np.argmax(similarities, axis=1)
        response_delta = np.mean(
            np.abs(
                source_responses[start:stop]
                - candidate_responses[matches]
            ),
            axis=1,
            dtype=np.float64,
        )
        consistency[start:stop] = np.exp(-response_delta / temperature)
    return np.clip(consistency, 0.0, 1.0)


def _cross_support_patch_consistency(
    supports: Sequence[BIRADResult],
    *,
    temperature: float,
    chunk_size: int,
    np: Any,
) -> tuple[tuple[Any, ...], float | None, bool]:
    if len(supports) < 2:
        return (
            tuple(
                np.zeros(result.patch_count, dtype=np.float64)
                for result in supports
            ),
            None,
            False,
        )
    patch_values = []
    image_values = []
    for index, source in enumerate(supports):
        candidates = [
            result for other_index, result in enumerate(supports)
            if other_index != index
        ]
        candidate_features, candidate_responses = _canonical_patch_bank(
            candidates, np
        )
        consistency = _nearest_response_consistency(
            source.normalized_patch_features,
            _patch_response_matrix(source, np),
            candidate_features,
            candidate_responses,
            temperature=temperature,
            chunk_size=chunk_size,
            np=np,
        )
        patch_values.append(consistency)
        image_values.append(_stable_dot(source.boundary_weights, consistency))
    aggregate = math.fsum(sorted(image_values)) / len(image_values)
    return tuple(patch_values), aggregate, True


def _permutation_stable_weighted_pool(
    patches: Any, weights: Any, np: Any
) -> Any:
    terms = np.ascontiguousarray(patches * weights[:, None], dtype=np.float64)
    order = sorted(range(terms.shape[0]), key=lambda index: terms[index].tobytes())
    return np.asarray(
        [
            math.fsum(float(terms[index, channel]) for index in order)
            for channel in range(terms.shape[1])
        ],
        dtype=np.float64,
    )


def _stable_nonnegative_mean(values: Sequence[float]) -> float:
    if not values:
        raise BIRADInputError("cannot average an empty BIR-AD signal")
    maximum = max(values)
    if maximum == 0.0:
        return 0.0
    normalized_mean = math.fsum(value / maximum for value in values) / len(values)
    if normalized_mean > math.nextafter(
        math.inf, 0.0
    ) / maximum:
        return math.nextafter(math.inf, 0.0)
    return maximum * normalized_mean


def _stable_root_mean_square(values: Sequence[float]) -> float:
    """Return sqrt(mean(x^2)) without squaring values at their original scale."""

    if not values:
        raise BIRADInputError("cannot scale an empty BIR-AD signal")
    maximum = max(values)
    if maximum == 0.0:
        return 0.0
    factor = math.sqrt(
        math.fsum((value / maximum) ** 2 for value in values) / len(values)
    )
    finite_maximum = math.nextafter(math.inf, 0.0)
    if factor > finite_maximum / maximum:
        return finite_maximum
    return maximum * factor


def _stable_vector_mean(values: Any, np: Any) -> Any:
    """Average a small [N,C] feature bank without overflowing its reduction."""

    scale = np.max(np.abs(values), axis=0)
    normalized = np.divide(
        values,
        scale[None, :],
        out=np.zeros_like(values, dtype=np.float64),
        where=scale[None, :] > 0.0,
    )
    factors = np.asarray(
        [
            math.fsum(float(value) for value in normalized[:, channel])
            / values.shape[0]
            for channel in range(values.shape[1])
        ],
        dtype=np.float64,
    )
    return scale * factors


def _support_sequence(value: Any, name: str) -> list[Any]:
    if isinstance(value, (str, bytes, Path)) or value is None:
        raise BIRADInputError(f"{name} must be a non-empty sequence")
    try:
        return [value[index] for index in range(len(value))]
    except (TypeError, IndexError) as exc:
        raise BIRADInputError(f"{name} must be a non-empty sequence") from exc


def _result_sort_key(result: BIRADResult) -> bytes:
    digest = hashlib.sha256()
    for value in (
        result.normalized_patch_features,
        result.structural_boundary,
        result.pixel_feature_boundary_disagreement,
        result.clarity,
        result.boundary_weights,
        result.clear_weights,
        result.ambiguous_weights,
        result.clear_representation,
        result.ambiguous_representation,
    ):
        digest.update(value.tobytes())
    digest.update(float(result.bai).hex().encode("ascii"))
    digest.update(float(result.bai_reliability).hex().encode("ascii"))
    return digest.digest()
