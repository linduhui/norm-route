"""Foreground/background decoupled normal prototypes for Stage 5 routing.

FBDP-AD builds both prototype banks exclusively from the current task's
normal support patch features.  Border and low-objectness patches form the
background pool; stable patches that are far from that pool form the
foreground pool.  Query patches are only evaluated after the support context
has been frozen, so a query can never alter prototype selection or the
support-derived objectness gate.

The implementation intentionally accepts feature arrays and geometry only.
It has no evaluator-side metadata input and does not modify an anomaly expert.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence


FBDP_AD_PROTOCOL_VERSION = "stage5.fbdp_ad.v2"
FBDP_AD_RESIDUAL_QUANTILES = (0.50, 0.90, 0.95, 0.99)
FBDP_AD_PROTOTYPE_METHODS = ("kmeans", "pooling")
FBDP_AD_CONSISTENCY_MODES = (
    "same_position",
    "local_window",
    "nearest_neighbor",
)
FBDP_AD_BACKGROUND_CANDIDATE_MODES = (
    "combined",
    "border_only",
    "low_objectness_only",
)
FBDP_AD_QUERY_BANK_MODES = ("decoupled", "single")
FBDP_AD_K1_GATE_POLICIES = ("neutral", "shrink", "disable")


class FBDPADInputError(ValueError):
    """Raised when FBDP-AD receives malformed or non-finite features."""


class FBDPADDependencyError(ImportError):
    """Raised when the optional Stage 5 numeric dependency is unavailable."""


@dataclass(frozen=True)
class FBDPADSupportContext:
    """Query-independent foreground/background model of one normal support set."""

    foreground_prototypes: Any = field(repr=False, compare=False)
    background_prototypes: Any = field(repr=False, compare=False)
    support_objectness: tuple[Any, ...] = field(repr=False, compare=False)
    support_cross_consistency: tuple[Any, ...] = field(repr=False, compare=False)
    foreground_candidate_masks: tuple[Any, ...] = field(repr=False, compare=False)
    background_candidate_masks: tuple[Any, ...] = field(repr=False, compare=False)
    border_mask: Any = field(repr=False, compare=False)
    patch_grid_shape: tuple[int, int]
    feature_dimension: int
    support_count: int
    foreground_candidate_count: int
    background_candidate_count: int
    foreground_background_confusion: float
    objectness_gate: float
    objectness_contrast: float
    support_reliability: float
    support_assignment_confidence: float
    prototype_compactness: float
    foreground_support_coverage: float
    leave_one_out_reconstruction_margin: float | None
    leave_one_out_reconstruction_valid: bool
    support_consistency_valid: bool
    foreground_candidate_fallback: bool
    foreground_candidate_ratio: float
    background_candidate_ratio: float
    prototype_method: str
    consistency_mode: str
    consistency_window_radius: int
    background_candidate_mode: str
    query_bank_mode: str
    k1_gate_policy: str
    k1_gate_scale: float
    fallback_gate_cap: float
    use_cross_support_consistency: bool
    use_fbc_in_gate: bool
    use_objectness_gate: bool
    temperature: float
    residual_quantile_levels: tuple[float, ...] = FBDP_AD_RESIDUAL_QUANTILES
    protocol_version: str = FBDP_AD_PROTOCOL_VERSION

    @property
    def fbc(self) -> float:
        """Compatibility shorthand for foreground-background confusion."""

        return self.foreground_background_confusion

    @property
    def foreground_prototype_count(self) -> int:
        return int(self.foreground_prototypes.shape[0])

    @property
    def background_prototype_count(self) -> int:
        return int(self.background_prototypes.shape[0])

    @property
    def g_obj(self) -> float:
        """Formula shorthand for the support-derived objectness gate."""

        return self.objectness_gate

    @property
    def reliability(self) -> float:
        return self.support_reliability


@dataclass(frozen=True)
class FBDPADResult:
    """Patch-level decoupled similarities and residual summaries for one query."""

    support_context: FBDPADSupportContext = field(repr=False, compare=False)
    foreground_similarity: Any
    background_similarity: Any
    foreground_probability: Any
    foreground_residual: Any
    background_residual: Any
    decoupled_similarity: Any
    residuals: Any
    residual_quantiles: Any
    gated_residuals: Any
    gated_residual_quantiles: Any
    foreground_background_margin: Any
    foreground_assignment_entropy: Any
    foreground_assignment_entropy_quantiles: Any
    foreground_background_margin_quantiles: Any
    patch_grid_shape: tuple[int, int]
    protocol_version: str = FBDP_AD_PROTOCOL_VERSION

    @property
    def patch_count(self) -> int:
        return int(self.residuals.shape[0])

    @property
    def fbc(self) -> float:
        return self.support_context.foreground_background_confusion

    @property
    def foreground_background_confusion(self) -> float:
        return self.support_context.foreground_background_confusion

    @property
    def objectness_gate(self) -> float:
        return self.support_context.objectness_gate

    @property
    def foreground_prototypes(self) -> Any:
        return self.support_context.foreground_prototypes

    @property
    def background_prototypes(self) -> Any:
        return self.support_context.background_prototypes

    @property
    def fbdp_vector(self) -> tuple[float, ...]:
        """Return compact Router-ready support and gated-query statistics."""

        return (
            self.fbc,
            self.objectness_gate,
            *(float(value) for value in self.gated_residual_quantiles),
        )

    # Formula-oriented and query-explicit aliases keep diagnostics and method
    # descriptions readable without duplicating stored arrays.
    @property
    def s_fg(self) -> Any:
        return self.foreground_similarity

    @property
    def s_bg(self) -> Any:
        return self.background_similarity

    @property
    def pi_fg(self) -> Any:
        return self.foreground_probability

    @property
    def r_fg(self) -> Any:
        return self.foreground_residual

    @property
    def r_bg(self) -> Any:
        return self.background_residual

    @property
    def margin_fg_bg(self) -> Any:
        return self.foreground_background_margin

    @property
    def query_decoupled_similarity(self) -> Any:
        return self.decoupled_similarity

    @property
    def query_residuals(self) -> Any:
        return self.residuals

    @property
    def query_residual_quantiles(self) -> Any:
        return self.residual_quantiles

    @property
    def gated_query_residual_quantiles(self) -> Any:
        return self.gated_residual_quantiles

    @property
    def query_assignment_entropy(self) -> Any:
        return self.foreground_assignment_entropy


def prepare_fbdp_ad_support_context(
    support_patch_features: Sequence[Any] | Any,
    *,
    patch_grid_shape: Sequence[int] | None = None,
    border_width: int = 1,
    background_objectness_quantile: float = 0.35,
    foreground_consistency_quantile: float = 0.50,
    foreground_distance_quantile: float = 0.65,
    num_foreground_prototypes: int = 4,
    num_background_prototypes: int = 4,
    prototype_method: str = "kmeans",
    kmeans_iterations: int = 20,
    consistency_mode: str = "local_window",
    consistency_window_radius: int = 1,
    use_cross_support_consistency: bool = True,
    background_candidate_mode: str = "combined",
    query_bank_mode: str = "decoupled",
    k1_gate_policy: str = "shrink",
    k1_gate_scale: float = 0.50,
    fallback_gate_cap: float = 0.25,
    use_fbc_in_gate: bool = True,
    use_objectness_gate: bool = True,
    temperature: float = 0.10,
    residual_quantiles: Sequence[float] = FBDP_AD_RESIDUAL_QUANTILES,
) -> FBDPADSupportContext:
    """Build a deterministic FBDP-AD context from normal support patches only.

    Objectness is a continuous support-derived score combining distance from
    border appearance, spatial centrality, and cross-support stability.  It is
    used only for candidate construction; no category list or target-query
    statistic controls the gate.
    """

    np = _numpy()
    supports = _support_matrices(support_patch_features, np)
    patch_count, dimension = supports[0].shape
    for index, patches in enumerate(supports[1:], start=1):
        if patches.shape != (patch_count, dimension):
            raise FBDPADInputError(
                f"support_patch_features[{index}] shape {patches.shape} disagrees "
                f"with {(patch_count, dimension)}"
            )

    grid_shape = _resolve_grid_shape(patch_count, patch_grid_shape)
    _positive_integer(border_width, "border_width")
    if border_width > min(grid_shape):
        raise FBDPADInputError("border_width cannot exceed the smaller grid dimension")
    _unit_quantile(
        background_objectness_quantile, "background_objectness_quantile"
    )
    _unit_quantile(
        foreground_consistency_quantile, "foreground_consistency_quantile"
    )
    _unit_quantile(foreground_distance_quantile, "foreground_distance_quantile")
    _positive_integer(num_foreground_prototypes, "num_foreground_prototypes")
    _positive_integer(num_background_prototypes, "num_background_prototypes")
    _positive_integer(kmeans_iterations, "kmeans_iterations")
    _nonnegative_integer(consistency_window_radius, "consistency_window_radius")
    _positive_finite(temperature, "temperature")
    method = str(prototype_method).strip().lower()
    if method not in FBDP_AD_PROTOTYPE_METHODS:
        raise FBDPADInputError(
            f"prototype_method must be one of {FBDP_AD_PROTOTYPE_METHODS}"
        )
    resolved_consistency_mode = str(consistency_mode).strip().lower()
    if resolved_consistency_mode not in FBDP_AD_CONSISTENCY_MODES:
        raise FBDPADInputError(
            f"consistency_mode must be one of {FBDP_AD_CONSISTENCY_MODES}"
        )
    resolved_background_mode = str(background_candidate_mode).strip().lower()
    if resolved_background_mode not in FBDP_AD_BACKGROUND_CANDIDATE_MODES:
        raise FBDPADInputError(
            "background_candidate_mode must be one of "
            f"{FBDP_AD_BACKGROUND_CANDIDATE_MODES}"
        )
    resolved_query_bank_mode = str(query_bank_mode).strip().lower()
    if resolved_query_bank_mode not in FBDP_AD_QUERY_BANK_MODES:
        raise FBDPADInputError(
            f"query_bank_mode must be one of {FBDP_AD_QUERY_BANK_MODES}"
        )
    resolved_k1_policy = str(k1_gate_policy).strip().lower()
    if resolved_k1_policy not in FBDP_AD_K1_GATE_POLICIES:
        raise FBDPADInputError(
            f"k1_gate_policy must be one of {FBDP_AD_K1_GATE_POLICIES}"
        )
    _unit_quantile(k1_gate_scale, "k1_gate_scale")
    _unit_quantile(fallback_gate_cap, "fallback_gate_cap")
    for name, value in (
        ("use_cross_support_consistency", use_cross_support_consistency),
        ("use_fbc_in_gate", use_fbc_in_gate),
        ("use_objectness_gate", use_objectness_gate),
    ):
        if not isinstance(value, bool):
            raise FBDPADInputError(f"{name} must be boolean")
    quantile_levels = _quantile_levels(residual_quantiles)

    normalized_original = [
        _l2_normalize_rows(patches, f"support_patch_features[{index}]", np)
        for index, patches in enumerate(supports)
    ]
    # Canonical support order makes reductions and prototypes exactly invariant
    # to support permutation.  Arrays exposed for diagnostics are mapped back to
    # the caller's order below.
    canonical_order = sorted(
        range(len(normalized_original)),
        key=lambda index: normalized_original[index].tobytes(),
    )
    normalized = [normalized_original[index] for index in canonical_order]
    stack = np.stack(normalized, axis=0).astype(np.float64, copy=False)

    consistency_valid = len(normalized) >= 2 and use_cross_support_consistency
    if consistency_valid:
        canonical_consistency = _cross_support_consistency(
            normalized,
            grid_shape,
            mode=resolved_consistency_mode,
            window_radius=consistency_window_radius,
            np=np,
        )
    else:
        # One-shot or explicit no-consistency ablations retain a neutral
        # candidate path while the validity bit distinguishes missing evidence.
        canonical_consistency = [
            np.ones(patch_count, dtype=np.float64) for _ in normalized
        ]

    border = _border_mask(grid_shape, border_width, np).reshape(-1)
    centrality = _spatial_centrality(grid_shape, np).reshape(-1)
    border_rows = np.concatenate(
        [patches[border] for patches in normalized], axis=0
    )
    initial_background = _build_prototypes(
        border_rows,
        num_background_prototypes,
        method=method,
        iterations=kmeans_iterations,
        np=np,
    )

    canonical_objectness: list[Any] = []
    for patches, consistency in zip(normalized, canonical_consistency):
        border_similarity = _maximum_similarity(patches, initial_background, np)
        feature_foregroundness = np.clip(1.0 - border_similarity, 0.0, 1.0)
        objectness = (
            feature_foregroundness
            * (0.5 + 0.5 * centrality)
            * (0.5 + 0.5 * consistency)
        )
        canonical_objectness.append(np.clip(objectness, 0.0, 1.0))

    flat_objectness = np.concatenate(canonical_objectness)
    low_objectness_cut = float(
        np.quantile(flat_objectness, background_objectness_quantile)
    )
    tiled_border = np.tile(border, len(normalized))
    low_objectness_mask = flat_objectness <= low_objectness_cut
    if resolved_background_mode == "combined":
        flat_background_mask = tiled_border | low_objectness_mask
    elif resolved_background_mode == "border_only":
        flat_background_mask = tiled_border.copy()
    else:
        flat_background_mask = low_objectness_mask
    if not bool(np.any(flat_background_mask)):
        raise FBDPADInputError("background candidate selection produced no patches")
    flat_supports = np.concatenate(normalized, axis=0)
    background_rows = flat_supports[flat_background_mask]
    background_prototypes = _build_prototypes(
        background_rows,
        num_background_prototypes,
        method=method,
        iterations=kmeans_iterations,
        np=np,
    )

    flat_consistency = np.concatenate(canonical_consistency)
    background_distance = np.clip(
        1.0 - _maximum_similarity(flat_supports, background_prototypes, np),
        0.0,
        1.0,
    )
    foreground_selection_pool = (~tiled_border) & (
        flat_objectness > low_objectness_cut
    )
    if not bool(np.any(foreground_selection_pool)):
        foreground_selection_pool = ~tiled_border
    if not bool(np.any(foreground_selection_pool)):
        foreground_selection_pool = np.ones(flat_supports.shape[0], dtype=bool)
    consistency_cut = float(
        np.quantile(
            flat_consistency[foreground_selection_pool],
            foreground_consistency_quantile,
        )
    )
    distance_cut = float(
        np.quantile(
            background_distance[foreground_selection_pool],
            foreground_distance_quantile,
        )
    )
    foreground_mask = (
        (flat_consistency >= consistency_cut)
        & (background_distance >= distance_cut)
        & (flat_objectness > low_objectness_cut)
        & ~tiled_border
    )
    fallback = not bool(np.any(foreground_mask))
    if fallback:
        # Degenerate grids and texture sets may have no strict interior
        # candidate.  Select the maximally stable/far patch, then let FBC and
        # the objectness contrast drive the support gate toward zero.
        candidate_score = (
            background_distance
            * (0.5 + 0.5 * flat_consistency)
            * (0.5 + 0.5 * flat_objectness)
        )
        best = int(np.argmax(candidate_score))
        foreground_mask = np.zeros(flat_supports.shape[0], dtype=bool)
        foreground_mask[best] = True

    foreground_rows = flat_supports[foreground_mask]
    foreground_prototypes = _build_prototypes(
        foreground_rows,
        num_foreground_prototypes,
        method=method,
        iterations=kmeans_iterations,
        np=np,
    )
    fbc = _foreground_background_confusion(
        foreground_prototypes, background_prototypes, np
    )
    foreground_objectness = flat_objectness[foreground_mask]
    background_objectness = flat_objectness[flat_background_mask]
    observed_range = float(flat_objectness.max() - flat_objectness.min())
    if observed_range <= np.finfo(np.float64).eps:
        objectness_contrast = 0.0
    else:
        objectness_contrast = float(
            np.clip(
                (
                    float(foreground_objectness.mean(dtype=np.float64))
                    - float(background_objectness.mean(dtype=np.float64))
                )
                / observed_range,
                0.0,
                1.0,
            )
        )
    foreground_stability = float(
        flat_consistency[foreground_mask].mean(dtype=np.float64)
    )
    assignment_confidence = _support_assignment_confidence(
        flat_supports,
        foreground_prototypes,
        background_prototypes,
        temperature=float(temperature),
        np=np,
    )
    prototype_compactness = _prototype_compactness(
        foreground_rows,
        background_rows,
        foreground_prototypes,
        background_prototypes,
        np,
    )
    canonical_foreground_masks_for_reliability = _split_flat_mask(
        foreground_mask, len(normalized), patch_count
    )
    canonical_background_masks_for_reliability = _split_flat_mask(
        flat_background_mask, len(normalized), patch_count
    )
    foreground_support_coverage = float(
        sum(bool(np.any(mask)) for mask in canonical_foreground_masks_for_reliability)
        / len(normalized)
    )
    loo_margin = _leave_one_out_reconstruction_margin(
        normalized,
        canonical_foreground_masks_for_reliability,
        canonical_background_masks_for_reliability,
        num_foreground_prototypes=num_foreground_prototypes,
        num_background_prototypes=num_background_prototypes,
        method=method,
        iterations=kmeans_iterations,
        np=np,
    )
    loo_valid = loo_margin is not None
    loo_reliability = (
        float(np.clip((loo_margin + 1.0) * 0.5, 0.0, 1.0))
        if loo_margin is not None
        else 1.0
    )
    support_reliability = _geometric_mean(
        (
            assignment_confidence,
            prototype_compactness,
            foreground_support_coverage,
            loo_reliability,
        )
    )
    separation = max(0.0, 1.0 - fbc) if use_fbc_in_gate else 1.0
    # The base evidence follows the v1 support-only gate.  The independent
    # reliability term adds assignment confidence, compactness, support
    # coverage, and leave-one-support-out reconstruction evidence.
    objectness_gate = float(
        np.clip(
            math.sqrt(separation * objectness_contrast * foreground_stability)
            * math.sqrt(support_reliability),
            0.0,
            1.0,
        )
    )
    if len(normalized) == 1:
        if resolved_k1_policy == "disable":
            objectness_gate = 0.0
        elif resolved_k1_policy == "shrink":
            objectness_gate *= float(k1_gate_scale)
    if fallback:
        objectness_gate = min(objectness_gate, float(fallback_gate_cap))
    if not use_objectness_gate:
        objectness_gate = 1.0
    if not bool(np.any(np.linalg.norm(flat_supports, axis=1) > 0.0)):
        objectness_gate = 0.0

    canonical_background_masks = _split_flat_mask(
        flat_background_mask, len(normalized), patch_count
    )
    canonical_foreground_masks = _split_flat_mask(
        foreground_mask, len(normalized), patch_count
    )
    objectness_original = _restore_order(canonical_objectness, canonical_order)
    consistency_original = _restore_order(canonical_consistency, canonical_order)
    background_masks_original = _restore_order(
        canonical_background_masks, canonical_order
    )
    foreground_masks_original = _restore_order(
        canonical_foreground_masks, canonical_order
    )

    return FBDPADSupportContext(
        foreground_prototypes=foreground_prototypes.astype(np.float32),
        background_prototypes=background_prototypes.astype(np.float32),
        support_objectness=tuple(
            np.asarray(values, dtype=np.float32) for values in objectness_original
        ),
        support_cross_consistency=tuple(
            np.asarray(values, dtype=np.float32) for values in consistency_original
        ),
        foreground_candidate_masks=tuple(
            np.asarray(values, dtype=bool) for values in foreground_masks_original
        ),
        background_candidate_masks=tuple(
            np.asarray(values, dtype=bool) for values in background_masks_original
        ),
        border_mask=border.astype(bool),
        patch_grid_shape=grid_shape,
        feature_dimension=dimension,
        support_count=len(supports),
        foreground_candidate_count=int(foreground_mask.sum()),
        background_candidate_count=int(flat_background_mask.sum()),
        foreground_background_confusion=fbc,
        objectness_gate=objectness_gate,
        objectness_contrast=objectness_contrast,
        support_reliability=support_reliability,
        support_assignment_confidence=assignment_confidence,
        prototype_compactness=prototype_compactness,
        foreground_support_coverage=foreground_support_coverage,
        leave_one_out_reconstruction_margin=loo_margin,
        leave_one_out_reconstruction_valid=loo_valid,
        support_consistency_valid=consistency_valid,
        foreground_candidate_fallback=fallback,
        foreground_candidate_ratio=float(foreground_mask.mean()),
        background_candidate_ratio=float(flat_background_mask.mean()),
        prototype_method=method,
        consistency_mode=resolved_consistency_mode,
        consistency_window_radius=int(consistency_window_radius),
        background_candidate_mode=resolved_background_mode,
        query_bank_mode=resolved_query_bank_mode,
        k1_gate_policy=resolved_k1_policy,
        k1_gate_scale=float(k1_gate_scale),
        fallback_gate_cap=float(fallback_gate_cap),
        use_cross_support_consistency=use_cross_support_consistency,
        use_fbc_in_gate=use_fbc_in_gate,
        use_objectness_gate=use_objectness_gate,
        temperature=float(temperature),
        residual_quantile_levels=quantile_levels,
    )


def compose_fbdp_ad_query(
    query_patch_features: Any,
    support_context: FBDPADSupportContext,
    *,
    temperature: float | None = None,
) -> FBDPADResult:
    """Evaluate query patches against a previously frozen support context."""

    if not isinstance(support_context, FBDPADSupportContext):
        raise TypeError("support_context must be FBDPADSupportContext")
    np = _numpy()
    query = _matrix(query_patch_features, "query_patch_features", np)
    if query.shape[1] != support_context.feature_dimension:
        raise FBDPADInputError(
            f"query feature dimension {query.shape[1]} disagrees with support "
            f"dimension {support_context.feature_dimension}"
        )
    expected_patch_count = math.prod(support_context.patch_grid_shape)
    if query.shape[0] != expected_patch_count:
        raise FBDPADInputError(
            f"query patch count {query.shape[0]} disagrees with support grid "
            f"{support_context.patch_grid_shape}"
        )
    tau = support_context.temperature if temperature is None else temperature
    _positive_finite(tau, "temperature")
    normalized_query = _l2_normalize_rows(query, "query_patch_features", np)
    foreground_similarity = _maximum_similarity(
        normalized_query, support_context.foreground_prototypes, np
    )
    background_similarity = _maximum_similarity(
        normalized_query, support_context.background_prototypes, np
    )
    margin = foreground_similarity - background_similarity
    foreground_probability = _sigmoid(margin / float(tau), np)
    if support_context.query_bank_mode == "single":
        single_similarity = np.maximum(
            foreground_similarity, background_similarity
        )
        foreground_residual = foreground_probability * (1.0 - single_similarity)
        background_residual = (1.0 - foreground_probability) * (
            1.0 - single_similarity
        )
        decoupled_similarity = single_similarity
    else:
        foreground_residual = foreground_probability * (
            1.0 - foreground_similarity
        )
        background_residual = (1.0 - foreground_probability) * (
            1.0 - background_similarity
        )
        decoupled_similarity = (
            foreground_probability * foreground_similarity
            + (1.0 - foreground_probability) * background_similarity
        )
    residuals = foreground_residual + background_residual
    # This identity is useful for detecting future formula drift.
    if not np.allclose(residuals, 1.0 - decoupled_similarity, rtol=1e-12, atol=1e-12):
        raise RuntimeError("FBDP-AD decoupled similarity identity failed")
    quantiles = np.quantile(residuals, support_context.residual_quantile_levels)
    gated = support_context.objectness_gate * residuals
    gated_quantiles = np.quantile(
        gated, support_context.residual_quantile_levels
    )
    assignment_entropy = _binary_entropy(foreground_probability, np)
    assignment_entropy_quantiles = np.quantile(
        assignment_entropy, support_context.residual_quantile_levels
    )
    margin_quantiles = np.quantile(
        margin, support_context.residual_quantile_levels
    )

    return FBDPADResult(
        support_context=support_context,
        # Retain float64 here: residuals close to zero lose their algebraic
        # relationship to ``1 - decoupled_similarity`` if both sides are
        # rounded independently to float32.
        foreground_similarity=foreground_similarity,
        background_similarity=background_similarity,
        foreground_probability=foreground_probability,
        foreground_residual=foreground_residual,
        background_residual=background_residual,
        decoupled_similarity=decoupled_similarity,
        residuals=residuals,
        residual_quantiles=np.asarray(quantiles, dtype=np.float64),
        gated_residuals=gated,
        gated_residual_quantiles=np.asarray(gated_quantiles, dtype=np.float64),
        foreground_background_margin=margin,
        foreground_assignment_entropy=assignment_entropy,
        foreground_assignment_entropy_quantiles=np.asarray(
            assignment_entropy_quantiles, dtype=np.float64
        ),
        foreground_background_margin_quantiles=np.asarray(
            margin_quantiles, dtype=np.float64
        ),
        patch_grid_shape=support_context.patch_grid_shape,
    )


def compute_fbdp_ad(
    query_patch_features: Any,
    support_patch_features: Sequence[Any] | Any,
    **kwargs: Any,
) -> FBDPADResult:
    """Build support prototypes and compute one query's FBDP-AD statistics."""

    context = prepare_fbdp_ad_support_context(support_patch_features, **kwargs)
    return compose_fbdp_ad_query(query_patch_features, context)


def compute_foreground_background_confusion(
    foreground_prototypes: Any, background_prototypes: Any
) -> float:
    """Return symmetric positive-cosine overlap between two prototype banks."""

    np = _numpy()
    foreground = _l2_normalize_rows(
        _matrix(foreground_prototypes, "foreground_prototypes", np),
        "foreground_prototypes",
        np,
    )
    background = _l2_normalize_rows(
        _matrix(background_prototypes, "background_prototypes", np),
        "background_prototypes",
        np,
    )
    if foreground.shape[1] != background.shape[1]:
        raise FBDPADInputError("foreground/background prototype dimensions disagree")
    return _foreground_background_confusion(foreground, background, np)


def _foreground_background_confusion(foreground: Any, background: Any, np: Any) -> float:
    similarities = np.clip(foreground @ background.T, 0.0, 1.0)
    forward = float(similarities.max(axis=1).mean(dtype=np.float64))
    backward = float(similarities.max(axis=0).mean(dtype=np.float64))
    return float(np.clip(0.5 * (forward + backward), 0.0, 1.0))


def _build_prototypes(
    rows: Any,
    count: int,
    *,
    method: str,
    iterations: int,
    np: Any,
) -> Any:
    candidates = _canonical_rows(
        _l2_normalize_rows(rows, "prototype candidates", np), np
    )
    cluster_count = min(int(count), candidates.shape[0])
    if method == "pooling":
        groups = np.array_split(candidates, cluster_count)
        centres = np.stack(
            [group.mean(axis=0, dtype=np.float64) for group in groups], axis=0
        )
        return _canonical_rows(
            _l2_normalize_rows(centres, "pooled prototypes", np), np
        )

    mean = _l2_normalize_rows(
        candidates.mean(axis=0, dtype=np.float64)[None, :],
        "prototype mean",
        np,
    )[0]
    first = int(np.argmin(candidates @ mean))
    selected = [first]
    while len(selected) < cluster_count:
        nearest_similarity = (candidates @ candidates[selected].T).max(axis=1)
        nearest_similarity[selected] = np.inf
        selected.append(int(np.argmin(nearest_similarity)))
    centres = candidates[selected].copy()

    for _ in range(iterations):
        assignments = np.argmax(candidates @ centres.T, axis=1)
        updated = centres.copy()
        for cluster in range(cluster_count):
            members = candidates[assignments == cluster]
            if members.shape[0]:
                updated[cluster] = members.mean(axis=0, dtype=np.float64)
        updated = _l2_normalize_rows(updated, "k-means prototypes", np)
        if np.array_equal(updated, centres):
            break
        centres = updated
    return _canonical_rows(centres, np)


def _maximum_similarity(rows: Any, prototypes: Any, np: Any) -> Any:
    return np.clip(
        np.asarray(rows, dtype=np.float64)
        @ np.asarray(prototypes, dtype=np.float64).T,
        -1.0,
        1.0,
    ).max(axis=1)


def _cross_support_consistency(
    supports: Sequence[Any],
    grid_shape: tuple[int, int],
    *,
    mode: str,
    window_radius: int,
    np: Any,
) -> list[Any]:
    patch_count = supports[0].shape[0]
    neighbours = (
        _window_neighbour_indices(grid_shape, window_radius)
        if mode == "local_window"
        else None
    )
    results: list[Any] = []
    for source_index, source in enumerate(supports):
        comparisons: list[Any] = []
        for target_index, target in enumerate(supports):
            if target_index == source_index:
                continue
            if mode == "same_position":
                similarities = np.sum(source * target, axis=1)
            elif mode == "nearest_neighbor":
                similarities = (source @ target.T).max(axis=1)
            else:
                similarities = np.empty(patch_count, dtype=np.float64)
                assert neighbours is not None
                for patch_index, candidate_indices in enumerate(neighbours):
                    similarities[patch_index] = np.max(
                        target[candidate_indices] @ source[patch_index]
                    )
            comparisons.append(np.clip(similarities, 0.0, 1.0))
        results.append(
            np.mean(np.stack(comparisons, axis=0), axis=0, dtype=np.float64)
        )
    return results


def _window_neighbour_indices(
    grid_shape: tuple[int, int], radius: int
) -> tuple[Any, ...]:
    rows, columns = grid_shape
    result = []
    for row in range(rows):
        for column in range(columns):
            indices = []
            for candidate_row in range(max(0, row - radius), min(rows, row + radius + 1)):
                for candidate_column in range(
                    max(0, column - radius), min(columns, column + radius + 1)
                ):
                    indices.append(candidate_row * columns + candidate_column)
            result.append(indices)
    return tuple(result)


def _binary_entropy(probabilities: Any, np: Any) -> Any:
    values = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        np.finfo(np.float64).eps,
        1.0 - np.finfo(np.float64).eps,
    )
    return -(
        values * np.log(values) + (1.0 - values) * np.log(1.0 - values)
    ) / math.log(2.0)


def _support_assignment_confidence(
    rows: Any,
    foreground_prototypes: Any,
    background_prototypes: Any,
    *,
    temperature: float,
    np: Any,
) -> float:
    margin = _maximum_similarity(rows, foreground_prototypes, np) - _maximum_similarity(
        rows, background_prototypes, np
    )
    probabilities = _sigmoid(margin / temperature, np)
    return float(np.clip(1.0 - _binary_entropy(probabilities, np).mean(), 0.0, 1.0))


def _prototype_compactness(
    foreground_rows: Any,
    background_rows: Any,
    foreground_prototypes: Any,
    background_prototypes: Any,
    np: Any,
) -> float:
    foreground = np.clip(
        _maximum_similarity(foreground_rows, foreground_prototypes, np), 0.0, 1.0
    )
    background = np.clip(
        _maximum_similarity(background_rows, background_prototypes, np), 0.0, 1.0
    )
    return float(
        np.clip(
            0.5
            * (
                foreground.mean(dtype=np.float64)
                + background.mean(dtype=np.float64)
            ),
            0.0,
            1.0,
        )
    )


def _leave_one_out_reconstruction_margin(
    supports: Sequence[Any],
    foreground_masks: Sequence[Any],
    background_masks: Sequence[Any],
    *,
    num_foreground_prototypes: int,
    num_background_prototypes: int,
    method: str,
    iterations: int,
    np: Any,
) -> float | None:
    if len(supports) < 2:
        return None
    margins: list[float] = []
    for held_out in range(len(supports)):
        foreground_parts = [
            supports[index][foreground_masks[index]]
            for index in range(len(supports))
            if index != held_out and bool(np.any(foreground_masks[index]))
        ]
        background_parts = [
            supports[index][background_masks[index]]
            for index in range(len(supports))
            if index != held_out and bool(np.any(background_masks[index]))
        ]
        if not foreground_parts or not background_parts:
            continue
        foreground_prototypes = _build_prototypes(
            np.concatenate(foreground_parts, axis=0),
            num_foreground_prototypes,
            method=method,
            iterations=iterations,
            np=np,
        )
        background_prototypes = _build_prototypes(
            np.concatenate(background_parts, axis=0),
            num_background_prototypes,
            method=method,
            iterations=iterations,
            np=np,
        )
        held_foreground = supports[held_out][foreground_masks[held_out]]
        held_background = supports[held_out][background_masks[held_out]]
        if held_foreground.shape[0]:
            margins.extend(
                (
                    _maximum_similarity(
                        held_foreground, foreground_prototypes, np
                    )
                    - _maximum_similarity(
                        held_foreground, background_prototypes, np
                    )
                ).tolist()
            )
        if held_background.shape[0]:
            margins.extend(
                (
                    _maximum_similarity(
                        held_background, background_prototypes, np
                    )
                    - _maximum_similarity(
                        held_background, foreground_prototypes, np
                    )
                ).tolist()
            )
    if not margins:
        return None
    return float(np.clip(math.fsum(sorted(margins)) / len(margins), -1.0, 1.0))


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    clipped = [min(1.0, max(0.0, float(value))) for value in values]
    if any(value == 0.0 for value in clipped):
        return 0.0
    return math.exp(math.fsum(math.log(value) for value in clipped) / len(clipped))


def _support_matrices(value: Sequence[Any] | Any, np: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        if value.ndim == 2:
            raw = [value]
        elif value.ndim == 3:
            raw = [value[index] for index in range(value.shape[0])]
        else:
            raise FBDPADInputError(
                "support_patch_features must be [K,P,C], [P,C], or a sequence"
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise FBDPADInputError("support_patch_features must be non-empty")
        try:
            sequence_array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            sequence_array = None
        if sequence_array is not None and sequence_array.ndim == 2:
            raw = [sequence_array]
        elif sequence_array is not None and sequence_array.ndim == 3:
            raw = [sequence_array[index] for index in range(sequence_array.shape[0])]
        else:
            raw = list(value)
    else:
        raise FBDPADInputError(
            "support_patch_features must be [K,P,C], [P,C], or a sequence"
        )
    return [
        _matrix(item, f"support_patch_features[{index}]", np)
        for index, item in enumerate(raw)
    ]


def _matrix(value: Any, name: str, np: Any) -> Any:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FBDPADInputError(f"{name} must be a numeric matrix") from exc
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise FBDPADInputError(f"{name} must have non-empty shape [P,C]")
    if not np.all(np.isfinite(array)):
        raise FBDPADInputError(f"{name} must contain only finite values")
    return array


def _l2_normalize_rows(rows: Any, name: str, np: Any) -> Any:
    array = np.asarray(rows, dtype=np.float64)
    scales = np.max(np.abs(array), axis=1, keepdims=True)
    safe_scales = np.where(scales > 0.0, scales, 1.0)
    scaled = array / safe_scales
    norms = np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))
    denominators = np.where(norms > 0.0, norms, 1.0)
    normalized = scaled / denominators
    if not np.all(np.isfinite(normalized)):
        raise FBDPADInputError(f"{name} could not be normalized safely")
    return normalized


def _canonical_rows(rows: Any, np: Any) -> Any:
    array = np.ascontiguousarray(rows, dtype=np.float64)
    keys = tuple(array[:, index] for index in reversed(range(array.shape[1])))
    return array[np.lexsort(keys)]


def _resolve_grid_shape(
    patch_count: int, patch_grid_shape: Sequence[int] | None
) -> tuple[int, int]:
    if patch_grid_shape is None:
        side = math.isqrt(patch_count)
        if side * side != patch_count:
            raise FBDPADInputError(
                "patch_grid_shape is required when patch count is not a perfect square"
            )
        return side, side
    if (
        isinstance(patch_grid_shape, (str, bytes))
        or len(patch_grid_shape) != 2
    ):
        raise FBDPADInputError("patch_grid_shape must contain rows and columns")
    rows, columns = patch_grid_shape
    _positive_integer(rows, "patch_grid_shape rows")
    _positive_integer(columns, "patch_grid_shape columns")
    resolved = (int(rows), int(columns))
    if math.prod(resolved) != patch_count:
        raise FBDPADInputError(
            f"patch_grid_shape {resolved} does not match {patch_count} patches"
        )
    return resolved


def _border_mask(grid_shape: tuple[int, int], width: int, np: Any) -> Any:
    rows, columns = grid_shape
    row_indices, column_indices = np.indices(grid_shape)
    return (
        (row_indices < width)
        | (row_indices >= rows - width)
        | (column_indices < width)
        | (column_indices >= columns - width)
    )


def _spatial_centrality(grid_shape: tuple[int, int], np: Any) -> Any:
    rows, columns = grid_shape
    row_indices, column_indices = np.indices(grid_shape)
    distance = np.minimum.reduce(
        (
            row_indices,
            column_indices,
            rows - 1 - row_indices,
            columns - 1 - column_indices,
        )
    ).astype(np.float64)
    maximum = float(distance.max())
    return distance / maximum if maximum > 0.0 else np.zeros_like(distance)


def _split_flat_mask(mask: Any, count: int, patch_count: int) -> list[Any]:
    return [
        mask[index * patch_count : (index + 1) * patch_count].copy()
        for index in range(count)
    ]


def _restore_order(canonical_values: Sequence[Any], canonical_order: Sequence[int]) -> list[Any]:
    restored: list[Any] = [None] * len(canonical_values)
    for canonical_index, original_index in enumerate(canonical_order):
        restored[original_index] = canonical_values[canonical_index]
    return restored


def _sigmoid(values: Any, np: Any) -> Any:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _quantile_levels(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not values:
        raise FBDPADInputError("residual_quantiles must be non-empty")
    levels = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in levels):
        raise FBDPADInputError("residual_quantiles must lie in [0, 1]")
    if levels != tuple(sorted(set(levels))):
        raise FBDPADInputError("residual_quantiles must be sorted and unique")
    return levels


def _unit_quantile(value: float, name: str) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FBDPADInputError(f"{name} must be numeric") from exc
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise FBDPADInputError(f"{name} must lie in [0, 1]")


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FBDPADInputError(f"{name} must be a positive integer")


def _nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FBDPADInputError(f"{name} must be a non-negative integer")


def _positive_finite(value: float, name: str) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FBDPADInputError(f"{name} must be numeric") from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise FBDPADInputError(f"{name} must be positive and finite")


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise FBDPADDependencyError("FBDP-AD requires NumPy") from exc
    return np


__all__ = [
    "FBDP_AD_BACKGROUND_CANDIDATE_MODES",
    "FBDP_AD_CONSISTENCY_MODES",
    "FBDP_AD_K1_GATE_POLICIES",
    "FBDP_AD_PROTOCOL_VERSION",
    "FBDP_AD_PROTOTYPE_METHODS",
    "FBDP_AD_QUERY_BANK_MODES",
    "FBDP_AD_RESIDUAL_QUANTILES",
    "FBDPADDependencyError",
    "FBDPADInputError",
    "FBDPADResult",
    "FBDPADSupportContext",
    "compose_fbdp_ad_query",
    "compute_fbdp_ad",
    "compute_foreground_background_confusion",
    "prepare_fbdp_ad_support_context",
]
