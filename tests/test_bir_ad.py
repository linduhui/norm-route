import csv
import math
from pathlib import Path

import pytest


np = pytest.importorskip("numpy")

from src.normroute.cli.diagnose_bir_ad import (  # noqa: E402
    PATCH_STATISTICS_NAME,
    SUMMARY_STATISTICS_NAME,
    write_bir_ad_diagnostics,
)
from src.normroute.router.bir_ad import (  # noqa: E402
    BIRADInputError,
    BIRADNormalizationStats,
    PatchAlignedImage,
    compute_bir_ad,
    compute_bir_ad_task,
    fit_bir_ad_normalization,
)


def _edge_image() -> np.ndarray:
    image = np.zeros((8, 8), dtype=np.float64)
    # The vertical edge lies inside the right patch cells, not on the grid split.
    image[:, 6:] = 1.0
    return image


def _patches(offset: float = 0.0) -> np.ndarray:
    return np.asarray(
        [
            [1.0 + offset, 0.0, 0.0, 0.0],
            [0.5 + offset, 0.5, 0.0, 0.0],
            [0.0, 1.0 + offset, 1.0, 0.0],
            [0.0, 0.0, 1.0 + offset, 2.0],
        ],
        dtype=np.float64,
    )


def _fitted_stats() -> BIRADNormalizationStats:
    return fit_bir_ad_normalization(
        [_patches(), _patches(0.5), _patches(1.0)],
        [_edge_image(), np.rot90(_edge_image()), np.rot90(_edge_image(), 2)],
        split="train",
    )


def test_formal_bai_weights_and_representations_are_normalized() -> None:
    patches = _patches()
    result = compute_bir_ad(patches, _edge_image())

    assert result.patch_grid_shape == (2, 2)
    assert result.patch_count == 4
    assert np.allclose(
        result.patch_l2_norm,
        np.linalg.norm(patches, axis=1) / math.sqrt(patches.shape[1]),
    )
    assert np.sum(result.boundary_weights) == pytest.approx(1.0)
    assert np.sum(result.clear_weights) == pytest.approx(1.0)
    assert np.sum(result.ambiguous_weights) == pytest.approx(1.0)
    assert np.all(result.boundary_weights >= 0.0)
    assert np.all(result.clear_weights >= 0.0)
    assert np.all(result.ambiguous_weights >= 0.0)
    assert result.clear_representation.shape == (4,)
    assert result.ambiguous_representation.shape == (4,)
    assert np.all((result.clarity > 0.0) & (result.clarity < 1.0))
    assert result.bai == pytest.approx(
        np.sum(result.boundary_weights * (1.0 - result.clarity))
    )
    assert np.sum(result.ambiguous_evidence) == pytest.approx(result.bai)
    assert np.sum(result.clear_evidence) == pytest.approx(1.0 - result.bai)
    assert result.bai_reliability == pytest.approx(
        np.sum(
            result.boundary_weights
            * result.boundary_orientation_consistency
        )
    )
    assert 0.0 <= result.bai <= 1.0
    assert 0.0 <= result.bai_reliability <= 1.0


def test_structural_boundary_matches_four_neighbour_cosine_definition() -> None:
    patches = _patches()
    result = compute_bir_ad(patches, _edge_image(), neighbor_connectivity=4)
    normalized = patches / np.linalg.norm(patches, axis=1, keepdims=True)
    # Top-left has top-right and bottom-left as its two four-neighbours.
    expected = 0.5 * (
        1.0 - float(np.dot(normalized[0], normalized[1]))
        + 1.0 - float(np.dot(normalized[0], normalized[2]))
    )

    assert result.structural_boundary[0] == pytest.approx(expected)
    assert np.max(result.structural_boundary) > np.min(result.structural_boundary)


def test_sobel_structure_tensor_and_two_sided_evidence_are_exposed() -> None:
    result = compute_bir_ad(_patches(), _edge_image())
    flat = compute_bir_ad(_patches(), np.zeros((8, 8), dtype=np.float64))

    assert np.max(result.sobel_edge_energy) > np.min(result.sobel_edge_energy)
    assert np.max(result.boundary_orientation_consistency) > 0.0
    assert np.all(
        (result.boundary_orientation_consistency >= 0.0)
        & (result.boundary_orientation_consistency <= 1.0)
    )
    assert np.all(
        (result.two_sided_feature_contrast >= 0.0)
        & (result.two_sided_feature_contrast <= 2.0)
    )
    assert np.all(flat.sobel_edge_energy == 0.0)
    assert np.all(flat.boundary_orientation_consistency == 0.0)
    assert np.all(flat.two_sided_feature_contrast == 0.0)
    assert flat.bai_reliability == 0.0


def test_strict_alignment_requires_verified_backbone_geometry() -> None:
    with pytest.raises(BIRADInputError, match="PatchAlignedImage"):
        compute_bir_ad(
            _patches(),
            _edge_image(),
            require_strict_alignment=True,
        )
    aligned = PatchAlignedImage(
        pixels=_edge_image(),
        patch_grid_shape=(2, 2),
        source_image_sha256="0" * 64,
        transform_fingerprint="frozen-transform-v1",
    )
    result = compute_bir_ad(
        _patches(),
        aligned,
        require_strict_alignment=True,
    )

    assert result.alignment_verified is True
    assert result.alignment_fingerprint == "frozen-transform-v1"
    assert result.source_image_sha256 == "0" * 64
    with pytest.raises(BIRADInputError, match="disagrees"):
        compute_bir_ad(
            _patches(),
            aligned,
            patch_grid_shape=(1, 4),
        )


def test_explicit_pixel_feature_disagreement_reduces_clarity() -> None:
    without_penalty = compute_bir_ad(
        _patches(),
        _edge_image(),
        disagreement_penalty=0.0,
    )
    with_penalty = compute_bir_ad(
        _patches(),
        _edge_image(),
        disagreement_penalty=2.0,
    )

    assert np.any(with_penalty.pixel_feature_boundary_disagreement > 0.0)
    assert np.allclose(
        with_penalty.pixel_feature_boundary_agreement,
        1.0 - with_penalty.pixel_feature_boundary_disagreement,
    )
    assert np.all(with_penalty.clarity <= without_penalty.clarity)
    assert with_penalty.weighted_pixel_feature_disagreement == pytest.approx(
        np.sum(
            with_penalty.boundary_weights
            * with_penalty.pixel_feature_boundary_disagreement
        )
    )


def test_training_normalization_is_frozen_and_permutation_invariant() -> None:
    patches = [_patches(), _patches(0.5), _patches(1.0)]
    images = [_edge_image(), np.rot90(_edge_image()), np.rot90(_edge_image(), 2)]
    first = fit_bir_ad_normalization(patches, images, split="train")
    order = [2, 0, 1]
    second = fit_bir_ad_normalization(
        [patches[index] for index in order],
        [images[index] for index in order],
        split="train",
    )

    assert first.is_fitted
    assert first.locations == second.locations
    assert first.scales == second.scales
    assert first.sample_count == second.sample_count == 12
    with pytest.raises(BIRADInputError, match="split='train'"):
        fit_bir_ad_normalization(patches, images, split="test")
    with pytest.raises(BIRADInputError, match="statistics are required"):
        compute_bir_ad(
            patches[0],
            images[0],
            require_fitted_normalization=True,
        )
    fitted_result = compute_bir_ad(
        patches[0],
        images[0],
        normalization_stats=first,
        require_fitted_normalization=True,
    )
    assert fitted_result.normalization_stats is first
    assert np.all(np.abs(fitted_result.normalized_structural_boundary) <= 5.0)


def test_single_patch_and_large_training_statistics_are_numerically_stable() -> None:
    single = compute_bir_ad(
        np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
    )
    assert single.structural_boundary.tolist() == [0.0]
    assert single.boundary_weights.tolist() == [1.0]

    large = _patches() * 1e300
    stats = fit_bir_ad_normalization(
        [large, large * 0.5],
        [_edge_image(), np.rot90(_edge_image())],
        split="train",
        two_sided_radius=2,
    )
    assert all(math.isfinite(value) for value in stats.locations)
    assert all(math.isfinite(value) and value > 0.0 for value in stats.scales)


def test_constant_tiny_and_large_inputs_remain_finite() -> None:
    cases = (
        np.zeros((4, 8), dtype=np.float64),
        np.full((4, 8), np.finfo(np.float64).tiny, dtype=np.float64),
        np.asarray(
            [
                [1e300, -1e300, 1e300, -1e300],
                [1e299, 1e299, -1e299, -1e299],
                [1e298, 0.0, 0.0, 0.0],
                [0.0, 1e297, 0.0, -1e297],
            ],
            dtype=np.float64,
        ),
    )
    for patches in cases:
        result = compute_bir_ad(patches, _edge_image())
        for value in (
            result.patch_channel_std,
            result.patch_l2_norm,
            result.structural_boundary,
            result.sobel_edge_energy,
            result.boundary_orientation_consistency,
            result.two_sided_feature_contrast,
            result.clarity,
            result.boundary_weights,
            result.clear_weights,
            result.ambiguous_weights,
            result.clear_representation,
            result.ambiguous_representation,
        ):
            assert np.all(np.isfinite(value))
        assert np.sum(result.boundary_weights) == pytest.approx(1.0)
        assert np.sum(result.clear_weights) == pytest.approx(1.0)
        assert np.sum(result.ambiguous_weights) == pytest.approx(1.0)
        assert math.isfinite(result.bai)
        assert math.isfinite(result.bai_reliability)


@pytest.mark.parametrize(
    "bad_features",
    [
        np.asarray([[np.nan, 0.0]]),
        np.asarray([[np.inf, 0.0]]),
        np.empty((0, 2)),
        np.asarray([1.0, 2.0]),
    ],
)
def test_invalid_numeric_inputs_fail_explicitly(bad_features: np.ndarray) -> None:
    with pytest.raises(BIRADInputError):
        compute_bir_ad(bad_features, np.zeros((2, 2)))


def test_support_permutation_is_exactly_invariant_and_vector_is_formal() -> None:
    query_patches = _patches(0.25)
    query_image = _edge_image()
    support_patches = [_patches(0.0), _patches(0.5), _patches(1.0)]
    support_images = [
        np.rot90(_edge_image(), 0),
        np.rot90(_edge_image(), 1),
        np.rot90(_edge_image(), 2),
    ]
    stats = _fitted_stats()
    first = compute_bir_ad_task(
        query_patches,
        query_image,
        support_patches,
        support_images,
        normalization_stats=stats,
    )
    permutation = [2, 0, 1]
    second = compute_bir_ad_task(
        query_patches,
        query_image,
        [support_patches[index] for index in permutation],
        [support_images[index] for index in permutation],
        normalization_stats=stats,
    )

    assert first.support_bai == second.support_bai
    assert first.support_bai_variance == second.support_bai_variance
    assert first.query_bai == second.query_bai
    assert first.query_support_boundary_shift == second.query_support_boundary_shift
    assert first.support_bai_std == second.support_bai_std
    assert first.support_bai_reliability == second.support_bai_reliability
    assert first.support_boundary_consistency == second.support_boundary_consistency
    assert (
        first.query_support_boundary_consistency
        == second.query_support_boundary_consistency
    )
    for first_values, second_values in zip(
        first.support_patch_consistency,
        second.support_patch_consistency,
    ):
        assert np.array_equal(first_values, second_values)
    assert np.array_equal(
        first.query_patch_support_consistency,
        second.query_patch_support_consistency,
    )
    assert [item.bai for item in first.supports] == [
        item.bai for item in second.supports
    ]
    assert first.query_support_boundary_shift == pytest.approx(
        first.query_bai - first.support_bai
    )
    assert first.bai_vector == (
        first.support_bai,
        first.support_bai_std,
        first.query_bai,
        first.query_support_boundary_shift,
        abs(first.query_support_boundary_shift),
    )


def test_patch_level_cross_support_consistency_has_explicit_k1_missingness() -> None:
    identical = compute_bir_ad_task(
        _patches(),
        _edge_image(),
        [_patches(), _patches()],
        [_edge_image(), _edge_image()],
    )
    one_shot = compute_bir_ad_task(
        _patches(),
        _edge_image(),
        [_patches()],
        [_edge_image()],
    )

    assert identical.support_boundary_consistency_valid is True
    assert identical.support_boundary_consistency == pytest.approx(1.0)
    assert all(np.allclose(values, 1.0) for values in identical.support_patch_consistency)
    assert identical.query_support_boundary_consistency == pytest.approx(1.0)
    assert one_shot.support_boundary_consistency_valid is False
    assert one_shot.support_boundary_consistency is None
    assert np.all(one_shot.support_patch_consistency[0] == 0.0)
    assert 0.0 <= one_shot.query_support_boundary_consistency <= 1.0


def test_diagnostics_write_weight_pngs_and_exhaustive_statistics(
    tmp_path: Path,
) -> None:
    task = compute_bir_ad_task(
        _patches(0.25),
        _edge_image(),
        [_patches(), _patches(0.5)],
        [_edge_image(), np.rot90(_edge_image())],
        normalization_stats=_fitted_stats(),
    )
    outputs = write_bir_ad_diagnostics(task, tmp_path, heatmap_cell_size=4)

    patch_path = tmp_path / PATCH_STATISTICS_NAME
    summary_path = tmp_path / SUMMARY_STATISTICS_NAME
    assert outputs["patch_statistics"] == str(patch_path)
    assert outputs["summary_statistics"] == str(summary_path)
    assert len(outputs["weight_visualizations"]) == 12
    assert all(Path(path).is_file() for path in outputs["weight_visualizations"])
    with patch_path.open(newline="", encoding="utf-8") as handle:
        patch_rows = list(csv.DictReader(handle))
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))

    assert len(patch_rows) == 12
    assert len(summary_rows) == 3
    assert {row["role"] for row in patch_rows} == {"query", "support"}
    assert all(float(row["boundary_weight"]) >= 0.0 for row in patch_rows)
    assert all(float(row["clear_weight"]) >= 0.0 for row in patch_rows)
    assert all(float(row["ambiguous_weight"]) >= 0.0 for row in patch_rows)
    assert all(
        0.0 <= float(row["pixel_feature_boundary_disagreement"]) <= 1.0
        for row in patch_rows
    )
    assert all(row["patch_support_consistency"] for row in patch_rows)
    assert all(row["support_bai"] for row in summary_rows)
    assert all(row["support_bai_variance"] for row in summary_rows)
    assert all(row["query_support_boundary_shift"] for row in summary_rows)
    assert all(row["query_support_boundary_consistency"] for row in summary_rows)
    assert all(row["support_boundary_consistency_valid"] == "True" for row in summary_rows)
    assert all(row["normalization_fitted"] == "True" for row in summary_rows)
