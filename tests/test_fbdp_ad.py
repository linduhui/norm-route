import csv
import inspect
import json
import math
from pathlib import Path

import pytest


np = pytest.importorskip("numpy")

from src.normroute.cli.diagnose_fbdp_ad import (  # noqa: E402
    PATCH_STATISTICS_NAME,
    PROTOTYPE_STATISTICS_NAME,
    RUN_RECORD_NAME,
    SUMMARY_STATISTICS_NAME,
    main,
    write_fbdp_ad_diagnostics,
)
from src.normroute.router.fbdp_ad import (  # noqa: E402
    FBDPADInputError,
    compose_fbdp_ad_query,
    compute_fbdp_ad,
    compute_foreground_background_confusion,
    prepare_fbdp_ad_support_context,
)


def _separated_supports(count: int = 3) -> list[np.ndarray]:
    supports = []
    for index in range(count):
        patches = np.tile(np.asarray([1.0, 0.0, 0.0]), (25, 1))
        for row in range(1, 4):
            for column in range(1, 4):
                patches[row * 5 + column] = np.asarray(
                    [0.0, 1.0, 0.01 * index]
                )
        supports.append(patches)
    return supports


def test_support_candidates_and_prototypes_follow_fbdp_contract() -> None:
    context = prepare_fbdp_ad_support_context(
        _separated_supports(),
        patch_grid_shape=(5, 5),
        num_foreground_prototypes=2,
        num_background_prototypes=2,
    )

    assert context.support_count == 3
    assert context.support_consistency_valid is True
    assert context.foreground_candidate_count > 0
    assert context.background_candidate_count >= 3 * 16
    assert context.foreground_prototype_count <= 2
    assert context.background_prototype_count <= 2
    assert context.foreground_prototypes.shape[1] == 3
    assert context.background_prototypes.shape[1] == 3
    assert 0.0 <= context.fbc <= 1.0
    assert 0.0 <= context.objectness_gate <= 1.0
    assert context.objectness_gate > 0.8
    for background_mask in context.background_candidate_masks:
        assert np.all(background_mask[context.border_mask])
    for foreground_mask, consistency, objectness in zip(
        context.foreground_candidate_masks,
        context.support_cross_consistency,
        context.support_objectness,
    ):
        assert np.all(consistency[foreground_mask] > 0.99)
        assert np.all(objectness[foreground_mask] > 0.0)


def test_decoupled_similarity_residual_formula_and_quantiles() -> None:
    supports = _separated_supports()
    query = supports[0].copy()
    query[12] = np.asarray([0.0, 0.0, 1.0])
    result = compute_fbdp_ad(query, supports, patch_grid_shape=(5, 5))

    expected_probability = 1.0 / (
        1.0
        + np.exp(
            -(
                result.foreground_similarity - result.background_similarity
            )
            / result.support_context.temperature
        )
    )
    assert np.allclose(result.foreground_probability, expected_probability)
    assert np.allclose(
        result.foreground_residual,
        result.foreground_probability * (1.0 - result.foreground_similarity),
    )
    assert np.allclose(
        result.background_residual,
        (1.0 - result.foreground_probability)
        * (1.0 - result.background_similarity),
    )
    assert np.allclose(
        result.residuals, result.foreground_residual + result.background_residual
    )
    assert np.allclose(result.residuals, 1.0 - result.decoupled_similarity)
    assert np.allclose(
        result.residual_quantiles,
        np.quantile(result.residuals, (0.50, 0.90, 0.95, 0.99)),
    )
    assert np.allclose(
        result.gated_residuals,
        result.objectness_gate * result.residuals,
    )
    assert result.residuals[12] > 0.9
    assert result.residuals[0] < 1e-3
    assert len(result.fbdp_vector) == 6
    assert result.s_fg is result.foreground_similarity
    assert result.s_bg is result.background_similarity
    assert result.pi_fg is result.foreground_probability
    assert result.r_fg is result.foreground_residual
    assert result.r_bg is result.background_residual
    assert result.query_decoupled_similarity is result.decoupled_similarity
    assert result.query_residual_quantiles is result.residual_quantiles


def test_well_explained_background_is_not_penalized_by_foreground_margin() -> None:
    context = prepare_fbdp_ad_support_context(
        _separated_supports(), patch_grid_shape=(5, 5), temperature=0.05
    )
    background_query = np.tile(np.asarray([1.0, 0.0, 0.0]), (25, 1))
    result = compose_fbdp_ad_query(background_query, context)

    assert np.all(result.background_similarity > 0.999)
    assert np.all(result.foreground_probability < 1e-6)
    assert np.max(result.residuals) < 1e-6


def test_texture_like_support_automatically_disables_fbdp_branch() -> None:
    texture = np.tile(np.asarray([1.0, 1.0, 0.0]), (25, 1))
    supports = [texture.copy(), texture.copy(), texture.copy()]
    context = prepare_fbdp_ad_support_context(
        supports, patch_grid_shape=(5, 5)
    )
    query = texture.copy()
    query[12] = np.asarray([0.0, 0.0, 1.0])
    result = compose_fbdp_ad_query(query, context)

    assert context.fbc == pytest.approx(1.0)
    assert context.objectness_contrast == 0.0
    assert context.objectness_gate == 0.0
    assert np.all(result.gated_residuals == 0.0)
    assert np.all(result.gated_residual_quantiles == 0.0)


def test_zero_feature_support_is_treated_as_uninformative_texture() -> None:
    supports = np.zeros((2, 25, 3), dtype=np.float64)
    result = compute_fbdp_ad(
        np.ones((25, 3), dtype=np.float64),
        supports.tolist(),
        patch_grid_shape=(5, 5),
    )

    assert result.support_context.g_obj == 0.0
    assert np.all(result.gated_residuals == 0.0)


def test_local_window_consistency_tolerates_one_patch_translation() -> None:
    first = np.tile(np.asarray([1.0, 0.0]), (25, 1))
    second = first.copy()
    first[12] = np.asarray([0.0, 1.0])
    second[13] = np.asarray([0.0, 1.0])
    same_position = prepare_fbdp_ad_support_context(
        [first, second],
        patch_grid_shape=(5, 5),
        consistency_mode="same_position",
    )
    local_window = prepare_fbdp_ad_support_context(
        [first, second],
        patch_grid_shape=(5, 5),
        consistency_mode="local_window",
        consistency_window_radius=1,
    )

    assert same_position.support_cross_consistency[0][12] == 0.0
    assert local_window.support_cross_consistency[0][12] == pytest.approx(1.0)


def test_k1_gate_policy_and_fallback_reliability_are_explicit() -> None:
    support = _separated_supports(1)
    neutral = prepare_fbdp_ad_support_context(
        support,
        patch_grid_shape=(5, 5),
        k1_gate_policy="neutral",
    )
    shrink = prepare_fbdp_ad_support_context(
        support,
        patch_grid_shape=(5, 5),
        k1_gate_policy="shrink",
        k1_gate_scale=0.5,
    )

    assert neutral.support_consistency_valid is False
    assert neutral.leave_one_out_reconstruction_valid is False
    assert shrink.objectness_gate == pytest.approx(0.5 * neutral.objectness_gate)
    assert 0.0 <= shrink.support_reliability <= 1.0
    if shrink.foreground_candidate_fallback:
        assert shrink.objectness_gate <= shrink.fallback_gate_cap


def test_background_candidate_and_single_bank_ablations_are_executable() -> None:
    supports = _separated_supports()
    combined = prepare_fbdp_ad_support_context(
        supports,
        patch_grid_shape=(5, 5),
        background_candidate_mode="combined",
    )
    border_only = prepare_fbdp_ad_support_context(
        supports,
        patch_grid_shape=(5, 5),
        background_candidate_mode="border_only",
        query_bank_mode="single",
    )
    result = compose_fbdp_ad_query(supports[0], border_only)

    assert combined.background_candidate_count >= border_only.background_candidate_count
    assert border_only.query_bank_mode == "single"
    assert np.allclose(
        result.decoupled_similarity,
        np.maximum(result.foreground_similarity, result.background_similarity),
    )
    assert np.all((result.foreground_assignment_entropy >= 0.0))
    assert np.all((result.foreground_assignment_entropy <= 1.0))
    assert result.foreground_assignment_entropy_quantiles.shape == (4,)
    assert result.foreground_background_margin_quantiles.shape == (4,)


@pytest.mark.parametrize("method", ["kmeans", "pooling"])
def test_small_prototype_builders_are_deterministic_and_support_invariant(
    method: str,
) -> None:
    supports = _separated_supports()
    first = prepare_fbdp_ad_support_context(
        supports, patch_grid_shape=(5, 5), prototype_method=method
    )
    second = prepare_fbdp_ad_support_context(
        [supports[2], supports[0], supports[1]],
        patch_grid_shape=(5, 5),
        prototype_method=method,
    )

    assert np.array_equal(first.foreground_prototypes, second.foreground_prototypes)
    assert np.array_equal(first.background_prototypes, second.background_prototypes)
    assert first.fbc == second.fbc
    assert first.objectness_gate == second.objectness_gate
    assert first.foreground_candidate_count == second.foreground_candidate_count
    assert first.background_candidate_count == second.background_candidate_count


def test_fbc_is_symmetric_and_bounded() -> None:
    foreground = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    background = np.asarray([[-1.0, 0.0], [0.0, -1.0]])
    assert compute_foreground_background_confusion(foreground, background) == 0.0
    assert compute_foreground_background_confusion(foreground, foreground) == 1.0
    assert compute_foreground_background_confusion(
        foreground, background
    ) == compute_foreground_background_confusion(background, foreground)


def test_query_cannot_change_support_context() -> None:
    context = prepare_fbdp_ad_support_context(
        _separated_supports(), patch_grid_shape=(5, 5)
    )
    foreground_before = context.foreground_prototypes.copy()
    background_before = context.background_prototypes.copy()
    compose_fbdp_ad_query(_separated_supports()[0], context)
    compose_fbdp_ad_query(np.ones((25, 3)), context)

    assert np.array_equal(context.foreground_prototypes, foreground_before)
    assert np.array_equal(context.background_prototypes, background_before)


def test_k1_and_large_values_are_finite() -> None:
    support = _separated_supports(1)[0] * 1e300
    result = compute_fbdp_ad(
        support.copy(), [support], patch_grid_shape=(5, 5)
    )

    assert result.support_context.support_consistency_valid is False
    assert all(
        np.all(np.isfinite(values))
        for values in (
            result.foreground_similarity,
            result.background_similarity,
            result.foreground_probability,
            result.decoupled_similarity,
            result.residuals,
            result.residual_quantiles,
        )
    )
    assert math.isfinite(result.fbc)
    assert math.isfinite(result.objectness_gate)


@pytest.mark.parametrize(
    "supports,query,kwargs,match",
    [
        ([np.ones((4, 2)), np.ones((5, 2))], np.ones((4, 2)), {}, "shape"),
        ([np.ones((4, 2))], np.ones((4, 3)), {}, "dimension"),
        ([np.ones((4, 2))], np.ones((4, 2)), {"temperature": 0.0}, "temperature"),
        ([np.ones((4, 2))], np.ones((4, 2)), {"prototype_method": "large"}, "prototype_method"),
        ([np.ones((4, 2))], np.ones((4, 2)), {"patch_grid_shape": (1, 3)}, "patch_grid_shape"),
        ([np.asarray([[np.nan, 0.0]])], np.ones((1, 2)), {}, "finite"),
    ],
)
def test_invalid_inputs_fail_explicitly(
    supports: list[np.ndarray],
    query: np.ndarray,
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(FBDPADInputError, match=match):
        compute_fbdp_ad(query, supports, **kwargs)


def test_public_api_has_no_evaluator_only_inputs() -> None:
    forbidden = ("label", "mask", "defect_type", "anomaly_type")
    for function in (
        compute_fbdp_ad,
        prepare_fbdp_ad_support_context,
        compose_fbdp_ad_query,
    ):
        parameters = set(inspect.signature(function).parameters)
        assert not parameters.intersection(forbidden)


def test_diagnostics_are_exhaustive_and_reproducible(tmp_path: Path) -> None:
    result = compute_fbdp_ad(
        _separated_supports()[0],
        _separated_supports(),
        patch_grid_shape=(5, 5),
    )
    outputs = write_fbdp_ad_diagnostics(result, tmp_path, heatmap_cell_size=2)

    assert outputs["patch_statistics"] == str(tmp_path / PATCH_STATISTICS_NAME)
    assert outputs["summary_statistics"] == str(tmp_path / SUMMARY_STATISTICS_NAME)
    assert outputs["prototype_statistics"] == str(
        tmp_path / PROTOTYPE_STATISTICS_NAME
    )
    assert len(outputs["visualizations"]) == 16
    assert all(Path(path).is_file() for path in outputs["visualizations"])
    with (tmp_path / PATCH_STATISTICS_NAME).open(
        newline="", encoding="utf-8"
    ) as handle:
        patch_rows = list(csv.DictReader(handle))
    with (tmp_path / SUMMARY_STATISTICS_NAME).open(
        newline="", encoding="utf-8"
    ) as handle:
        summary_rows = list(csv.DictReader(handle))

    assert len(patch_rows) == 100
    assert {row["role"] for row in patch_rows} == {"query", "support"}
    assert len(summary_rows) == 1
    assert float(summary_rows[0]["objectness_gate"]) > 0.0
    assert json.loads(summary_rows[0]["residual_quantiles"])


def test_diagnostic_cli_records_failure_instead_of_skipping(tmp_path: Path) -> None:
    output_dir = tmp_path / "diagnostics"
    status = main(
        [
            "--query-patches",
            str(tmp_path / "missing-query.npy"),
            "--support-patches",
            str(tmp_path / "missing-support.npy"),
            "--output-dir",
            str(output_dir),
            "--seed",
            "17",
        ]
    )

    assert status == 1
    run = json.loads((output_dir / RUN_RECORD_NAME).read_text(encoding="utf-8"))
    assert run["ok"] is False
    assert run["seed"] == 17
    assert run["git_commit"]
    assert run["environment"]["python_version"]
    assert run["predictions"] is None
    assert len(run["failures"]) == 1


def test_diagnostic_cli_records_success_provenance_and_prediction(
    tmp_path: Path,
) -> None:
    query_path = tmp_path / "query.npy"
    support_paths = [tmp_path / f"support-{index}.npy" for index in range(2)]
    np.save(query_path, _separated_supports()[0], allow_pickle=False)
    for path, support in zip(support_paths, _separated_supports(2)):
        np.save(path, support, allow_pickle=False)
    output_dir = tmp_path / "diagnostics"

    status = main(
        [
            "--query-patches",
            str(query_path),
            "--support-patches",
            str(support_paths[0]),
            "--support-patches",
            str(support_paths[1]),
            "--grid-shape",
            "5",
            "5",
            "--output-dir",
            str(output_dir),
            "--seed",
            "23",
        ]
    )

    assert status == 0
    run = json.loads((output_dir / RUN_RECORD_NAME).read_text(encoding="utf-8"))
    assert run["ok"] is True
    assert run["seed"] == 23
    assert run["predictions"] == str(output_dir / SUMMARY_STATISTICS_NAME)
    assert run["failures"] == []
    assert Path(run["predictions"]).is_file()
