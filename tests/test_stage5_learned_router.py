from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.normroute.cli.run_stage5_router import (
    TaskOutcome,
    _assert_feature_names_safe,
    balanced_query_task_weights,
    capability_objectives,
    empirical_routing_losses,
    fit_train_runtime_scale,
    fit_train_score_calibrations,
    main,
    read_evaluator_outcomes,
    read_router_features,
)
from src.normroute.router.feature_bundle import (
    ROUTER_FEATURE_BUNDLE_PROTOCOL_VERSION,
)
from src.normroute.router.learned_router import (
    LinearRouterModel,
    Stage5RouterError,
    _balanced_class_weights,
    fit_linear_router,
)


@pytest.mark.parametrize(
    "name",
    (
        "finalScore",
        "imagescore",
        "expertoutcome",
        "teacherprobability",
        "query_anomaly_type",
        "mylabelhint",
        "anomalyscorefeature",
        "utilityloss",
        "outcomeflag",
    ),
)
def test_formal_router_feature_boundary_reuses_inference_policy(name: str) -> None:
    with pytest.raises(Stage5RouterError, match="inference isolation"):
        _assert_feature_names_safe((name,))


def _calibration_fixture() -> tuple[
    tuple[str, ...], dict[str, TaskOutcome], tuple[str, ...], dict[str, float]
]:
    task_ids = tuple(f"task-{index}" for index in range(6))
    experts = ("first", "second")
    labels = (0, 0, 0, 1, 1, 1)
    outcomes = {
        task_id: TaskOutcome(
            label=label,
            scores={
                "first": (0.1, 0.2, 0.3, 0.7, 0.8, 0.9)[index],
                "second": (2.0, 2.5, 3.0, 7.0, 7.5, 8.0)[index],
            },
            runtimes_ms={"first": 100.0, "second": 10.0},
        )
        for index, (task_id, label) in enumerate(zip(task_ids, labels))
    }
    return task_ids, outcomes, experts, {task_id: 1.0 for task_id in task_ids}


def test_router_uses_shared_platt_protocol_and_is_per_expert_affine_invariant() -> None:
    task_ids, outcomes, experts, weights = _calibration_fixture()
    transformed = {
        task_id: TaskOutcome(
            label=outcome.label,
            scores={
                "first": 10.0 * outcome.scores["first"] + 7.0,
                "second": 3.25 * outcome.scores["second"] - 11.0,
            },
            runtimes_ms=outcome.runtimes_ms,
        )
        for task_id, outcome in outcomes.items()
    }
    first = fit_train_score_calibrations(
        task_ids,
        outcomes,
        experts,
        task_weights=weights,
        fit_categories=("train-a",),
    )
    second = fit_train_score_calibrations(
        task_ids,
        transformed,
        experts,
        task_weights=weights,
        fit_categories=("train-a",),
    )

    for task_id in task_ids:
        for expert in experts:
            assert first[expert].predict(outcomes[task_id].scores[expert]) == pytest.approx(
                second[expert].predict(transformed[task_id].scores[expert])
            )


def test_grouped_weights_make_duplicate_query_calibration_invariant() -> None:
    experts = ("first", "second")
    base_ids = ("normal-k1", "anomaly-k1")
    duplicate_ids = ("normal-k1", "normal-k4", "anomaly-k1")
    manifest = {
        "normal-k1": {"dataset": "mvtec", "category": "bottle", "sample_id": "normal"},
        "normal-k4": {"dataset": "mvtec", "category": "bottle", "sample_id": "normal"},
        "anomaly-k1": {"dataset": "mvtec", "category": "bottle", "sample_id": "anomaly"},
    }
    outcomes = {
        "normal-k1": TaskOutcome(0, {"first": 0.1, "second": 1.0}, {"first": 10.0, "second": 20.0}),
        "normal-k4": TaskOutcome(0, {"first": 0.1, "second": 1.0}, {"first": 10.0, "second": 20.0}),
        "anomaly-k1": TaskOutcome(1, {"first": 0.9, "second": 9.0}, {"first": 10.0, "second": 20.0}),
    }
    base_weights = balanced_query_task_weights(base_ids, manifest)
    duplicate_weights = balanced_query_task_weights(duplicate_ids, manifest)
    base = fit_train_score_calibrations(
        base_ids,
        outcomes,
        experts,
        task_weights=base_weights,
        fit_categories=("bottle",),
    )
    duplicate = fit_train_score_calibrations(
        duplicate_ids,
        outcomes,
        experts,
        task_weights=duplicate_weights,
        fit_categories=("bottle",),
    )

    assert sum(duplicate_weights[name] for name in ("normal-k1", "normal-k4")) == pytest.approx(
        duplicate_weights["anomaly-k1"]
    )
    for expert in experts:
        for task_id in base_ids:
            assert base[expert].predict(outcomes[task_id].scores[expert]) == pytest.approx(
                duplicate[expert].predict(outcomes[task_id].scores[expert])
            )


def test_score_calibration_handles_zero_variance_and_runtime_is_complete() -> None:
    task_ids = ("normal", "anomaly")
    experts = ("a", "b")
    outcomes = {
        "normal": TaskOutcome(0, {"a": 1.0, "b": 2.0}, {"a": 0.0, "b": 1.0}),
        "anomaly": TaskOutcome(1, {"a": 1.0, "b": 2.0}, {"a": 0.0, "b": 1.0}),
    }
    calibrations = fit_train_score_calibrations(
        task_ids,
        outcomes,
        experts,
        task_weights={"normal": 0.5, "anomaly": 0.5},
        fit_categories=("bottle",),
    )
    assert all(math.isfinite(item.predict(outcomes["normal"].scores[name])) for name, item in calibrations.items())
    assert fit_train_runtime_scale(task_ids, outcomes, experts) == 1.0

    incomplete = dict(outcomes)
    incomplete["normal"] = TaskOutcome(0, {"a": 1.0, "b": 2.0}, {"a": None, "b": 1.0})
    with pytest.raises(Stage5RouterError, match="incomplete"):
        fit_train_runtime_scale(task_ids, incomplete, experts)


def test_empirical_cost_tradeoff_can_change_winner() -> None:
    task_ids, outcomes, experts, weights = _calibration_fixture()
    calibrations = fit_train_score_calibrations(
        task_ids,
        outcomes,
        experts,
        task_weights=weights,
        fit_categories=("train-a",),
    )
    no_cost = empirical_routing_losses(
        (task_ids[-1],), outcomes, experts, calibrations,
        runtime_scale=100.0, runtime_tradeoff=0.0,
    )
    with_cost = empirical_routing_losses(
        (task_ids[-1],), outcomes, experts, calibrations,
        runtime_scale=100.0, runtime_tradeoff=1.0,
    )

    assert int(np.argmin(no_cost[0])) == 0
    assert int(np.argmin(with_cost[0])) == 1


def test_capability_objectives_reject_one_expert_and_malformed_inputs() -> None:
    profile = {
        "overall_skill": 0.6,
        "boundary_skill": 0.6,
        "fgbg_skill": 0.6,
        "lowshot_skill": 0.6,
        "texture_skill": 0.6,
        "latency_p95": 10.0,
        "failure_rate": 0.0,
        "confidence_intervals": {
            "overall_skill": {"lower": 0.5, "upper": 0.7}
        },
    }
    common = {
        "feature_values": np.zeros((1, 1)),
        "task_ids": ("task",),
        "manifest_by_task": {"task": {"k_shot": "1"}},
        "feature_names": ("normal_signal",),
        "capability_weight": 0.0,
        "uncertainty_weight": 1.0,
        "cost_weight": 0.0,
        "capability_mode": "static",
    }
    with pytest.raises(Stage5RouterError, match="at least two"):
        capability_objectives(
            np.asarray([[1.0]]),
            experts=("only",),
            bank={"profiles": {"only": profile}},
            **common,
        )

    profiles = {"a": dict(profile), "b": dict(profile)}
    with pytest.raises(Stage5RouterError, match="sum to one"):
        capability_objectives(
            np.asarray([[0.8, 0.8]]),
            experts=("a", "b"),
            bank={"profiles": profiles},
            **common,
        )
    malformed = json.loads(json.dumps(profiles))
    malformed["b"]["confidence_intervals"]["overall_skill"] = {
        "lower": 0.8,
        "upper": 0.2,
    }
    with pytest.raises(Stage5RouterError, match="confidence interval"):
        capability_objectives(
            np.asarray([[0.5, 0.5]]),
            experts=("a", "b"),
            bank={"profiles": malformed},
            **common,
        )


def test_linear_router_is_deterministic_and_uses_train_normalization() -> None:
    train_x = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 2.5, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 2.5],
        ],
        dtype=np.float32,
    )
    train_y = np.asarray([0, 0, 1, 1, 2, 2])
    validation_x = train_x + 0.1
    validation_y = train_y.copy()
    kwargs = {
        "feature_names": ("normal_a", "normal_b", "bir_query_bai"),
        "experts": ("anomalydino", "patchcore", "winclip"),
        "l2_grid": (0.0, 0.001),
        "epochs": 120,
        "batch_size": 6,
        "learning_rate": 0.1,
        "seed": 17,
        "device": "cpu",
    }

    first = fit_linear_router(
        train_x, train_y, validation_x, validation_y, **kwargs
    )
    second = fit_linear_router(
        train_x, train_y, validation_x, validation_y, **kwargs
    )

    assert first.model.to_dict() == second.model.to_dict()
    assert first.model.location == pytest.approx(
        tuple(np.mean(train_x, axis=0))
    )
    assert np.array_equal(first.model.predict_indices(validation_x), validation_y)
    restored = LinearRouterModel.from_dict(first.model.to_dict())
    assert restored.predict_proba(validation_x) == pytest.approx(
        first.model.predict_proba(validation_x)
    )


def test_linear_router_accepts_soft_targets_and_query_weights() -> None:
    train_x = np.asarray([[3.0, 0.0], [2.0, 0.0], [0.0, 3.0], [0.0, 2.0]])
    soft_y = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
    validation_y = np.asarray([0, 0, 1, 1])
    fit = fit_linear_router(
        train_x,
        soft_y,
        train_x,
        validation_y,
        feature_names=("normal_a", "bir_query_bai"),
        experts=("patchcore", "winclip"),
        train_sample_weights=np.asarray([0.5, 0.5, 1.0, 1.0]),
        validation_sample_weights=np.ones(4),
        l2_grid=(0.0,),
        epochs=100,
        batch_size=4,
        learning_rate=0.1,
        seed=3,
    )

    assert np.array_equal(fit.model.predict_indices(train_x), validation_y)
    assert fit.frontier[0]["validation_cross_entropy"] > 0.0


@pytest.mark.parametrize("soft_targets", [False, True])
def test_linear_router_class_balance_is_invariant_to_grouped_query_repeats(
    soft_targets: bool,
) -> None:
    base_x = np.asarray(
        [[2.0, 0.0], [0.0, 2.0], [1.5, 0.2], [0.2, 1.5]],
        dtype=np.float64,
    )
    if soft_targets:
        base_y = np.asarray(
            [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]],
            dtype=np.float64,
        )
    else:
        base_y = np.asarray([0, 1, 0, 1], dtype=np.int64)
    base_weights = np.ones(4, dtype=np.float64)

    # Split the first query's unit mass across two identical K/seed variants.
    repeated_x = np.concatenate((base_x, base_x[[0]]), axis=0)
    repeated_y = np.concatenate((base_y, base_y[[0]]), axis=0)
    repeated_weights = np.asarray([0.5, 1.0, 1.0, 1.0, 0.5])

    target_matrix = (
        base_y
        if soft_targets
        else np.eye(2, dtype=np.float64)[base_y]
    )
    repeated_target_matrix = (
        repeated_y
        if soft_targets
        else np.eye(2, dtype=np.float64)[repeated_y]
    )
    base_class_weights = _balanced_class_weights(
        target_matrix, base_weights, 2, np
    )
    repeated_class_weights = _balanced_class_weights(
        repeated_target_matrix, repeated_weights, 2, np
    )
    assert repeated_class_weights == pytest.approx(base_class_weights)

    kwargs = {
        "feature_names": ("normal_a", "bir_query_bai"),
        "experts": ("patchcore", "winclip"),
        "train_sample_weights": base_weights,
        "validation_sample_weights": np.ones(4),
        "l2_grid": (0.0,),
        "epochs": 80,
        "batch_size": 32,
        "learning_rate": 0.05,
        "seed": 23,
    }
    base_fit = fit_linear_router(
        base_x,
        base_y,
        base_x,
        base_y,
        **kwargs,
    )
    repeated_fit = fit_linear_router(
        repeated_x,
        repeated_y,
        base_x,
        base_y,
        **{**kwargs, "train_sample_weights": repeated_weights},
    )

    assert repeated_fit.model.location == pytest.approx(base_fit.model.location)
    assert repeated_fit.model.scale == pytest.approx(base_fit.model.scale)
    assert np.asarray(repeated_fit.model.weights) == pytest.approx(
        np.asarray(base_fit.model.weights)
    )
    assert repeated_fit.model.bias == pytest.approx(base_fit.model.bias)


def test_router_feature_reader_rejects_schema_drift_and_forbidden_input(
    tmp_path: Path,
) -> None:
    base = {
        "protocol_version": ROUTER_FEATURE_BUNDLE_PROTOCOL_VERSION,
        "ablation_name": "full",
        "dataset": "mvtec",
        "category": "bottle",
        "k_shot": 1,
        "seed": 0,
        "support_set_id": "support-0",
        "feature_names": ["normal_a", "bir_query_bai"],
        "values": [0.1, 0.2],
    }
    manifest = {
        "task-0": {
            "task_id": "task-0",
            "dataset": "mvtec",
            "category": "bottle",
            "k_shot": "1",
            "seed": "0",
            "support_set_id": "support-0",
        },
        "task-1": {
            "task_id": "task-1",
            "dataset": "mvtec",
            "category": "bottle",
            "k_shot": "1",
            "seed": "0",
            "support_set_id": "support-0",
        },
    }
    path = tmp_path / "router_features.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    **base,
                    "task_id": task_id,
                    "feature_names": names,
                }
            )
            for task_id, names in (
                ("task-0", ["normal_a", "bir_query_bai"]),
                ("task-1", ["normal_a", "bir_query_bai_reliability"]),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Stage5RouterError, match="schema changed"):
        read_router_features(path, manifest, feature_view="all")

    forbidden = {
        **base,
        "task_id": "task-0",
        "feature_names": ["normal_a", "target_label"],
    }
    path.write_text(json.dumps(forbidden) + "\n", encoding="utf-8")
    with pytest.raises(Stage5RouterError, match="forbidden tokens"):
        read_router_features(
            path,
            {"task-0": manifest["task-0"]},
            feature_view="all",
        )


def test_router_feature_reader_projects_all_fbdp_views(tmp_path: Path) -> None:
    task_id = "task-0"
    manifest = {
        task_id: {
            "task_id": task_id,
            "dataset": "mvtec",
            "category": "bottle",
            "k_shot": "2",
            "seed": "0",
            "support_set_id": "support-0",
        }
    }
    path = tmp_path / "router_features.jsonl"
    path.write_text(
        json.dumps(
            {
                "protocol_version": ROUTER_FEATURE_BUNDLE_PROTOCOL_VERSION,
                "task_id": task_id,
                "dataset": "mvtec",
                "category": "bottle",
                "k_shot": 2,
                "seed": 0,
                "support_set_id": "support-0",
                "ablation_name": "full",
                "fbdp_ablation_name": "full",
                "feature_view": "normal_bir_fbdp",
                "feature_names": ["normal_a", "bir_a", "fbdp_a"],
                "values": [0.1, 0.2, 0.3],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    expected = {
        "all": ("normal_a", "bir_a", "fbdp_a"),
        "normal_only": ("normal_a",),
        "normal_bir": ("normal_a", "bir_a"),
        "normal_fbdp": ("normal_a", "fbdp_a"),
        "normal_bir_fbdp": ("normal_a", "bir_a", "fbdp_a"),
    }
    for view, names in expected.items():
        artifact = read_router_features(path, manifest, feature_view=view)
        assert artifact.feature_names == names
        assert artifact.source_feature_view == "normal_bir_fbdp"
        assert artifact.source_ablation == "full"
        assert artifact.source_fbdp_ablation == "full"
        assert artifact.source_protocol_version == ROUTER_FEATURE_BUNDLE_PROTOCOL_VERSION

    legacy = json.loads(path.read_text(encoding="utf-8"))
    legacy["protocol_version"] = "stage5.router_feature_bundle.v2"
    legacy["feature_names"] = ["normal_a", "bir_a"]
    legacy["values"] = [0.1, 0.2]
    legacy.pop("feature_view")
    legacy.pop("fbdp_ablation_name")
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    artifact = read_router_features(path, manifest, feature_view="normal_bir")
    assert artifact.source_protocol_version == "stage5.router_feature_bundle.v2"

    legacy["feature_names"].append("fbdp_a")
    legacy["values"].append(0.3)
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    with pytest.raises(Stage5RouterError, match="legacy protocol"):
        read_router_features(path, manifest, feature_view="all")


def test_router_evaluator_requires_runtime_ms(tmp_path: Path) -> None:
    task_id = "image-0|mvtec|bottle|support-0|1|0"
    manifest = {
        task_id: {
            "task_id": task_id,
            "dataset": "mvtec",
            "category": "bottle",
            "k_shot": "1",
            "seed": "0",
            "support_set_id": "support-0",
        }
    }
    matrix = tmp_path / "routing_matrix_long.csv"
    _write_csv(
        matrix,
        (
            "image_id",
            "dataset",
            "category",
            "support_set_id",
            "k_shot",
            "seed",
            "expert_name",
            "final_score",
            "label",
            "status",
        ),
        [
            {
                "image_id": "image-0",
                "dataset": "mvtec",
                "category": "bottle",
                "support_set_id": "support-0",
                "k_shot": 1,
                "seed": 0,
                "expert_name": expert,
                "final_score": score,
                "label": 0,
                "status": "ok",
            }
            for expert, score in (("patchcore", 0.1), ("winclip", 0.2))
        ],
    )
    with pytest.raises(Stage5RouterError, match="runtime_ms"):
        read_evaluator_outcomes(matrix, manifest)


def test_capability_cost_uses_positive_train_only_latency() -> None:
    experts = ("patchcore", "winclip")
    profile = {
        "overall_skill": 0.6,
        "boundary_skill": 0.6,
        "fgbg_skill": 0.6,
        "lowshot_skill": 0.6,
        "texture_skill": 0.6,
        "failure_rate": 0.0,
        "confidence_intervals": {},
    }
    bank = {
        "profiles": {
            "patchcore": {**profile, "latency": 10.0, "latency_p95": 10.0},
            "winclip": {**profile, "latency": 100.0, "latency_p95": 100.0},
        },
        "difficulty_thresholds": {
            "boundary": 1.0,
            "fgbg": 1.0,
            "texture": 1.0,
        },
        "lowshot_k": 1,
    }
    objectives = capability_objectives(
        np.asarray([[0.5, 0.5]]),
        np.asarray([[0.0]]),
        ("task-0",),
        {"task-0": {"k_shot": "1"}},
        experts,
        ("normal_signal",),
        bank,
        capability_weight=0.0,
        uncertainty_weight=0.0,
        cost_weight=1.0,
    )
    assert objectives[0, 0] < objectives[0, 1]

    invalid = json.loads(json.dumps(bank))
    invalid["profiles"]["patchcore"]["latency"] = None
    invalid["profiles"]["patchcore"]["latency_p95"] = None
    with pytest.raises(Stage5RouterError, match="latency"):
        capability_objectives(
            np.asarray([[0.5, 0.5]]),
            np.asarray([[0.0]]),
            ("task-0",),
            {"task-0": {"k_shot": "1"}},
            experts,
            ("normal_signal",),
            invalid,
            capability_weight=0.0,
            uncertainty_weight=0.0,
            cost_weight=1.0,
        )


def test_entropy_scaled_lcb_uncertainty_is_query_and_expert_specific() -> None:
    experts = ("stable", "uncertain")

    def profile(lower: float, upper: float) -> dict[str, object]:
        return {
            "overall_skill": 0.6,
            "boundary_skill": 0.6,
            "fgbg_skill": 0.6,
            "lowshot_skill": 0.6,
            "texture_skill": 0.6,
            "latency": 10.0,
            "latency_p95": 10.0,
            "failure_rate": 0.0,
            "confidence_intervals": {
                "overall_skill": {"lower": lower, "upper": upper},
                "failure_rate": {"lower": 0.0, "upper": 0.0},
            },
        }

    bank = {
        "profiles": {
            "stable": profile(0.55, 0.65),
            "uncertain": profile(0.10, 0.90),
        },
        "difficulty_thresholds": {
            "boundary": 1.0,
            "fgbg": 1.0,
            "texture": 1.0,
        },
        "lowshot_k": 1,
    }
    task_ids = ("ambiguous", "confident")
    probabilities = np.asarray([[0.5, 0.5], [0.999, 0.001]])
    common = {
        "feature_values": np.zeros((2, 1)),
        "task_ids": task_ids,
        "manifest_by_task": {
            task_id: {"k_shot": "4"} for task_id in task_ids
        },
        "experts": experts,
        "feature_names": ("normal_signal",),
        "bank": bank,
        "capability_weight": 0.0,
        "cost_weight": 0.0,
        "capability_mode": "static",
    }
    without_uncertainty = capability_objectives(
        probabilities,
        uncertainty_weight=0.0,
        **common,
    )
    with_uncertainty = capability_objectives(
        probabilities,
        uncertainty_weight=1.0,
        **common,
    )
    penalty = with_uncertainty - without_uncertainty

    assert penalty[0, 1] > penalty[0, 0] > 0.0
    assert penalty[1, 1] < penalty[0, 1]
    assert np.allclose(
        without_uncertainty,
        capability_objectives(
            probabilities,
            uncertainty_weight=0.0,
            uncertainty_mode="legacy_interval_width",
            **common,
        ),
    )


def test_lcb_uncertainty_is_relative_not_only_absolute_interval_width() -> None:
    experts = ("high_skill", "low_skill")

    def profile(point: float, lower: float, upper: float) -> dict[str, object]:
        return {
            "overall_skill": point,
            "boundary_skill": point,
            "fgbg_skill": point,
            "lowshot_skill": point,
            "texture_skill": point,
            "latency": 10.0,
            "latency_p95": 10.0,
            "failure_rate": 0.0,
            "confidence_intervals": {
                "overall_skill": {"lower": lower, "upper": upper}
            },
        }

    bank = {
        "profiles": {
            "high_skill": profile(0.8, 0.7, 0.9),
            "low_skill": profile(0.3, 0.2, 0.4),
        },
        "difficulty_thresholds": {
            "boundary": 1.0,
            "fgbg": 1.0,
            "texture": 1.0,
        },
        "lowshot_k": 1,
    }
    kwargs = {
        "feature_values": np.zeros((1, 1)),
        "task_ids": ("task-0",),
        "manifest_by_task": {"task-0": {"k_shot": "4"}},
        "experts": experts,
        "feature_names": ("normal_signal",),
        "bank": bank,
        "capability_weight": 0.0,
        "cost_weight": 0.0,
        "capability_mode": "static",
    }
    baseline = capability_objectives(
        np.asarray([[0.5, 0.5]]), uncertainty_weight=0.0, **kwargs
    )
    guarded = capability_objectives(
        np.asarray([[0.5, 0.5]]), uncertainty_weight=1.0, **kwargs
    )
    penalty = guarded - baseline

    # Both intervals have width 0.2, but a 0.1 drop is more material at skill
    # 0.3 than at skill 0.8.
    assert penalty[0, 1] > penalty[0, 0] > 0.0


def test_stage5_router_cli_writes_label_free_predictions_and_evaluator_metrics(
    tmp_path: Path,
) -> None:
    folds_path = tmp_path / "fold_manifest.csv"
    features_path = tmp_path / "router_features.jsonl"
    matrix_path = tmp_path / "routing_matrix_long.csv"
    output_dir = tmp_path / "run"
    experts = ("anomalydino", "patchcore", "winclip")
    splits = ("train",) * 6 + ("val",) * 3 + ("test",) * 3
    tasks = []
    for index, split in enumerate(splits):
        category = {
            "train": "bottle",
            "val": "cable",
            "test": "capsule",
        }[split]
        image_id = f"image-{index}"
        support_id = f"support-{index}"
        seed = str(index)
        task_id = f"{image_id}|mvtec|{category}|{support_id}|1|{seed}"
        tasks.append(
            {
                "fold": "fold0",
                "split": split,
                "task_id": task_id,
                "sample_id": image_id,
                "dataset": "mvtec",
                "category": category,
                "k_shot": "1",
                "seed": seed,
                "support_set_id": support_id,
                "target": index % 3,
                "label": index % 2,
            }
        )
    _write_csv(
        folds_path,
        (
            "fold",
            "split",
            "task_id",
            "sample_id",
            "dataset",
            "category",
            "k_shot",
            "seed",
            "support_set_id",
        ),
        tasks,
    )
    with features_path.open("w", encoding="utf-8") as handle:
        for row in tasks:
            values = [0.0, 0.0, 0.0]
            values[int(row["target"])] = 3.0
            handle.write(
                json.dumps(
                    {
                        "protocol_version": (
                            ROUTER_FEATURE_BUNDLE_PROTOCOL_VERSION
                        ),
                        "task_id": row["task_id"],
                        "ablation_name": "full",
                        "dataset": row["dataset"],
                        "category": row["category"],
                        "k_shot": int(row["k_shot"]),
                        "seed": int(row["seed"]),
                        "support_set_id": row["support_set_id"],
                        "feature_names": [
                            "normal_signal_0",
                            "normal_signal_1",
                            "bir_query_bai",
                        ],
                        "values": values,
                    }
                )
                + "\n"
            )
    evaluator_rows = []
    for row in tasks:
        target = int(row["target"])
        label = int(row["label"])
        for expert_index, expert in enumerate(experts):
            if label == 1:
                score = 0.9 if expert_index == target else 0.1 + expert_index * 0.1
            else:
                score = 0.1 if expert_index == target else 0.8 + expert_index * 0.05
            evaluator_rows.append(
                {
                    "image_id": row["sample_id"],
                    "dataset": row["dataset"],
                    "category": row["category"],
                    "support_set_id": row["support_set_id"],
                    "k_shot": row["k_shot"],
                    "seed": row["seed"],
                    "expert_name": expert,
                    "final_score": score,
                    "runtime_ms": 10 + expert_index,
                    "label": label,
                    "status": "ok",
                }
            )
    _write_csv(
        matrix_path,
        (
            "image_id",
            "dataset",
            "category",
            "support_set_id",
            "k_shot",
            "seed",
            "expert_name",
            "final_score",
            "runtime_ms",
            "label",
            "status",
        ),
        evaluator_rows,
    )

    return_code = main(
        [
            "--router-features",
            str(features_path),
            "--fold-manifest",
            str(folds_path),
            "--routing-matrix",
            str(matrix_path),
            "--fold",
            "fold0",
            "--variant",
            "full",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--epochs",
            "120",
            "--batch-size",
            "6",
            "--learning-rate",
            "0.1",
            "--l2-grid",
            "0",
            "--seed",
            "11",
        ]
    )

    assert return_code == 0
    predictions = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(predictions) == 3
    assert all("label" not in row and "oracle_expert" not in row for row in predictions)
    assert all(set(row["expert_probabilities"]) == set(experts) for row in predictions)
    metrics = json.loads(
        (output_dir / "evaluator_only" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert metrics["methods"]["learned_router"]["expert_calls_per_task"] == 1
    assert metrics["methods"]["learned_router"]["failure_rate"] == 0.0
    assert metrics["methods"]["learned_router"]["average_runtime_ms"] is not None
    assert metrics["runtime_source"] == "evaluator_only_routing_matrix.runtime_ms"
    assert metrics["uncertainty_mode"] == "entropy_scaled_lcb"
    run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert run["ok"] is True
    assert run["config"]["uncertainty_mode"] == "entropy_scaled_lcb"
    assert run["predictions"] == str(output_dir / "predictions.jsonl")
    assert json.loads(
        (output_dir / "failures.json").read_text(encoding="utf-8")
    ) == []


def _write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
