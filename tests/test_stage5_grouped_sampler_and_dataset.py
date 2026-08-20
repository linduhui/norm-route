from __future__ import annotations

from collections import defaultdict

import pytest

from src.normroute.router.dataset import RouterInferenceDataset, Stage5DatasetError
from src.normroute.router.sampler import GroupedSampler, inverse_frequency_weights


def _features() -> list[dict]:
    rows = []
    for query in ("q0", "q1"):
        for k_shot, seed in ((1, 0), (2, 1), (4, 2)):
            rows.append(
                {
                    "task_id": f"{query}|mvtec|bottle|support-{k_shot}-{seed}|{k_shot}|{seed}",
                    "dataset": "mvtec",
                    "category": "bottle",
                    "sample_id": query,
                    "k_shot": k_shot,
                    "seed": seed,
                    "feature_names": ["normal_niv", "bir_query_bai"],
                    "values": [float(k_shot), float(seed)],
                }
            )
    return rows


def test_grouped_sampler_selects_one_k_seed_variant_per_query() -> None:
    dataset = RouterInferenceDataset(_features())
    sampler = GroupedSampler(dataset, seed=7, shuffle=True)

    first = list(sampler)
    assert len(first) == 2
    selected_groups = [dataset.group_ids[index] for index in first]
    assert len(set(selected_groups)) == 2
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert len(list(sampler)) == 2


def test_inverse_frequency_weights_give_each_query_equal_total_mass() -> None:
    groups = ("q0", "q0", "q0", "q1", "q1", "q2")
    weights = inverse_frequency_weights(groups)
    totals = defaultdict(float)
    for group, weight in zip(groups, weights):
        totals[group] += weight
    assert totals == pytest.approx({"q0": 1.0, "q1": 1.0, "q2": 1.0})


def test_inference_dataset_has_no_label_expert_score_or_teacher() -> None:
    dataset = RouterInferenceDataset(_features())
    assert set(dataset[0]) == {"task_id", "features"}

    for forbidden_field in ("teacher_utility", "patchcore_score", "ground_truth"):
        leaked = _features()
        leaked[0][forbidden_field] = 1.0
        with pytest.raises(Stage5DatasetError, match="forbidden inference fields"):
            RouterInferenceDataset(leaked)


@pytest.mark.parametrize(
    "forbidden_feature_name",
    [
        "final_score",
        "image_score",
        "anomaly_score",
        "finalScore",
        "calibrated-utility",
        "expertOutcome",
        "raw_scores_v2",
        "finalscore",
        "imagescore",
        "anomalyscore",
        "expertutility",
        "expertoutcome",
        "finalScore2",
        "scorevalue",
        "teacherprobability",
        "mylabelhint",
        "query_defect_type",
        "expertutilityvalue",
        "query_anomaly_type",
        "predicted_anomaly_type_hint",
        "groundTruthLabel",
        "anomalyscorefeature",
        "utilityloss",
        "outcomeflag",
        "scorecard_width",
        "utilityroom_distance",
    ],
)
def test_inference_dataset_rejects_any_score_utility_or_outcome_feature_name(
    forbidden_feature_name: str,
) -> None:
    leaked = _features()
    leaked[0]["feature_names"][0] = forbidden_feature_name

    with pytest.raises(Stage5DatasetError, match="forbidden inference fields"):
        RouterInferenceDataset(leaked)


@pytest.mark.parametrize("raw_field", ["final_score", "image_score", "anomaly_score"])
def test_inference_dataset_rejects_nested_raw_score_fields(raw_field: str) -> None:
    leaked = _features()
    leaked[0]["raw_evaluator_payload"] = {raw_field: 0.5}

    with pytest.raises(Stage5DatasetError, match=raw_field):
        RouterInferenceDataset(leaked)


def test_inference_dataset_score_policy_does_not_match_unrelated_stems() -> None:
    records = _features()
    for row in records:
        row["feature_names"] = ["scoring_width", "utilitarian_distance"]

    dataset = RouterInferenceDataset(records)

    assert dataset.feature_names == ("scoring_width", "utilitarian_distance")
