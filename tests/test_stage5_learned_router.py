from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.normroute.cli.run_stage5_router import (
    main,
    read_router_features,
)
from src.normroute.router.feature_bundle import (
    ROUTER_FEATURE_BUNDLE_PROTOCOL_VERSION,
)
from src.normroute.router.learned_router import (
    LinearRouterModel,
    Stage5RouterError,
    fit_linear_router,
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
    run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert run["ok"] is True
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
