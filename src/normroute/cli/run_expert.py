"""Run a Stage 2 expert through the shared no-leakage interface."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normroute.evaluation.export import export_run_outputs
from src.normroute.experts.base import (
    DummyExpert,
    Expert,
    ExpertInput,
    ExpertPrediction,
    validate_expert_input_fields,
)
from src.normroute.experts.patchcore import PatchCoreExpert
from src.normroute.experts.winclip import WinCLIPConfig, WinCLIPExpert


AGENT_COLUMNS = ["image_id", "dataset", "category", "split", "image_path"]
SUPPORT_COLUMNS = [
    "support_set_id",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_rank",
    "image_id",
    "image_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Stage 2 visual expert.")
    parser.add_argument("--expert", default="dummy", choices=["dummy", "patchcore", "winclip"])
    parser.add_argument("--config")
    parser.add_argument("--agent-input-csv", required=True)
    parser.add_argument("--support-set-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--category")
    parser.add_argument("--k-shot", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--support-set-id")
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path, metrics_path, failures_path = run_expert(
        expert_name=args.expert,
        config_path=args.config,
        agent_input_csv=args.agent_input_csv,
        support_set_csv=args.support_set_csv,
        output_dir=args.output_dir,
        dataset=args.dataset,
        category=args.category,
        k_shot=args.k_shot,
        seed=args.seed,
        support_set_id=args.support_set_id,
        budget=args.budget,
        limit=args.limit,
    )
    print(f"Wrote {predictions_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {failures_path}")


def run_expert(
    *,
    expert_name: str,
    agent_input_csv: str | Path,
    support_set_csv: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    dataset: str | None = None,
    category: str | None = None,
    k_shot: int | None = None,
    seed: int | None = None,
    support_set_id: str | None = None,
    budget: int = 1,
    limit: int | None = None,
) -> tuple[Path, Path, Path]:
    """Run one Stage 2 expert combination and write standard output artifacts."""

    expert = _build_expert(expert_name, output_dir=str(output_dir), config_path=config_path)
    agent_rows = _read_csv(agent_input_csv, AGENT_COLUMNS, "agent input CSV")
    support_rows = _read_csv(support_set_csv, SUPPORT_COLUMNS, "support set CSV")
    inputs = _build_inputs(
        agent_rows=agent_rows,
        support_rows=support_rows,
        dataset=dataset,
        category=category,
        k_shot=k_shot,
        seed=seed,
        support_set_id=support_set_id,
        budget=budget,
        limit=limit,
    )

    predictions: list[ExpertPrediction] = []
    try:
        expert.fit(inputs)
        for expert_input in inputs:
            try:
                predictions.append(expert.predict(expert_input))
            except Exception as exc:  # pragma: no cover - real experts will use this path.
                predictions.append(_failed_prediction(expert.name, expert_input, exc))
    except Exception as exc:
        for expert_input in inputs:
            predictions.append(_failed_prediction(expert.name, expert_input, exc))

    return export_run_outputs(
        output_dir=output_dir,
        predictions=predictions,
    )


def _build_expert(name: str, *, output_dir: str, config_path: str | Path | None = None) -> Expert:
    if name == "dummy":
        return DummyExpert()
    if name == "patchcore":
        return PatchCoreExpert(output_dir=output_dir)
    if name == "winclip":
        default_config = PROJECT_ROOT / "configs" / "stage2" / "winclip_mvtec.yaml"
        resolved_config = Path(config_path) if config_path else default_config
        return WinCLIPExpert(
            output_dir=output_dir,
            config=WinCLIPConfig.from_yaml(resolved_config),
        )
    raise ValueError(f"Unsupported expert: {name}")


def _read_csv(path: str | Path, required_columns: list[str], name: str) -> list[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        validate_expert_input_fields({column: "" for column in fieldnames}, context=name)
        missing = [column for column in required_columns if column not in fieldnames]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def _build_inputs(
    *,
    agent_rows: list[dict[str, str]],
    support_rows: list[dict[str, str]],
    dataset: str | None,
    category: str | None,
    k_shot: int | None,
    seed: int | None,
    support_set_id: str | None,
    budget: int,
    limit: int | None,
) -> list[ExpertInput]:
    support_by_category: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in support_rows:
        validate_expert_input_fields(row, context="support set row")
        if k_shot is not None and int(row["k_shot"]) != k_shot:
            continue
        if seed is not None and int(row["seed"]) != seed:
            continue
        if support_set_id and row["support_set_id"] != support_set_id:
            continue
        key = (row["dataset"], row["category"])
        support_by_category.setdefault(key, []).append(row)

    inputs: list[ExpertInput] = []
    for row in agent_rows:
        validate_expert_input_fields(row, context="agent input row")
        if row["split"] != "test":
            continue
        if dataset and row["dataset"] != dataset:
            continue
        if category and row["category"] != category:
            continue

        support = sorted(
            support_by_category.get((row["dataset"], row["category"]), []),
            key=lambda item: int(item["support_rank"]),
        )
        if not support:
            raise ValueError(
                f"No support rows for dataset={row['dataset']!r}, category={row['category']!r}"
            )
        support_set_ids = {item["support_set_id"] for item in support}
        k_shots = {item["k_shot"] for item in support}
        seeds = {item["seed"] for item in support}
        if len(support_set_ids) != 1 or len(k_shots) != 1 or len(seeds) != 1:
            raise ValueError(
                f"Ambiguous support rows for dataset={row['dataset']!r}, category={row['category']!r}"
            )

        inputs.append(
            ExpertInput.from_dict(
                {
                    "image_id": row["image_id"],
                    "query_path": row["image_path"],
                    "dataset": row["dataset"],
                    "category": row["category"],
                    "support_set_id": next(iter(support_set_ids)),
                    "k_shot": next(iter(k_shots)),
                    "seed": next(iter(seeds)),
                    "budget": budget,
                    "support_paths": [item["image_path"] for item in support],
                }
            )
        )
        if limit is not None and len(inputs) >= limit:
            break

    return inputs


def _failed_prediction(expert_name: str, expert_input: ExpertInput, exc: Exception) -> ExpertPrediction:
    return ExpertPrediction(
        image_id=expert_input.image_id,
        expert_name=expert_name,
        dataset=expert_input.dataset,
        category=expert_input.category,
        support_set_id=expert_input.support_set_id,
        k_shot=expert_input.k_shot,
        seed=expert_input.seed,
        final_score=0.0,
        final_decision="abstain",
        anomaly_map_path="",
        actions="",
        tool_calls=0,
        runtime_ms=0.0,
        status="error",
        error_message=str(exc),
    )


if __name__ == "__main__":
    main()
