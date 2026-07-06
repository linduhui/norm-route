"""Run Stage 2 experts across a fixed no-leakage parameter grid."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import traceback
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normroute.cli.run_expert import SUPPORT_COLUMNS, _read_csv, run_expert


RESULT_FILES = ("predictions.csv", "metrics.json", "failures.json")
UNRESOLVED_SUPPORT_SET_ID = "support_set_id_unresolved"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Stage 2 expert grid.")
    parser.add_argument(
        "--experts",
        nargs="+",
        required=True,
        choices=["dummy", "patchcore", "winclip", "anomalydino"],
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--k-shots", nargs="+", required=True, type=int)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--agent-input-csv")
    parser.add_argument("--support-set-dir", default="data/support_sets")
    parser.add_argument(
        "--support-set-csv-template",
        default="{dataset}_k{k_shot}_seed{seed}.csv",
        help="Template resolved under --support-set-dir unless it is absolute.",
    )
    parser.add_argument("--output-root", default="outputs/stage2")
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_grid(
        experts=args.experts,
        dataset=args.dataset,
        categories=args.categories,
        k_shots=args.k_shots,
        seeds=args.seeds,
        agent_input_csv=Path(args.agent_input_csv)
        if args.agent_input_csv
        else Path("data") / "manifests" / f"{args.dataset}_agent_input.csv",
        support_set_dir=Path(args.support_set_dir),
        support_set_csv_template=args.support_set_csv_template,
        output_root=Path(args.output_root),
        budget=args.budget,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"Wrote {result['grid_log_path']}")
    if result["num_failed"]:
        raise SystemExit(1)


def run_grid(
    *,
    experts: list[str],
    dataset: str,
    categories: list[str],
    k_shots: list[int],
    seeds: list[int],
    agent_input_csv: Path,
    support_set_dir: Path,
    support_set_csv_template: str,
    output_root: Path,
    budget: int = 1,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run every expert/category/k/seed combination once."""

    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for expert, category, k_shot, seed in itertools.product(experts, categories, k_shots, seeds):
        support_set_csv = _support_set_csv_path(
            support_set_dir=support_set_dir,
            template=support_set_csv_template,
            dataset=dataset,
            category=category,
            k_shot=k_shot,
            seed=seed,
        )
        support_set_id = _resolve_support_set_id(
            support_set_csv=support_set_csv,
            dataset=dataset,
            category=category,
            k_shot=k_shot,
            seed=seed,
        )
        combo_dir = _combo_output_dir(
            output_root=output_root,
            expert=expert,
            dataset=dataset,
            category=category,
            k_shot=k_shot,
            seed=seed,
            support_set_id=support_set_id or UNRESOLVED_SUPPORT_SET_ID,
        )
        base_record = {
            "expert": expert,
            "dataset": dataset,
            "category": category,
            "k_shot": k_shot,
            "seed": seed,
            "support_set_id": support_set_id or "",
            "support_set_csv": str(support_set_csv),
            "output_dir": str(combo_dir),
        }

        if _has_existing_results(combo_dir) and not overwrite:
            records.append({**base_record, "status": "skipped_existing", "error_message": ""})
            continue

        try:
            if support_set_id is None:
                raise ValueError(
                    "Unable to resolve one support_set_id for "
                    f"dataset={dataset!r}, category={category!r}, k_shot={k_shot}, seed={seed}"
                )
            run_expert(
                expert_name=expert,
                agent_input_csv=agent_input_csv,
                support_set_csv=support_set_csv,
                output_dir=combo_dir,
                dataset=dataset,
                category=category,
                k_shot=k_shot,
                seed=seed,
                support_set_id=support_set_id,
                budget=budget,
                limit=limit,
            )
            num_failed = _num_failed(combo_dir / "metrics.json")
            status = "failed_predictions" if num_failed else "ok"
            records.append({**base_record, "status": status, "error_message": ""})
        except Exception as exc:
            combo_dir.mkdir(parents=True, exist_ok=True)
            _write_combo_failure(combo_dir / "failures.json", base_record, exc)
            records.append(
                {
                    **base_record,
                    "status": "error",
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    failed_statuses = {"error", "failed_predictions"}
    log = {
        "records": records,
        "num_total": len(records),
        "num_ok": sum(1 for record in records if record["status"] == "ok"),
        "num_failed": sum(1 for record in records if record["status"] in failed_statuses),
        "num_skipped_existing": sum(
            1 for record in records if record["status"] == "skipped_existing"
        ),
    }
    grid_log_path = output_root / "grid_log.json"
    grid_log_path.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**log, "grid_log_path": grid_log_path}


def _support_set_csv_path(
    *,
    support_set_dir: Path,
    template: str,
    dataset: str,
    category: str,
    k_shot: int,
    seed: int,
) -> Path:
    path = Path(
        template.format(dataset=dataset, category=category, k_shot=k_shot, seed=seed)
    )
    return path if path.is_absolute() else support_set_dir / path


def _resolve_support_set_id(
    *,
    support_set_csv: Path,
    dataset: str,
    category: str,
    k_shot: int,
    seed: int,
) -> str | None:
    try:
        rows = _read_csv(support_set_csv, SUPPORT_COLUMNS, "support set CSV")
    except Exception:
        return None
    support_set_ids = {
        row["support_set_id"]
        for row in rows
        if row["dataset"] == dataset
        and row["category"] == category
        and int(row["k_shot"]) == k_shot
        and int(row["seed"]) == seed
    }
    if len(support_set_ids) != 1:
        return None
    return next(iter(support_set_ids))


def _combo_output_dir(
    *,
    output_root: Path,
    expert: str,
    dataset: str,
    category: str,
    k_shot: int,
    seed: int,
    support_set_id: str,
) -> Path:
    return output_root / expert / dataset / category / f"k{k_shot}" / f"seed{seed}" / support_set_id


def _has_existing_results(output_dir: Path) -> bool:
    return any((output_dir / file_name).exists() for file_name in RESULT_FILES)


def _num_failed(metrics_path: Path) -> int:
    if not metrics_path.exists():
        return 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return int(metrics.get("num_failed", 0))


def _write_combo_failure(path: Path, combo: dict[str, Any], exc: Exception) -> None:
    payload = {
        "failed_predictions": [],
        "grid_failure": {
            "combo": combo,
            "status": "error",
            "error_message": str(exc),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
