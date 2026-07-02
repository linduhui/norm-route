"""Build deterministic few-normal-shot support set CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.support_sampler import (
    DEFAULT_K_SHOTS,
    DEFAULT_SEEDS,
    build_support_set_rows,
    write_support_set_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["mvtec", "visa", "all"],
        default="all",
        help="Dataset to build. Defaults to both MVTec and VisA.",
    )
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--output-dir", default="data/support_sets")
    parser.add_argument("--k-shots", nargs="*", type=int, default=list(DEFAULT_K_SHOTS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = ["mvtec", "visa"] if args.dataset == "all" else [args.dataset]
    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)

    for dataset in datasets:
        agent_input_path = manifest_dir / f"{dataset}_agent_input.csv"
        for k_shot in args.k_shots:
            for seed in args.seeds:
                rows = build_support_set_rows(
                    agent_input_path,
                    dataset=dataset,
                    k_shot=k_shot,
                    seed=seed,
                )
                output_path = output_dir / f"{dataset}_k{k_shot}_seed{seed}.csv"
                write_support_set_csv(output_path, rows)
                print(f"Wrote {output_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
