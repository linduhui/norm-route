"""Build Stage 3 evaluator-only routing matrices."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ..routing.join_predictions import (
    RoutingMatrixError,
    build_routing_matrices,
    write_routing_matrices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 3 evaluator-only routing matrices.")
    parser.add_argument("--stage2-root", default="outputs/stage2")
    parser.add_argument("--evaluator-csv", default="data/manifests/mvtec_evaluator.csv")
    parser.add_argument("--output-dir", default="outputs/stage3/evaluator_only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        matrices = build_routing_matrices(
            stage2_root=args.stage2_root,
            evaluator_csv=args.evaluator_csv,
        )
        long_path, wide_path = write_routing_matrices(matrices, args.output_dir)
    except RoutingMatrixError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Wrote {long_path}")
    print(f"Wrote {wide_path}")


if __name__ == "__main__":
    main()
