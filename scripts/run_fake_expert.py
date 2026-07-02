"""Run the deterministic FakeExpert smoke path."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experts.fake_expert import FakeExpert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FakeExpert on one image.")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--support-set-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--support-path", action="append", default=[])
    parser.add_argument("--jsonl-name", default="predictions.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expert = FakeExpert()
    expert.fit_support(
        support_paths=args.support_path,
        support_set_id=args.support_set_id,
        seed=args.seed,
    )
    result = expert.predict(
        image_path=args.image_path,
        image_id=args.image_id,
        output_dir=args.output_dir,
    )
    result.append_jsonl(f"{args.output_dir}/{args.jsonl_name}")
    print(result.to_jsonl_line(), end="")


if __name__ == "__main__":
    main()
