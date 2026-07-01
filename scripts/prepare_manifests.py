"""Prepare dataset manifests without loading models or modifying raw data."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.manifest import write_manifest_outputs
from src.data.mvtec import build_mvtec_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mvtec"], default="mvtec")
    parser.add_argument("--mvtec-root", required=True, help="Path to the raw MVTec AD root.")
    parser.add_argument("--categories", nargs="*", help="Optional MVTec categories to scan.")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--audit-path", default="outputs/stage1_gate/mvtec_audit.json")
    parser.add_argument(
        "--path-style",
        choices=["absolute", "relative"],
        default="absolute",
        help="How image and mask paths are written in manifests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset != "mvtec":
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    rows = build_mvtec_manifest(
        args.mvtec_root,
        path_style=args.path_style,
        categories=args.categories,
    )
    rows.audit["run"] = _run_metadata(args, rows.audit)
    agent_path, evaluator_path, audit_path = write_manifest_outputs(
        agent_rows=rows.agent_input,
        evaluator_rows=rows.evaluator,
        audit=rows.audit,
        manifest_dir=Path(args.manifest_dir),
        audit_path=Path(args.audit_path),
    )
    print(f"Wrote {agent_path}")
    print(f"Wrote {evaluator_path}")
    print(f"Wrote {audit_path}")
    _print_audit_summary(rows.audit)


def _run_metadata(args: argparse.Namespace, audit: dict) -> dict:
    return {
        "config": {
            "dataset": args.dataset,
            "mvtec_root": args.mvtec_root,
            "categories": args.categories,
            "manifest_dir": args.manifest_dir,
            "audit_path": args.audit_path,
            "path_style": args.path_style,
        },
        "seed": None,
        "git_commit": _git_commit(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "predictions": "not_applicable_manifest_only",
        "failures": {
            "missing_masks": audit.get("missing_masks", []),
            "duplicate_image_ids": audit.get("duplicate_image_ids", []),
            "broken_image_files": audit.get("broken_image_files", []),
            "mask_size_mismatches": audit.get("mask_size_mismatches", []),
            "broken_mask_files": audit.get("broken_mask_files", []),
        },
    }


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip()


def _print_audit_summary(audit: dict) -> None:
    counts = audit["counts"]
    print("Audit summary:")
    print(f"  categories: {counts['categories']}")
    print(f"  train_good_images: {counts['train_good_images']}")
    print(f"  test_good_images: {counts['test_good_images']}")
    print(f"  test_anomaly_images: {counts['test_anomaly_images']}")
    print(f"  missing_masks: {counts['missing_masks']}")
    print(f"  duplicate_image_ids: {counts['duplicate_image_ids']}")
    print(f"  broken_image_files: {counts['broken_image_files']}")
    print(f"  mask_size_mismatches: {counts['mask_size_mismatches']}")
    print(f"  broken_mask_files: {counts['broken_mask_files']}")


if __name__ == "__main__":
    main()
