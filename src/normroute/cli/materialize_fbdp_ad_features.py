"""Materialize Router views from immutable FBDP-AD signatures."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

from .build_bir_ad import _read_normal_signatures
from ..router.bir_ad_ablation import BIR_AD_ABLATIONS
from ..router.fbdp_ad_ablation import (
    FBDP_AD_ABLATIONS,
    get_fbdp_ad_ablation,
    validate_fbdp_ad_ablation_record,
)
from ..router.feature_bundle import (
    build_router_feature_bundles,
    write_router_feature_bundles_jsonl,
)


FBDP_AD_MATERIALIZE_PROTOCOL_VERSION = "stage5.fbdp_ad_materialize.v1"
ROUTER_FEATURES_NAME = "router_features.jsonl"
RUN_RECORD_NAME = "fbdp_ad_materialize_run.json"
FAILURES_NAME = "fbdp_ad_materialize_failures.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-signatures", required=True)
    parser.add_argument("--fbdp-signatures", required=True)
    parser.add_argument(
        "--fbdp-ablation", choices=tuple(FBDP_AD_ABLATIONS), required=True
    )
    parser.add_argument("--bir-signatures")
    parser.add_argument(
        "--bir-ablation", choices=tuple(BIR_AD_ABLATIONS), default="full"
    )
    parser.add_argument(
        "--feature-view",
        choices=("normal_fbdp", "normal_bir_fbdp"),
        default="normal_fbdp",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    output_path: Path | None = None
    inputs = {
        "normal_signatures": args.normal_signatures,
        "fbdp_signatures": args.fbdp_signatures,
        **({"bir_signatures": args.bir_signatures} if args.bir_signatures else {}),
    }
    try:
        if args.feature_view == "normal_bir_fbdp" and not args.bir_signatures:
            raise ValueError(
                "--bir-signatures is required for feature_view=normal_bir_fbdp"
            )
        normal_records = _read_normal_signatures(args.normal_signatures)
        fbdp_records = _read_jsonl(args.fbdp_signatures)
        fbdp_spec = get_fbdp_ad_ablation(args.fbdp_ablation)
        for record in fbdp_records:
            validate_fbdp_ad_ablation_record(record, fbdp_spec)
        bir_records = (
            _read_jsonl(args.bir_signatures) if args.bir_signatures else None
        )
        bundles = build_router_feature_bundles(
            normal_records,
            bir_records,
            fbdp_ad_signatures=fbdp_records,
            ablation=args.bir_ablation,
            fbdp_ablation=fbdp_spec,
            feature_view=args.feature_view,
        )
        output_path = write_router_feature_bundles_jsonl(
            bundles, output_dir / ROUTER_FEATURES_NAME
        )
    except Exception as exc:
        failures.append({"code": type(exc).__name__, "message": str(exc)})

    failures_path = _atomic_write_json(output_dir / FAILURES_NAME, failures)
    run_record = {
        "protocol_version": FBDP_AD_MATERIALIZE_PROTOCOL_VERSION,
        "run_kind": "router_feature_materialization",
        "ok": not failures,
        "config": dict(vars(args)),
        "seed": args.seed,
        "git_commit": _git_commit(),
        "environment": _environment_record(),
        "input_hashes": {
            name: _file_sha256(path)
            for name, path in inputs.items()
            if path and Path(path).is_file()
        },
        "predictions": str(output_path) if output_path else None,
        "outputs": {
            "router_features": str(output_path) if output_path else None,
            "failures": str(failures_path),
        },
        "failures": failures,
    }
    _atomic_write_json(output_dir / RUN_RECORD_NAME, run_record)
    if failures:
        print(
            f"FBDP feature materialization failed: {failures[0]['message']}",
            file=sys.stderr,
        )
        return 1
    print(f"FBDP Router features PASS: {output_path}")
    return 0


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{source}:{line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{source} is empty")
    return rows


def _environment_record() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "torch", "torchvision", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> Path:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


__all__ = [
    "FAILURES_NAME",
    "FBDP_AD_MATERIALIZE_PROTOCOL_VERSION",
    "ROUTER_FEATURES_NAME",
    "RUN_RECORD_NAME",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
