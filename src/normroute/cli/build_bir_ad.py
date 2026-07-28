"""Build fold-normalized BIR-AD and optional Router feature artifacts."""

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

from .diagnose_bir_ad import write_bir_ad_diagnostics
from ..router.bir_ad_ablation import BIR_AD_ABLATIONS, get_bir_ad_ablation
from ..router.bir_ad_pipeline import (
    BIR_AD_FAILURES_NAME,
    BIR_AD_SIGNATURES_NAME,
    BIRADTaskEncoder,
    fit_fold_bir_ad_normalization,
    load_fold_normalization_artifact,
    read_fold_manifest_csv,
    read_supports_csv,
    read_tasks_jsonl,
    write_bir_ad_signatures_jsonl,
)
from ..router.feature_bundle import (
    build_router_feature_bundles,
    write_router_feature_bundles_jsonl,
)
from ..router.feature_cache import FeatureCache
from ..router.feature_provider import (
    FrozenVisualBackboneProvider,
    load_router_backbone_config,
)


BIR_AD_BUILD_PROTOCOL_VERSION = "stage5.bir_ad_build.v2"
BIR_AD_BUILD_RUN_NAME = "bir_ad_build_run.json"
BIR_AD_NORMALIZATION_NAME = "bir_ad_fold_normalization.json"
BIR_AD_ROUTER_FEATURES_NAME = "router_features.jsonl"
BIR_AD_ABLATION_RECORD_NAME = "bir_ad_ablation.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="Stage 5 task JSONL.")
    parser.add_argument(
        "--supports", required=True, help="Official train/good support CSV."
    )
    normalization = parser.add_mutually_exclusive_group(required=True)
    normalization.add_argument(
        "--fold-manifest",
        help="Fold CSV used to fit normalization from train tasks only.",
    )
    normalization.add_argument(
        "--normalization-artifact",
        help="Existing frozen fold-normalization JSON.",
    )
    parser.add_argument(
        "--fold",
        help="Fold id; required with --fold-manifest.",
    )
    parser.add_argument(
        "--backbone-config",
        required=True,
        help="Verified local frozen-backbone YAML/JSON.",
    )
    parser.add_argument(
        "--feature-cache",
        default="outputs/stage5/features",
        help="Content-addressed feature cache directory.",
    )
    parser.add_argument(
        "--normal-signatures",
        help="Optional normal_signatures.parquet to join into Router features.",
    )
    parser.add_argument(
        "--ablation",
        choices=tuple(BIR_AD_ABLATIONS),
        default="full",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Frozen-backbone device; CUDA also selects torch BIR consistency in auto mode.",
    )
    parser.add_argument(
        "--bir-consistency-backend",
        choices=("auto", "numpy", "torch"),
        default="auto",
        help="Nearest-patch consistency backend; auto follows the effective BIR device.",
    )
    parser.add_argument(
        "--bir-device",
        help="BIR consistency device; defaults to --device for torch and cpu for NumPy.",
    )
    parser.add_argument(
        "--bir-consistency-dtype",
        choices=("float64", "float32"),
        default="float64",
        help="float64 preserves the reference numerical path; float32 is opt-in.",
    )
    parser.add_argument(
        "--consistency-chunk-size",
        type=int,
        default=1024,
        help="Rows per nearest-patch similarity chunk.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--grid-shape",
        nargs=2,
        type=int,
        metavar=("ROWS", "COLUMNS"),
    )
    parser.add_argument(
        "--diagnostics-limit",
        type=int,
        default=0,
        help="Write per-patch CSV/PNG for the first N canonical tasks.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="outputs/stage5/bir_ad",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / BIR_AD_FAILURES_NAME
    outputs: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    runtime_statistics: dict[str, Any] = {}
    config_record = {
        key: value
        for key, value in vars(args).items()
    }
    try:
        _validate_args(args)
        consistency_backend, consistency_device = _resolve_bir_runtime(args)
        config_record.update(
            {
                "resolved_bir_consistency_backend": consistency_backend,
                "resolved_bir_device": consistency_device,
            }
        )
        tasks = read_tasks_jsonl(args.tasks)
        supports = read_supports_csv(args.supports)
        backbone_config = load_router_backbone_config(args.backbone_config)
        provider = FrozenVisualBackboneProvider(
            backbone_config,
            device=args.device,
        )
        feature_cache = FeatureCache.for_encoder(
            args.feature_cache,
            provider,
        )
        if args.fold_manifest:
            artifact = fit_fold_bir_ad_normalization(
                tasks,
                supports,
                read_fold_manifest_csv(args.fold_manifest),
                fold=args.fold,
                feature_cache=feature_cache,
                feature_encoder=provider,
                pixel_aligner=provider,
                patch_grid_shape=_grid_shape(args.grid_shape),
                batch_size=args.batch_size,
                output_path=output_dir / BIR_AD_NORMALIZATION_NAME,
            )
        else:
            artifact = load_fold_normalization_artifact(
                args.normalization_artifact
            )
            normalization_copy = output_dir / BIR_AD_NORMALIZATION_NAME
            _atomic_write_json(normalization_copy, artifact.to_dict())
        ablation = get_bir_ad_ablation(args.ablation)
        _atomic_write_json(
            output_dir / BIR_AD_ABLATION_RECORD_NAME,
            ablation.to_dict(),
        )
        encoder = BIRADTaskEncoder(
            feature_cache,
            provider,
            normalization_artifact=artifact,
            pixel_aligner=provider,
            batch_size=args.batch_size,
            patch_grid_shape=_grid_shape(args.grid_shape),
            bir_ad_kwargs={
                **ablation.compute_kwargs(),
                "consistency_chunk_size": args.consistency_chunk_size,
            },
            consistency_backend=consistency_backend,
            consistency_device=consistency_device,
            consistency_dtype=args.bir_consistency_dtype,
        )
        signatures = encoder.encode_tasks(tasks, supports)
        runtime_statistics = dict(encoder.last_run_statistics)
        signatures_path = write_bir_ad_signatures_jsonl(
            signatures,
            output_dir / BIR_AD_SIGNATURES_NAME,
        )
        _atomic_write_json(failures_path, [])
        outputs.update(
            {
                "normalization": str(output_dir / BIR_AD_NORMALIZATION_NAME),
                "signatures": str(signatures_path),
                "failures": str(failures_path),
                "feature_cache_manifest": str(feature_cache.write_manifest()),
                "ablation": str(output_dir / BIR_AD_ABLATION_RECORD_NAME),
            }
        )
        if args.normal_signatures:
            normal_records = _read_normal_signatures(args.normal_signatures)
            bundles = build_router_feature_bundles(
                normal_records,
                signatures,
                ablation=ablation,
            )
            router_path = write_router_feature_bundles_jsonl(
                bundles,
                output_dir / BIR_AD_ROUTER_FEATURES_NAME,
            )
            outputs["router_features"] = str(router_path)
        if args.diagnostics_limit:
            diagnostics = {}
            for signature in signatures[: args.diagnostics_limit]:
                token = hashlib.sha256(
                    signature.task_id.encode("utf-8")
                ).hexdigest()[:12]
                task_dir = output_dir / "diagnostics" / token
                diagnostics[signature.task_id] = write_bir_ad_diagnostics(
                    signature.result,
                    task_dir,
                )
            outputs["diagnostics"] = diagnostics
    except Exception as exc:
        failures = [{"code": type(exc).__name__, "message": str(exc)}]
        _atomic_write_json(failures_path, failures)
    run_record = {
        "protocol_version": BIR_AD_BUILD_PROTOCOL_VERSION,
        "run_kind": "feature_preprocessing",
        "config": config_record,
        "seed": args.seed,
        "git_commit": _git_commit(),
        "environment": _environment_record(),
        "runtime_statistics": runtime_statistics,
        "outputs": outputs,
        "failures": failures,
        "predictions": None,
    }
    _atomic_write_json(output_dir / BIR_AD_BUILD_RUN_NAME, run_record)
    if failures:
        print(f"ERROR: {failures[0]['message']}", file=sys.stderr)
        return 1
    print(
        f"BIR-AD build PASS: signatures={outputs['signatures']}, "
        f"normalization={outputs['normalization']}"
    )
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.fold_manifest and not isinstance(args.fold, str):
        raise ValueError("--fold is required with --fold-manifest")
    if args.normalization_artifact and args.fold:
        raise ValueError("--fold cannot be used with --normalization-artifact")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.diagnostics_limit < 0:
        raise ValueError("--diagnostics-limit must be non-negative")
    if args.consistency_chunk_size <= 0:
        raise ValueError("--consistency-chunk-size must be positive")
    _resolve_bir_runtime(args)
    _grid_shape(args.grid_shape)


def _resolve_bir_runtime(args: argparse.Namespace) -> tuple[str, str]:
    requested_device = str(args.bir_device or args.device).strip()
    if not requested_device:
        raise ValueError("BIR consistency device must be non-empty")
    backend = args.bir_consistency_backend
    if backend == "auto":
        backend = (
            "torch"
            if requested_device.casefold().split(":", 1)[0] == "cuda"
            else "numpy"
        )
    if backend == "numpy":
        if args.bir_device and requested_device.casefold() not in {"cpu", "cpu:0"}:
            raise ValueError("--bir-device must be cpu with the NumPy backend")
        if args.bir_consistency_dtype != "float64":
            raise ValueError(
                "the NumPy consistency backend requires --bir-consistency-dtype float64"
            )
        return "numpy", "cpu"
    return "torch", requested_device


def _grid_shape(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    rows, columns = (int(item) for item in value)
    if rows <= 0 or columns <= 0:
        raise ValueError("--grid-shape values must be positive")
    return rows, columns


def _read_normal_signatures(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"normal signatures do not exist: {source}")
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "joining normal signatures requires pyarrow from the stage5 extras"
        ) from exc
    rows = parquet.read_table(source).to_pylist()
    if not rows:
        raise ValueError(f"normal signatures are empty: {source}")
    return rows


def _environment_record() -> dict[str, Any]:
    packages = {}
    for package in ("numpy", "Pillow", "pyarrow", "timm", "torch"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    record = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
        },
    }
    try:
        import torch

        record["torch_cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available() else None
            ),
            "device_count": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
            "devices": (
                [
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "capability": list(
                            torch.cuda.get_device_capability(index)
                        ),
                    }
                    for index in range(torch.cuda.device_count())
                ]
                if torch.cuda.is_available() else []
            ),
        }
    except Exception as exc:
        record["torch_cuda"] = {
            "available": False,
            "torch_cuda_version": None,
            "cudnn_version": None,
            "device_count": 0,
            "devices": [],
            "inspection_error": f"{type(exc).__name__}: {exc}",
        }
    return record


def _git_commit() -> str:
    project_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return (
        completed.stdout.strip()
        if completed.returncode == 0
        else "unknown"
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    text = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
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


if __name__ == "__main__":
    raise SystemExit(main())
