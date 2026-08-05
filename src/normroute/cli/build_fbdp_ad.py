"""Build leakage-safe FBDP-AD signatures and optional batch diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from .diagnose_fbdp_ad import write_fbdp_ad_diagnostics
from .build_bir_ad import _read_normal_signatures
from ..router.bir_ad_ablation import BIR_AD_ABLATIONS
from ..router.fbdp_ad_ablation import (
    FBDP_AD_ABLATIONS,
    get_fbdp_ad_ablation,
)
from ..router.fbdp_ad_pipeline import (
    FBDP_AD_FAILURES_NAME,
    FBDP_AD_SIGNATURES_NAME,
    FBDPADTaskEncoder,
    read_supports_csv,
    read_tasks_jsonl,
    write_fbdp_ad_signatures_jsonl,
)
from ..router.feature_cache import FeatureCache
from ..router.feature_bundle import (
    build_router_feature_bundles,
    write_router_feature_bundles_jsonl,
)
from ..router.feature_provider import (
    FrozenVisualBackboneProvider,
    load_router_backbone_config,
)


FBDP_AD_BUILD_PROTOCOL_VERSION = "stage5.fbdp_ad_build.v1"
FBDP_AD_BUILD_RUN_NAME = "fbdp_ad_build_run.json"
FBDP_AD_ABLATION_RECORD_NAME = "fbdp_ad_ablation.json"
FBDP_AD_STATISTICS_NAME = "fbdp_statistics.csv"
FBDP_AD_ROUTER_FEATURES_NAME = "router_features.jsonl"

FBDP_AD_STATISTIC_COLUMNS = (
    "task_id",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
    "foreground_candidate_ratio",
    "background_candidate_ratio",
    "foreground_prototype_count",
    "background_prototype_count",
    "foreground_background_confusion",
    "objectness_gate",
    "objectness_contrast",
    "support_reliability",
    "support_assignment_confidence",
    "prototype_compactness",
    "foreground_support_coverage",
    "leave_one_out_reconstruction_margin",
    "leave_one_out_reconstruction_valid",
    "support_consistency_valid",
    "foreground_candidate_fallback",
    "residual_q50",
    "residual_q90",
    "residual_q95",
    "residual_q99",
    "gated_residual_q50",
    "gated_residual_q90",
    "gated_residual_q95",
    "gated_residual_q99",
    "residual_mean",
    "residual_max",
    "gated_residual_mean",
    "gated_residual_max",
    "assignment_entropy_mean",
    "assignment_entropy_max",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="Stage 5 task JSONL.")
    parser.add_argument(
        "--supports", required=True, help="Official train/good support CSV."
    )
    parser.add_argument(
        "--backbone-config",
        required=True,
        help="Verified local frozen-backbone YAML/JSON.",
    )
    parser.add_argument(
        "--feature-cache",
        "--feature-root",
        dest="feature_cache",
        default="outputs/stage5/features",
        help="Content-addressed feature cache directory.",
    )
    parser.add_argument(
        "--ablation", choices=tuple(FBDP_AD_ABLATIONS), default="full"
    )
    parser.add_argument(
        "--normal-signatures",
        help="Optional normal_signatures.parquet used to build Router features.",
    )
    parser.add_argument(
        "--bir-signatures",
        help="BIR signature JSONL required for the normal_bir_fbdp view.",
    )
    parser.add_argument(
        "--bir-ablation", choices=tuple(BIR_AD_ABLATIONS), default="full"
    )
    parser.add_argument(
        "--feature-view",
        choices=("normal_fbdp", "normal_bir_fbdp"),
        default="normal_fbdp",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--grid-shape", nargs=2, type=int, metavar=("ROWS", "COLUMNS")
    )
    parser.add_argument("--border-width", type=int, default=1)
    parser.add_argument("--background-objectness-quantile", type=float, default=0.35)
    parser.add_argument("--foreground-consistency-quantile", type=float, default=0.50)
    parser.add_argument("--foreground-distance-quantile", type=float, default=0.65)
    parser.add_argument("--num-foreground-prototypes", type=int, default=4)
    parser.add_argument("--num-background-prototypes", type=int, default=4)
    parser.add_argument("--kmeans-iterations", type=int, default=20)
    parser.add_argument("--consistency-window-radius", type=int, default=1)
    parser.add_argument(
        "--k1-gate-policy", choices=("neutral", "shrink", "disable"), default="shrink"
    )
    parser.add_argument("--k1-gate-scale", type=float, default=0.5)
    parser.add_argument("--fallback-gate-cap", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument(
        "--diagnostics-limit",
        "--limit",
        dest="diagnostics_limit",
        type=int,
        default=0,
        help="Write patch/prototype PNG diagnostics for the first N signatures.",
    )
    parser.add_argument(
        "--output-dir", default="outputs/stage5/fbdp_ad"
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / FBDP_AD_FAILURES_NAME
    outputs: dict[str, Any] = {
        "signatures": None,
        "statistics": None,
        "failures": str(failures_path),
    }
    failures: list[dict[str, Any]] = []
    runtime_statistics: dict[str, Any] = {}
    task_rows: list[Mapping[str, Any]] = []
    config_record = dict(vars(args))
    try:
        _validate_args(args)
        task_rows = read_tasks_jsonl(args.tasks)
        support_rows = read_supports_csv(args.supports)
        backbone_config = load_router_backbone_config(args.backbone_config)
        provider = FrozenVisualBackboneProvider(backbone_config, device=args.device)
        feature_cache = FeatureCache.for_encoder(args.feature_cache, provider)
        ablation = get_fbdp_ad_ablation(args.ablation)
        _atomic_write_json(
            output_dir / FBDP_AD_ABLATION_RECORD_NAME, ablation.to_dict()
        )
        fbdp_kwargs = {
            "border_width": args.border_width,
            "background_objectness_quantile": args.background_objectness_quantile,
            "foreground_consistency_quantile": args.foreground_consistency_quantile,
            "foreground_distance_quantile": args.foreground_distance_quantile,
            "num_foreground_prototypes": args.num_foreground_prototypes,
            "num_background_prototypes": args.num_background_prototypes,
            "kmeans_iterations": args.kmeans_iterations,
            "consistency_window_radius": args.consistency_window_radius,
            "k1_gate_policy": args.k1_gate_policy,
            "k1_gate_scale": args.k1_gate_scale,
            "fallback_gate_cap": args.fallback_gate_cap,
            "temperature": args.temperature,
            **ablation.compute_kwargs(),
        }
        encoder = FBDPADTaskEncoder(
            feature_cache,
            provider,
            batch_size=args.batch_size,
            patch_grid_shape=_grid_shape(args.grid_shape),
            fbdp_ad_kwargs=fbdp_kwargs,
        )
        signatures = encoder.encode_tasks(task_rows, support_rows)
        if len(signatures) != len(task_rows):
            raise RuntimeError("FBDP signature coverage disagrees with task count")
        signature_path = write_fbdp_ad_signatures_jsonl(
            signatures, output_dir / FBDP_AD_SIGNATURES_NAME
        )
        records = [signature.to_record() for signature in signatures]
        statistics_path = write_fbdp_statistics_csv(
            records, output_dir / FBDP_AD_STATISTICS_NAME
        )
        _atomic_write_json(failures_path, [])
        outputs.update(
            {
                "signatures": str(signature_path),
                "statistics": str(statistics_path),
                "feature_cache_manifest": str(feature_cache.write_manifest()),
                "ablation": str(output_dir / FBDP_AD_ABLATION_RECORD_NAME),
            }
        )
        runtime_statistics = {
            **encoder.last_run_statistics,
            "prediction_count": len(signatures),
            "failure_count": 0,
            "coverage_count": len(signatures),
        }
        if args.normal_signatures:
            normal_records = _read_normal_signatures(args.normal_signatures)
            bir_records = (
                _read_jsonl_records(args.bir_signatures)
                if args.bir_signatures
                else None
            )
            bundles = build_router_feature_bundles(
                normal_records,
                bir_records,
                fbdp_ad_signatures=signatures,
                ablation=args.bir_ablation,
                fbdp_ablation=ablation,
                feature_view=args.feature_view,
            )
            router_path = write_router_feature_bundles_jsonl(
                bundles, output_dir / FBDP_AD_ROUTER_FEATURES_NAME
            )
            outputs["router_features"] = str(router_path)
        if args.diagnostics_limit:
            diagnostics = {}
            for signature in signatures[: args.diagnostics_limit]:
                token = hashlib.sha256(
                    signature.task_id.encode("utf-8")
                ).hexdigest()[:12]
                diagnostics[signature.task_id] = write_fbdp_ad_diagnostics(
                    signature.result, output_dir / "diagnostics" / token
                )
            outputs["diagnostics"] = diagnostics
    except Exception as exc:
        failures = _task_failures(task_rows, exc)
        _atomic_write_json(failures_path, failures)
        runtime_statistics = {
            "prediction_count": 0,
            "failure_count": len(failures),
            "coverage_count": len(failures),
        }

    run_record = {
        "protocol_version": FBDP_AD_BUILD_PROTOCOL_VERSION,
        "run_kind": "feature_preprocessing",
        "ok": not failures,
        "config": config_record,
        "seed": args.seed,
        "git_commit": _git_commit(),
        "environment": _environment_record(),
        "runtime_statistics": runtime_statistics,
        "outputs": outputs,
        "predictions": outputs["signatures"],
        "failures": failures,
    }
    _atomic_write_json(output_dir / FBDP_AD_BUILD_RUN_NAME, run_record)
    if failures:
        print(f"FBDP-AD build failed: {failures[0]['message']}", file=sys.stderr)
        return 1
    print(
        f"FBDP-AD build PASS: signatures={outputs['signatures']}, "
        f"statistics={outputs['statistics']}"
    )
    return 0


def write_fbdp_statistics_csv(
    records: Sequence[Mapping[str, Any]], output_path: str | Path
) -> Path:
    if isinstance(records, (str, bytes)) or not records:
        raise ValueError("FBDP statistics records must be non-empty")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {column: record.get(column, "") for column in FBDP_AD_STATISTIC_COLUMNS}
        for record in sorted(records, key=lambda item: str(item.get("task_id", "")))
    ]
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(
            handle,
            fieldnames=FBDP_AD_STATISTIC_COLUMNS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "border_width", "num_foreground_prototypes", "num_background_prototypes", "kmeans_iterations"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.diagnostics_limit < 0:
        raise ValueError("--diagnostics-limit must be non-negative")
    if args.consistency_window_radius < 0:
        raise ValueError("--consistency-window-radius must be non-negative")
    if args.grid_shape and (len(args.grid_shape) != 2 or any(value <= 0 for value in args.grid_shape)):
        raise ValueError("--grid-shape must contain positive rows and columns")
    if args.feature_view == "normal_bir_fbdp" and not args.bir_signatures:
        raise ValueError("--bir-signatures is required for normal_bir_fbdp")
    if args.bir_signatures and not args.normal_signatures:
        raise ValueError("--bir-signatures requires --normal-signatures")


def _grid_shape(value: Sequence[int] | None) -> tuple[int, int] | None:
    return tuple(value) if value is not None else None  # type: ignore[return-value]


def _task_failures(
    tasks: Sequence[Mapping[str, Any]], exc: Exception
) -> list[dict[str, Any]]:
    if not tasks:
        return [{"task_id": None, "code": type(exc).__name__, "message": str(exc)}]
    return [
        {
            "task_id": str(task.get("task_id", f"task-index-{index}")),
            "code": type(exc).__name__,
            "message": str(exc),
        }
        for index, task in enumerate(tasks)
    ]


def _read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows = []
    try:
        with source.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"{source}:{line_number} is blank")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{source}:{line_number} is not an object")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSONL {source}: {exc}") from exc
    if not rows:
        raise ValueError(f"JSONL is empty: {source}")
    return rows


def _environment_record() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "torch", "timm", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _git_commit() -> str:
    project_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
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


if __name__ == "__main__":
    raise SystemExit(main())
