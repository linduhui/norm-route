"""Run leakage-safe FBDP-AD diagnostics from cached patch feature arrays."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from ..router.fbdp_ad import (
    FBDP_AD_PROTOCOL_VERSION,
    FBDP_AD_RESIDUAL_QUANTILES,
    FBDPADResult,
    compute_fbdp_ad,
)


FBDP_AD_DIAGNOSTIC_PROTOCOL_VERSION = "stage5.fbdp_ad_diagnostic.v2"
PATCH_STATISTICS_NAME = "fbdp_ad_patch_statistics.csv"
SUMMARY_STATISTICS_NAME = "fbdp_ad_summary_statistics.csv"
PROTOTYPE_STATISTICS_NAME = "fbdp_ad_prototypes.csv"
RUN_RECORD_NAME = "fbdp_ad_diagnostic_run.json"

PATCH_STATISTIC_COLUMNS = (
    "protocol_version",
    "role",
    "image_index",
    "patch_index",
    "grid_row",
    "grid_column",
    "objectness",
    "cross_support_consistency",
    "border_candidate",
    "foreground_candidate",
    "background_candidate",
    "foreground_similarity",
    "background_similarity",
    "foreground_probability",
    "foreground_assignment_entropy",
    "foreground_background_margin",
    "foreground_residual",
    "background_residual",
    "decoupled_similarity",
    "residual",
    "gated_residual",
)

SUMMARY_STATISTIC_COLUMNS = (
    "protocol_version",
    "patch_count",
    "grid_rows",
    "grid_columns",
    "support_count",
    "foreground_candidate_count",
    "background_candidate_count",
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
    "foreground_candidate_ratio",
    "background_candidate_ratio",
    "prototype_method",
    "consistency_mode",
    "consistency_window_radius",
    "background_candidate_mode",
    "query_bank_mode",
    "k1_gate_policy",
    "k1_gate_scale",
    "fallback_gate_cap",
    "use_cross_support_consistency",
    "use_fbc_in_gate",
    "use_objectness_gate",
    "temperature",
    "residual_quantile_levels",
    "residual_quantiles",
    "gated_residual_quantiles",
    "assignment_entropy_quantiles",
    "foreground_background_margin_quantiles",
    "residual_mean",
    "residual_max",
    "gated_residual_mean",
    "gated_residual_max",
)

PROTOTYPE_STATISTIC_COLUMNS = (
    "protocol_version",
    "prototype_role",
    "prototype_index",
    "feature_dimension",
    "value",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-patches", required=True, help="Query [P,C] .npy file.")
    parser.add_argument(
        "--support-patches",
        action="append",
        required=True,
        help="Normal train/good support [P,C] .npy file; repeat per support.",
    )
    parser.add_argument(
        "--grid-shape",
        nargs=2,
        type=int,
        metavar=("ROWS", "COLUMNS"),
        help="Patch grid; omitted only when P is a perfect square.",
    )
    parser.add_argument("--border-width", type=int, default=1)
    parser.add_argument("--background-objectness-quantile", type=float, default=0.35)
    parser.add_argument("--foreground-consistency-quantile", type=float, default=0.50)
    parser.add_argument("--foreground-distance-quantile", type=float, default=0.65)
    parser.add_argument("--num-foreground-prototypes", type=int, default=4)
    parser.add_argument("--num-background-prototypes", type=int, default=4)
    parser.add_argument(
        "--prototype-method", choices=("kmeans", "pooling"), default="kmeans"
    )
    parser.add_argument("--kmeans-iterations", type=int, default=20)
    parser.add_argument(
        "--consistency-mode",
        choices=("same_position", "local_window", "nearest_neighbor"),
        default="local_window",
    )
    parser.add_argument("--consistency-window-radius", type=int, default=1)
    parser.add_argument(
        "--background-candidate-mode",
        choices=("combined", "border_only", "low_objectness_only"),
        default="combined",
    )
    parser.add_argument(
        "--query-bank-mode", choices=("decoupled", "single"), default="decoupled"
    )
    parser.add_argument(
        "--k1-gate-policy", choices=("neutral", "shrink", "disable"), default="shrink"
    )
    parser.add_argument("--k1-gate-scale", type=float, default=0.5)
    parser.add_argument("--fallback-gate-cap", type=float, default=0.25)
    parser.add_argument("--disable-cross-support-consistency", action="store_true")
    parser.add_argument("--disable-fbc-in-gate", action="store_true")
    parser.add_argument("--disable-objectness-gate", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument(
        "--residual-quantiles",
        nargs="+",
        type=float,
        default=FBDP_AD_RESIDUAL_QUANTILES,
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/stage5/fbdp_ad_diagnostics",
        help="Directory for CSV, PNG, and run-record artifacts.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Recorded diagnostic seed.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    outputs: dict[str, Any] = {}
    try:
        result = compute_fbdp_ad(
            _load_npy(Path(args.query_patches)),
            [_load_npy(Path(path)) for path in args.support_patches],
            patch_grid_shape=args.grid_shape,
            border_width=args.border_width,
            background_objectness_quantile=args.background_objectness_quantile,
            foreground_consistency_quantile=args.foreground_consistency_quantile,
            foreground_distance_quantile=args.foreground_distance_quantile,
            num_foreground_prototypes=args.num_foreground_prototypes,
            num_background_prototypes=args.num_background_prototypes,
            prototype_method=args.prototype_method,
            kmeans_iterations=args.kmeans_iterations,
            consistency_mode=args.consistency_mode,
            consistency_window_radius=args.consistency_window_radius,
            use_cross_support_consistency=(
                not args.disable_cross_support_consistency
            ),
            background_candidate_mode=args.background_candidate_mode,
            query_bank_mode=args.query_bank_mode,
            k1_gate_policy=args.k1_gate_policy,
            k1_gate_scale=args.k1_gate_scale,
            fallback_gate_cap=args.fallback_gate_cap,
            use_fbc_in_gate=not args.disable_fbc_in_gate,
            use_objectness_gate=not args.disable_objectness_gate,
            temperature=args.temperature,
            residual_quantiles=args.residual_quantiles,
        )
        outputs = write_fbdp_ad_diagnostics(result, output_dir)
    except Exception as exc:
        failures.append({"code": type(exc).__name__, "message": str(exc)})

    run_record = {
        "protocol_version": FBDP_AD_DIAGNOSTIC_PROTOCOL_VERSION,
        "fbdp_ad_protocol_version": FBDP_AD_PROTOCOL_VERSION,
        "ok": not failures,
        "config": {
            "query_patches": args.query_patches,
            "support_patches": list(args.support_patches),
            "grid_shape": args.grid_shape,
            "border_width": args.border_width,
            "background_objectness_quantile": args.background_objectness_quantile,
            "foreground_consistency_quantile": args.foreground_consistency_quantile,
            "foreground_distance_quantile": args.foreground_distance_quantile,
            "num_foreground_prototypes": args.num_foreground_prototypes,
            "num_background_prototypes": args.num_background_prototypes,
            "prototype_method": args.prototype_method,
            "kmeans_iterations": args.kmeans_iterations,
            "consistency_mode": args.consistency_mode,
            "consistency_window_radius": args.consistency_window_radius,
            "use_cross_support_consistency": (
                not args.disable_cross_support_consistency
            ),
            "background_candidate_mode": args.background_candidate_mode,
            "query_bank_mode": args.query_bank_mode,
            "k1_gate_policy": args.k1_gate_policy,
            "k1_gate_scale": args.k1_gate_scale,
            "fallback_gate_cap": args.fallback_gate_cap,
            "use_fbc_in_gate": not args.disable_fbc_in_gate,
            "use_objectness_gate": not args.disable_objectness_gate,
            "temperature": args.temperature,
            "residual_quantiles": list(args.residual_quantiles),
            "output_dir": str(output_dir),
        },
        "seed": args.seed,
        "git_commit": _git_commit(),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "predictions": outputs.get("summary_statistics"),
        "outputs": outputs,
        "failures": failures,
    }
    _atomic_write_text(
        output_dir / RUN_RECORD_NAME,
        json.dumps(run_record, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )
    if failures:
        for failure in failures:
            print(f"ERROR: {failure['message']}", file=sys.stderr)
        return 1
    print(
        "FBDP-AD diagnostics PASS: "
        f"patch_statistics={outputs['patch_statistics']}, "
        f"summary_statistics={outputs['summary_statistics']}"
    )
    return 0


def write_fbdp_ad_diagnostics(
    result: FBDPADResult,
    output_dir: str | Path,
    *,
    heatmap_cell_size: int = 32,
) -> dict[str, Any]:
    """Write exhaustive patch/prototype tables and support/query heatmaps."""

    if not isinstance(result, FBDPADResult):
        raise TypeError("result must be FBDPADResult")
    if (
        isinstance(heatmap_cell_size, bool)
        or not isinstance(heatmap_cell_size, int)
        or heatmap_cell_size <= 0
    ):
        raise ValueError("heatmap_cell_size must be a positive integer")
    np = _numpy()
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    context = result.support_context
    columns = result.patch_grid_shape[1]
    patch_rows: list[dict[str, Any]] = []
    visualization_paths: list[str] = []

    for image_index in range(context.support_count):
        objectness = context.support_objectness[image_index]
        consistency = context.support_cross_consistency[image_index]
        foreground_mask = context.foreground_candidate_masks[image_index]
        background_mask = context.background_candidate_masks[image_index]
        for patch_index in range(objectness.shape[0]):
            patch_rows.append(
                _support_patch_row(
                    result,
                    image_index,
                    patch_index,
                    columns,
                    objectness,
                    consistency,
                    foreground_mask,
                    background_mask,
                )
            )
        for name, values in (
            ("objectness", objectness),
            ("cross_support_consistency", consistency),
            ("foreground_candidate", foreground_mask.astype(float)),
            ("background_candidate", background_mask.astype(float)),
        ):
            path = directory / f"support_{image_index:03d}_{name}.png"
            _write_heatmap(
                path, values, result.patch_grid_shape, cell_size=heatmap_cell_size
            )
            visualization_paths.append(str(path))

    for patch_index in range(result.patch_count):
        patch_rows.append(_query_patch_row(result, patch_index, columns))
    for name, values in (
        ("foreground_probability", result.foreground_probability),
        ("decoupled_similarity", result.decoupled_similarity),
        ("residual", result.residuals),
        ("gated_residual", result.gated_residuals),
    ):
        path = directory / f"query_{name}.png"
        _write_heatmap(
            path, values, result.patch_grid_shape, cell_size=heatmap_cell_size
        )
        visualization_paths.append(str(path))

    summary_rows = [_summary_row(result, np)]
    prototype_rows = _prototype_rows(result)
    patch_path = directory / PATCH_STATISTICS_NAME
    summary_path = directory / SUMMARY_STATISTICS_NAME
    prototype_path = directory / PROTOTYPE_STATISTICS_NAME
    _atomic_write_csv(patch_path, PATCH_STATISTIC_COLUMNS, patch_rows)
    _atomic_write_csv(summary_path, SUMMARY_STATISTIC_COLUMNS, summary_rows)
    _atomic_write_csv(
        prototype_path, PROTOTYPE_STATISTIC_COLUMNS, prototype_rows
    )
    return {
        "patch_statistics": str(patch_path),
        "summary_statistics": str(summary_path),
        "prototype_statistics": str(prototype_path),
        "visualizations": visualization_paths,
    }


def _support_patch_row(
    result: FBDPADResult,
    image_index: int,
    patch_index: int,
    columns: int,
    objectness: Any,
    consistency: Any,
    foreground_mask: Any,
    background_mask: Any,
) -> dict[str, Any]:
    return {
        "protocol_version": result.protocol_version,
        "role": "support",
        "image_index": image_index,
        "patch_index": patch_index,
        "grid_row": patch_index // columns,
        "grid_column": patch_index % columns,
        "objectness": float(objectness[patch_index]),
        "cross_support_consistency": float(consistency[patch_index]),
        "border_candidate": bool(result.support_context.border_mask[patch_index]),
        "foreground_candidate": bool(foreground_mask[patch_index]),
        "background_candidate": bool(background_mask[patch_index]),
        "foreground_similarity": "",
        "background_similarity": "",
        "foreground_probability": "",
        "foreground_assignment_entropy": "",
        "foreground_background_margin": "",
        "foreground_residual": "",
        "background_residual": "",
        "decoupled_similarity": "",
        "residual": "",
        "gated_residual": "",
    }


def _query_patch_row(
    result: FBDPADResult, patch_index: int, columns: int
) -> dict[str, Any]:
    return {
        "protocol_version": result.protocol_version,
        "role": "query",
        "image_index": 0,
        "patch_index": patch_index,
        "grid_row": patch_index // columns,
        "grid_column": patch_index % columns,
        "objectness": "",
        "cross_support_consistency": "",
        "border_candidate": "",
        "foreground_candidate": "",
        "background_candidate": "",
        "foreground_similarity": float(result.foreground_similarity[patch_index]),
        "background_similarity": float(result.background_similarity[patch_index]),
        "foreground_probability": float(result.foreground_probability[patch_index]),
        "foreground_assignment_entropy": float(
            result.foreground_assignment_entropy[patch_index]
        ),
        "foreground_background_margin": float(
            result.foreground_background_margin[patch_index]
        ),
        "foreground_residual": float(result.foreground_residual[patch_index]),
        "background_residual": float(result.background_residual[patch_index]),
        "decoupled_similarity": float(result.decoupled_similarity[patch_index]),
        "residual": float(result.residuals[patch_index]),
        "gated_residual": float(result.gated_residuals[patch_index]),
    }


def _summary_row(result: FBDPADResult, np: Any) -> dict[str, Any]:
    context = result.support_context
    return {
        "protocol_version": result.protocol_version,
        "patch_count": result.patch_count,
        "grid_rows": result.patch_grid_shape[0],
        "grid_columns": result.patch_grid_shape[1],
        "support_count": context.support_count,
        "foreground_candidate_count": context.foreground_candidate_count,
        "background_candidate_count": context.background_candidate_count,
        "foreground_prototype_count": context.foreground_prototype_count,
        "background_prototype_count": context.background_prototype_count,
        "foreground_background_confusion": context.fbc,
        "objectness_gate": context.objectness_gate,
        "objectness_contrast": context.objectness_contrast,
        "support_reliability": context.support_reliability,
        "support_assignment_confidence": context.support_assignment_confidence,
        "prototype_compactness": context.prototype_compactness,
        "foreground_support_coverage": context.foreground_support_coverage,
        "leave_one_out_reconstruction_margin": (
            context.leave_one_out_reconstruction_margin
            if context.leave_one_out_reconstruction_margin is not None
            else ""
        ),
        "leave_one_out_reconstruction_valid": (
            context.leave_one_out_reconstruction_valid
        ),
        "support_consistency_valid": context.support_consistency_valid,
        "foreground_candidate_fallback": context.foreground_candidate_fallback,
        "foreground_candidate_ratio": context.foreground_candidate_ratio,
        "background_candidate_ratio": context.background_candidate_ratio,
        "prototype_method": context.prototype_method,
        "consistency_mode": context.consistency_mode,
        "consistency_window_radius": context.consistency_window_radius,
        "background_candidate_mode": context.background_candidate_mode,
        "query_bank_mode": context.query_bank_mode,
        "k1_gate_policy": context.k1_gate_policy,
        "k1_gate_scale": context.k1_gate_scale,
        "fallback_gate_cap": context.fallback_gate_cap,
        "use_cross_support_consistency": (
            context.use_cross_support_consistency
        ),
        "use_fbc_in_gate": context.use_fbc_in_gate,
        "use_objectness_gate": context.use_objectness_gate,
        "temperature": context.temperature,
        "residual_quantile_levels": json.dumps(context.residual_quantile_levels),
        "residual_quantiles": json.dumps(
            [float(value) for value in result.residual_quantiles]
        ),
        "gated_residual_quantiles": json.dumps(
            [float(value) for value in result.gated_residual_quantiles]
        ),
        "assignment_entropy_quantiles": json.dumps(
            [
                float(value)
                for value in result.foreground_assignment_entropy_quantiles
            ]
        ),
        "foreground_background_margin_quantiles": json.dumps(
            [
                float(value)
                for value in result.foreground_background_margin_quantiles
            ]
        ),
        "residual_mean": float(result.residuals.mean(dtype=np.float64)),
        "residual_max": float(result.residuals.max()),
        "gated_residual_mean": float(
            result.gated_residuals.mean(dtype=np.float64)
        ),
        "gated_residual_max": float(result.gated_residuals.max()),
    }


def _prototype_rows(result: FBDPADResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, prototypes in (
        ("foreground", result.foreground_prototypes),
        ("background", result.background_prototypes),
    ):
        for prototype_index in range(prototypes.shape[0]):
            for dimension in range(prototypes.shape[1]):
                rows.append(
                    {
                        "protocol_version": result.protocol_version,
                        "prototype_role": role,
                        "prototype_index": prototype_index,
                        "feature_dimension": dimension,
                        "value": float(prototypes[prototype_index, dimension]),
                    }
                )
    return rows


def _write_heatmap(
    path: Path,
    values: Any,
    grid_shape: tuple[int, int],
    *,
    cell_size: int,
) -> None:
    np = _numpy()
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("FBDP-AD diagnostics require Pillow") from exc
    array = np.asarray(values, dtype=np.float64).reshape(grid_shape)
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum > minimum:
        normalized = (array - minimum) / (maximum - minimum)
    else:
        normalized = np.zeros_like(array)
    red = np.rint(255.0 * normalized)
    green = np.rint(255.0 * (1.0 - np.abs(2.0 * normalized - 1.0)))
    blue = np.rint(255.0 * (1.0 - normalized))
    rgb = np.stack((red, green, blue), axis=-1).clip(0.0, 255.0).astype(np.uint8)
    image = Image.fromarray(rgb).resize(
        (grid_shape[1] * cell_size, grid_shape[0] * cell_size),
        resample=getattr(Image, "Resampling", Image).NEAREST,
    )
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, content: str) -> None:
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
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_npy(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"FBDP-AD patch feature file does not exist: {path}")
    try:
        return _numpy().load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"could not load FBDP-AD patch features {path}: {exc}") from exc


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("FBDP-AD diagnostics require NumPy") from exc
    return np


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


if __name__ == "__main__":
    raise SystemExit(main())
