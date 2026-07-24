"""Generate auditable BIR-AD patch-weight visualizations and statistics."""

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

from ..router.bir_ad import (
    BIR_AD_CLARITY_COMPONENT_NAMES,
    BIR_AD_PROTOCOL_VERSION,
    BIRADNormalizationStats,
    BIRADResult,
    BIRADTaskResult,
    compute_bir_ad,
    compute_bir_ad_task,
    load_bir_ad_normalization_stats,
)


BIR_AD_DIAGNOSTIC_PROTOCOL_VERSION = "stage5.bir_ad_diagnostic.v3"
PATCH_STATISTICS_NAME = "bir_ad_patch_statistics.csv"
SUMMARY_STATISTICS_NAME = "bir_ad_summary_statistics.csv"
RUN_RECORD_NAME = "bir_ad_diagnostic_run.json"

PATCH_STATISTIC_COLUMNS = (
    "protocol_version",
    "role",
    "image_index",
    "patch_index",
    "grid_row",
    "grid_column",
    "patch_channel_std",
    "patch_l2_norm",
    "structural_boundary",
    "sobel_edge_energy",
    "boundary_orientation_consistency",
    "two_sided_feature_contrast",
    "normalized_patch_channel_std",
    "normalized_patch_l2_norm",
    "normalized_structural_boundary",
    "normalized_sobel_edge_energy",
    "normalized_boundary_orientation_consistency",
    "normalized_two_sided_feature_contrast",
    "pixel_feature_boundary_disagreement",
    "pixel_feature_boundary_agreement",
    "clarity",
    "boundary_weight",
    "patch_support_consistency",
    "clear_evidence",
    "ambiguous_evidence",
    "clear_weight",
    "ambiguous_weight",
)
SUMMARY_STATISTIC_COLUMNS = (
    "protocol_version",
    "role",
    "image_index",
    "patch_count",
    "grid_rows",
    "grid_columns",
    "bai",
    "bai_reliability",
    "support_bai",
    "support_bai_variance",
    "support_bai_std",
    "query_bai",
    "query_support_boundary_shift",
    "absolute_boundary_shift",
    "support_bai_reliability",
    "query_bai_reliability",
    "weighted_pixel_feature_disagreement",
    "support_pixel_feature_disagreement",
    "query_pixel_feature_disagreement",
    "support_boundary_consistency",
    "support_boundary_consistency_valid",
    "query_support_boundary_consistency",
    "alignment_verified",
    "alignment_fingerprint",
    "source_image_sha256",
    "use_structural_boundary_weighting",
    "disagreement_penalty",
    "normalization_source",
    "normalization_sample_count",
    "normalization_fitted",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-patches", required=True, help="Query [P,C] .npy file.")
    parser.add_argument("--query-image", required=True, help="Query image path.")
    parser.add_argument(
        "--support-patches",
        action="append",
        default=[],
        help="Normal support [P,C] .npy file; repeat once per support.",
    )
    parser.add_argument(
        "--support-image",
        action="append",
        default=[],
        help="Official train/good support image; repeat once per support.",
    )
    parser.add_argument(
        "--grid-shape",
        nargs=2,
        type=int,
        metavar=("ROWS", "COLUMNS"),
        help="Patch grid shape; omitted only when P is a perfect square.",
    )
    parser.add_argument(
        "--clarity-weights",
        "--component-weights",
        dest="clarity_weights",
        nargs=5,
        type=float,
        default=(0.2, 0.2, 0.2, 0.2, 0.2),
        metavar=tuple(BIR_AD_CLARITY_COMPONENT_NAMES),
    )
    parser.add_argument(
        "--normalization-stats",
        help="Frozen training-category BIR-AD normalization JSON.",
    )
    parser.add_argument(
        "--require-fitted-normalization",
        action="store_true",
        help="Fail instead of using the recorded fixed identity smoke transform.",
    )
    parser.add_argument("--boundary-temperature", type=float, default=1.0)
    parser.add_argument("--neighbor-connectivity", type=int, choices=(4, 8), default=4)
    parser.add_argument("--two-sided-radius", type=int, default=1)
    parser.add_argument("--structure-tensor-sigma", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default="outputs/stage5/bir_ad_diagnostics",
        help="Directory for PNG, CSV, and run-record artifacts.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Recorded diagnostic seed.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    outputs: dict[str, Any] = {}
    stats: BIRADNormalizationStats | None = None
    try:
        if len(args.support_patches) != len(args.support_image):
            raise ValueError(
                "--support-patches and --support-image must be repeated equally"
            )
        stats = (
            load_bir_ad_normalization_stats(args.normalization_stats)
            if args.normalization_stats
            else BIRADNormalizationStats.identity()
        )
        kwargs = {
            "patch_grid_shape": args.grid_shape,
            "normalization_stats": stats,
            "clarity_weights": args.clarity_weights,
            "boundary_temperature": args.boundary_temperature,
            "neighbor_connectivity": args.neighbor_connectivity,
            "two_sided_radius": args.two_sided_radius,
            "structure_tensor_sigma": args.structure_tensor_sigma,
            "require_fitted_normalization": args.require_fitted_normalization,
        }
        query_patches = _load_npy(Path(args.query_patches))
        if args.support_patches:
            result: BIRADResult | BIRADTaskResult = compute_bir_ad_task(
                query_patches,
                Path(args.query_image),
                [_load_npy(Path(path)) for path in args.support_patches],
                [Path(path) for path in args.support_image],
                **kwargs,
            )
        else:
            result = compute_bir_ad(
                query_patches,
                Path(args.query_image),
                **kwargs,
            )
        outputs = write_bir_ad_diagnostics(result, output_dir)
    except Exception as exc:
        failures.append({"code": type(exc).__name__, "message": str(exc)})

    run_record = {
        "protocol_version": BIR_AD_DIAGNOSTIC_PROTOCOL_VERSION,
        "bir_ad_protocol_version": BIR_AD_PROTOCOL_VERSION,
        "ok": not failures,
        "config": {
            "query_patches": args.query_patches,
            "query_image": args.query_image,
            "support_patches": list(args.support_patches),
            "support_images": list(args.support_image),
            "grid_shape": args.grid_shape,
            "clarity_weights": list(args.clarity_weights),
            "normalization_stats_path": args.normalization_stats,
            "require_fitted_normalization": args.require_fitted_normalization,
            "boundary_temperature": args.boundary_temperature,
            "neighbor_connectivity": args.neighbor_connectivity,
            "two_sided_radius": args.two_sided_radius,
            "structure_tensor_sigma": args.structure_tensor_sigma,
            "output_dir": str(output_dir),
        },
        "normalization": stats.to_dict() if stats is not None else None,
        "seed": args.seed,
        "git_commit": _git_commit(),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
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
        f"BIR-AD diagnostics PASS: patch_statistics={outputs['patch_statistics']}, "
        f"summary_statistics={outputs['summary_statistics']}"
    )
    return 0


def write_bir_ad_diagnostics(
    result: BIRADResult | BIRADTaskResult,
    output_dir: str | Path,
    *,
    heatmap_cell_size: int = 32,
) -> dict[str, Any]:
    """Write clear/ambiguous PNG maps and exhaustive deterministic CSV files."""

    if (
        isinstance(heatmap_cell_size, bool)
        or not isinstance(heatmap_cell_size, int)
        or heatmap_cell_size <= 0
    ):
        raise ValueError("heatmap_cell_size must be a positive integer")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(result, BIRADTaskResult):
        named_results = [("query", 0, result.query)]
        named_results.extend(
            ("support", index, support)
            for index, support in enumerate(result.supports)
        )
        task_result: BIRADTaskResult | None = result
    elif isinstance(result, BIRADResult):
        named_results = [("query", 0, result)]
        task_result = None
    else:
        raise TypeError("result must be BIRADResult or BIRADTaskResult")

    visualization_paths: list[str] = []
    patch_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for role, image_index, image_result in named_results:
        stem = role if role == "query" else f"support_{image_index:03d}"
        if task_result is None:
            patch_support_consistency = None
        elif role == "query":
            patch_support_consistency = (
                task_result.query_patch_support_consistency
            )
        else:
            patch_support_consistency = (
                task_result.support_patch_consistency[image_index]
            )
        for kind, weights in (
            ("boundary", image_result.boundary_weights),
            ("clear", image_result.clear_weights),
            ("ambiguous", image_result.ambiguous_weights),
            (
                "pixel_feature_disagreement",
                image_result.pixel_feature_boundary_disagreement,
            ),
        ):
            path = directory / f"{stem}_{kind}_weight.png"
            _write_weight_heatmap(
                path,
                weights,
                image_result.patch_grid_shape,
                cell_size=heatmap_cell_size,
            )
            visualization_paths.append(str(path))
        patch_rows.extend(
            _patch_rows(
                role,
                image_index,
                image_result,
                patch_support_consistency=patch_support_consistency,
            )
        )
        summary_rows.append(
            _summary_row(role, image_index, image_result, task_result)
        )
    patch_path = directory / PATCH_STATISTICS_NAME
    summary_path = directory / SUMMARY_STATISTICS_NAME
    _atomic_write_csv(patch_path, PATCH_STATISTIC_COLUMNS, patch_rows)
    _atomic_write_csv(summary_path, SUMMARY_STATISTIC_COLUMNS, summary_rows)
    return {
        "patch_statistics": str(patch_path),
        "summary_statistics": str(summary_path),
        "weight_visualizations": visualization_paths,
    }


def _patch_rows(
    role: str,
    image_index: int,
    result: BIRADResult,
    *,
    patch_support_consistency: Any | None = None,
) -> list[dict[str, Any]]:
    columns = result.patch_grid_shape[1]
    rows = []
    for index in range(result.patch_count):
        rows.append(
            {
                "protocol_version": result.protocol_version,
                "role": role,
                "image_index": image_index,
                "patch_index": index,
                "grid_row": index // columns,
                "grid_column": index % columns,
                "patch_channel_std": float(result.patch_channel_std[index]),
                "patch_l2_norm": float(result.patch_l2_norm[index]),
                "structural_boundary": float(result.structural_boundary[index]),
                "sobel_edge_energy": float(result.sobel_edge_energy[index]),
                "boundary_orientation_consistency": float(
                    result.boundary_orientation_consistency[index]
                ),
                "two_sided_feature_contrast": float(
                    result.two_sided_feature_contrast[index]
                ),
                "normalized_patch_channel_std": float(
                    result.normalized_patch_channel_std[index]
                ),
                "normalized_patch_l2_norm": float(
                    result.normalized_patch_l2_norm[index]
                ),
                "normalized_structural_boundary": float(
                    result.normalized_structural_boundary[index]
                ),
                "normalized_sobel_edge_energy": float(
                    result.normalized_sobel_edge_energy[index]
                ),
                "normalized_boundary_orientation_consistency": float(
                    result.normalized_boundary_orientation_consistency[index]
                ),
                "normalized_two_sided_feature_contrast": float(
                    result.normalized_two_sided_feature_contrast[index]
                ),
                "pixel_feature_boundary_disagreement": float(
                    result.pixel_feature_boundary_disagreement[index]
                ),
                "pixel_feature_boundary_agreement": float(
                    result.pixel_feature_boundary_agreement[index]
                ),
                "clarity": float(result.clarity[index]),
                "boundary_weight": float(result.boundary_weights[index]),
                "patch_support_consistency": (
                    float(patch_support_consistency[index])
                    if patch_support_consistency is not None
                    else ""
                ),
                "clear_evidence": float(result.clear_evidence[index]),
                "ambiguous_evidence": float(result.ambiguous_evidence[index]),
                "clear_weight": float(result.clear_weights[index]),
                "ambiguous_weight": float(result.ambiguous_weights[index]),
            }
        )
    return rows


def _summary_row(
    role: str,
    image_index: int,
    result: BIRADResult,
    task_result: BIRADTaskResult | None,
) -> dict[str, Any]:
    stats = result.normalization_stats
    return {
        "protocol_version": result.protocol_version,
        "role": role,
        "image_index": image_index,
        "patch_count": result.patch_count,
        "grid_rows": result.patch_grid_shape[0],
        "grid_columns": result.patch_grid_shape[1],
        "bai": result.bai,
        "bai_reliability": result.bai_reliability,
        "support_bai": task_result.support_bai if task_result else "",
        "support_bai_variance": (
            task_result.support_bai_variance if task_result else ""
        ),
        "support_bai_std": task_result.support_bai_std if task_result else "",
        "query_bai": task_result.query_bai if task_result else result.bai,
        "query_support_boundary_shift": (
            task_result.query_support_boundary_shift if task_result else ""
        ),
        "absolute_boundary_shift": (
            task_result.absolute_boundary_shift if task_result else ""
        ),
        "support_bai_reliability": (
            task_result.support_bai_reliability if task_result else ""
        ),
        "query_bai_reliability": (
            task_result.query_bai_reliability
            if task_result
            else result.bai_reliability
        ),
        "weighted_pixel_feature_disagreement": (
            result.weighted_pixel_feature_disagreement
        ),
        "support_pixel_feature_disagreement": (
            task_result.support_pixel_feature_disagreement
            if task_result
            else ""
        ),
        "query_pixel_feature_disagreement": (
            task_result.query_pixel_feature_disagreement
            if task_result
            else result.weighted_pixel_feature_disagreement
        ),
        "support_boundary_consistency": (
            task_result.support_boundary_consistency
            if task_result
            and task_result.support_boundary_consistency is not None
            else ""
        ),
        "support_boundary_consistency_valid": (
            task_result.support_boundary_consistency_valid
            if task_result
            else ""
        ),
        "query_support_boundary_consistency": (
            task_result.query_support_boundary_consistency
            if task_result
            else ""
        ),
        "alignment_verified": result.alignment_verified,
        "alignment_fingerprint": result.alignment_fingerprint or "",
        "source_image_sha256": result.source_image_sha256 or "",
        "use_structural_boundary_weighting": (
            result.use_structural_boundary_weighting
        ),
        "disagreement_penalty": result.disagreement_penalty,
        "normalization_source": stats.source,
        "normalization_sample_count": stats.sample_count,
        "normalization_fitted": stats.is_fitted,
    }


def _write_weight_heatmap(
    path: Path,
    weights: Any,
    grid_shape: tuple[int, int],
    *,
    cell_size: int,
) -> None:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("BIR-AD diagnostics require NumPy and Pillow") from exc
    array = np.asarray(weights, dtype=np.float64).reshape(grid_shape)
    maximum = float(array.max())
    normalized = array / maximum if maximum > 0.0 else np.zeros_like(array)
    red = np.rint(255.0 * normalized)
    green = np.rint(255.0 * (1.0 - np.abs(2.0 * normalized - 1.0)))
    blue = np.rint(255.0 * (1.0 - normalized))
    rgb = np.stack((red, green, blue), axis=-1).clip(0.0, 255.0).astype(np.uint8)
    image = Image.fromarray(rgb).resize(
        (grid_shape[1] * cell_size, grid_shape[0] * cell_size),
        resample=Image.Resampling.NEAREST,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _atomic_write_text(path: Path, text: str) -> None:
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


def _load_npy(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"BIR-AD patch feature file does not exist: {path}")
    try:
        import numpy as np

        return np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"could not load BIR-AD patch feature file {path}: {exc}") from exc


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
