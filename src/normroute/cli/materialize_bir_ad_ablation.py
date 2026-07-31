"""Materialize a Router-only BIR-AD ablation from equivalent BIR signatures.

This command is intentionally limited to ablations whose BIR computation
configuration exactly matches the source run.  It never recomputes an image,
changes an expert, or weakens task/support provenance checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

from .build_bir_ad import (
    BIR_AD_ABLATION_RECORD_NAME,
    BIR_AD_BUILD_RUN_NAME,
    BIR_AD_NORMALIZATION_NAME,
    BIR_AD_ROUTER_FEATURES_NAME,
    _environment_record,
    _git_commit,
    _read_normal_signatures,
)
from ..router.bir_ad_ablation import BIR_AD_ABLATIONS, get_bir_ad_ablation
from ..router.bir_ad_pipeline import (
    BIR_AD_FAILURES_NAME,
    BIR_AD_SIGNATURE_COLUMNS,
    BIR_AD_SIGNATURE_PROTOCOL_VERSION,
    BIR_AD_SIGNATURES_NAME,
)
from ..router.feature_bundle import (
    build_router_feature_bundles,
    write_router_feature_bundles_jsonl,
)


BIR_AD_MATERIALIZE_PROTOCOL_VERSION = "stage5.bir_ad_materialize.v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-output",
        required=True,
        help="Completed BIR-AD build directory with equivalent compute settings.",
    )
    parser.add_argument(
        "--normal-signatures",
        required=True,
        help="normal_signatures.parquet joined into Router features.",
    )
    parser.add_argument(
        "--ablation",
        required=True,
        choices=tuple(BIR_AD_ABLATIONS),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_dir = Path(args.source_output)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / BIR_AD_FAILURES_NAME
    failures: list[dict[str, str]] = []
    outputs: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    try:
        target_ablation = get_bir_ad_ablation(args.ablation)
        source_ablation = _read_json_object(
            source_dir / BIR_AD_ABLATION_RECORD_NAME
        )
        source_run = _read_json_object(source_dir / BIR_AD_BUILD_RUN_NAME)
        source_failures = _read_json_value(
            source_dir / BIR_AD_FAILURES_NAME
        )
        if source_failures != [] or source_run.get("failures") != []:
            raise ValueError(
                "source BIR-AD run is not a complete failure-free build"
            )
        _validate_compute_compatibility(source_ablation, target_ablation.to_dict())
        if source_run.get("config", {}).get("ablation") != source_ablation.get(
            "name"
        ):
            raise ValueError("source run and source ablation record disagree")

        source_signatures_path = source_dir / BIR_AD_SIGNATURES_NAME
        signature_text = _read_text(source_signatures_path)
        bir_records = _parse_and_validate_signatures(signature_text)
        normal_records = _read_normal_signatures(args.normal_signatures)
        bundles = build_router_feature_bundles(
            normal_records,
            bir_records,
            ablation=target_ablation,
        )

        source_normalization_path = source_dir / BIR_AD_NORMALIZATION_NAME
        normalization_text = _read_text(source_normalization_path)
        _read_json_object(source_normalization_path)
        _atomic_write_text(
            output_dir / BIR_AD_SIGNATURES_NAME, signature_text
        )
        _atomic_write_text(
            output_dir / BIR_AD_NORMALIZATION_NAME, normalization_text
        )
        _atomic_write_json(
            output_dir / BIR_AD_ABLATION_RECORD_NAME,
            target_ablation.to_dict(),
        )
        router_path = write_router_feature_bundles_jsonl(
            bundles,
            output_dir / BIR_AD_ROUTER_FEATURES_NAME,
        )
        _atomic_write_json(failures_path, [])
        outputs = {
            "normalization": str(output_dir / BIR_AD_NORMALIZATION_NAME),
            "signatures": str(output_dir / BIR_AD_SIGNATURES_NAME),
            "router_features": str(router_path),
            "ablation": str(output_dir / BIR_AD_ABLATION_RECORD_NAME),
            "failures": str(failures_path),
        }
        source_hashes = {
            BIR_AD_SIGNATURES_NAME: _file_sha256(source_signatures_path),
            BIR_AD_NORMALIZATION_NAME: _file_sha256(
                source_normalization_path
            ),
            BIR_AD_ABLATION_RECORD_NAME: _file_sha256(
                source_dir / BIR_AD_ABLATION_RECORD_NAME
            ),
            BIR_AD_BUILD_RUN_NAME: _file_sha256(
                source_dir / BIR_AD_BUILD_RUN_NAME
            ),
        }
    except Exception as exc:
        failures = [{"code": type(exc).__name__, "message": str(exc)}]
        _atomic_write_json(failures_path, failures)

    run_record = {
        "protocol_version": BIR_AD_MATERIALIZE_PROTOCOL_VERSION,
        "run_kind": "router_feature_materialization",
        "config": vars(args),
        "seed": args.seed,
        "git_commit": _git_commit(),
        "environment": _environment_record(),
        "source_hashes": source_hashes,
        "outputs": outputs,
        "failures": failures,
        "predictions": None,
    }
    _atomic_write_json(output_dir / BIR_AD_BUILD_RUN_NAME, run_record)
    if failures:
        print(f"ERROR: {failures[0]['message']}", file=sys.stderr)
        return 1
    print(
        f"BIR-AD materialization PASS: ablation={args.ablation}, "
        f"router_features={outputs['router_features']}"
    )
    return 0


def _validate_compute_compatibility(
    source_ablation: Mapping[str, Any],
    target_ablation: Mapping[str, Any],
) -> None:
    source_protocol = source_ablation.get("protocol_version")
    target_protocol = target_ablation.get("protocol_version")
    compatible_protocols = {
        "stage5.bir_ad_ablation.v1",
        "stage5.bir_ad_ablation.v2",
    }
    if (
        source_protocol not in compatible_protocols
        or target_protocol not in compatible_protocols
    ):
        raise ValueError("source or target ablation protocol is incompatible")
    source_compute = _canonical_json(source_ablation.get("compute_kwargs"))
    target_compute = _canonical_json(target_ablation.get("compute_kwargs"))
    if source_compute != target_compute:
        raise ValueError(
            "source and target BIR compute settings differ; "
            "materialization is not valid"
        )


def _parse_and_validate_signatures(text: str) -> list[dict[str, Any]]:
    rows = []
    seen_task_ids: set[str] = set()
    consistency_runtime: tuple[str, str, str] | None = None
    normalization_hash: str | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"BIR-AD signatures line {line_number} is blank")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid BIR-AD signature JSON at line {line_number}"
            ) from exc
        if not isinstance(row, dict) or set(row) != set(BIR_AD_SIGNATURE_COLUMNS):
            raise ValueError(
                f"BIR-AD signature schema mismatch at line {line_number}"
            )
        if row.get("protocol_version") != BIR_AD_SIGNATURE_PROTOCOL_VERSION:
            raise ValueError(
                f"incompatible BIR-AD signature protocol at line {line_number}"
            )
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in seen_task_ids:
            raise ValueError("BIR-AD signatures contain an invalid/duplicate task_id")
        seen_task_ids.add(task_id)
        observed_runtime = (
            row["consistency_backend"],
            row["consistency_device"],
            row["consistency_dtype"],
        )
        if consistency_runtime is None:
            consistency_runtime = observed_runtime
        elif observed_runtime != consistency_runtime:
            raise ValueError("source signatures mix consistency runtimes")
        observed_normalization = row["normalization_sha256"]
        if normalization_hash is None:
            normalization_hash = observed_normalization
        elif observed_normalization != normalization_hash:
            raise ValueError("source signatures mix fold normalizations")
        rows.append(row)
    if not rows:
        raise ValueError("source BIR-AD signatures are empty")
    return rows


def _read_json_object(path: Path) -> dict[str, Any]:
    value = _read_json_value(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_json_value(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON artifact {path}: {exc}") from exc


def _read_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"could not read source artifact {path}: {exc}") from exc
    if not text:
        raise ValueError(f"source artifact is empty: {path}")
    return text


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"could not hash source artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        )
        + "\n",
    )


def _atomic_write_text(path: Path, text: str) -> None:
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
