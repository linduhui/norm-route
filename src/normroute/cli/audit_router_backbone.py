"""Audit a Stage 5 router backbone checkpoint and runtime freeze state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Callable, Mapping

from ..router.feature_provider import (
    FrozenVisualBackboneProvider,
    RouterBackboneConfig,
    load_router_backbone_config,
    verify_local_checkpoint,
)


BACKBONE_AUDIT_PROTOCOL_VERSION = "stage5.router_backbone_audit.v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/stage5/router_backbone.example.yaml",
        help="Router backbone JSON/YAML config path.",
    )
    parser.add_argument(
        "--report",
        default="outputs/stage5/audits/router_backbone_audit.json",
        help="Machine-readable audit report path.",
    )
    parser.add_argument("--device", default="cpu", help="Local torch device used for loading.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_router_backbone(
        config_path=args.config,
        report_path=args.report,
        device=args.device,
    )
    if not report["ok"]:
        for failure in report["failures"]:
            print(f"ERROR: {failure['message']}", file=sys.stderr)
        return 1
    print(f"Stage 5 router backbone audit PASS: report={args.report}")
    return 0


def audit_router_backbone(
    *,
    config_path: str | Path,
    report_path: str | Path | None = None,
    device: str = "cpu",
    provider_factory: Callable[[RouterBackboneConfig], Any] | None = None,
) -> dict[str, Any]:
    """Verify provenance, local bytes, and the loaded model's freeze state."""

    config_file = Path(config_path)
    failures: list[dict[str, str]] = []
    config: RouterBackboneConfig | None = None
    checkpoint_record: dict[str, Any] | None = None
    freeze_status = {
        "configured_frozen": None,
        "model_loaded": False,
        "all_parameters_frozen": False,
        "eval_mode": False,
    }

    try:
        config = load_router_backbone_config(config_file)
        freeze_status["configured_frozen"] = config.frozen
        checkpoint_record = verify_local_checkpoint(config)
        factory = provider_factory or (
            lambda item: FrozenVisualBackboneProvider(item, device=device)
        )
        provider = factory(config)
        freeze_status.update(
            {
                "model_loaded": True,
                "all_parameters_frozen": bool(provider.is_frozen),
                "eval_mode": bool(provider.is_eval),
            }
        )
        if not freeze_status["all_parameters_frozen"]:
            failures.append(_failure("BACKBONE_NOT_FROZEN", "backbone parameters are not frozen"))
        if not freeze_status["eval_mode"]:
            failures.append(_failure("BACKBONE_NOT_EVAL", "backbone is not in eval mode"))
    except Exception as exc:
        failures.append(_failure(type(exc).__name__, str(exc)))

    report: dict[str, Any] = {
        "protocol_version": BACKBONE_AUDIT_PROTOCOL_VERSION,
        "ok": not failures,
        "config_path": str(config_file),
        "config_sha256": _optional_sha256(config_file),
        "backbone_config": config.to_dict() if config is not None else None,
        "checkpoint": checkpoint_record,
        "freeze_status": freeze_status,
        "failure_count": len(failures),
        "failures": failures,
        "provenance": {
            "git_commit": _git_commit(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    if report_path is not None:
        _write_report(Path(report_path), report)
    return report


def _failure(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _optional_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


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
