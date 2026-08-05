import json
from pathlib import Path

import pytest

from src.normroute.cli.build_fbdp_ad import (
    FBDP_AD_BUILD_RUN_NAME,
    _validate_args,
    main,
    parse_args,
)
from src.normroute.router.fbdp_ad_pipeline import FBDP_AD_FAILURES_NAME


def _base_args() -> list[str]:
    return [
        "--tasks",
        "tasks.jsonl",
        "--supports",
        "supports.csv",
        "--backbone-config",
        "backbone.yaml",
    ]


def test_batch_cli_supports_remote_diagnostic_aliases() -> None:
    args = parse_args(
        [
            *_base_args(),
            "--feature-root",
            "outputs/stage5/features",
            "--limit",
            "100",
            "--grid-shape",
            "37",
            "37",
        ]
    )
    _validate_args(args)

    assert args.feature_cache == "outputs/stage5/features"
    assert args.diagnostics_limit == 100
    assert args.grid_shape == [37, 37]


def test_joint_router_view_requires_bir_signatures() -> None:
    args = parse_args(
        [
            *_base_args(),
            "--normal-signatures",
            "normal.parquet",
            "--feature-view",
            "normal_bir_fbdp",
        ]
    )
    with pytest.raises(ValueError, match="--bir-signatures"):
        _validate_args(args)


def test_batch_cli_records_fatal_failure_with_reproducibility_fields(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    status = main(
        [
            "--tasks",
            str(tmp_path / "missing-tasks.jsonl"),
            "--supports",
            str(tmp_path / "missing-supports.csv"),
            "--backbone-config",
            str(tmp_path / "missing-backbone.yaml"),
            "--output-dir",
            str(output_dir),
            "--seed",
            "19",
        ]
    )

    assert status == 1
    failures = json.loads(
        (output_dir / FBDP_AD_FAILURES_NAME).read_text(encoding="utf-8")
    )
    run = json.loads(
        (output_dir / FBDP_AD_BUILD_RUN_NAME).read_text(encoding="utf-8")
    )
    assert len(failures) == 1
    assert run["failures"] == failures
    assert run["seed"] == 19
    assert run["git_commit"]
    assert run["environment"]["python"]
    assert run["predictions"] is None
