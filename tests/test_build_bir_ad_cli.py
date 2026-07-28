import json
from pathlib import Path

import pytest

from src.normroute.cli.build_bir_ad import (
    BIR_AD_BUILD_RUN_NAME,
    _resolve_bir_runtime,
    _validate_args,
    main,
    parse_args,
)
from src.normroute.router.bir_ad_pipeline import BIR_AD_FAILURES_NAME


def test_batch_bir_ad_cli_parses_fold_and_ablation() -> None:
    args = parse_args(
        [
            "--tasks",
            "tasks.jsonl",
            "--supports",
            "supports.csv",
            "--fold-manifest",
            "folds.csv",
            "--fold",
            "fold0",
            "--backbone-config",
            "backbone.yaml",
            "--ablation",
            "full",
            "--grid-shape",
            "37",
            "37",
        ]
    )
    _validate_args(args)
    assert args.fold == "fold0"
    assert args.grid_shape == [37, 37]
    assert _resolve_bir_runtime(args) == ("numpy", "cpu")


def test_batch_bir_ad_cli_auto_selects_cuda_consistency() -> None:
    args = parse_args(
        [
            "--tasks",
            "tasks.jsonl",
            "--supports",
            "supports.csv",
            "--normalization-artifact",
            "normalization.json",
            "--backbone-config",
            "backbone.yaml",
            "--device",
            "cuda:3",
            "--consistency-chunk-size",
            "4096",
        ]
    )

    _validate_args(args)

    assert _resolve_bir_runtime(args) == ("torch", "cuda:3")
    assert args.bir_consistency_dtype == "float64"
    assert args.consistency_chunk_size == 4096


def test_batch_bir_ad_cli_requires_fold_for_fitting() -> None:
    args = parse_args(
        [
            "--tasks",
            "tasks.jsonl",
            "--supports",
            "supports.csv",
            "--fold-manifest",
            "folds.csv",
            "--backbone-config",
            "backbone.yaml",
        ]
    )
    with pytest.raises(ValueError, match="--fold is required"):
        _validate_args(args)


def test_batch_bir_ad_cli_records_fatal_failure(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    status = main(
        [
            "--tasks",
            str(tmp_path / "missing-tasks.jsonl"),
            "--supports",
            str(tmp_path / "missing-supports.csv"),
            "--normalization-artifact",
            str(tmp_path / "missing-normalization.json"),
            "--backbone-config",
            str(tmp_path / "missing-backbone.yaml"),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 1
    failures = json.loads(
        (output_dir / BIR_AD_FAILURES_NAME).read_text(encoding="utf-8")
    )
    run = json.loads(
        (output_dir / BIR_AD_BUILD_RUN_NAME).read_text(encoding="utf-8")
    )
    assert len(failures) == 1
    assert run["failures"] == failures
    assert run["seed"] == 0
    assert run["git_commit"]
    assert run["environment"]["python"]
    assert run["predictions"] is None
