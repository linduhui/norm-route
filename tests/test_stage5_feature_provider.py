import hashlib
from pathlib import Path

import pytest

from src.normroute.cli.audit_router_backbone import audit_router_backbone
from src.normroute.router.feature_provider import (
    CheckpointHashError,
    CheckpointNotFoundError,
    FrozenVisualBackboneProvider,
    RouterBackboneConfig,
    RouterBackboneConfigError,
    checkpoint_sha256,
    freeze_visual_backbone,
    load_router_backbone_config,
    verify_local_checkpoint,
    visual_backbone_is_frozen,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = True

    def requires_grad_(self, value: bool) -> "_FakeParameter":
        self.requires_grad = value
        return self


class _FakeModel:
    def __init__(self) -> None:
        self.training = True
        self._parameters = [_FakeParameter(), _FakeParameter()]

    def parameters(self):
        return iter(self._parameters)

    def eval(self) -> "_FakeModel":
        self.training = False
        return self


class _AuditedProvider:
    def __init__(self, config: RouterBackboneConfig) -> None:
        self.config = config
        self.is_frozen = True
        self.is_eval = True


def _write_config(path: Path, checkpoint_name: str, sha256: str) -> None:
    path.write_text(
        "\n".join(
            [
                "protocol_version: stage5.router_backbone.v1",
                f"checkpoint_path: {checkpoint_name}",
                f'sha256: "{sha256}"',
                "architecture: dinov2_vits14",
                "input_size: 518",
                "frozen: true",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_pyproject_exposes_stage5_optional_environment() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    stage5_section = pyproject.split("[project.optional-dependencies]", 1)[1].split(
        "[tool.setuptools.packages.find]", 1
    )[0]

    assert '"torch>=' in stage5_section
    assert '"torchvision>=' in stage5_section
    assert '"timm>=' in stage5_section


def test_config_resolves_local_checkpoint_and_records_required_provenance(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "dinov2_vits14.pth"
    checkpoint.write_bytes(b"local-stage5-checkpoint")
    expected_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config_path = tmp_path / "router_backbone.yaml"
    _write_config(config_path, checkpoint.name, expected_hash)

    config = load_router_backbone_config(config_path)
    record = verify_local_checkpoint(config)

    assert config.to_dict() == {
        "protocol_version": "stage5.router_backbone.v1",
        "checkpoint_path": str(checkpoint),
        "sha256": expected_hash,
        "architecture": "dinov2_vits14",
        "input_size": 518,
        "frozen": True,
    }
    assert checkpoint_sha256(checkpoint) == expected_hash
    assert record["checkpoint_path"] == str(checkpoint.resolve())
    assert record["hash_matches"] is True


def test_missing_checkpoint_fails_explicitly_without_dependency_import(
    tmp_path: Path,
) -> None:
    config = RouterBackboneConfig(
        checkpoint_path=tmp_path / "missing.pth",
        sha256="0" * 64,
    )

    with pytest.raises(CheckpointNotFoundError, match="automatic downloads are disabled"):
        verify_local_checkpoint(config)

    with pytest.raises(CheckpointNotFoundError, match="automatic downloads are disabled"):
        FrozenVisualBackboneProvider(config)


def test_checkpoint_hash_mismatch_is_a_hard_failure(tmp_path: Path) -> None:
    checkpoint = tmp_path / "backbone.pth"
    checkpoint.write_bytes(b"unexpected bytes")
    config = RouterBackboneConfig(checkpoint_path=checkpoint, sha256="0" * 64)

    with pytest.raises(CheckpointHashError, match="SHA-256 mismatch"):
        verify_local_checkpoint(config)


def test_remote_checkpoint_references_are_forbidden() -> None:
    with pytest.raises(RouterBackboneConfigError, match="remote references are forbidden"):
        RouterBackboneConfig(
            checkpoint_path="https://example.invalid/dinov2.pth",
            sha256="0" * 64,
        )


def test_visual_backbone_is_forced_into_frozen_eval_state() -> None:
    model = _FakeModel()

    returned = freeze_visual_backbone(model)

    assert returned is model
    assert visual_backbone_is_frozen(model)
    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model._parameters)


def test_config_rejects_unfrozen_router_backbone(tmp_path: Path) -> None:
    with pytest.raises(RouterBackboneConfigError, match="frozen: true"):
        RouterBackboneConfig(
            checkpoint_path=tmp_path / "backbone.pth",
            sha256="0" * 64,
            frozen=False,
        )


def test_backbone_audit_records_path_hash_and_runtime_freeze_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "backbone.pth"
    checkpoint.write_bytes(b"audited checkpoint")
    expected_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config_path = tmp_path / "router_backbone.yaml"
    report_path = tmp_path / "router_backbone_audit.json"
    _write_config(config_path, checkpoint.name, expected_hash)

    report = audit_router_backbone(
        config_path=config_path,
        report_path=report_path,
        provider_factory=_AuditedProvider,
    )

    assert report["ok"] is True
    assert report["backbone_config"]["checkpoint_path"] == str(checkpoint)
    assert report["checkpoint"]["observed_sha256"] == expected_hash
    assert report["freeze_status"] == {
        "configured_frozen": True,
        "model_loaded": True,
        "all_parameters_frozen": True,
        "eval_mode": True,
    }
    assert report_path.is_file()


def test_backbone_audit_reports_missing_weight_as_failure(tmp_path: Path) -> None:
    config_path = tmp_path / "router_backbone.yaml"
    _write_config(config_path, "missing.pth", "0" * 64)

    report = audit_router_backbone(config_path=config_path)

    assert report["ok"] is False
    assert report["failure_count"] == 1
    assert report["failures"][0]["code"] == "CheckpointNotFoundError"
    assert "automatic downloads are disabled" in report["failures"][0]["message"]
