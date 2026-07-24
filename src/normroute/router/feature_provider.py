"""Frozen visual feature providers for Stage 5 routers.

The provider deliberately separates local checkpoint verification from model
construction.  It never asks ``timm`` for pretrained weights: the configured
checkpoint must exist on the local filesystem and match its recorded SHA-256
before any optional deep-learning dependency is imported.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence


ROUTER_BACKBONE_PROTOCOL_VERSION = "stage5.router_backbone.v1"
DINOV2_S14_ARCHITECTURE = "dinov2_vits14"
DEFAULT_INPUT_SIZE = 518

_TIMM_ARCHITECTURE_ALIASES = {
    DINOV2_S14_ARCHITECTURE: "vit_small_patch14_dinov2.lvd142m",
    "dinov2-s/14": "vit_small_patch14_dinov2.lvd142m",
    "vit_small_patch14_dinov2": "vit_small_patch14_dinov2.lvd142m",
}
_REQUIRED_CONFIG_FIELDS = frozenset(
    {"checkpoint_path", "sha256", "architecture", "input_size"}
)
_OPTIONAL_CONFIG_FIELDS = frozenset({"protocol_version", "frozen"})
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_REMOTE_PATH_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


class RouterBackboneError(RuntimeError):
    """Base error for Stage 5 visual backbone failures."""


class RouterBackboneConfigError(RouterBackboneError, ValueError):
    """Raised when backbone provenance is incomplete or invalid."""


class CheckpointNotFoundError(RouterBackboneError, FileNotFoundError):
    """Raised when the required local checkpoint is absent."""


class CheckpointHashError(RouterBackboneError):
    """Raised when checkpoint bytes disagree with the configured hash."""


class Stage5DependencyError(RouterBackboneError, ImportError):
    """Raised when the optional Stage 5 environment is not installed."""


@dataclass(frozen=True)
class RouterBackboneConfig:
    """Auditable configuration for one frozen visual backbone."""

    checkpoint_path: Path
    sha256: str
    architecture: str = DINOV2_S14_ARCHITECTURE
    input_size: int = DEFAULT_INPUT_SIZE
    frozen: bool = True
    protocol_version: str = ROUTER_BACKBONE_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        checkpoint_path = _validate_local_path(self.checkpoint_path)
        if not isinstance(self.sha256, str):
            raise RouterBackboneConfigError("sha256 must be a string")
        if not isinstance(self.architecture, str):
            raise RouterBackboneConfigError("architecture must be a string")
        sha256 = self.sha256.strip().lower()
        architecture = self.architecture.strip()
        if not _SHA256_PATTERN.fullmatch(sha256):
            raise RouterBackboneConfigError("sha256 must contain exactly 64 hexadecimal characters")
        if not architecture:
            raise RouterBackboneConfigError("architecture must be a non-empty string")
        if isinstance(self.input_size, bool) or not isinstance(self.input_size, int):
            raise RouterBackboneConfigError("input_size must be an integer")
        if self.input_size <= 0:
            raise RouterBackboneConfigError("input_size must be greater than zero")
        if self.frozen is not True:
            raise RouterBackboneConfigError("Stage 5 router backbones must set frozen: true")
        if self.protocol_version != ROUTER_BACKBONE_PROTOCOL_VERSION:
            raise RouterBackboneConfigError(
                f"protocol_version must be {ROUTER_BACKBONE_PROTOCOL_VERSION!r}"
            )
        object.__setattr__(self, "checkpoint_path", checkpoint_path)
        object.__setattr__(self, "sha256", sha256)
        object.__setattr__(self, "architecture", architecture)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        base_dir: str | Path | None = None,
    ) -> "RouterBackboneConfig":
        """Validate a config mapping and resolve its relative local path."""

        if not isinstance(value, Mapping):
            raise RouterBackboneConfigError("router backbone config must be a mapping")
        fields = set(value)
        missing = sorted(_REQUIRED_CONFIG_FIELDS - fields)
        extra = sorted(fields - _REQUIRED_CONFIG_FIELDS - _OPTIONAL_CONFIG_FIELDS)
        if missing or extra:
            raise RouterBackboneConfigError(
                f"router backbone config has invalid fields; missing={missing}, extra={extra}"
            )
        raw_path = _validate_local_path(value["checkpoint_path"])
        if base_dir is not None and not raw_path.is_absolute():
            raw_path = Path(base_dir) / raw_path
        return cls(
            checkpoint_path=raw_path,
            sha256=value["sha256"],
            architecture=value["architecture"],
            input_size=value["input_size"],
            frozen=value.get("frozen", True),
            protocol_version=value.get(
                "protocol_version", ROUTER_BACKBONE_PROTOCOL_VERSION
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-safe provenance record."""

        return {
            "protocol_version": self.protocol_version,
            "checkpoint_path": str(self.checkpoint_path),
            "sha256": self.sha256,
            "architecture": self.architecture,
            "input_size": self.input_size,
            "frozen": self.frozen,
        }


VisualBackboneConfig = RouterBackboneConfig
"""Compatibility name emphasizing that this is a visual encoder config."""


class FeatureProvider(ABC):
    """Minimal router-facing feature encoder abstraction."""

    @property
    @abstractmethod
    def config(self) -> RouterBackboneConfig:
        """Return the immutable, auditable provider configuration."""

    @abstractmethod
    def encode(self, image_paths: Sequence[str | Path]) -> Any:
        """Encode every image or fail explicitly without skipping a sample."""

    def encode_global_and_patches(self, image_paths: Sequence[str | Path]) -> Any:
        """Return batched global and patch features when the provider supports it."""

        raise RouterBackboneError(
            "this feature provider does not expose global and patch features"
        )

    def align_images_for_patches(
        self,
        image_paths: Sequence[str | Path],
        *,
        patch_grid_shape: tuple[int, int],
    ) -> Any:
        """Return pixel views after the exact patch-token spatial transform."""

        raise RouterBackboneError(
            "this feature provider does not expose patch-aligned pixel views"
        )


def load_router_backbone_config(path: str | Path) -> RouterBackboneConfig:
    """Load JSON/YAML config without loading a model or touching the network."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise RouterBackboneConfigError(f"router backbone config does not exist: {config_path}")
    try:
        text = config_path.read_text(encoding="utf-8-sig")
        if config_path.suffix.lower() == ".json":
            payload = json.loads(text)
        elif config_path.suffix.lower() in {".yaml", ".yml"}:
            payload = _load_yaml_mapping(text)
        else:
            raise RouterBackboneConfigError(
                f"router backbone config must be JSON or YAML: {config_path}"
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RouterBackboneConfigError(f"could not read {config_path}: {exc}") from exc
    return RouterBackboneConfig.from_mapping(payload, base_dir=config_path.parent)


def checkpoint_sha256(path: str | Path) -> str:
    """Hash a local checkpoint without following any remote reference."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise CheckpointNotFoundError(
            f"local router backbone checkpoint does not exist: {checkpoint_path}"
        )
    digest = hashlib.sha256()
    try:
        with checkpoint_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RouterBackboneError(f"could not read checkpoint {checkpoint_path}: {exc}") from exc
    return digest.hexdigest()


def verify_local_checkpoint(config: RouterBackboneConfig) -> dict[str, Any]:
    """Require a local file whose exact bytes match the frozen config."""

    path = config.checkpoint_path.expanduser()
    if not path.is_file():
        raise CheckpointNotFoundError(
            f"local router backbone checkpoint does not exist: {path}; "
            "automatic downloads are disabled"
        )
    resolved = path.resolve()
    observed = checkpoint_sha256(resolved)
    if observed != config.sha256:
        raise CheckpointHashError(
            f"checkpoint SHA-256 mismatch for {resolved}: "
            f"expected {config.sha256}, observed {observed}"
        )
    return {
        "checkpoint_path": str(resolved),
        "expected_sha256": config.sha256,
        "observed_sha256": observed,
        "size_bytes": resolved.stat().st_size,
        "hash_matches": True,
    }


def freeze_visual_backbone(model: Any) -> Any:
    """Put a model in eval mode and permanently disable parameter gradients."""

    parameters = list(_model_parameters(model))
    if not parameters:
        raise RouterBackboneError("visual backbone exposes no parameters to freeze")
    for parameter in parameters:
        if hasattr(parameter, "requires_grad_"):
            parameter.requires_grad_(False)
        else:
            parameter.requires_grad = False
    if not hasattr(model, "eval"):
        raise RouterBackboneError("visual backbone does not implement eval()")
    model.eval()
    if not visual_backbone_is_frozen(model):
        raise RouterBackboneError("visual backbone freeze verification failed")
    return model


def visual_backbone_is_frozen(model: Any) -> bool:
    """Return true only when all parameters are frozen and the model is in eval mode."""

    try:
        parameters = list(_model_parameters(model))
    except (TypeError, AttributeError):
        return False
    return bool(parameters) and all(
        getattr(parameter, "requires_grad", True) is False for parameter in parameters
    ) and getattr(model, "training", True) is False


class FrozenVisualBackboneProvider(FeatureProvider):
    """Local-checkpoint-only frozen visual encoder backed by PyTorch/timm."""

    def __init__(
        self,
        config: RouterBackboneConfig,
        *,
        device: str = "cpu",
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._config = config
        self._checkpoint_record = verify_local_checkpoint(config)
        torch, timm = _load_stage5_dependencies()
        self._torch = torch
        self._device = torch.device(device)

        timm_architecture = _TIMM_ARCHITECTURE_ALIASES.get(
            config.architecture.casefold(), config.architecture
        )
        try:
            if model_factory is None:
                model = timm.create_model(
                    timm_architecture,
                    pretrained=False,
                    num_classes=0,
                )
            else:
                model = model_factory(timm_architecture)
        except Exception as exc:
            raise RouterBackboneError(
                f"could not construct local architecture {config.architecture!r}: {exc}"
            ) from exc

        state_dict = _load_checkpoint_state_dict(
            torch, Path(self._checkpoint_record["checkpoint_path"])
        )
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise RouterBackboneError(
                f"checkpoint is incompatible with architecture {config.architecture!r}: {exc}"
            ) from exc

        try:
            model.to(self._device)
            freeze_visual_backbone(model)
            data_config = timm.data.resolve_model_data_config(model)
            data_config["input_size"] = (3, config.input_size, config.input_size)
            self._data_config = dict(data_config)
            self._normalization_mean = tuple(
                float(value) for value in data_config.get("mean", (0.0, 0.0, 0.0))
            )
            self._normalization_std = tuple(
                float(value) for value in data_config.get("std", (1.0, 1.0, 1.0))
            )
            self._transform = timm.data.create_transform(
                **data_config,
                is_training=False,
            )
            transform_payload = {
                "provider_config": config.to_dict(),
                "data_config": {
                    key: data_config.get(key)
                    for key in (
                        "input_size",
                        "interpolation",
                        "crop_pct",
                        "crop_mode",
                        "mean",
                        "std",
                    )
                },
                "is_training": False,
            }
            self._spatial_transform_fingerprint = hashlib.sha256(
                json.dumps(
                    transform_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
        except Exception as exc:
            if isinstance(exc, RouterBackboneError):
                raise
            raise RouterBackboneError(f"could not initialize frozen visual backbone: {exc}") from exc
        self._model = model

    @property
    def config(self) -> RouterBackboneConfig:
        return self._config

    @property
    def checkpoint_record(self) -> dict[str, Any]:
        return dict(self._checkpoint_record)

    @property
    def is_frozen(self) -> bool:
        return visual_backbone_is_frozen(self._model)

    @property
    def is_eval(self) -> bool:
        return getattr(self._model, "training", True) is False

    def encode(self, image_paths: Sequence[str | Path]) -> Any:
        if isinstance(image_paths, (str, bytes, Path)) or not image_paths:
            raise RouterBackboneError("image_paths must be a non-empty sequence")
        if not self.is_frozen:
            raise RouterBackboneError("visual backbone is no longer frozen")

        images = []
        try:
            from PIL import Image

            for raw_path in image_paths:
                image_path = Path(raw_path)
                if not image_path.is_file():
                    raise FileNotFoundError(f"router input image does not exist: {image_path}")
                with Image.open(image_path) as image:
                    images.append(self._transform(image.convert("RGB")))
            batch = self._torch.stack(images).to(self._device)
            with self._torch.inference_mode():
                embeddings = self._model(batch)
            embeddings = _canonical_embeddings(embeddings)
            return embeddings.detach().cpu()
        except Exception as exc:
            if isinstance(exc, RouterBackboneError):
                raise
            raise RouterBackboneError(f"visual feature encoding failed: {exc}") from exc

    def encode_global_and_patches(self, image_paths: Sequence[str | Path]) -> Any:
        """Extract the CLS/global vector and every spatial patch token."""

        if isinstance(image_paths, (str, bytes, Path)) or not image_paths:
            raise RouterBackboneError("image_paths must be a non-empty sequence")
        if not self.is_frozen:
            raise RouterBackboneError("visual backbone is no longer frozen")

        images = []
        try:
            from PIL import Image

            for raw_path in image_paths:
                image_path = Path(raw_path)
                if not image_path.is_file():
                    raise FileNotFoundError(f"router input image does not exist: {image_path}")
                with Image.open(image_path) as image:
                    images.append(self._transform(image.convert("RGB")))
            batch = self._torch.stack(images).to(self._device)
            with self._torch.inference_mode():
                if not hasattr(self._model, "forward_features"):
                    raise RouterBackboneError(
                        "visual backbone does not expose patch-level forward_features()"
                    )
                output = self._model.forward_features(batch)
            global_features, patch_features = _canonical_global_patch_embeddings(output)
            return {
                "global_features": global_features.detach().cpu(),
                "patch_features": patch_features.detach().cpu(),
            }
        except Exception as exc:
            if isinstance(exc, RouterBackboneError):
                raise
            raise RouterBackboneError(
                f"visual global/patch feature encoding failed: {exc}"
            ) from exc

    @property
    def spatial_transform_fingerprint(self) -> str:
        return self._spatial_transform_fingerprint

    def align_images_for_patches(
        self,
        image_paths: Sequence[str | Path],
        *,
        patch_grid_shape: tuple[int, int],
    ) -> Any:
        """Apply the identical resize/crop and return de-normalized RGB views."""

        if isinstance(image_paths, (str, bytes, Path)) or not image_paths:
            raise RouterBackboneError("image_paths must be a non-empty sequence")
        rows, columns = _positive_grid_shape(patch_grid_shape)
        try:
            from PIL import Image

            from .bir_ad import PatchAlignedImage
            from .feature_cache import file_sha256

            aligned = []
            for raw_path in image_paths:
                image_path = Path(raw_path)
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"router input image does not exist: {image_path}"
                    )
                with Image.open(image_path) as image:
                    transformed = self._transform(image.convert("RGB"))
                pixels = canonical_aligned_pixels(
                    transformed,
                    mean=self._normalization_mean,
                    std=self._normalization_std,
                )
                if pixels.shape[0] < rows or pixels.shape[1] < columns:
                    raise RouterBackboneError(
                        f"aligned pixel shape {pixels.shape[:2]} is smaller than "
                        f"patch grid {(rows, columns)}"
                    )
                aligned.append(
                    PatchAlignedImage(
                        pixels=pixels,
                        patch_grid_shape=(rows, columns),
                        source_image_sha256=file_sha256(image_path),
                        transform_fingerprint=self.spatial_transform_fingerprint,
                    )
                )
            return aligned
        except Exception as exc:
            if isinstance(exc, RouterBackboneError):
                raise
            raise RouterBackboneError(
                f"visual pixel/patch alignment failed: {exc}"
            ) from exc

    def audit_state(self) -> dict[str, Any]:
        """Return checkpoint and freeze facts for the Stage 5 audit."""

        return {
            "config": self.config.to_dict(),
            "checkpoint": self.checkpoint_record,
            "frozen": self.is_frozen,
            "eval_mode": self.is_eval,
            "spatial_transform_fingerprint": self.spatial_transform_fingerprint,
        }


def canonical_aligned_pixels(
    transformed: Any,
    *,
    mean: Sequence[float],
    std: Sequence[float],
) -> Any:
    """Convert a normalized [C,H,W] transform result into aligned RGB [H,W,C]."""

    try:
        import numpy as np
    except ImportError as exc:
        raise Stage5DependencyError(
            "patch-aligned pixel extraction requires NumPy"
        ) from exc
    value = transformed
    for method in ("detach", "cpu"):
        if hasattr(value, method):
            value = getattr(value, method)()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] not in (1, 3):
        raise RouterBackboneError(
            f"transformed image must have shape [C,H,W], observed {array.shape}"
        )
    mean_values = np.asarray(tuple(mean), dtype=np.float64)
    std_values = np.asarray(tuple(std), dtype=np.float64)
    if mean_values.shape != (array.shape[0],) or std_values.shape != (
        array.shape[0],
    ):
        raise RouterBackboneError("transform mean/std channel counts disagree")
    if not np.isfinite(mean_values).all() or not np.isfinite(std_values).all():
        raise RouterBackboneError("transform mean/std must be finite")
    if np.any(std_values <= 0.0) or not np.isfinite(array).all():
        raise RouterBackboneError("transformed pixels/std must be finite and positive")
    pixels = array * std_values[:, None, None] + mean_values[:, None, None]
    return np.ascontiguousarray(
        np.moveaxis(np.clip(pixels, 0.0, 1.0), 0, -1),
        dtype=np.float32,
    )


def _positive_grid_shape(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise RouterBackboneError(
            "patch_grid_shape must contain two positive integers"
        )
    return value


def _validate_local_path(value: Any) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise RouterBackboneConfigError("checkpoint_path must be a non-empty local path")
    raw = str(value).strip()
    lowered = raw.casefold()
    if _REMOTE_PATH_PATTERN.match(raw) or lowered.startswith(
        ("hf_hub:", "s3:", "gs:", "ftp:")
    ):
        raise RouterBackboneConfigError(
            "checkpoint_path must be a local filesystem path; remote references are forbidden"
        )
    return Path(raw)


def _model_parameters(model: Any) -> Any:
    if not hasattr(model, "parameters"):
        raise RouterBackboneError("visual backbone does not implement parameters()")
    return model.parameters()


def _load_stage5_dependencies() -> tuple[Any, Any]:
    try:
        import timm
        import torch
    except ImportError as exc:
        raise Stage5DependencyError(
            "Stage 5 visual dependencies are missing; install the local project with "
            "the 'stage5' optional dependency group"
        ) from exc
    return torch, timm


def _load_checkpoint_state_dict(torch: Any, checkpoint_path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as exc:
        raise Stage5DependencyError(
            "torch>=2.2 with weights_only checkpoint loading is required"
        ) from exc
    except Exception as exc:
        raise RouterBackboneError(f"could not load local checkpoint {checkpoint_path}: {exc}") from exc

    state_dict = _extract_state_dict(payload)
    normalized: dict[str, Any] = {}
    for raw_key, tensor in state_dict.items():
        key = str(raw_key)
        prefixes = (
            "module.",
            "teacher.backbone.",
            "student.backbone.",
            "backbone.",
        )
        while True:
            prefix = next((item for item in prefixes if key.startswith(item)), None)
            if prefix is None:
                break
            key = key[len(prefix) :]
        if key in normalized:
            raise RouterBackboneError(f"checkpoint contains duplicate normalized key {key!r}")
        normalized[key] = tensor
    if not normalized:
        raise RouterBackboneError("checkpoint state dict is empty")
    return normalized


def _extract_state_dict(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise RouterBackboneError("checkpoint must contain a state-dict mapping")
    for key in ("state_dict", "model", "teacher", "student"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    return payload


def _canonical_embeddings(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in ("x_norm_clstoken", "pre_logits", "features"):
            if key in value:
                value = value[key]
                break
        else:
            raise RouterBackboneError("visual backbone returned an unsupported mapping")
    if isinstance(value, (tuple, list)):
        if not value:
            raise RouterBackboneError("visual backbone returned no embeddings")
        value = value[0]
    if not hasattr(value, "ndim"):
        raise RouterBackboneError("visual backbone output is not a tensor")
    if value.ndim == 3:
        value = value[:, 0, :]
    elif value.ndim > 3:
        value = value.flatten(start_dim=2).mean(dim=2)
    if value.ndim != 2:
        raise RouterBackboneError(
            f"visual backbone must return one feature vector per image; shape={tuple(value.shape)}"
        )
    return value


def _canonical_global_patch_embeddings(value: Any) -> tuple[Any, Any]:
    """Normalize common timm/DINO forward-feature formats to [B,D]/[B,P,D]."""

    if isinstance(value, Mapping):
        global_value = None
        patch_value = None
        for key in ("x_norm_clstoken", "cls_token", "global_features", "pre_logits"):
            if key in value:
                global_value = value[key]
                break
        for key in ("x_norm_patchtokens", "patch_tokens", "patch_features", "features"):
            if key in value:
                patch_value = value[key]
                break
        if global_value is None or patch_value is None:
            raise RouterBackboneError(
                "visual backbone forward_features mapping lacks global or patch tokens"
            )
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        global_value, patch_value = value
    else:
        if not hasattr(value, "ndim"):
            raise RouterBackboneError("visual backbone patch output is not a tensor")
        if value.ndim == 3 and value.shape[1] >= 2:
            global_value = value[:, 0, :]
            patch_value = value[:, 1:, :]
        elif value.ndim == 4:
            # timm CNN-style [B,C,H,W] output: spatial mean is the global feature.
            patch_value = value.flatten(start_dim=2).transpose(1, 2)
            global_value = patch_value.mean(dim=1)
        else:
            raise RouterBackboneError(
                "visual backbone must return token [B,T,D] or map [B,C,H,W] features"
            )

    if not hasattr(global_value, "ndim") or not hasattr(patch_value, "ndim"):
        raise RouterBackboneError("visual backbone global/patch outputs are not tensors")
    if global_value.ndim == 3 and global_value.shape[1] == 1:
        global_value = global_value[:, 0, :]
    if patch_value.ndim == 4:
        patch_value = patch_value.flatten(start_dim=2).transpose(1, 2)
    if global_value.ndim != 2 or patch_value.ndim != 3:
        raise RouterBackboneError(
            "visual backbone global/patch output must have shapes [B,D] and [B,P,D]"
        )
    if global_value.shape[0] != patch_value.shape[0]:
        raise RouterBackboneError("global and patch feature batch sizes disagree")
    if global_value.shape[-1] != patch_value.shape[-1] or patch_value.shape[1] == 0:
        raise RouterBackboneError("global and patch feature dimensions disagree")
    return global_value, patch_value


def _load_yaml_mapping(text: str) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError:
        return _parse_flat_yaml(text)
    try:
        payload = yaml.safe_load(text)
    except Exception as exc:
        raise RouterBackboneConfigError(f"invalid YAML router backbone config: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RouterBackboneConfigError("router backbone YAML root must be a mapping")
    return payload


def _parse_flat_yaml(text: str) -> Mapping[str, Any]:
    """Parse the intentionally flat checked-in config without requiring PyYAML."""

    result: dict[str, Any] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        content = raw_line.split("#", 1)[0].strip()
        if not content:
            continue
        if raw_line[: len(raw_line) - len(raw_line.lstrip())]:
            raise RouterBackboneConfigError(
                f"fallback YAML parser only accepts a flat mapping (line {line_number})"
            )
        if ":" not in content:
            raise RouterBackboneConfigError(f"invalid YAML entry at line {line_number}")
        key, raw_value = (item.strip() for item in content.split(":", 1))
        if not key or not raw_value or key in result:
            raise RouterBackboneConfigError(f"invalid YAML key at line {line_number}")
        if raw_value[:1] in {"'", '"'} and raw_value[-1:] == raw_value[:1]:
            value: Any = raw_value[1:-1]
        elif raw_value.casefold() in {"true", "false"}:
            value = raw_value.casefold() == "true"
        else:
            try:
                value = int(raw_value)
            except ValueError:
                value = raw_value
        result[key] = value
    if not result:
        raise RouterBackboneConfigError("router backbone YAML config is empty")
    return result
