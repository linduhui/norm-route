"""WinCLIP-style expert for Stage 2 few-normal-shot smoke runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import time
from typing import Any

from PIL import Image, ImageStat

from src.normroute.experts.base import (
    Expert,
    ExpertInput,
    ExpertPrediction,
    validate_expert_input_fields,
)


@dataclass(frozen=True)
class WinCLIPConfig:
    """Prompt-guided WinCLIP parameters with no external weights or downloads."""

    image_size: int = 64
    grid_size: int = 8
    anomaly_threshold: float = 0.5
    class_name_map: dict[str, str] = field(default_factory=dict)
    normal_prompt_templates: tuple[str, ...] = (
        "a photo of a normal {class_name}",
        "a photo of an intact {class_name}",
    )
    anomaly_prompt_templates: tuple[str, ...] = (
        "a photo of an anomalous {class_name}",
        "a photo of a damaged {class_name}",
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WinCLIPConfig":
        winclip = dict(payload.get("winclip", payload))
        return cls(
            image_size=int(winclip.get("image_size", cls.image_size)),
            grid_size=int(winclip.get("grid_size", cls.grid_size)),
            anomaly_threshold=float(winclip.get("anomaly_threshold", cls.anomaly_threshold)),
            class_name_map={
                str(key): str(value)
                for key, value in dict(winclip.get("class_name_map", {})).items()
            },
            normal_prompt_templates=tuple(
                str(item)
                for item in winclip.get(
                    "normal_prompt_templates", cls.normal_prompt_templates
                )
            ),
            anomaly_prompt_templates=tuple(
                str(item)
                for item in winclip.get(
                    "anomaly_prompt_templates", cls.anomaly_prompt_templates
                )
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WinCLIPConfig":
        return cls.from_dict(_parse_simple_yaml(Path(path).read_text(encoding="utf-8")))


class WinCLIPExpert(Expert):
    """A no-download WinCLIP wrapper using class prompts and normal support images.

    The wrapper intentionally does not call an external baseline or fetch CLIP
    weights. It exercises the WinCLIP Stage 2 contract with category-name prompts,
    support-only visual references, and the shared no-leakage expert interface.
    """

    name = "winclip"

    def __init__(
        self,
        *,
        output_dir: str | Path,
        config: WinCLIPConfig | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config or WinCLIPConfig()
        self._support_features: list[tuple[float, ...]] = []
        self._support_centroid: tuple[float, ...] = ()
        self._reference_scale = 1.0

    def fit(self, inputs: list[ExpertInput]) -> None:
        for expert_input in inputs:
            validate_expert_input_fields(expert_input.__dict__)

        support_paths = _unique_support_paths(inputs)
        if not support_paths:
            raise ValueError("WinCLIPExpert requires at least one normal support image")

        features: list[tuple[float, ...]] = []
        for support_path in support_paths:
            _validate_support_path(support_path)
            features.append(self._extract_global_features(Path(support_path)))

        self._support_features = features
        self._support_centroid = _centroid(features)
        self._reference_scale = _mean_centroid_distance(features, self._support_centroid)

    def predict(self, expert_input: ExpertInput) -> ExpertPrediction:
        validate_expert_input_fields(expert_input.__dict__)
        if not self._support_centroid:
            raise RuntimeError("WinCLIPExpert.fit must be called before predict")

        start = time.perf_counter()
        query_path = Path(expert_input.query_path)
        if not query_path.is_file():
            raise FileNotFoundError(f"Query image does not exist: {query_path}")

        class_name = self._class_name(expert_input.category)
        normal_prompts = self._render_prompts(self.config.normal_prompt_templates, class_name)
        anomaly_prompts = self._render_prompts(self.config.anomaly_prompt_templates, class_name)
        query_features = self._extract_global_features(query_path)
        visual_distance = _euclidean(query_features, self._support_centroid)
        visual_score = visual_distance / (visual_distance + self._reference_scale)
        prompt_score = _prompt_contrast(query_features, normal_prompts, anomaly_prompts)
        final_score = max(0.0, min(1.0, 0.85 * visual_score + 0.15 * prompt_score))
        cell_scores = self._cell_scores(query_path, prompt_score)
        anomaly_map_path = self._write_anomaly_map(expert_input.image_id, cell_scores)
        runtime_ms = (time.perf_counter() - start) * 1000.0

        return ExpertPrediction(
            image_id=expert_input.image_id,
            expert_name=self.name,
            dataset=expert_input.dataset,
            category=expert_input.category,
            support_set_id=expert_input.support_set_id,
            k_shot=expert_input.k_shot,
            seed=expert_input.seed,
            final_score=final_score,
            final_decision="anomaly"
            if final_score >= self.config.anomaly_threshold
            else "normal",
            anomaly_map_path=str(anomaly_map_path),
            actions=(
                "WINCLIP_PROMPT_SIM "
                f"image_size={self.config.image_size} "
                f"grid_size={self.config.grid_size} "
                f"class_name={class_name} "
                f"normal_prompts={len(normal_prompts)} "
                f"anomaly_prompts={len(anomaly_prompts)}"
            ),
            tool_calls=1,
            runtime_ms=runtime_ms,
            status="ok",
            error_message="",
        )

    def _class_name(self, category: str) -> str:
        return self.config.class_name_map.get(category, category.replace("_", " "))

    def _render_prompts(self, templates: tuple[str, ...], class_name: str) -> tuple[str, ...]:
        return tuple(template.format(class_name=class_name) for template in templates)

    def _extract_global_features(self, image_path: Path) -> tuple[float, ...]:
        image = Image.open(image_path).convert("RGB").resize(
            (self.config.image_size, self.config.image_size)
        )
        stat = ImageStat.Stat(image)
        mean = tuple(value / 255.0 for value in stat.mean)
        stddev = tuple(value / 255.0 for value in stat.stddev)
        return (*mean, *stddev)

    def _cell_scores(self, image_path: Path, prompt_score: float) -> list[float]:
        image = Image.open(image_path).convert("RGB").resize(
            (self.config.image_size, self.config.image_size)
        )
        cell_size = max(1, self.config.image_size // self.config.grid_size)
        scores: list[float] = []
        for top in range(0, self.config.image_size, cell_size):
            for left in range(0, self.config.image_size, cell_size):
                cell = image.crop((left, top, left + cell_size, top + cell_size))
                stat = ImageStat.Stat(cell)
                cell_feature = tuple(value / 255.0 for value in (*stat.mean, *stat.stddev))
                distance = _euclidean(cell_feature, self._support_centroid)
                visual_score = distance / (distance + self._reference_scale)
                scores.append(max(0.0, min(1.0, 0.85 * visual_score + 0.15 * prompt_score)))
        return scores

    def _write_anomaly_map(self, image_id: str, cell_scores: list[float]) -> Path:
        maps_dir = self.output_dir / "anomaly_maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        cells_per_side = int(math.sqrt(len(cell_scores)))
        map_image = Image.new("L", (cells_per_side, cells_per_side))
        map_image.putdata([int(255.0 * score) for score in cell_scores])
        map_image = map_image.resize(
            (self.config.image_size, self.config.image_size),
            resample=Image.Resampling.BILINEAR,
        )
        output_path = maps_dir / f"{_safe_stem(image_id)}.png"
        map_image.save(output_path)
        return output_path


def _unique_support_paths(inputs: list[ExpertInput]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for expert_input in inputs:
        for support_path in expert_input.support_paths:
            if support_path not in seen:
                paths.append(support_path)
                seen.add(support_path)
    return paths


def _validate_support_path(support_path: str) -> None:
    path = Path(support_path)
    if not path.is_file():
        raise FileNotFoundError(f"Support image does not exist: {path}")

    parts = {part.lower() for part in path.parts}
    if "train" not in parts or "good" not in parts:
        raise ValueError(f"Support image must come from train/good: {path}")


def _centroid(features: list[tuple[float, ...]]) -> tuple[float, ...]:
    width = len(features[0])
    return tuple(sum(feature[index] for feature in features) / len(features) for index in range(width))


def _mean_centroid_distance(
    features: list[tuple[float, ...]], centroid: tuple[float, ...]
) -> float:
    distances = [_euclidean(feature, centroid) for feature in features]
    mean_distance = sum(distances) / len(distances)
    return max(mean_distance, 1e-6)


def _prompt_contrast(
    image_features: tuple[float, ...],
    normal_prompts: tuple[str, ...],
    anomaly_prompts: tuple[str, ...],
) -> float:
    normal_similarity = _mean_prompt_similarity(image_features, normal_prompts)
    anomaly_similarity = _mean_prompt_similarity(image_features, anomaly_prompts)
    return 1.0 / (1.0 + math.exp(normal_similarity - anomaly_similarity))


def _mean_prompt_similarity(image_features: tuple[float, ...], prompts: tuple[str, ...]) -> float:
    if not prompts:
        return 0.0
    return sum(_prompt_similarity(image_features, prompt) for prompt in prompts) / len(prompts)


def _prompt_similarity(image_features: tuple[float, ...], prompt: str) -> float:
    text_features = _text_features(prompt, len(image_features))
    return sum(left * right for left, right in zip(image_features, text_features))


def _text_features(text: str, length: int) -> tuple[float, ...]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return tuple((digest[index % len(digest)] / 255.0) for index in range(length))


def _euclidean(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right)))


def _safe_stem(image_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in image_id)
    return safe or "image"


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    current_list_key: tuple[int, dict[str, Any], str] | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if stripped.startswith("- "):
            if current_list_key is None:
                raise ValueError("YAML list item without a parent key")
            list_indent, parent, key = current_list_key
            if indent <= list_indent:
                raise ValueError(f"YAML list item has invalid indentation: {raw_line}")
            if parent[key] == {}:
                parent[key] = []
            if not isinstance(parent[key], list):
                raise ValueError(f"YAML key is not a list: {key}")
            parent[key].append(_parse_scalar(stripped[2:].strip()))
            continue

        key, separator, value = stripped.partition(":")
        if not separator:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _parse_scalar(value)
            current_list_key = None
        else:
            next_container: dict[str, Any] = {}
            parent[key] = next_container
            stack.append((indent, next_container))
            current_list_key = (indent, parent, key)

    _collapse_empty_list_containers(root)
    return root


def _collapse_empty_list_containers(value: Any) -> None:
    if not isinstance(value, dict):
        return
    for key, child in list(value.items()):
        if child == {}:
            value[key] = []
        else:
            _collapse_empty_list_containers(child)


def _parse_scalar(value: str) -> Any:
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
