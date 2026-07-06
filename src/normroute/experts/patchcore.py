"""PatchCore-style expert for Stage 2 few-normal-shot smoke runs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time

from PIL import Image, ImageStat

from src.normroute.experts.base import (
    Expert,
    ExpertInput,
    ExpertPrediction,
    validate_expert_input_fields,
)


@dataclass(frozen=True)
class PatchCoreConfig:
    """Lightweight PatchCore parameters with no external weights or downloads."""

    image_size: int = 64
    patch_size: int = 16
    stride: int = 8
    anomaly_threshold: float = 0.5


class PatchCoreExpert(Expert):
    """A no-download PatchCore wrapper using support patch nearest-neighbor scores.

    This class intentionally keeps the Stage 2 contract separate from any external
    baseline implementation. It consumes only normal support paths and query paths.
    """

    name = "patchcore"

    def __init__(
        self,
        *,
        output_dir: str | Path,
        config: PatchCoreConfig | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config or PatchCoreConfig()
        self._memory_bank: list[tuple[float, ...]] = []
        self._reference_scale = 1.0

    def fit(self, inputs: list[ExpertInput]) -> None:
        for expert_input in inputs:
            validate_expert_input_fields(expert_input.__dict__)

        support_paths = _unique_support_paths(inputs)
        if not support_paths:
            raise ValueError("PatchCoreExpert requires at least one normal support image")

        memory_bank: list[tuple[float, ...]] = []
        for support_path in support_paths:
            _validate_support_path(support_path)
            memory_bank.extend(self._extract_features(Path(support_path)))

        if not memory_bank:
            raise ValueError("PatchCoreExpert could not extract support patch features")

        self._memory_bank = memory_bank
        self._reference_scale = _mean_pairwise_step(memory_bank)

    def predict(self, expert_input: ExpertInput) -> ExpertPrediction:
        validate_expert_input_fields(expert_input.__dict__)
        if not self._memory_bank:
            raise RuntimeError("PatchCoreExpert.fit must be called before predict")

        start = time.perf_counter()
        query_path = Path(expert_input.query_path)
        if not query_path.is_file():
            raise FileNotFoundError(f"Query image does not exist: {query_path}")

        query_features = self._extract_features(query_path)
        if not query_features:
            raise ValueError(f"Could not extract query patch features: {query_path}")

        patch_scores = [_nearest_distance(feature, self._memory_bank) for feature in query_features]
        raw_score = max(patch_scores)
        final_score = raw_score / (raw_score + self._reference_scale)
        anomaly_map_path = self._write_anomaly_map(expert_input.image_id, patch_scores)
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
            pixel_score_path="",
            actions=(
                "PATCHCORE_NN "
                f"image_size={self.config.image_size} "
                f"patch_size={self.config.patch_size} "
                f"stride={self.config.stride}"
            ),
            tool_calls=1,
            runtime_ms=runtime_ms,
            status="ok",
            error_message="",
        )

    def _extract_features(self, image_path: Path) -> list[tuple[float, ...]]:
        image = Image.open(image_path).convert("RGB").resize(
            (self.config.image_size, self.config.image_size)
        )
        features: list[tuple[float, ...]] = []
        limit = self.config.image_size - self.config.patch_size
        for top in range(0, limit + 1, self.config.stride):
            for left in range(0, limit + 1, self.config.stride):
                patch = image.crop(
                    (left, top, left + self.config.patch_size, top + self.config.patch_size)
                )
                stat = ImageStat.Stat(patch)
                mean = tuple(value / 255.0 for value in stat.mean)
                stddev = tuple(value / 255.0 for value in stat.stddev)
                features.append(
                    (
                        *mean,
                        *stddev,
                        left / max(1, limit),
                        top / max(1, limit),
                    )
                )
        return features

    def _write_anomaly_map(self, image_id: str, patch_scores: list[float]) -> Path:
        maps_dir = self.output_dir / "anomaly_maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        cells_per_side = int(math.sqrt(len(patch_scores)))
        normalized = [
            int(255.0 * score / (score + self._reference_scale)) for score in patch_scores
        ]
        map_image = Image.new("L", (cells_per_side, cells_per_side))
        map_image.putdata(normalized)
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


def _nearest_distance(feature: tuple[float, ...], memory_bank: list[tuple[float, ...]]) -> float:
    return min(_euclidean(feature, support_feature) for support_feature in memory_bank)


def _euclidean(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right)))


def _mean_pairwise_step(memory_bank: list[tuple[float, ...]]) -> float:
    if len(memory_bank) < 2:
        return 1.0

    distances = [
        _euclidean(memory_bank[index], memory_bank[index - 1])
        for index in range(1, min(len(memory_bank), 64))
    ]
    mean_distance = sum(distances) / len(distances)
    return max(mean_distance, 1e-6)


def _safe_stem(image_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in image_id)
    return safe or "image"
