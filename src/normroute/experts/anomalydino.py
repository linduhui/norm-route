"""AnomalyDINO-style expert for Stage 2 few-normal-shot smoke runs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any

from PIL import Image, ImageFilter, ImageStat

from src.normroute.experts.base import (
    Expert,
    ExpertInput,
    ExpertPrediction,
    validate_expert_input_fields,
)
from src.normroute.experts.winclip import _parse_simple_yaml


@dataclass(frozen=True)
class AnomalyDINOConfig:
    """Support-guided token parameters with no external weights or downloads."""

    image_size: int = 64
    token_grid_size: int = 8
    anomaly_threshold: float = 0.5
    image_score_top_fraction: float = 0.10

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnomalyDINOConfig":
        anomalydino = dict(payload.get("anomalydino", payload))
        return cls(
            image_size=int(anomalydino.get("image_size", cls.image_size)),
            token_grid_size=int(anomalydino.get("token_grid_size", cls.token_grid_size)),
            anomaly_threshold=float(
                anomalydino.get("anomaly_threshold", cls.anomaly_threshold)
            ),
            image_score_top_fraction=float(
                anomalydino.get("image_score_top_fraction", cls.image_score_top_fraction)
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnomalyDINOConfig":
        return cls.from_dict(_parse_simple_yaml(Path(path).read_text(encoding="utf-8")))


class AnomalyDINOExpert(Expert):
    """A no-download AnomalyDINO wrapper using support token nearest neighbors.

    This wrapper exercises the same Stage 2 contract as the real baseline would:
    it fits only on train/good support images, scores query images, and stores
    optional pixel-level scores outside predictions.csv.
    """

    name = "anomalydino"

    def __init__(
        self,
        *,
        output_dir: str | Path,
        config: AnomalyDINOConfig | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config or AnomalyDINOConfig()
        self._token_bank: list[tuple[float, ...]] = []
        self._reference_scale = 1.0

    def fit(self, inputs: list[ExpertInput]) -> None:
        for expert_input in inputs:
            validate_expert_input_fields(expert_input.__dict__)

        support_paths = _unique_support_paths(inputs)
        if not support_paths:
            raise ValueError("AnomalyDINOExpert requires at least one normal support image")

        token_bank: list[tuple[float, ...]] = []
        for support_path in support_paths:
            _validate_support_path(support_path)
            token_bank.extend(self._extract_tokens(Path(support_path)))

        if not token_bank:
            raise ValueError("AnomalyDINOExpert could not extract support tokens")

        self._token_bank = token_bank
        self._reference_scale = _mean_pairwise_step(token_bank)

    def predict(self, expert_input: ExpertInput) -> ExpertPrediction:
        validate_expert_input_fields(expert_input.__dict__)
        if not self._token_bank:
            raise RuntimeError("AnomalyDINOExpert.fit must be called before predict")

        start = time.perf_counter()
        query_path = Path(expert_input.query_path)
        if not query_path.is_file():
            raise FileNotFoundError(f"Query image does not exist: {query_path}")

        query_tokens = self._extract_tokens(query_path)
        if not query_tokens:
            raise ValueError(f"Could not extract query tokens: {query_path}")

        token_scores = [
            _nearest_distance(token, self._token_bank) / self._reference_scale
            for token in query_tokens
        ]
        normalized_scores = [score / (score + 1.0) for score in token_scores]
        final_score = _top_fraction_mean(
            normalized_scores,
            fraction=self.config.image_score_top_fraction,
        )
        anomaly_map_path = self._write_anomaly_map(expert_input.image_id, normalized_scores)
        pixel_score_path = self._write_pixel_scores(expert_input.image_id, normalized_scores)
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
            pixel_score_path=str(pixel_score_path),
            actions=(
                "ANOMALYDINO_TOKEN_NN "
                f"image_size={self.config.image_size} "
                f"token_grid_size={self.config.token_grid_size} "
                f"image_score_top_fraction={self.config.image_score_top_fraction}"
            ),
            tool_calls=1,
            runtime_ms=runtime_ms,
            status="ok",
            error_message="",
        )

    def _extract_tokens(self, image_path: Path) -> list[tuple[float, ...]]:
        image = Image.open(image_path).convert("RGB").resize(
            (self.config.image_size, self.config.image_size)
        )
        edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
        cell_size = max(1, self.config.image_size // self.config.token_grid_size)
        tokens: list[tuple[float, ...]] = []
        for top in range(0, self.config.image_size, cell_size):
            for left in range(0, self.config.image_size, cell_size):
                box = (left, top, left + cell_size, top + cell_size)
                cell = image.crop(box)
                edge_cell = edges.crop(box)
                stat = ImageStat.Stat(cell)
                edge_stat = ImageStat.Stat(edge_cell)
                mean = tuple(value / 255.0 for value in stat.mean)
                stddev = tuple(value / 255.0 for value in stat.stddev)
                edge_mean = edge_stat.mean[0] / 255.0
                tokens.append(
                    (
                        *mean,
                        *stddev,
                        edge_mean,
                        left / max(1, self.config.image_size - cell_size),
                        top / max(1, self.config.image_size - cell_size),
                    )
                )
        return tokens

    def _write_anomaly_map(self, image_id: str, scores: list[float]) -> Path:
        maps_dir = self.output_dir / "anomaly_maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        cells_per_side = int(math.sqrt(len(scores)))
        map_image = Image.new("L", (cells_per_side, cells_per_side))
        map_image.putdata([int(255.0 * max(0.0, min(1.0, score))) for score in scores])
        map_image = map_image.resize(
            (self.config.image_size, self.config.image_size),
            resample=Image.Resampling.BILINEAR,
        )
        output_path = maps_dir / f"{_safe_stem(image_id)}.png"
        map_image.save(output_path)
        return output_path

    def _write_pixel_scores(self, image_id: str, scores: list[float]) -> Path:
        scores_dir = self.output_dir / "pixel_scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        output_path = scores_dir / f"{_safe_stem(image_id)}.csv"
        cell_size = max(1, self.config.image_size // self.config.token_grid_size)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["x", "y", "score"])
            writer.writeheader()
            index = 0
            for top in range(0, self.config.image_size, cell_size):
                for left in range(0, self.config.image_size, cell_size):
                    writer.writerow(
                        {
                            "x": left,
                            "y": top,
                            "score": max(0.0, min(1.0, scores[index])),
                        }
                    )
                    index += 1
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
    return math.sqrt(
        sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right))
    )


def _mean_pairwise_step(memory_bank: list[tuple[float, ...]]) -> float:
    if len(memory_bank) < 2:
        return 1.0

    distances = [
        _euclidean(memory_bank[index], memory_bank[index - 1])
        for index in range(1, min(len(memory_bank), 64))
    ]
    mean_distance = sum(distances) / len(distances)
    return max(mean_distance, 1e-6)


def _top_fraction_mean(scores: list[float], *, fraction: float) -> float:
    if not scores:
        return 0.0
    count = max(1, math.ceil(len(scores) * max(0.0, min(1.0, fraction))))
    top_scores = sorted(scores, reverse=True)[:count]
    return sum(top_scores) / len(top_scores)


def _safe_stem(image_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in image_id)
    return safe or "image"
