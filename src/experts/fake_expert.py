"""Deterministic fake expert for pipeline tests only."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from src.experts.results import ExpertResult


class FakeExpert:
    """A no-model expert that produces stable fake scores and anomaly maps."""

    expert_name = "fake_expert"
    model_version = "fake-0"
    weight_hash = "no-weights"

    def __init__(self) -> None:
        self._support_paths: list[str] | None = None
        self._support_set_id: str | None = None
        self._seed: int | None = None

    def fit_support(self, support_paths: list[str], support_set_id: str, seed: int) -> None:
        self._support_paths = list(support_paths)
        self._support_set_id = support_set_id
        self._seed = seed

    def predict(self, image_path: str, image_id: str, output_dir: str) -> ExpertResult:
        if self._support_paths is None or self._support_set_id is None or self._seed is None:
            raise RuntimeError("FakeExpert.fit_support must be called before predict")

        Path(image_path)
        output_path = Path(output_dir)
        maps_dir = output_path / "anomaly_maps"
        maps_dir.mkdir(parents=True, exist_ok=True)

        digest = hashlib.sha256(f"{image_id}|{self._seed}".encode("utf-8")).digest()
        raw_score = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
        normalized_score = raw_score
        anomaly_map_path = maps_dir / f"{_safe_stem(image_id)}.png"
        _write_fake_map(anomaly_map_path, digest)

        return ExpertResult(
            schema_version="1.0",
            expert_name=self.expert_name,
            image_id=image_id,
            support_set_id=self._support_set_id,
            raw_score=raw_score,
            normalized_score=normalized_score,
            anomaly_map_path=str(anomaly_map_path),
            runtime_ms=0.0,
            peak_memory_mb=0.0,
            model_version=self.model_version,
            weight_hash=self.weight_hash,
            status="ok",
            error_message="",
        )


def _safe_stem(image_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in image_id)
    return safe or "image"


def _write_fake_map(path: Path, digest: bytes) -> None:
    width = 8
    height = 8
    pixels = []
    for index in range(width * height):
        value = digest[index % len(digest)]
        pixels.append((value, value, value))

    image = Image.new("RGB", (width, height))
    image.putdata(pixels)
    image.save(path)
