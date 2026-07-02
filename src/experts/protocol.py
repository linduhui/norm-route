"""Protocol implemented by visual anomaly inspection experts."""

from __future__ import annotations

from typing import Protocol

from src.experts.results import ExpertResult


class ExpertProtocol(Protocol):
    """Unified interface for all NORM-Route visual experts."""

    def fit_support(self, support_paths: list[str], support_set_id: str, seed: int) -> None:
        """Prepare the expert with official train/good support images."""

    def predict(self, image_path: str, image_id: str, output_dir: str) -> ExpertResult:
        """Score one query image and return a validated result."""
