"""Abstract policy interface and the Stage 4 smoke policy."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol import AgentTask, RouteDecision


TrainRecord = AgentTask | Mapping[str, Any]


class Policy(ABC):
    """Common persistence and selection contract for Stage 4 policies."""

    name: str

    @abstractmethod
    def fit(self, train_records: Sequence[TrainRecord]) -> None:
        """Fit from records belonging only to the current fold's train split."""

    @abstractmethod
    def select(self, task: AgentTask) -> RouteDecision:
        """Select exactly one candidate expert for one leakage-safe task."""

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        """Persist policy state and return the written path."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "Policy":
        """Load policy state from a path written by :meth:`save`."""


class AlwaysAnomalyDINOPolicy(Policy):
    """Deterministic smoke policy that always selects AnomalyDINO."""

    name = "always_anomalydino"
    state_version = "stage4.policy.always_anomalydino.v1"

    def fit(self, train_records: Sequence[TrainRecord]) -> None:
        # Deliberately outcome-free. Materializing the sequence is unnecessary and
        # could accidentally encourage future code to inspect provenance fields.
        del train_records

    def select(self, task: AgentTask) -> RouteDecision:
        if "AnomalyDINO" not in task.candidate_experts:
            raise ValueError("AgentTask does not allow AnomalyDINO")
        return RouteDecision(
            task_id=task.task_id,
            dataset=task.dataset,
            category=task.category,
            k_shot=task.k_shot,
            seed=task.seed,
            support_set_id=task.support_set_id,
            policy_name=self.name,
            selected_expert="AnomalyDINO",
            decision_reason="Smoke policy always selects AnomalyDINO.",
            estimated_cost_ms=0.0,
            tool_calls=1,
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.is_dir():
            destination = destination / "policy.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy_name": self.name,
            "state_version": self.state_version,
        }
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "AlwaysAnomalyDINOPolicy":
        source = Path(path)
        if source.is_dir():
            source = source / "policy.json"
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        expected = {
            "policy_name": cls.name,
            "state_version": cls.state_version,
        }
        if payload != expected:
            raise ValueError(f"Invalid {cls.name} policy state in {source}")
        return cls()


AlwaysAnomalyDinoPolicy = AlwaysAnomalyDINOPolicy
"""Compatibility alias using conventional CamelCase capitalization."""
