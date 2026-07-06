"""Unified expert interface for Stage 2 baseline integrations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import time
from typing import Any, Protocol


FORBIDDEN_EXPERT_INPUT_FIELDS = frozenset(
    {"label", "mask", "mask_path", "defect_type", "anomaly_type", "test_statistics"}
)


class ExpertInputError(ValueError):
    """Raised when expert-visible input contains evaluator-only fields."""


def validate_expert_input_fields(payload: dict[str, Any], *, context: str = "expert input") -> None:
    """Reject evaluator-only fields before a baseline can consume the payload."""

    forbidden = sorted(FORBIDDEN_EXPERT_INPUT_FIELDS.intersection(payload))
    if forbidden:
        raise ExpertInputError(f"{context} contains forbidden fields: {forbidden}")


@dataclass(frozen=True)
class ExpertInput:
    """Agent-visible input for one target image."""

    image_id: str
    query_path: str
    dataset: str
    category: str
    support_set_id: str
    k_shot: int
    seed: int
    budget: int
    support_paths: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExpertInput":
        validate_expert_input_fields(payload)
        return cls(
            image_id=str(payload["image_id"]),
            query_path=str(payload["query_path"]),
            dataset=str(payload["dataset"]),
            category=str(payload["category"]),
            support_set_id=str(payload["support_set_id"]),
            k_shot=int(payload["k_shot"]),
            seed=int(payload["seed"]),
            budget=int(payload["budget"]),
            support_paths=tuple(str(path) for path in payload.get("support_paths", ())),
        )


@dataclass(frozen=True)
class ExpertPrediction:
    """Serializable Stage 2 prediction record."""

    image_id: str
    expert_name: str
    dataset: str
    category: str
    support_set_id: str
    k_shot: int
    seed: int
    final_score: float
    final_decision: str
    anomaly_map_path: str
    pixel_score_path: str
    actions: str
    tool_calls: int
    runtime_ms: float
    status: str
    error_message: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        validate_expert_input_fields(payload, context="expert prediction")
        return payload


class Expert(Protocol):
    """Common wrapper contract for real and smoke-test visual experts."""

    name: str

    def fit(self, inputs: list[ExpertInput]) -> None:
        """Prepare the expert from fixed train/good support images only."""

    def predict(self, expert_input: ExpertInput) -> ExpertPrediction:
        """Score one query image without evaluator-only ground truth."""


class DummyExpert:
    """Deterministic smoke expert used to exercise Stage 2 plumbing."""

    name = "dummy"

    def fit(self, inputs: list[ExpertInput]) -> None:
        for expert_input in inputs:
            validate_expert_input_fields(expert_input.__dict__)

    def predict(self, expert_input: ExpertInput) -> ExpertPrediction:
        validate_expert_input_fields(expert_input.__dict__)
        start = time.perf_counter()
        digest = hashlib.sha256(
            f"{expert_input.image_id}\n{expert_input.support_set_id}\n{expert_input.seed}".encode(
                "utf-8"
            )
        ).hexdigest()
        score = int(digest[:8], 16) / 0xFFFFFFFF
        runtime_ms = (time.perf_counter() - start) * 1000.0
        return ExpertPrediction(
            image_id=expert_input.image_id,
            expert_name=self.name,
            dataset=expert_input.dataset,
            category=expert_input.category,
            support_set_id=expert_input.support_set_id,
            k_shot=expert_input.k_shot,
            seed=expert_input.seed,
            final_score=score,
            final_decision="normal" if score < 0.5 else "anomaly",
            anomaly_map_path="",
            pixel_score_path="",
            actions="DUMMY_EXPERT",
            tool_calls=1,
            runtime_ms=runtime_ms,
            status="ok",
            error_message="",
        )
