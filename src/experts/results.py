"""Result schema shared by all visual experts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any


FORBIDDEN_EXPERT_RESULT_FIELDS = frozenset({"label", "mask_path", "defect_type"})


class ExpertResultValidationError(ValueError):
    """Raised when an expert result violates the public schema."""


@dataclass(frozen=True)
class ExpertResult:
    """Serializable output returned by every visual expert."""

    schema_version: str
    expert_name: str
    image_id: str
    support_set_id: str
    raw_score: float
    normalized_score: float
    anomaly_map_path: str
    runtime_ms: float
    peak_memory_mb: float
    model_version: str
    weight_hash: str
    status: str
    error_message: str

    @classmethod
    def field_names(cls) -> set[str]:
        return {field.name for field in fields(cls)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExpertResult":
        forbidden = FORBIDDEN_EXPERT_RESULT_FIELDS.intersection(payload)
        if forbidden:
            joined = ", ".join(sorted(forbidden))
            raise ExpertResultValidationError(f"Forbidden ExpertResult fields: {joined}")

        required = cls.field_names()
        missing = required.difference(payload)
        if missing:
            joined = ", ".join(sorted(missing))
            raise ExpertResultValidationError(f"Missing ExpertResult fields: {joined}")

        unknown = set(payload).difference(required)
        if unknown:
            joined = ", ".join(sorted(unknown))
            raise ExpertResultValidationError(f"Unknown ExpertResult fields: {joined}")

        return cls(**{name: payload[name] for name in required})

    @classmethod
    def from_jsonl_line(cls, line: str) -> "ExpertResult":
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ExpertResultValidationError("ExpertResult JSONL line must contain an object")
        return cls.from_dict(payload)

    def validate(self) -> None:
        ExpertResult.from_dict(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"

    def append_jsonl(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(self.to_jsonl_line())
