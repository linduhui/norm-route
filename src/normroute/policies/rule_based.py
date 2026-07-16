"""Fold-calibrated, leakage-safe Stage 4 rule policies.

The policies in this module never fit from replay tasks.  They load a compact
artifact produced from one fold's training runs by ``calibrate_policy`` and
route using only the Stage 4 policy feature allowlist: dataset, category, and
k-shot.
"""

from __future__ import annotations

import json
from pathlib import Path
import math
from typing import Any, Mapping, Sequence

from ..agent.policy import Policy, TrainRecord
from ..agent.protocol import AgentTask, CANDIDATE_EXPERTS, RouteDecision


RULE_POLICY_PROTOCOL_VERSION = "stage4.rule_policy.v1"
CATEGORY_PRIOR = "category_prior"
CATEGORY_SHOT_PRIOR = "category_shot_prior"
RULE_POLICY_NAMES = (CATEGORY_PRIOR, CATEGORY_SHOT_PRIOR)
SUPPORTED_METRICS = ("image_auroc", "image_ap")
RUNTIME_TIE_BREAK = "lower_mean_training_runtime_ms"

_ARTIFACT_FIELDS = frozenset(
    {
        "protocol_version",
        "policy_name",
        "fold",
        "train_seeds",
        "metric",
        "tie_break",
        "tie_break_details",
        "git_commit",
        "global_best",
        "category_rules",
        "category_shot_rules",
        "fallback_order",
        "num_training_runs",
        "num_training_rows",
        "training_rows_sha256",
        "training_manifest_runs_sha256",
    }
)
_FORBIDDEN_ARTIFACT_KEYS = frozenset(
    {
        "label",
        "labels",
        "mask",
        "mask_path",
        "defect_type",
        "anomaly_type",
        "ground_truth",
        "oracle_best_expert",
        "test_label",
        "test_labels",
        "test_quality",
    }
)


class RulePolicyArtifactError(ValueError):
    """Raised when a rule policy artifact is malformed or unsafe."""


class _RuleBasedPolicy(Policy):
    """Shared artifact loading, persistence, and fallback behavior."""

    name = ""

    def __init__(self, artifact: Mapping[str, Any] | str | Path | None = None) -> None:
        self._artifact: dict[str, Any] | None = None
        self._category_rules: dict[tuple[str, str], str] = {}
        self._category_shot_rules: dict[tuple[str, str, int], str] = {}
        if artifact is not None:
            payload = load_rule_policy_artifact(artifact)
            if payload["policy_name"] != self.name:
                raise RulePolicyArtifactError(
                    f"Artifact policy_name={payload['policy_name']!r} cannot configure {self.name!r}"
                )
            self._install(payload)

    def fit(self, train_records: Sequence[TrainRecord]) -> None:
        # Calibration is intentionally a separate evaluator-side step.  Replay
        # passes only allowlisted task features here; they cannot alter the
        # already frozen fold artifact.
        del train_records
        if self._artifact is None:
            raise RulePolicyArtifactError(
                f"{self.name} requires a fold-specific policy_artifact.json"
            )

    def select(self, task: AgentTask) -> RouteDecision:
        artifact = self._require_artifact()
        features = task.policy_features
        selected, level = self._select_from_features(features)
        if selected not in task.candidate_experts:
            raise RulePolicyArtifactError(
                f"Artifact selected {selected!r}, which is not allowed by this task"
            )
        return RouteDecision(
            task_id=task.task_id,
            dataset=task.dataset,
            category=task.category,
            k_shot=task.k_shot,
            seed=task.seed,
            support_set_id=task.support_set_id,
            policy_name=self.name,
            selected_expert=selected,
            decision_reason=(
                f"Fold {artifact['fold']} {self.name} matched {level}; "
                f"calibrated on train seeds {artifact['train_seeds']}."
            ),
            estimated_cost_ms=0.0,
            tool_calls=1,
        )

    def configuration(self) -> dict[str, Any]:
        artifact = self._require_artifact()
        return {
            "fold": artifact["fold"],
            "train_seeds": list(artifact["train_seeds"]),
            "metric": artifact["metric"],
            "tie_break": artifact["tie_break"],
            "artifact_protocol_version": artifact["protocol_version"],
        }

    def save(self, path: str | Path) -> Path:
        destination = _artifact_path(path, for_write=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(
                    self._require_artifact(),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "_RuleBasedPolicy":
        return cls(path)

    def _install(self, payload: dict[str, Any]) -> None:
        self._artifact = payload
        self._category_rules = {
            (row["dataset"], row["category"]): row["selected_expert"]
            for row in payload["category_rules"]
        }
        self._category_shot_rules = {
            (row["dataset"], row["category"], row["k_shot"]): row["selected_expert"]
            for row in payload["category_shot_rules"]
        }

    def _require_artifact(self) -> dict[str, Any]:
        if self._artifact is None:
            raise RulePolicyArtifactError(
                f"{self.name} requires a fold-specific policy_artifact.json"
            )
        return self._artifact

    def _select_from_features(self, features: Mapping[str, Any]) -> tuple[str, str]:
        raise NotImplementedError


class CategoryPriorPolicy(_RuleBasedPolicy):
    """Use a train-fold category rule, then the train-fold global best."""

    name = CATEGORY_PRIOR

    def _select_from_features(self, features: Mapping[str, Any]) -> tuple[str, str]:
        key = (str(features["dataset"]), str(features["category"]))
        selected = self._category_rules.get(key)
        if selected is not None:
            return selected, "category"
        return str(self._require_artifact()["global_best"]), "global best fallback"


class CategoryShotPriorPolicy(_RuleBasedPolicy):
    """Use category+shot, then category, then the train-fold global best."""

    name = CATEGORY_SHOT_PRIOR

    def _select_from_features(self, features: Mapping[str, Any]) -> tuple[str, str]:
        dataset = str(features["dataset"])
        category = str(features["category"])
        k_shot = int(features["k_shot"])
        selected = self._category_shot_rules.get((dataset, category, k_shot))
        if selected is not None:
            return selected, "category+shot"
        selected = self._category_rules.get((dataset, category))
        if selected is not None:
            return selected, "category fallback"
        return str(self._require_artifact()["global_best"]), "global best fallback"


def load_rule_policy_artifact(
    source: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Load and strictly validate a fold-specific rule policy artifact."""

    if isinstance(source, Mapping):
        payload: Any = dict(source)
        context = "rule policy artifact"
    else:
        path = _artifact_path(source, for_write=False)
        context = str(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RulePolicyArtifactError(f"Could not read {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RulePolicyArtifactError(f"{context} must be a JSON object")
    _reject_forbidden_artifact_keys(payload, context=context)
    missing = sorted(_ARTIFACT_FIELDS - set(payload))
    extra = sorted(set(payload) - _ARTIFACT_FIELDS)
    if missing or extra:
        raise RulePolicyArtifactError(
            f"{context} has invalid fields; missing={missing}, extra={extra}"
        )
    if payload["protocol_version"] != RULE_POLICY_PROTOCOL_VERSION:
        raise RulePolicyArtifactError(
            f"{context} protocol_version must be {RULE_POLICY_PROTOCOL_VERSION!r}"
        )
    policy_name = _nonempty_string(payload["policy_name"], "policy_name", context)
    if policy_name not in RULE_POLICY_NAMES:
        raise RulePolicyArtifactError(f"{context} has unknown policy_name={policy_name!r}")
    _nonempty_string(payload["fold"], "fold", context)
    _nonempty_string(payload["git_commit"], "git_commit", context)
    if payload["metric"] not in SUPPORTED_METRICS:
        raise RulePolicyArtifactError(
            f"{context} metric must be one of {list(SUPPORTED_METRICS)!r}"
        )
    if payload["tie_break"] != RUNTIME_TIE_BREAK:
        raise RulePolicyArtifactError(
            f"{context} tie_break must be {RUNTIME_TIE_BREAK!r}"
        )
    train_seeds = payload["train_seeds"]
    if (
        not isinstance(train_seeds, list)
        or not train_seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in train_seeds)
        or train_seeds != sorted(set(train_seeds))
    ):
        raise RulePolicyArtifactError(
            f"{context} train_seeds must be a sorted, unique, non-empty integer list"
        )
    _canonical_expert(payload["global_best"], context=f"{context}.global_best")
    _positive_int(payload["num_training_runs"], "num_training_runs", context)
    _positive_int(payload["num_training_rows"], "num_training_rows", context)
    for field in ("training_rows_sha256", "training_manifest_runs_sha256"):
        value = _nonempty_string(payload[field], field, context)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise RulePolicyArtifactError(f"{context} field {field!r} must be a SHA-256 hex digest")

    category_rules = _validate_rule_rows(payload["category_rules"], shot=False, context=context)
    shot_rules = _validate_rule_rows(payload["category_shot_rules"], shot=True, context=context)
    if not category_rules:
        raise RulePolicyArtifactError(f"{context} must contain at least one category rule")
    expected_fallback = (
        ["category", "global_best"]
        if policy_name == CATEGORY_PRIOR
        else ["category+shot", "category", "global_best"]
    )
    if payload["fallback_order"] != expected_fallback:
        raise RulePolicyArtifactError(
            f"{context} fallback_order must be {expected_fallback!r}"
        )
    if policy_name == CATEGORY_PRIOR and shot_rules:
        raise RulePolicyArtifactError(
            f"{context} category_prior must not contain category+shot rules"
        )
    if policy_name == CATEGORY_SHOT_PRIOR and not shot_rules:
        raise RulePolicyArtifactError(
            f"{context} category_shot_prior must contain category+shot rules"
        )
    _validate_tie_break_details(payload["tie_break_details"], context=context)
    return payload


def _validate_rule_rows(value: Any, *, shot: bool, context: str) -> set[tuple[Any, ...]]:
    label = "category_shot_rules" if shot else "category_rules"
    fields = {"dataset", "category", "selected_expert"}
    if shot:
        fields.add("k_shot")
    if not isinstance(value, list):
        raise RulePolicyArtifactError(f"{context} {label} must be a list")
    keys: set[tuple[Any, ...]] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != fields:
            raise RulePolicyArtifactError(
                f"{context} {label}[{index}] must contain exactly {sorted(fields)!r}"
            )
        dataset = _nonempty_string(row["dataset"], "dataset", context)
        category = _nonempty_string(row["category"], "category", context)
        expert = _canonical_expert(
            row["selected_expert"], context=f"{context}.{label}[{index}]"
        )
        key: tuple[Any, ...] = (dataset, category)
        if shot:
            key = (*key, _positive_int(row["k_shot"], "k_shot", context))
        if key in keys:
            raise RulePolicyArtifactError(f"{context} has duplicate {label} key {key!r}")
        keys.add(key)
        if row["selected_expert"] != expert:
            raise RulePolicyArtifactError(
                f"{context} must use canonical expert display name {expert!r}"
            )
    return keys


def _validate_tie_break_details(value: Any, *, context: str) -> None:
    expected_fields = {
        "primary",
        "secondary",
        "tertiary",
        "candidate_expert_order",
        "float_tolerance",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise RulePolicyArtifactError(
            f"{context} tie_break_details must contain exactly {sorted(expected_fields)!r}"
        )
    if value["primary"] != "maximize_mean_training_metric":
        raise RulePolicyArtifactError(f"{context} has invalid primary tie-break detail")
    if value["secondary"] != RUNTIME_TIE_BREAK:
        raise RulePolicyArtifactError(f"{context} has invalid secondary tie-break detail")
    if value["tertiary"] != "candidate_expert_order":
        raise RulePolicyArtifactError(f"{context} has invalid tertiary tie-break detail")
    if value["candidate_expert_order"] != list(CANDIDATE_EXPERTS):
        raise RulePolicyArtifactError(f"{context} has invalid candidate_expert_order")
    tolerance = value["float_tolerance"]
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or tolerance < 0
    ):
        raise RulePolicyArtifactError(f"{context} float_tolerance must be a number >= 0")


def _artifact_path(path: str | Path, *, for_write: bool) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        artifact = candidate / "policy_artifact.json"
        if not for_write and not artifact.is_file():
            fallback = candidate / "policy.json"
            return fallback if fallback.is_file() else artifact
        return artifact
    return candidate


def _reject_forbidden_artifact_keys(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(
            key for key in value if str(key).strip().lower() in _FORBIDDEN_ARTIFACT_KEYS
        )
        if forbidden:
            raise RulePolicyArtifactError(
                f"{context} contains forbidden evaluator/test fields: {forbidden}"
            )
        for child in value.values():
            _reject_forbidden_artifact_keys(child, context=context)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_artifact_keys(child, context=context)


def _canonical_expert(value: Any, *, context: str) -> str:
    normalized = "".join(character for character in str(value).lower() if character.isalnum())
    mapping = {
        "".join(character for character in expert.lower() if character.isalnum()): expert
        for expert in CANDIDATE_EXPERTS
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise RulePolicyArtifactError(f"{context} has unknown expert {value!r}") from exc


def _nonempty_string(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RulePolicyArtifactError(f"{context} field {field!r} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, field: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RulePolicyArtifactError(f"{context} field {field!r} must be an integer >= 1")
    return value


__all__ = [
    "CATEGORY_PRIOR",
    "CATEGORY_SHOT_PRIOR",
    "CategoryPriorPolicy",
    "CategoryShotPriorPolicy",
    "RULE_POLICY_NAMES",
    "RULE_POLICY_PROTOCOL_VERSION",
    "RUNTIME_TIE_BREAK",
    "RulePolicyArtifactError",
    "SUPPORTED_METRICS",
    "load_rule_policy_artifact",
]
