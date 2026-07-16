"""Fixed, seeded-random, and runtime-only Stage 4 routing baselines.

No policy in this module consumes evaluator labels, masks, defect types, test
quality metrics, or realized test scores.  ``fastest_expert`` accepts runtime
only from records explicitly marked as the current fold's training split, or
from a predeclared expert cost card.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence

from ..agent.policy import AlwaysAnomalyDINOPolicy, Policy, TrainRecord
from ..agent.protocol import AgentTask, CANDIDATE_EXPERTS, RouteDecision


EXPERT_COST_CARD_VERSION = "stage4.expert_cost_card.v1"
TRAINING_RUNTIME_SOURCE = "current_fold_training_run"
_QUALITY_FIELD_TOKENS = (
    "label",
    "mask",
    "defect",
    "anomaly_type",
    "oracle",
    "ground_truth",
    "final_score",
    "final_decision",
    "auroc",
    "average_precision",
    "image_ap",
    "f1",
    "quality",
)
_QUALITY_FIELD_NAMES = frozenset(
    {
        "ap",
        "auc",
        "accuracy",
        "average_precision",
        "f1",
        "precision",
        "recall",
        "roc_auc",
        "score",
    }
)


class FixedExpertPolicy(Policy):
    """Base class for deterministic one-expert policies."""

    expert_name: str
    state_version = "stage4.policy.fixed.v1"

    def fit(self, train_records: Sequence[TrainRecord]) -> None:
        del train_records

    def select(self, task: AgentTask) -> RouteDecision:
        if self.expert_name not in task.candidate_experts:
            raise ValueError(f"AgentTask does not allow {self.expert_name}")
        return _decision(
            task=task,
            policy_name=self.name,
            selected_expert=self.expert_name,
            reason=f"Fixed policy always selects {self.expert_name}.",
            estimated_cost_ms=0.0,
        )

    def save(self, path: str | Path) -> Path:
        return _write_policy_state(
            path,
            {
                "policy_name": self.name,
                "state_version": self.state_version,
                "selected_expert": self.expert_name,
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "FixedExpertPolicy":
        payload = _read_policy_state(path)
        expected = {
            "policy_name": cls.name,
            "state_version": cls.state_version,
            "selected_expert": cls.expert_name,
        }
        if payload != expected:
            raise ValueError(f"Invalid {cls.name} policy state")
        return cls()


class AlwaysPatchCorePolicy(FixedExpertPolicy):
    name = "always_patchcore"
    expert_name = "PatchCore"
    state_version = "stage4.policy.always_patchcore.v1"


class AlwaysWinCLIPPolicy(FixedExpertPolicy):
    name = "always_winclip"
    expert_name = "WinCLIP"
    state_version = "stage4.policy.always_winclip.v1"


class RandomSeededPolicy(Policy):
    """Select one expert using a private RNG controlled only by policy seed."""

    name = "random_seeded"
    state_version = "stage4.policy.random_seeded.v1"

    def __init__(self, seed: int = 0) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("random_seeded seed must be an integer >= 0")
        self.seed = seed
        self._rng = random.Random(seed)
        self._selections_made = 0

    def fit(self, train_records: Sequence[TrainRecord]) -> None:
        del train_records

    def select(self, task: AgentTask) -> RouteDecision:
        candidates = tuple(task.candidate_experts)
        if not candidates:
            raise ValueError("AgentTask has no candidate experts")
        selected = self._rng.choice(candidates)
        self._selections_made += 1
        return _decision(
            task=task,
            policy_name=self.name,
            selected_expert=selected,
            reason=f"Seeded random selection using configured seed {self.seed}.",
            estimated_cost_ms=0.0,
        )

    def configuration(self) -> dict[str, Any]:
        return {"seed": self.seed}

    def save(self, path: str | Path) -> Path:
        return _write_policy_state(
            path,
            {
                "policy_name": self.name,
                "state_version": self.state_version,
                "seed": self.seed,
                "selections_made": self._selections_made,
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "RandomSeededPolicy":
        payload = _read_policy_state(path)
        if payload.get("policy_name") != cls.name or payload.get("state_version") != cls.state_version:
            raise ValueError(f"Invalid {cls.name} policy state")
        seed = payload.get("seed")
        selections_made = payload.get("selections_made")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or isinstance(selections_made, bool)
            or not isinstance(selections_made, int)
            or selections_made < 0
        ):
            raise ValueError(f"Invalid {cls.name} RNG state")
        policy = cls(seed=seed)
        for _ in range(selections_made):
            policy._rng.choice(CANDIDATE_EXPERTS)
        policy._selections_made = selections_made
        return policy


class FastestExpertPolicy(Policy):
    """Choose the lowest-runtime expert without using any quality metric."""

    name = "fastest_expert"
    state_version = "stage4.policy.fastest_expert.v1"

    def __init__(
        self,
        *,
        cost_card: Mapping[str, Any] | str | Path | None = None,
    ) -> None:
        self._card_global: dict[str, float] = {}
        self._card_context: dict[tuple[str, str, int], dict[str, float]] = {}
        self._train_global: dict[str, float] = {}
        self._train_context: dict[tuple[str, str, int], dict[str, float]] = {}
        self._training_fold = ""
        if cost_card is not None:
            payload = load_expert_cost_card(cost_card) if isinstance(cost_card, (str, Path)) else dict(cost_card)
            self._card_global, self._card_context = _parse_cost_card(payload)

    def fit(self, train_records: Sequence[TrainRecord]) -> None:
        totals: dict[str, list[float]] = defaultdict(list)
        context_totals: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        folds: set[str] = set()
        for index, record in enumerate(train_records):
            if isinstance(record, AgentTask):
                raise ValueError(
                    "fastest_expert requires runtime-only training records, not AgentTask objects"
                )
            if not isinstance(record, Mapping):
                raise ValueError(f"fastest_expert training record {index} must be a mapping")
            if "runtime_ms" not in record and "expert_name" not in record and "expert" not in record:
                raise ValueError(
                    "fastest_expert requires current-fold training runtime records or a cost card"
                )
            _reject_quality_fields(record, context=f"training runtime record {index}")
            split = str(record.get("split", "")).strip().lower()
            if split != "train":
                raise ValueError(
                    f"fastest_expert runtime record {index} must have split='train'; got {split!r}"
                )
            source = str(record.get("runtime_source", TRAINING_RUNTIME_SOURCE)).strip()
            if source != TRAINING_RUNTIME_SOURCE:
                raise ValueError(
                    f"fastest_expert runtime record {index} has forbidden source {source!r}"
                )
            fold = str(record.get("fold", "")).strip()
            if fold:
                folds.add(fold)
            expert = _canonical_expert(record.get("expert_name", record.get("expert")))
            runtime = _finite_nonnegative(record.get("runtime_ms"), "runtime_ms")
            context = _optional_context_key(
                record, context=f"training runtime record {index}"
            )
            totals[expert].append(runtime)
            if context is not None:
                context_totals[context][expert].append(runtime)

        if len(folds) > 1:
            raise ValueError(
                f"fastest_expert training runtimes mix folds: {sorted(folds)!r}"
            )
        self._training_fold = next(iter(folds), "")
        self._train_global = {
            expert: sum(values) / len(values) for expert, values in totals.items()
        }
        self._train_context = {
            context: {
                expert: sum(values) / len(values)
                for expert, values in expert_values.items()
            }
            for context, expert_values in context_totals.items()
        }
        if not self._train_global and not self._card_global and not self._card_context:
            raise ValueError(
                "fastest_expert has no legal runtime source; provide current-fold training "
                "runs or a predeclared expert cost card"
            )

    def select(self, task: AgentTask) -> RouteDecision:
        context = (task.dataset, task.category, task.k_shot)
        costs, source = self._costs_for(context)
        missing = [expert for expert in task.candidate_experts if expert not in costs]
        if missing:
            raise ValueError(
                "fastest_expert lacks legal runtime estimates for candidates: "
                f"{missing}; context={context!r}"
            )
        order = {expert: index for index, expert in enumerate(task.candidate_experts)}
        selected = min(task.candidate_experts, key=lambda expert: (costs[expert], order[expert]))
        return _decision(
            task=task,
            policy_name=self.name,
            selected_expert=selected,
            reason=(
                f"Lowest estimated runtime from {source}; no test-fold quality metric was read."
            ),
            estimated_cost_ms=costs[selected],
        )

    def configuration(self) -> dict[str, Any]:
        return {
            "runtime_sources": [
                source
                for source, present in (
                    (TRAINING_RUNTIME_SOURCE, bool(self._train_global or self._train_context)),
                    ("predeclared_expert_cost_card", bool(self._card_global or self._card_context)),
                )
                if present
            ],
            "training_fold": self._training_fold,
        }

    def save(self, path: str | Path) -> Path:
        return _write_policy_state(
            path,
            {
                "policy_name": self.name,
                "state_version": self.state_version,
                "training_fold": self._training_fold,
                "training_global_costs_ms": self._train_global,
                "training_context_costs_ms": _serialize_context_costs(self._train_context),
                "cost_card_global_costs_ms": self._card_global,
                "cost_card_context_costs_ms": _serialize_context_costs(self._card_context),
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "FastestExpertPolicy":
        payload = _read_policy_state(path)
        if payload.get("policy_name") != cls.name or payload.get("state_version") != cls.state_version:
            raise ValueError(f"Invalid {cls.name} policy state")
        policy = cls()
        policy._training_fold = str(payload.get("training_fold", ""))
        policy._train_global = _parse_saved_global(payload.get("training_global_costs_ms"))
        policy._card_global = _parse_saved_global(payload.get("cost_card_global_costs_ms"))
        policy._train_context = _parse_saved_context(payload.get("training_context_costs_ms"))
        policy._card_context = _parse_saved_context(payload.get("cost_card_context_costs_ms"))
        if not any((policy._train_global, policy._train_context, policy._card_global, policy._card_context)):
            raise ValueError(f"Invalid {cls.name} policy state: no runtime costs")
        return policy

    def _costs_for(self, context: tuple[str, str, int]) -> tuple[dict[str, float], str]:
        merged: dict[str, float] = {}
        sources: list[str] = []
        for costs, source in (
            (self._card_global, "predeclared expert cost card"),
            (self._train_global, "current fold training runs"),
            (self._card_context.get(context, {}), "contextual predeclared expert cost card"),
            (self._train_context.get(context, {}), "contextual current fold training runs"),
        ):
            if costs:
                merged.update(costs)
                sources.append(source)
        return merged, " + ".join(sources) or "no runtime source"


AlwaysAnomalyDinoPolicy = AlwaysAnomalyDINOPolicy


def load_expert_cost_card(path: str | Path) -> dict[str, Any]:
    """Read a predeclared JSON expert cost card."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read expert cost card {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Expert cost card root must be a JSON object")
    return payload


def _parse_cost_card(
    payload: Mapping[str, Any],
) -> tuple[dict[str, float], dict[tuple[str, str, int], dict[str, float]]]:
    _reject_quality_fields(payload, context="expert cost card")
    if "costs_ms" not in payload and all(_looks_like_expert(key) for key in payload):
        return _parse_cost_mapping(payload, "expert cost card"), {}
    if payload.get("protocol_version") != EXPERT_COST_CARD_VERSION:
        raise ValueError(
            f"Expert cost card protocol_version must be {EXPERT_COST_CARD_VERSION!r}"
        )
    global_costs = _parse_cost_mapping(payload.get("costs_ms", {}), "expert cost card costs_ms")
    contexts_value = payload.get("context_costs", [])
    if not isinstance(contexts_value, list):
        raise ValueError("Expert cost card context_costs must be a list")
    contextual: dict[tuple[str, str, int], dict[str, float]] = {}
    for index, row in enumerate(contexts_value):
        if not isinstance(row, Mapping):
            raise ValueError(f"Expert cost card context_costs[{index}] must be an object")
        key = _context_key(row, context=f"expert cost card context_costs[{index}]")
        if key in contextual:
            raise ValueError(f"Expert cost card has duplicate context {key!r}")
        contextual[key] = _parse_cost_mapping(
            row.get("costs_ms", {}), f"expert cost card context_costs[{index}].costs_ms"
        )
    if not global_costs and not contextual:
        raise ValueError("Expert cost card contains no costs")
    return global_costs, contextual


def _parse_cost_mapping(value: Any, context: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    costs: dict[str, float] = {}
    for raw_expert, raw_cost in value.items():
        expert = _canonical_expert(raw_expert)
        if expert in costs:
            raise ValueError(f"{context} has duplicate expert {expert}")
        costs[expert] = _finite_nonnegative(raw_cost, f"{context}.{raw_expert}")
    return costs


def _decision(
    *,
    task: AgentTask,
    policy_name: str,
    selected_expert: str,
    reason: str,
    estimated_cost_ms: float,
) -> RouteDecision:
    return RouteDecision(
        task_id=task.task_id,
        dataset=task.dataset,
        category=task.category,
        k_shot=task.k_shot,
        seed=task.seed,
        support_set_id=task.support_set_id,
        policy_name=policy_name,
        selected_expert=selected_expert,
        decision_reason=reason,
        estimated_cost_ms=estimated_cost_ms,
        tool_calls=1,
    )


def _context_key(record: Mapping[str, Any], *, context: str) -> tuple[str, str, int]:
    dataset = str(record.get("dataset", "")).strip()
    category = str(record.get("category", "")).strip()
    raw_k = record.get("k_shot")
    if not dataset or not category:
        raise ValueError(f"{context} requires non-empty dataset and category")
    if isinstance(raw_k, bool):
        raise ValueError(f"{context} k_shot must be an integer >= 1")
    try:
        k_shot = int(raw_k)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} k_shot must be an integer >= 1") from exc
    if k_shot < 1:
        raise ValueError(f"{context} k_shot must be an integer >= 1")
    return dataset, category, k_shot


def _optional_context_key(
    record: Mapping[str, Any], *, context: str
) -> tuple[str, str, int] | None:
    present = [field in record for field in ("dataset", "category", "k_shot")]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            f"{context} must provide dataset, category, and k_shot together"
        )
    return _context_key(record, context=context)


def _canonical_expert(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    mapping = {re.sub(r"[^a-z0-9]+", "", expert.lower()): expert for expert in CANDIDATE_EXPERTS}
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown expert name: {value!r}") from exc


def _looks_like_expert(value: Any) -> bool:
    try:
        _canonical_expert(value)
    except ValueError:
        return False
    return True


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number >= 0")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number >= 0") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be a finite number >= 0")
    return parsed


def _reject_quality_fields(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _QUALITY_FIELD_NAMES or any(
                token in normalized for token in _QUALITY_FIELD_TOKENS
            ):
                raise ValueError(f"{context} contains forbidden quality/evaluator field {key!r}")
            _reject_quality_fields(child, context=context)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _reject_quality_fields(child, context=context)


def _serialize_context_costs(
    costs: Mapping[tuple[str, str, int], Mapping[str, float]],
) -> list[dict[str, Any]]:
    return [
        {
            "dataset": context[0],
            "category": context[1],
            "k_shot": context[2],
            "costs_ms": dict(sorted(expert_costs.items())),
        }
        for context, expert_costs in sorted(costs.items())
    ]


def _parse_saved_global(value: Any) -> dict[str, float]:
    return _parse_cost_mapping(value or {}, "saved policy global costs")


def _parse_saved_context(value: Any) -> dict[tuple[str, str, int], dict[str, float]]:
    if not isinstance(value, list):
        raise ValueError("Saved policy contextual costs must be a list")
    result: dict[tuple[str, str, int], dict[str, float]] = {}
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise ValueError(f"Saved policy contextual cost {index} must be an object")
        key = _context_key(row, context=f"saved policy contextual cost {index}")
        if key in result:
            raise ValueError(f"Saved policy has duplicate context {key!r}")
        result[key] = _parse_cost_mapping(row.get("costs_ms", {}), "saved contextual costs")
    return result


def _policy_state_path(path: str | Path) -> Path:
    destination = Path(path)
    return destination / "policy.json" if destination.is_dir() else destination


def _write_policy_state(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = _policy_state_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def _read_policy_state(path: str | Path) -> dict[str, Any]:
    source = _policy_state_path(path)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Policy state {source} must be a JSON object")
    return payload
