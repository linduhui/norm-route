"""Registry for constructing Stage 4 routing policies by stable name."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from types import MappingProxyType

from .policy import AlwaysAnomalyDINOPolicy, Policy
from src.normroute.policies.fixed import (
    AlwaysPatchCorePolicy,
    AlwaysWinCLIPPolicy,
    FastestExpertPolicy,
    RandomSeededPolicy,
)
from src.normroute.policies.cost_aware import CostAwarePolicy
from src.normroute.policies.rule_based import (
    CategoryPriorPolicy,
    CategoryShotPriorPolicy,
)
from src.normroute.policies.learned import (
    DecisionTreeMetadataPolicy,
    MultinomialLogisticMetadataPolicy,
)


PolicyFactory = Callable[..., Policy]
_POLICY_REGISTRY: dict[str, PolicyFactory] = {}
POLICY_REGISTRY = MappingProxyType(_POLICY_REGISTRY)
"""Read-only view of the registered policy factories."""


def register_policy(name: str, factory: PolicyFactory) -> None:
    """Register a policy factory and reject ambiguous duplicate names."""

    key = _normalize_name(name)
    if key in _POLICY_REGISTRY:
        raise ValueError(f"Policy is already registered: {key}")
    _POLICY_REGISTRY[key] = factory


def create_policy(name: str, **configuration: Any) -> Policy:
    """Construct a fresh policy instance by registered name."""

    key = _normalize_name(name)
    try:
        factory = _POLICY_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown policy {name!r}; available policies: {list_policies()}"
        ) from exc
    try:
        policy = factory(**configuration)
    except TypeError as exc:
        if configuration:
            raise ValueError(
                f"Policy {key!r} does not accept configuration keys "
                f"{sorted(configuration)!r}: {exc}"
            ) from exc
        raise
    if policy.name != key:
        raise ValueError(
            f"Policy factory registered as {key!r} produced name={policy.name!r}"
        )
    return policy


def get_policy(name: str, **configuration: Any) -> Policy:
    """Compatibility alias for :func:`create_policy`."""

    return create_policy(name, **configuration)


def list_policies() -> tuple[str, ...]:
    return tuple(sorted(_POLICY_REGISTRY))


def _normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Policy name must be a non-empty string")
    return name.strip().lower()


register_policy(AlwaysAnomalyDINOPolicy.name, AlwaysAnomalyDINOPolicy)
register_policy(AlwaysPatchCorePolicy.name, AlwaysPatchCorePolicy)
register_policy(AlwaysWinCLIPPolicy.name, AlwaysWinCLIPPolicy)
register_policy(RandomSeededPolicy.name, RandomSeededPolicy)
register_policy(FastestExpertPolicy.name, FastestExpertPolicy)
register_policy(CategoryPriorPolicy.name, CategoryPriorPolicy)
register_policy(CategoryShotPriorPolicy.name, CategoryShotPriorPolicy)
register_policy(CostAwarePolicy.name, CostAwarePolicy)
register_policy(DecisionTreeMetadataPolicy.name, DecisionTreeMetadataPolicy)
register_policy(MultinomialLogisticMetadataPolicy.name, MultinomialLogisticMetadataPolicy)
