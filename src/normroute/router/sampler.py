"""Query-group-balanced samplers for repeated K/seed Stage 5 tasks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Iterator, Sequence
import math
import random
from typing import Any


class GroupedSamplerError(ValueError):
    """Raised when grouped sampling inputs are invalid."""


class GroupedQuerySampler(Iterable[int]):
    """Choose exactly one K/seed variant from every query group per epoch.

    ``set_epoch`` makes distributed/training-loop behavior explicit and
    reproducible.  The number of sampled items equals the number of unique
    queries, so a query repeated across many K/seed combinations receives no
    more influence than a query with one row.
    """

    def __init__(self, source: Any, *, seed: int = 0, shuffle: bool = True) -> None:
        group_ids = getattr(source, "group_ids", source)
        if isinstance(group_ids, (str, bytes)) or not isinstance(group_ids, Sequence):
            raise GroupedSamplerError("source must be a dataset or sequence of group ids")
        if not group_ids:
            raise GroupedSamplerError("group ids must be non-empty")
        groups: dict[Hashable, list[int]] = {}
        for index, group_id in enumerate(group_ids):
            if not isinstance(group_id, Hashable):
                raise GroupedSamplerError(f"group id at index {index} is not hashable")
            groups.setdefault(group_id, []).append(index)
        self._groups = tuple(
            tuple(indices)
            for _, indices in sorted(groups.items(), key=lambda item: repr(item[0]))
        )
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise GroupedSamplerError("epoch must be a non-negative integer")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        selected = [indices[rng.randrange(len(indices))] for indices in self._groups]
        if self.shuffle:
            rng.shuffle(selected)
        return iter(selected)

    def __len__(self) -> int:
        return len(self._groups)


GroupedSampler = GroupedQuerySampler


def inverse_frequency_weights(group_ids: Sequence[Hashable]) -> tuple[float, ...]:
    """Return per-row weights whose total mass is one for every query."""

    if isinstance(group_ids, (str, bytes)) or not group_ids:
        raise GroupedSamplerError("group_ids must be a non-empty sequence")
    try:
        counts = Counter(group_ids)
    except TypeError as exc:
        raise GroupedSamplerError("all group ids must be hashable") from exc
    weights = tuple(1.0 / counts[group_id] for group_id in group_ids)
    if not all(math.isfinite(value) and value > 0.0 for value in weights):
        raise GroupedSamplerError("could not construct inverse-frequency weights")
    return weights


def query_group_counts(group_ids: Sequence[Hashable]) -> dict[Hashable, int]:
    """Return an auditable copy of the query repeat counts."""

    if isinstance(group_ids, (str, bytes)) or not group_ids:
        raise GroupedSamplerError("group_ids must be a non-empty sequence")
    try:
        return dict(Counter(group_ids))
    except TypeError as exc:
        raise GroupedSamplerError("all group ids must be hashable") from exc


__all__ = [
    "GroupedQuerySampler",
    "GroupedSampler",
    "GroupedSamplerError",
    "inverse_frequency_weights",
    "query_group_counts",
]
