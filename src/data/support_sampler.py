"""Few-normal-shot support set sampling from agent-input manifests only."""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path

from src.data.manifest import AGENT_INPUT_COLUMNS, FORBIDDEN_AGENT_COLUMNS


SUPPORT_SET_COLUMNS = [
    "support_set_id",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_rank",
    "image_id",
    "image_path",
]
DEFAULT_K_SHOTS = (1, 2, 4, 8)
DEFAULT_SEEDS = tuple(range(5))


def build_support_set_rows(
    agent_input_path: str | Path,
    *,
    dataset: str,
    k_shot: int,
    seed: int,
) -> list[dict[str, str]]:
    """Sample one support set per category from train/good rows.

    The sampler intentionally reads only the agent-input manifest. It rejects
    evaluator-only columns if they appear in the input CSV.
    """

    if k_shot <= 0:
        raise ValueError(f"k_shot must be positive, got {k_shot}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    rows = _read_agent_input(agent_input_path)
    dataset_rows = [row for row in rows if row["dataset"] == dataset]
    if not dataset_rows:
        raise ValueError(f"No rows for dataset {dataset!r} in {agent_input_path}")

    categories = sorted({row["category"] for row in dataset_rows})
    support_rows: list[dict[str, str]] = []
    for category in categories:
        candidates = [
            row
            for row in dataset_rows
            if row["category"] == category and _is_train_good_agent_row(row, dataset)
        ]
        candidates = sorted(candidates, key=lambda row: (row["image_id"], row["image_path"]))
        if len(candidates) < k_shot:
            raise ValueError(
                f"Category {category!r} in dataset {dataset!r} has only {len(candidates)} "
                f"train/good support candidates; requested K={k_shot}."
            )

        sampled = random.Random(_category_seed(dataset, category, k_shot, seed)).sample(
            candidates, k_shot
        )
        support_set_id = make_support_set_id(dataset=dataset, category=category, k_shot=k_shot, seed=seed)
        seen_ids: set[str] = set()
        for rank, row in enumerate(sampled, start=1):
            image_id = row["image_id"]
            if image_id in seen_ids:
                raise ValueError(f"Duplicate image_id {image_id!r} inside {support_set_id}")
            seen_ids.add(image_id)
            support_rows.append(
                {
                    "support_set_id": support_set_id,
                    "dataset": dataset,
                    "category": category,
                    "k_shot": str(k_shot),
                    "seed": str(seed),
                    "support_rank": str(rank),
                    "image_id": image_id,
                    "image_path": row["image_path"],
                }
            )

    return support_rows


def write_support_set_csv(path: str | Path, rows: list[dict[str, str]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUPPORT_SET_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def make_support_set_id(*, dataset: str, category: str, k_shot: int, seed: int) -> str:
    category_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", category).strip("-")
    if not category_slug:
        raise ValueError(f"Cannot build support_set_id for empty category: {category!r}")
    return f"{dataset}_{category_slug}_k{k_shot}_seed{seed}"


def _read_agent_input(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Agent input manifest does not exist: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Agent input manifest has no header: {manifest_path}")
        fieldnames = set(reader.fieldnames)
        forbidden = sorted(fieldnames & FORBIDDEN_AGENT_COLUMNS)
        if forbidden:
            raise ValueError(f"Agent input manifest contains forbidden fields: {forbidden}")
        missing = [column for column in AGENT_INPUT_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"Agent input manifest is missing fields: {missing}")
        return [
            {column: (row[column] or "").strip() for column in AGENT_INPUT_COLUMNS}
            for row in reader
        ]


def _is_train_good_agent_row(row: dict[str, str], dataset: str) -> bool:
    if row["split"] != "train":
        return False

    normalized_parts = _normalized_path_parts(row["image_path"])
    if dataset == "mvtec":
        return _has_adjacent_parts(normalized_parts, "train", "good")
    if dataset == "visa":
        return (
            _has_adjacent_parts(normalized_parts, "train", "good")
            or _has_adjacent_parts(normalized_parts, "train", "normal")
            or _has_subsequence(normalized_parts, ("data", "images", "normal"))
        )
    raise ValueError(f"Unsupported dataset: {dataset!r}")


def _category_seed(dataset: str, category: str, k_shot: int, seed: int) -> str:
    return f"{dataset}\n{category}\n{k_shot}\n{seed}"


def _normalized_path_parts(path_value: str) -> tuple[str, ...]:
    return tuple(part.lower() for part in re.split(r"[\\/]+", path_value) if part)


def _has_adjacent_parts(parts: tuple[str, ...], first: str, second: str) -> bool:
    return any(left == first and right == second for left, right in zip(parts, parts[1:]))


def _has_subsequence(parts: tuple[str, ...], subsequence: tuple[str, ...]) -> bool:
    if not subsequence:
        return True
    for start in range(0, len(parts) - len(subsequence) + 1):
        if parts[start : start + len(subsequence)] == subsequence:
            return True
    return False
