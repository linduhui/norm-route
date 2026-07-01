"""MVTec AD manifest generation.

This module only enumerates dataset paths and ground-truth metadata needed by
the evaluator. It does not create support sets, tune thresholds, or run models.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

from src.data.common import display_path, inspect_image, iter_image_files, relative_posix


DATASET_NAME = "mvtec"


@dataclass(frozen=True)
class ManifestRows:
    agent_input: list[dict[str, str]]
    evaluator: list[dict[str, str | int]]
    audit: dict[str, Any]


def build_mvtec_manifest(
    mvtec_root: str | Path,
    path_style: str = "absolute",
    categories: list[str] | None = None,
) -> ManifestRows:
    root = Path(mvtec_root)
    if not root.is_dir():
        raise FileNotFoundError(f"MVTec root does not exist or is not a directory: {root}")

    agent_rows: list[dict[str, str]] = []
    evaluator_rows: list[dict[str, str | int]] = []
    audit = _empty_audit(root, path_style)
    image_id_counts: dict[str, int] = {}

    selected_categories = _discover_categories(root, categories)
    audit["counts"]["categories"] = len(selected_categories)
    audit["categories"] = selected_categories

    for category in selected_categories:
        category_dir = root / category
        train_good = iter_image_files(category_dir / "train" / "good")
        test_good = iter_image_files(category_dir / "test" / "good")
        test_anomaly_dirs = _discover_test_anomaly_dirs(category_dir)

        audit["counts"]["train_good_images"] += len(train_good)
        audit["counts"]["test_good_images"] += len(test_good)

        for image_path in train_good:
            _add_sample(
                root=root,
                image_path=image_path,
                category=category,
                split="train",
                label=0,
                defect_type="good",
                mask_path=None,
                path_style=path_style,
                agent_rows=agent_rows,
                evaluator_rows=evaluator_rows,
                audit=audit,
                image_id_counts=image_id_counts,
            )

        for image_path in test_good:
            _add_sample(
                root=root,
                image_path=image_path,
                category=category,
                split="test",
                label=0,
                defect_type="good",
                mask_path=None,
                path_style=path_style,
                agent_rows=agent_rows,
                evaluator_rows=evaluator_rows,
                audit=audit,
                image_id_counts=image_id_counts,
            )

        for defect_type, defect_dir in test_anomaly_dirs:
            anomaly_images = iter_image_files(defect_dir)
            audit["counts"]["test_anomaly_images"] += len(anomaly_images)
            for image_path in anomaly_images:
                mask_path = _find_mask_path(category_dir, defect_type, defect_dir, image_path)
                if mask_path is None:
                    audit["counts"]["missing_masks"] += 1
                    audit["missing_masks"].append(
                        {
                            "category": category,
                            "defect_type": defect_type,
                            "image_path": display_path(image_path, root, path_style),
                        }
                    )

                _add_sample(
                    root=root,
                    image_path=image_path,
                    category=category,
                    split="test",
                    label=1,
                    defect_type=defect_type,
                    mask_path=mask_path,
                    path_style=path_style,
                    agent_rows=agent_rows,
                    evaluator_rows=evaluator_rows,
                    audit=audit,
                    image_id_counts=image_id_counts,
                )

    duplicate_ids = sorted(image_id for image_id, count in image_id_counts.items() if count > 1)
    audit["counts"]["duplicate_image_ids"] = len(duplicate_ids)
    audit["duplicate_image_ids"] = duplicate_ids

    return ManifestRows(agent_input=agent_rows, evaluator=evaluator_rows, audit=audit)


def _empty_audit(root: Path, path_style: str) -> dict[str, Any]:
    return {
        "dataset": DATASET_NAME,
        "mvtec_root": str(root),
        "path_style": path_style,
        "counts": {
            "categories": 0,
            "train_good_images": 0,
            "test_good_images": 0,
            "test_anomaly_images": 0,
            "missing_masks": 0,
            "duplicate_image_ids": 0,
            "broken_image_files": 0,
            "mask_size_mismatches": 0,
            "broken_mask_files": 0,
        },
        "categories": [],
        "missing_masks": [],
        "duplicate_image_ids": [],
        "broken_image_files": [],
        "mask_size_mismatches": [],
        "broken_mask_files": [],
    }


def _discover_categories(root: Path, selected: list[str] | None) -> list[str]:
    discovered = sorted(path.name for path in root.iterdir() if path.is_dir())
    if selected is None:
        return discovered

    missing = sorted(category for category in selected if category not in discovered)
    if missing:
        raise FileNotFoundError(f"MVTec categories not found under {root}: {missing}")
    return sorted(selected)


def _discover_test_anomaly_dirs(category_dir: Path) -> list[tuple[str, Path]]:
    test_dir = category_dir / "test"
    if not test_dir.is_dir():
        return []
    return sorted(
        (path.name, path)
        for path in test_dir.iterdir()
        if path.is_dir() and path.name != "good"
    )


def _add_sample(
    *,
    root: Path,
    image_path: Path,
    category: str,
    split: str,
    label: int,
    defect_type: str,
    mask_path: Path | None,
    path_style: str,
    agent_rows: list[dict[str, str]],
    evaluator_rows: list[dict[str, str | int]],
    audit: dict[str, Any],
    image_id_counts: dict[str, int],
) -> None:
    image_id = _image_id(root, image_path)
    image_id_counts[image_id] = image_id_counts.get(image_id, 0) + 1

    image_check = inspect_image(image_path)
    if not image_check.ok:
        audit["counts"]["broken_image_files"] += 1
        audit["broken_image_files"].append(
            {
                "image_path": display_path(image_path, root, path_style),
                "error": image_check.error,
            }
        )

    mask_value = ""
    if mask_path is not None:
        mask_value = display_path(mask_path, root, path_style)
        mask_check = inspect_image(mask_path)
        if not mask_check.ok:
            audit["counts"]["broken_mask_files"] += 1
            audit["broken_mask_files"].append(
                {
                    "mask_path": mask_value,
                    "error": mask_check.error,
                }
            )
        elif image_check.ok and image_check.size != mask_check.size:
            audit["counts"]["mask_size_mismatches"] += 1
            audit["mask_size_mismatches"].append(
                {
                    "image_path": display_path(image_path, root, path_style),
                    "mask_path": mask_value,
                    "image_size": list(image_check.size or ()),
                    "mask_size": list(mask_check.size or ()),
                }
            )

    agent_rows.append(
        {
            "image_id": image_id,
            "dataset": DATASET_NAME,
            "category": category,
            "split": split,
            "image_path": display_path(image_path, root, path_style),
        }
    )
    evaluator_rows.append(
        {
            "image_id": image_id,
            "label": label,
            "mask_path": mask_value,
            "defect_type": defect_type,
        }
    )


def _image_id(root: Path, image_path: Path) -> str:
    key = f"{DATASET_NAME}/{relative_posix(image_path, root)}"
    digest = sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{DATASET_NAME}_{digest}"


def _find_mask_path(
    category_dir: Path,
    defect_type: str,
    defect_dir: Path,
    image_path: Path,
) -> Path | None:
    image_relative_to_defect = image_path.relative_to(defect_dir)
    stem_relative = image_relative_to_defect.with_suffix("")

    candidates = [
        category_dir / "ground_truth" / defect_type / f"{image_path.stem}_mask.png",
        category_dir / "ground_truth" / defect_type / stem_relative.parent / f"{image_path.stem}_mask.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
