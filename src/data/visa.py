"""VisA manifest generation.

This module only enumerates dataset paths and evaluator metadata. It does not
create support sets, tune thresholds, run models, download data, or modify raw
dataset files.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

from src.data.common import display_path, inspect_image, iter_image_files, relative_posix


DATASET_NAME = "visa"
SPLIT_DIR_NAMES = {"train", "training", "test", "testing", "val", "validation"}
GOOD_DIR_NAMES = {"good", "normal", "ok"}
ANOMALY_DIR_NAMES = {"bad", "abnormal", "anomaly", "anomalous", "defect", "defective", "ng"}
MASK_DIR_NAMES = {"ground_truth", "gt", "mask", "masks"}
NON_CATEGORY_DIR_NAMES = {"split_csv"} | SPLIT_DIR_NAMES | GOOD_DIR_NAMES | ANOMALY_DIR_NAMES


@dataclass(frozen=True)
class ManifestRows:
    agent_input: list[dict[str, str]]
    evaluator: list[dict[str, str | int]]
    audit: dict[str, Any]


def build_visa_manifest(
    visa_root: str | Path,
    path_style: str = "absolute",
    categories: list[str] | None = None,
    split_csv_path: str | Path | None = None,
) -> ManifestRows:
    root = Path(visa_root)
    if not root.is_dir():
        raise FileNotFoundError(f"VisA root does not exist or is not a directory: {root}")

    agent_rows: list[dict[str, str]] = []
    evaluator_rows: list[dict[str, str | int]] = []
    audit = _empty_audit(root, path_style, split_csv_path)
    image_id_counts: dict[str, int] = {}

    selected_categories = _discover_categories(root, categories)
    audit["counts"]["categories"] = len(selected_categories)
    audit["categories"] = selected_categories

    selected_set = set(selected_categories)
    split_csv = _resolve_split_csv(root, split_csv_path)
    if split_csv is not None:
        audit["split_csv"] = display_path(split_csv, root, path_style)
        records = _read_split_csv(split_csv, root)
        for record in records:
            if record["category"] not in selected_set:
                continue
            _add_sample(
                root=root,
                image_path=record["image_path"],
                category=record["category"],
                split=record["split"],
                label=record["label"],
                defect_type=record["defect_type"],
                mask_path=record["mask_path"] or _find_mask_path(root, record["image_path"]),
                path_style=path_style,
                agent_rows=agent_rows,
                evaluator_rows=evaluator_rows,
                audit=audit,
                image_id_counts=image_id_counts,
            )
    else:
        _scan_explicit_split_layout(
            root=root,
            categories=selected_categories,
            path_style=path_style,
            agent_rows=agent_rows,
            evaluator_rows=evaluator_rows,
            audit=audit,
            image_id_counts=image_id_counts,
        )

    duplicate_ids = sorted(image_id for image_id, count in image_id_counts.items() if count > 1)
    audit["counts"]["duplicate_image_ids"] = len(duplicate_ids)
    audit["duplicate_image_ids"] = duplicate_ids
    _fill_split_counts(audit, agent_rows, evaluator_rows)

    return ManifestRows(agent_input=agent_rows, evaluator=evaluator_rows, audit=audit)


def _empty_audit(root: Path, path_style: str, split_csv_path: str | Path | None) -> dict[str, Any]:
    return {
        "dataset": DATASET_NAME,
        "visa_root": str(root),
        "path_style": path_style,
        "requested_split_csv": str(split_csv_path) if split_csv_path is not None else None,
        "split_csv": None,
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
    discovered = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and _normalized(path.name) not in NON_CATEGORY_DIR_NAMES
    )
    if selected is None:
        return discovered

    missing = sorted(category for category in selected if category not in discovered)
    if missing:
        raise FileNotFoundError(f"VisA categories not found under {root}: {missing}")
    return sorted(selected)


def _resolve_split_csv(root: Path, split_csv_path: str | Path | None) -> Path | None:
    if split_csv_path is not None:
        path = Path(split_csv_path)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f"VisA split CSV does not exist: {path}")
        return path

    default_path = root / "split_csv" / "1cls.csv"
    if default_path.is_file():
        return default_path

    csv_paths = sorted((root / "split_csv").glob("*.csv")) if (root / "split_csv").is_dir() else []
    if len(csv_paths) == 1:
        return csv_paths[0]
    if len(csv_paths) > 1:
        raise ValueError(
            f"Multiple VisA split CSV files found under {root / 'split_csv'}; "
            "pass --visa-split-csv to choose one explicitly."
        )
    return None


def _read_split_csv(path: Path, root: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"VisA split CSV has no header: {path}")
        return [_split_record(row, root, path, index + 2) for index, row in enumerate(reader)]


def _split_record(row: dict[str, str], root: Path, csv_path: Path, line_number: int) -> dict[str, Any]:
    image_value = _first_value(row, ["image_path", "image", "img_path", "path", "filename", "file"])
    if not image_value:
        raise ValueError(f"{csv_path}:{line_number} is missing an image path column")
    image_path = _resolve_dataset_path(root, image_value)

    category = _first_value(row, ["category", "object", "class", "cls"]) or _infer_category(root, image_path)
    if not category:
        raise ValueError(f"{csv_path}:{line_number} is missing category/object and cannot infer it")

    split = _normalize_split(_first_value(row, ["split", "set", "subset"]), csv_path, line_number)
    label = _normalize_label(_first_value(row, ["label", "is_anomaly", "anomaly"]), image_path)
    defect_type = _first_value(row, ["defect_type", "defect", "anomaly_type", "label_name"])
    if label == 0:
        defect_type = "good"
    elif not defect_type or _normalized(defect_type) in ANOMALY_DIR_NAMES:
        defect_type = _infer_defect_type(root, image_path)

    mask_value = _first_value(row, ["mask_path", "mask", "gt", "ground_truth"])
    mask_path = _resolve_dataset_path(root, mask_value) if mask_value else None

    return {
        "category": category,
        "split": split,
        "label": label,
        "defect_type": defect_type,
        "image_path": image_path,
        "mask_path": mask_path,
    }


def _scan_explicit_split_layout(
    *,
    root: Path,
    categories: list[str],
    path_style: str,
    agent_rows: list[dict[str, str]],
    evaluator_rows: list[dict[str, str | int]],
    audit: dict[str, Any],
    image_id_counts: dict[str, int],
) -> None:
    added_any = False
    for category in categories:
        category_dir = root / category
        train_good = _good_images_under(category_dir / "train")
        test_good = _good_images_under(category_dir / "test")
        test_anomaly_dirs = _anomaly_dirs_under(category_dir / "test")

        for image_path in train_good:
            added_any = True
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
            added_any = True
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
            for image_path in iter_image_files(defect_dir):
                added_any = True
                _add_sample(
                    root=root,
                    image_path=image_path,
                    category=category,
                    split="test",
                    label=1,
                    defect_type=defect_type,
                    mask_path=_find_mask_path(root, image_path),
                    path_style=path_style,
                    agent_rows=agent_rows,
                    evaluator_rows=evaluator_rows,
                    audit=audit,
                    image_id_counts=image_id_counts,
                )

    if not added_any:
        raise ValueError(
            "No VisA split CSV was found and no explicit train/test image layout was detected. "
            "Pass --visa-split-csv for the detected Data/Images layout."
        )


def _good_images_under(split_dir: Path) -> list[Path]:
    images: list[Path] = []
    for child in _child_dirs(split_dir):
        if _normalized(child.name) in GOOD_DIR_NAMES:
            images.extend(iter_image_files(child))
    return sorted(images)


def _anomaly_dirs_under(split_dir: Path) -> list[tuple[str, Path]]:
    if not split_dir.is_dir():
        return []
    return sorted(
        (path.name, path)
        for path in split_dir.iterdir()
        if path.is_dir() and _normalized(path.name) not in GOOD_DIR_NAMES | MASK_DIR_NAMES
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

    image_value = display_path(image_path, root, path_style)
    image_check = inspect_image(image_path)
    if not image_check.ok:
        audit["counts"]["broken_image_files"] += 1
        audit["broken_image_files"].append({"image_path": image_value, "error": image_check.error})

    mask_value = ""
    if label == 1:
        if mask_path is None or not mask_path.is_file():
            audit["counts"]["missing_masks"] += 1
            audit["missing_masks"].append(
                {"category": category, "defect_type": defect_type, "image_path": image_value}
            )
            mask_path = None
        else:
            mask_value = display_path(mask_path, root, path_style)
            mask_check = inspect_image(mask_path)
            if not mask_check.ok:
                audit["counts"]["broken_mask_files"] += 1
                audit["broken_mask_files"].append({"mask_path": mask_value, "error": mask_check.error})
            elif image_check.ok and image_check.size != mask_check.size:
                audit["counts"]["mask_size_mismatches"] += 1
                audit["mask_size_mismatches"].append(
                    {
                        "image_path": image_value,
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
            "image_path": image_value,
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


def _fill_split_counts(
    audit: dict[str, Any],
    agent_rows: list[dict[str, str]],
    evaluator_rows: list[dict[str, str | int]],
) -> None:
    evaluator_by_id = {row["image_id"]: row for row in evaluator_rows}
    for row in agent_rows:
        evaluator = evaluator_by_id[row["image_id"]]
        if row["split"] == "train" and evaluator["label"] == 0:
            audit["counts"]["train_good_images"] += 1
        elif row["split"] == "test" and evaluator["label"] == 0:
            audit["counts"]["test_good_images"] += 1
        elif row["split"] == "test" and evaluator["label"] == 1:
            audit["counts"]["test_anomaly_images"] += 1
        elif row["split"] == "train" and evaluator["label"] == 1:
            raise ValueError(f"VisA train split contains anomaly sample: {row['image_path']}")
        else:
            raise ValueError(
                f"Unsupported VisA split/label combination for {row['image_path']}: "
                f"split={row['split']!r}, label={evaluator['label']!r}"
            )


def _find_mask_path(root: Path, image_path: Path) -> Path | None:
    relative = image_path.relative_to(root)
    parts = list(relative.parts)
    candidates: list[Path] = []

    if "Images" in parts:
        images_index = parts.index("Images")
        mask_parts = parts[:images_index] + ["Masks"] + parts[images_index + 1 :]
        mask_base = root.joinpath(*mask_parts).with_suffix("")
        candidates.extend(_mask_name_candidates(mask_base, image_path))

    for marker in ("test", "Data"):
        if marker in parts:
            marker_index = parts.index(marker)
            suffix = Path(*parts[marker_index + 1 :]).with_suffix("")
            category_dir = root / parts[0]
            for mask_root_name in ("ground_truth", "masks", "Masks", "GT"):
                candidates.extend(_mask_name_candidates(category_dir / mask_root_name / suffix, image_path))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _mask_name_candidates(base_without_suffix: Path, image_path: Path) -> list[Path]:
    candidates = [base_without_suffix.with_suffix(suffix) for suffix in (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")]
    candidates.append(base_without_suffix.parent / f"{base_without_suffix.name}_mask.png")
    candidates.append(base_without_suffix.parent / f"{image_path.stem}_mask.png")
    return candidates


def _image_id(root: Path, image_path: Path) -> str:
    try:
        relative = relative_posix(image_path, root)
    except ValueError:
        relative = str(image_path)
    key = f"{DATASET_NAME}/{relative}"
    digest = sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{DATASET_NAME}_{digest}"


def _first_value(row: dict[str, str], names: list[str]) -> str:
    lowered = {_normalized(key): value for key, value in row.items()}
    for name in names:
        value = lowered.get(_normalized(name))
        if value is not None and value.strip():
            return value.strip()
    return ""


def _resolve_dataset_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _normalize_split(value: str, csv_path: Path, line_number: int) -> str:
    normalized = _normalized(value)
    if normalized in {"train", "training"}:
        return "train"
    if normalized in {"test", "testing", "val", "validation"}:
        return "test"
    raise ValueError(f"{csv_path}:{line_number} has unsupported split value: {value!r}")


def _normalize_label(value: str, image_path: Path) -> int:
    normalized = _normalized(value)
    if normalized in {"0", "good", "normal", "ok", "false"}:
        return 0
    if normalized in {"1", "bad", "abnormal", "anomaly", "anomalous", "defect", "true"}:
        return 1

    path_parts = {_normalized(part) for part in image_path.parts}
    if path_parts & GOOD_DIR_NAMES:
        return 0
    if path_parts & ANOMALY_DIR_NAMES:
        return 1
    raise ValueError(f"Cannot determine VisA label for image path: {image_path}")


def _infer_category(root: Path, image_path: Path) -> str:
    try:
        return image_path.relative_to(root).parts[0]
    except (ValueError, IndexError):
        return ""


def _infer_defect_type(root: Path, image_path: Path) -> str:
    try:
        parts = image_path.relative_to(root).parts
    except ValueError:
        return "anomaly"
    normalized_parts = [_normalized(part) for part in parts]
    for marker in ("test", "images"):
        if marker in normalized_parts:
            index = normalized_parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    return "anomaly"


def _child_dirs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted((child for child in path.iterdir() if child.is_dir()), key=lambda child: child.name)


def _normalized(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")
