"""Inspect a VisA dataset directory structure without generating manifests.

The report is intentionally descriptive: it records observed folders and image
extension counts so a later parser can be designed without assuming a fixed
layout. The raw dataset tree is only read, never modified.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = Path("outputs/stage1_gate/visa_structure_report.json")
IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
TRAIN_NAMES = {"train", "training"}
TEST_NAMES = {"test", "testing", "val", "validation"}
GOOD_NAMES = {"good", "normal", "ok"}
BAD_NAMES = {"bad", "abnormal", "anomaly", "anomalous", "defect", "defective", "ng"}
MASK_KEYWORDS = {
    "annotation",
    "annotations",
    "ground_truth",
    "gt",
    "label",
    "labels",
    "mask",
    "masks",
    "segmentation",
}
NON_CATEGORY_NAMES = TRAIN_NAMES | TEST_NAMES | GOOD_NAMES | BAD_NAMES | MASK_KEYWORDS | {
    "csv",
    "image",
    "images",
    "readme",
    "split",
    "splits",
    "split_csv",
}


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = inspect_visa_structure(Path(args.visa_root))
    except (FileNotFoundError, NotADirectoryError) as exc:
        parser.error(str(exc))
    output_path = Path(args.output_json)
    write_report(output_path, report)
    print(f"Wrote {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visa-root", required=True, help="Path to the raw VisA dataset root.")
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path for the read-only structure report JSON.",
    )
    return parser


def inspect_visa_structure(visa_root: Path) -> dict[str, Any]:
    root = visa_root.expanduser()
    if not root.exists():
        raise FileNotFoundError(f"VisA root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"VisA root is not a directory: {root}")

    top_level_dirs = sorted(_child_dir_names(root))
    image_counts = _count_images(root)
    role_folders = _detect_role_folders(root)
    category_candidates = _detect_category_candidates(root)

    return {
        "dataset": "visa",
        "visa_root": str(root),
        "notes": [
            "read_only_structure_probe",
            "no_manifest_generated",
            "no_models_run",
            "no_downloads",
        ],
        "top_level_folders": top_level_dirs,
        "category_folders": category_candidates,
        "detected_folders": role_folders,
        "image_file_counts": image_counts,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _detect_category_candidates(root: Path) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}

    for child in _child_dirs(root):
        if child.name.startswith(".") or _normalized_name(child) in NON_CATEGORY_NAMES:
            continue
        candidates[child.name] = _category_record(root, child, source="top_level")

    for split_dir in _matching_dirs(root, TRAIN_NAMES | TEST_NAMES):
        if split_dir.parent != root:
            continue
        for child in _child_dirs(split_dir):
            if child.name.startswith(".") or _normalized_name(child) in NON_CATEGORY_NAMES:
                continue
            key = child.name
            if key not in candidates:
                candidates[key] = _category_record(root, child, source=f"{split_dir.name}_child")

    return [candidates[name] for name in sorted(candidates)]


def _category_record(root: Path, category_dir: Path, *, source: str) -> dict[str, Any]:
    return {
        "name": category_dir.name,
        "relative_path": _relative_posix(category_dir, root),
        "source": source,
        "direct_children": sorted(_child_dir_names(category_dir)),
        "train_folders": _relative_matches(root, category_dir, TRAIN_NAMES),
        "test_folders": _relative_matches(root, category_dir, TEST_NAMES),
        "good_folders": _relative_matches(root, category_dir, GOOD_NAMES),
        "bad_or_anomaly_folders": _relative_matches(root, category_dir, BAD_NAMES),
        "mask_like_folders": _relative_keyword_matches(root, category_dir, MASK_KEYWORDS),
        "image_file_counts": _count_images(category_dir),
    }


def _detect_role_folders(root: Path) -> dict[str, list[str]]:
    return {
        "train": _relative_matches(root, root, TRAIN_NAMES),
        "test": _relative_matches(root, root, TEST_NAMES),
        "good": _relative_matches(root, root, GOOD_NAMES),
        "bad_or_anomaly": _relative_matches(root, root, BAD_NAMES),
        "mask_like": _relative_keyword_matches(root, root, MASK_KEYWORDS),
    }


def _count_images(root: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    total = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            counts[ext] += 1
            total += 1
    return {
        "total": total,
        "by_extension": {ext: counts[ext] for ext in sorted(counts)},
    }


def _relative_matches(root: Path, search_root: Path, names: set[str]) -> list[str]:
    return [_relative_posix(path, root) for path in _matching_dirs(search_root, names)]


def _relative_keyword_matches(root: Path, search_root: Path, keywords: set[str]) -> list[str]:
    matches = []
    for path in _all_dirs(search_root):
        normalized = _normalized_name(path)
        if normalized in keywords or any(keyword in normalized for keyword in keywords):
            matches.append(_relative_posix(path, root))
    return sorted(matches)


def _matching_dirs(root: Path, names: set[str]) -> list[Path]:
    return sorted(
        (path for path in _all_dirs(root) if _normalized_name(path) in names),
        key=lambda path: path.as_posix(),
    )


def _all_dirs(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_dir()]


def _child_dirs(root: Path) -> list[Path]:
    return sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)


def _child_dir_names(root: Path) -> list[str]:
    return [path.name for path in _child_dirs(root)]


def _normalized_name(path: Path) -> str:
    return path.name.lower().replace("-", "_").replace(" ", "_")


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


if __name__ == "__main__":
    main()
