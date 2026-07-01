"""Shared helpers for dataset manifest generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


@dataclass(frozen=True)
class ImageInspection:
    """Result of opening an image file with PIL."""

    ok: bool
    size: tuple[int, int] | None = None
    error: str | None = None


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def iter_image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if is_image_file(path))


def inspect_image(path: Path) -> ImageInspection:
    """Open and load an image to catch broken files early."""

    try:
        with Image.open(path) as image:
            image.load()
            return ImageInspection(ok=True, size=image.size)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return ImageInspection(ok=False, error=f"{type(exc).__name__}: {exc}")


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def display_path(path: Path, root: Path, path_style: str) -> str:
    if path_style == "absolute":
        return str(path.resolve())
    if path_style == "relative":
        return relative_posix(path, root)
    raise ValueError(f"Unsupported path_style: {path_style}")
