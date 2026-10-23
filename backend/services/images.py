"""Синхронная проверка изображения через Pillow.

Вызывать только из потока (asyncio.to_thread): не блокирует event loop.
Файл читается с диска; JPEG декодируется в уменьшенном масштабе (draft),
поэтому память не раздувается. Число пикселей ограничено заранее."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .photo_errors import (
    CorruptImage,
    ImageResolutionTooLarge,
    ImageTooSmall,
    UnsupportedImageFormat,
)

ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
ALLOWED_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
EXIF_ORIENTATION_TAG = 0x0112
_ORIENTATIONS_SWAPPING_SIDES = frozenset({5, 6, 7, 8})
_READ_ERRORS = (OSError, SyntaxError, ValueError, EOFError)  # включает UnidentifiedImageError


@dataclass(frozen=True, slots=True)
class ImageLimits:
    max_bytes: int
    max_pixels: int = 25_000_000
    min_side: int = 64


@dataclass(frozen=True, slots=True)
class ImageInfo:
    format: str
    width: int  # размеры с учётом EXIF-ориентации (как увидит человек)
    height: int
    size_bytes: int
    exif_orientation: int | None

    @property
    def needs_rotation(self) -> bool:
        return self.exif_orientation not in (None, 1)


def _read_header(path: Path, limits: ImageLimits) -> tuple[str, int, int, int | None]:
    with Image.open(path) as img:  # ленивое чтение: декодирования ещё нет
        fmt = img.format or ""
        if fmt not in ALLOWED_FORMATS:
            raise UnsupportedImageFormat()
        width, height = img.size
        if width * height > limits.max_pixels:
            raise ImageResolutionTooLarge(limits.max_pixels)
        if min(width, height) < limits.min_side:
            raise ImageTooSmall(limits.min_side)
        if getattr(img, "n_frames", 1) > 1:
            raise UnsupportedImageFormat("Анимированные изображения не поддерживаются.")
        try:
            orientation = img.getexif().get(EXIF_ORIENTATION_TAG)
        except Exception:  # noqa: BLE001 - битые метаданные не повод отклонять картинку
            orientation = None
    return fmt, width, height, orientation if isinstance(orientation, int) else None


def _verify_structure(path: Path) -> None:
    # verify() обязан идти сразу после open() и делает объект непригодным.
    with Image.open(path) as img:
        img.verify()  # для PNG проверяет контрольные суммы блоков


def _decode_fully(path: Path, width: int, height: int) -> None:
    """Полное декодирование: ловит обрезанные и испорченные данные."""
    with Image.open(path) as img:
        if img.format == "JPEG":
            img.draft(None, (max(1, width // 8), max(1, height // 8)))
        img.load()


def inspect_image(path: Path, limits: ImageLimits, size_bytes: int) -> ImageInfo:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        try:
            fmt, width, height, orientation = _read_header(path, limits)
            _verify_structure(path)
            _decode_fully(path, width, height)
        except Image.DecompressionBombError:
            raise ImageResolutionTooLarge(limits.max_pixels) from None
        except _READ_ERRORS as exc:
            raise CorruptImage() from exc

    if orientation in _ORIENTATIONS_SWAPPING_SIDES:
        width, height = height, width
    return ImageInfo(fmt, width, height, size_bytes, orientation)
