"""Чистая логика проверки входящих фото (без зависимости от aiogram)."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar


class _PhotoSizeLike(Protocol):
    width: int
    height: int
    file_size: int | None


P = TypeVar("P", bound=_PhotoSizeLike)


@dataclass(frozen=True, slots=True)
class PhotoCheck:
    ok: bool
    reason: str | None = None


def pick_largest_photo(photos: Sequence[P]) -> P | None:
    """Telegram присылает несколько размеров одного фото; берём самый большой."""
    if not photos:
        return None
    return max(photos, key=lambda p: (p.width * p.height, p.file_size or 0))


def format_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.1f} МБ"


def check_photo_size(file_size: int | None, max_bytes: int) -> PhotoCheck:
    """Если Telegram не сообщил размер, пропускаем: на этапе 1 файл не скачивается.
    Повторная проверка при скачивании — задача следующего этапа."""
    if file_size is not None and file_size > max_bytes:
        return PhotoCheck(
            ok=False,
            reason=(
                f"Фото слишком большое ({format_mb(file_size)}). "
                f"Максимум — {format_mb(max_bytes)}. Отправьте фото меньшего размера."
            ),
        )
    return PhotoCheck(ok=True)
