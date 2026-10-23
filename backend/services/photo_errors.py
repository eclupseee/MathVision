"""Ожидаемые ошибки приёма фото. Тексты user_message безопасны для показа
пользователю и не содержат токенов, путей или деталей исключений."""
from __future__ import annotations

from .validation import format_mb


class PhotoError(Exception):
    user_message = "Не удалось обработать фото. Попробуйте отправить другое."

    def __init__(self, user_message: str | None = None) -> None:
        super().__init__(type(self).__name__)
        if user_message is not None:
            self.user_message = user_message


class ImageTooLarge(PhotoError):
    def __init__(self, size: int | None, limit: int) -> None:
        shown = f" ({format_mb(size)})" if size else ""
        super().__init__(
            f"Фото слишком большое{shown}. Максимум — {format_mb(limit)}. "
            "Отправьте файл меньшего размера."
        )


class ImageResolutionTooLarge(PhotoError):
    def __init__(self, max_pixels: int) -> None:
        super().__init__(
            f"Разрешение слишком большое (максимум — {max_pixels // 1_000_000} МП). "
            "Уменьшите изображение и отправьте снова."
        )


class ImageTooSmall(PhotoError):
    def __init__(self, min_side: int) -> None:
        super().__init__(
            f"Изображение слишком маленькое (минимум {min_side} px по каждой стороне)."
        )


class UnsupportedImageFormat(PhotoError):
    def __init__(self, user_message: str | None = None) -> None:
        super().__init__(
            user_message
            or "Этот формат не поддерживается. Отправьте изображение JPEG, PNG или WebP."
        )


class CorruptImage(PhotoError):
    user_message = (
        "Не удалось прочитать изображение: файл повреждён или не является картинкой. "
        "Попробуйте отправить фото ещё раз."
    )


class DownloadFailed(PhotoError):
    user_message = (
        "Не удалось получить фото от Telegram (проблема сети или сервера). "
        "Попробуйте отправить его ещё раз через минуту."
    )
