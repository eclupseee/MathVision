"""Общие типы распознавания условия задачи: результат, ошибки, очистка текста.

Конкретные движки: services/tesseract_engine.py (по умолчанию) и
services/pix2tex_engine.py (опционально, экспериментально)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

MAX_CONDITION_CHARS = 3500  # с запасом до лимита Telegram (4096) вместе с рамкой сообщения
MIN_MEANINGFUL_CHARS = 3


class RecognitionError(Exception):
    """Ожидаемая ошибка распознавания. user_message безопасен для показа:
    без путей, токенов и текста исключений."""

    user_message = "Не удалось распознать условие. Попробуйте отправить другое фото."

    def __init__(self, user_message: str | None = None) -> None:
        super().__init__(type(self).__name__)
        if user_message is not None:
            self.user_message = user_message


class OcrUnavailable(RecognitionError):
    user_message = (
        "Распознавание сейчас недоступно: движок OCR не установлен или не настроен. "
        "Сообщите администратору бота."
    )


class OcrTimeout(RecognitionError):
    user_message = (
        "Распознавание заняло слишком много времени. "
        "Попробуйте фото меньшего размера или с более чётким текстом."
    )


class OcrFailed(RecognitionError):
    user_message = (
        "Движок распознавания завершился с ошибкой. Попробуйте ещё раз позже "
        "или отправьте другое фото."
    )


class OcrNoText(RecognitionError):
    user_message = (
        "Не удалось найти текст на фото. Сфотографируйте условие ровнее и при хорошем "
        "освещении, без бликов и теней, и отправьте снова."
    )


class ModelDownloadFailed(RecognitionError):
    user_message = (
        "Не удалось загрузить модель распознавания (нужен интернет при первом запуске). "
        "Проверьте соединение и попробуйте позже."
    )


class OllamaUnavailable(RecognitionError):
    def __init__(self, user_message: str | None = None) -> None:
        super().__init__(user_message)
        self._custom_msg = user_message or "Облачное распознавание Ollama недоступно: проверьте установку Ollama, вход в аккаунт и подключение к интернету."

    def __str__(self) -> str:
        return self._custom_msg




class OllamaTimeout(RecognitionError):
    user_message = "Облачная модель не ответила вовремя. Попробуйте ещё раз позже."


class OllamaFailed(RecognitionError):
    user_message = (
        "Не удалось получить ответ от облачной модели. Проверьте доступ к Ollama Cloud "
        "и попробуйте позже."
    )


@dataclass(frozen=True, slots=True)
class RecognizedCondition:
    text: str
    engine: str
    truncated: bool = False


class Recognizer(Protocol):
    name: str

    async def recognize(self, image_path: Path) -> RecognizedCondition:
        """Распознать условие по файлу изображения. Ошибки — только RecognitionError."""

    def diagnose(self) -> str | None:
        """None, если движок готов; иначе — краткое описание проблемы (для лога)."""


_MULTI_BLANK = re.compile(r"\n{3,}")
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_THINKING_PREFIX = re.compile(
    r"^\s*Thinking\.\.\..*?^\s*(?:Ответ|Answer)\s*:\s*",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)


def clean_text(raw: str) -> str:
    """Нормализует текст и удаляет служебный вывод терминала и модели."""
    text = _ANSI_ESCAPE.sub("", raw)
    text = _THINKING_PREFIX.sub("", text, count=1)
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = "".join(
        ch for ch in text if ch in "\n\t" or unicodedata.category(ch) not in ("Cc", "Cs")
    )
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _MULTI_BLANK.sub("\n\n", text).strip()


def build_condition(raw: str, engine: str) -> RecognizedCondition:
    """Из «сырого» вывода движка делает результат или бросает OcrNoText."""
    text = clean_text(raw)
    if sum(ch.isalnum() for ch in text) < MIN_MEANINGFUL_CHARS:
        raise OcrNoText()
    truncated = len(text) > MAX_CONDITION_CHARS
    if truncated:
        text = text[:MAX_CONDITION_CHARS].rstrip()
    return RecognizedCondition(text=text, engine=engine, truncated=truncated)
