"""Локальное распознавание текста через Tesseract (внешняя программа, без pip-зависимостей).

Запуск — asyncio-подпроцесс без shell: event loop не блокируется, процесс
убивается по таймауту и при отмене. Текст распознанного условия не логируется."""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from config import OcrSettings

from .ocr_preprocess import prepare_image_for_ocr
from .recognition import (
    OcrFailed,
    OcrTimeout,
    OcrUnavailable,
    RecognizedCondition,
    build_condition,
)

logger = logging.getLogger(__name__)

_WINDOWS_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)
_LANG_ERROR_MARKERS = ("failed loading language", "error opening data file")


def find_tesseract(
    command: str | None,
    *,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[str], bool] = os.path.isfile,
    candidates: tuple[str, ...] = _WINDOWS_CANDIDATES,
) -> str | None:
    """Путь к tesseract: явная настройка -> PATH -> стандартные папки Windows."""
    if command:
        return which(command) or (command if exists(command) else None)
    found = which("tesseract")
    if found:
        return found
    return next((c for c in candidates if exists(c)), None)


class TesseractRecognizer:
    name = "tesseract"

    def __init__(self, settings: OcrSettings, *, spawn=asyncio.create_subprocess_exec) -> None:
        self._settings = settings
        self._spawn = spawn

    def _executable(self) -> str:
        exe = find_tesseract(self._settings.tesseract_cmd)
        if exe is None:
            raise OcrUnavailable(
                "Распознавание недоступно: на сервере бота не найден Tesseract OCR. "
                "Сообщите администратору бота."
            )
        return exe

    def diagnose(self) -> str | None:
        exe = find_tesseract(self._settings.tesseract_cmd)
        if exe is None:
            return "Tesseract не найден (установите его или задайте TESSERACT_CMD)"
        try:
            result = subprocess.run(
                [exe, "--list-langs"], capture_output=True, text=True, timeout=15, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return "Tesseract найден, но не запускается"
        available = {ln.strip() for ln in result.stdout.splitlines()[1:]}
        missing = [lang for lang in self._settings.langs.split("+") if lang not in available]
        if missing:
            return f"не установлены языковые пакеты Tesseract: {', '.join(missing)}"
        return None

    async def recognize(self, image_path: Path) -> RecognizedCondition:
        exe = self._executable()
        prepared = image_path.with_name("ocr_input.png")  # лежит в той же временной папке
        try:
            await asyncio.to_thread(prepare_image_for_ocr, image_path, prepared)
        except (OSError, ValueError, SyntaxError) as exc:
            logger.warning("Не удалось подготовить изображение к OCR: %s", type(exc).__name__)
            raise OcrFailed() from None

        args = [
            exe, str(prepared), "stdout",
            "-l", self._settings.langs, "--psm", "6", "--dpi", "300",
        ]
        try:
            proc = await self._spawn(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        except OSError as exc:
            logger.warning("Не удалось запустить Tesseract: %s", type(exc).__name__)
            raise OcrUnavailable() from None

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._settings.timeout_seconds
            )
        except asyncio.TimeoutError:
            await self._kill(proc)
            logger.warning("Tesseract: превышено время ожидания")
            raise OcrTimeout() from None
        except BaseException:  # включая CancelledError: не оставляем процесс-сироту
            await self._kill(proc)
            raise

        err = stderr.decode("utf-8", errors="replace").lower()
        # Tesseract 5 при отсутствии языкового пакета может вернуть код 0 и молча
        # продолжить на оставшихся языках (для русского текста это даст мусор),
        # поэтому stderr проверяется всегда, а не только при ошибке.
        if any(marker in err for marker in _LANG_ERROR_MARKERS):
            logger.warning("Tesseract: не найден языковой пакет")
            raise OcrUnavailable(
                "Распознавание недоступно: не установлен языковой пакет Tesseract "
                f"({self._settings.langs}). Сообщите администратору бота."
            )
        if proc.returncode != 0:
            logger.warning("Tesseract завершился с кодом %s", proc.returncode)
            raise OcrFailed()
        return build_condition(stdout.decode("utf-8", errors="replace"), self.name)

    @staticmethod
    async def _kill(proc) -> None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.shield(proc.wait())
        except Exception:  # noqa: BLE001 - очистка не должна маскировать исходную ошибку
            pass
