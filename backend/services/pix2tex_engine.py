"""ЭКСПЕРИМЕНТАЛЬНЫЙ движок: pix2tex (формула на изображении -> LaTeX).

Не входит в обязательные зависимости. pix2tex рассчитан на ОДНУ формулу в блоке
и плохо читает обычный текст; на Python 3.13 его совместимость не гарантирована
(известна проблема с albumentations 2.x). Модель и torch тяжёлые, веса
скачиваются при первом запуске. См. README."""
from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path

from PIL import Image, ImageOps

from config import OcrSettings

from .recognition import (
    ModelDownloadFailed,
    OcrFailed,
    OcrTimeout,
    OcrUnavailable,
    RecognizedCondition,
    build_condition,
)

logger = logging.getLogger(__name__)
_NETWORK_MODULES = {"requests", "urllib3", "httpx", "http", "socket", "ssl"}


def _default_loader() -> Callable[[Image.Image], str]:
    from pix2tex.cli import LatexOCR  # импорт здесь: пакет необязателен

    return LatexOCR()


def _is_network_error(exc: BaseException | None, depth: int = 0) -> bool:
    if exc is None or depth > 5:
        return False
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if type(exc).__module__.split(".")[0] in _NETWORK_MODULES:
        return True
    return _is_network_error(exc.__cause__ or exc.__context__, depth + 1)


class Pix2TexRecognizer:
    name = "pix2tex"

    def __init__(
        self,
        settings: OcrSettings,
        *,
        loader: Callable[[], Callable[[Image.Image], str]] = _default_loader,
    ) -> None:
        self._timeout = settings.timeout_seconds
        self._loader = loader
        self._model: Callable[[Image.Image], str] | None = None
        self._lock = threading.Lock()

    def diagnose(self) -> str | None:
        if self._loader is _default_loader and find_spec("pix2tex") is None:
            return "пакет pix2tex не установлен (см. README, раздел про pix2tex)"
        return None

    def _run(self, image_path: Path) -> str:
        with self._lock:  # модель загружается один раз; инференс не параллелим
            if self._model is None:
                self._model = self._loader()
            model = self._model
            with Image.open(image_path) as src:
                img = ImageOps.exif_transpose(src)
                img.load()  # файл закрыт до инференса
            return str(model(img))

    async def recognize(self, image_path: Path) -> RecognizedCondition:
        try:
            latex = await asyncio.wait_for(
                asyncio.to_thread(self._run, image_path), timeout=self._timeout
            )
        except asyncio.TimeoutError:
            raise OcrTimeout() from None
        except ImportError:
            raise OcrUnavailable(
                "Движок pix2tex не установлен. Сообщите администратору бота."
            ) from None
        except Exception as exc:  # noqa: BLE001 - любая ошибка модели -> понятное сообщение
            logger.warning("Ошибка pix2tex: %s", type(exc).__name__)
            if _is_network_error(exc):
                raise ModelDownloadFailed() from None
            raise OcrFailed() from None
        return build_condition(latex, self.name)
