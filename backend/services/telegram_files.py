"""Скачивание файлов из Telegram с переводом сетевых ошибок в DownloadFailed."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import aiohttp
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest

from .photo_errors import DownloadFailed, ImageTooLarge, PhotoError

logger = logging.getLogger(__name__)


class TelegramDownloader:
    def __init__(self, bot: Bot, timeout: float = 30.0) -> None:
        self._bot = bot
        self._timeout = timeout

    async def download(self, file_id: str, destination: Path, max_bytes: int) -> None:
        try:
            tg_file = await self._bot.get_file(file_id)
            if tg_file.file_size is not None and tg_file.file_size > max_bytes:
                raise ImageTooLarge(tg_file.file_size, max_bytes)
            if not tg_file.file_path:
                raise DownloadFailed()
            # Файл пишется потоково прямо на диск (не целиком в память).
            await self._bot.download_file(
                tg_file.file_path, destination=destination, timeout=self._timeout
            )
        except PhotoError:
            raise
        except TelegramBadRequest as exc:
            if "too big" in str(exc).lower():
                raise ImageTooLarge(None, max_bytes) from None
            logger.warning("Telegram отклонил запрос файла: %s", type(exc).__name__)
            raise DownloadFailed() from None
        except (TelegramAPIError, aiohttp.ClientError, TimeoutError, OSError) as exc:
            # from None + только имя класса: в тексте исключений может быть URL с токеном
            logger.warning("Ошибка скачивания файла: %s", type(exc).__name__)
            raise DownloadFailed() from None
