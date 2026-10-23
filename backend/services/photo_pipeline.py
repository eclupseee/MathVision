"""Приём фото: скачивание во временную папку, проверка, гарантированная очистка."""
from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .images import ImageInfo, ImageLimits, inspect_image
from .photo_errors import CorruptImage, DownloadFailed, ImageTooLarge
from .validation import check_photo_size

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IncomingImage:
    file_id: str
    file_size: int | None  # по данным Telegram; может быть неизвестен


class FileDownloader(Protocol):
    async def download(self, file_id: str, destination: Path, max_bytes: int) -> None:
        """Скачать файл в destination. Ожидаемые сбои — как PhotoError."""


def _remove_dir(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("Не удалось удалить временную папку (приём фото)")


@dataclass(frozen=True, slots=True)
class AcceptedImage:
    path: Path  # доступен только внутри `async with`; потом папка удаляется
    info: ImageInfo


@asynccontextmanager
async def acquired_image(
    downloader: FileDownloader,
    incoming: IncomingImage,
    limits: ImageLimits,
    *,
    temp_root: Path | None = None,
) -> AsyncIterator[AcceptedImage]:
    """Скачивает и проверяет фото, отдаёт путь к временному файлу.

    Временная папка удаляется при выходе из блока: после успеха, при ошибке
    (в том числе внутри блока) и при отмене (CancelledError). Имя файла
    фиксированное, не от пользователя."""
    if not check_photo_size(incoming.file_size, limits.max_bytes).ok:
        raise ImageTooLarge(incoming.file_size, limits.max_bytes)

    tmp_dir = Path(tempfile.mkdtemp(prefix="mathvision_", dir=temp_root))
    try:
        destination = tmp_dir / "incoming.bin"
        await downloader.download(incoming.file_id, destination, limits.max_bytes)

        try:
            size = destination.stat().st_size
        except FileNotFoundError:
            raise DownloadFailed() from None
        if size > limits.max_bytes:
            raise ImageTooLarge(size, limits.max_bytes)
        if size == 0:
            raise CorruptImage()

        worker = asyncio.ensure_future(
            asyncio.to_thread(inspect_image, destination, limits, size)
        )
        try:
            # shield: при отмене поток доработает и закроет файл до удаления папки
            info = await asyncio.shield(worker)
        except asyncio.CancelledError:
            await asyncio.wait({worker})
            if not worker.cancelled():
                worker.exception()  # пометить исключение как обработанное
            raise
        yield AcceptedImage(path=destination, info=info)
    finally:
        _remove_dir(tmp_dir)


async def process_image(
    downloader: FileDownloader,
    incoming: IncomingImage,
    limits: ImageLimits,
    *,
    temp_root: Path | None = None,
) -> ImageInfo:
    """Только проверка фото (этап 2); временные файлы удаляются сразу."""
    async with acquired_image(downloader, incoming, limits, temp_root=temp_root) as accepted:
        return accepted.info
