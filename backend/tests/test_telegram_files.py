import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import aiohttp  # noqa: F401
    from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
    from aiogram.methods import GetFile

    from services.telegram_files import TelegramDownloader

    HAS_AIOGRAM = True
except ImportError:
    HAS_AIOGRAM = False

from services.photo_errors import DownloadFailed, ImageTooLarge

LIMIT = 1024


class FakeBot:
    def __init__(self, file_size=100, file_path="photos/a.jpg", get_error=None, dl_error=None):
        self.file_size, self.file_path = file_size, file_path
        self.get_error, self.dl_error = get_error, dl_error
        self.downloaded = None

    async def get_file(self, file_id):
        if self.get_error:
            raise self.get_error
        return SimpleNamespace(file_size=self.file_size, file_path=self.file_path)

    async def download_file(self, file_path, destination, timeout):
        if self.dl_error:
            raise self.dl_error
        Path(destination).write_bytes(b"data")
        self.downloaded = file_path


@unittest.skipUnless(HAS_AIOGRAM, "aiogram не установлен")
class TelegramDownloaderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dest = Path(self._tmp.name) / "incoming.bin"

    async def download(self, bot):
        await TelegramDownloader(bot).download("fid", self.dest, LIMIT)

    def method(self):
        return GetFile(file_id="fid")

    async def test_success(self):
        bot = FakeBot()
        await self.download(bot)
        self.assertEqual(self.dest.read_bytes(), b"data")
        self.assertEqual(bot.downloaded, "photos/a.jpg")

    async def test_network_error_on_get_file(self):
        err = TelegramNetworkError(method=self.method(), message="https://api/bot123:SECRET")
        with self.assertRaises(DownloadFailed) as ctx:
            await self.download(FakeBot(get_error=err))
        self.assertIsNone(ctx.exception.__cause__)  # URL/токен не тянется в traceback

    async def test_network_error_on_download(self):
        err = TelegramNetworkError(method=self.method(), message="timeout")
        with self.assertRaises(DownloadFailed):
            await self.download(FakeBot(dl_error=err))

    async def test_timeout(self):
        with self.assertRaises(DownloadFailed):
            await self.download(FakeBot(dl_error=asyncio.TimeoutError()))

    async def test_telegram_file_is_too_big(self):
        err = TelegramBadRequest(method=self.method(), message="Bad Request: file is too big")
        with self.assertRaises(ImageTooLarge):
            await self.download(FakeBot(get_error=err))

    async def test_other_bad_request(self):
        err = TelegramBadRequest(method=self.method(), message="Bad Request: wrong file_id")
        with self.assertRaises(DownloadFailed):
            await self.download(FakeBot(get_error=err))

    async def test_reported_size_over_limit(self):
        with self.assertRaises(ImageTooLarge):
            await self.download(FakeBot(file_size=LIMIT + 1))


if __name__ == "__main__":
    unittest.main()
