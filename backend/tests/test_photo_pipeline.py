import asyncio
import tempfile
import unittest
from pathlib import Path

from services.images import ImageLimits
from services.photo_errors import (
    CorruptImage,
    DownloadFailed,
    ImageTooLarge,
)
from services.photo_pipeline import IncomingImage, acquired_image, process_image
from tests.imgutil import make_image_bytes

LIMITS = ImageLimits(max_bytes=1024 * 1024)


class FakeDownloader:
    def __init__(self, data=b"", error=None, hang=False, write=True):
        self.data, self.error, self.hang, self.write = data, error, hang, write
        self.calls = 0
        self.started = asyncio.Event()

    async def download(self, file_id, destination: Path, max_bytes):
        self.calls += 1
        if self.write:
            destination.write_bytes(self.data)
        self.started.set()
        if self.hang:
            await asyncio.Event().wait()
        if self.error:
            raise self.error


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def leftovers(self):
        return list(self.root.iterdir())

    async def run_pipeline(self, downloader, file_size=1000):
        return await process_image(
            downloader, IncomingImage("file-id", file_size), LIMITS, temp_root=self.root
        )

    async def test_success_and_temp_removed(self):
        dl = FakeDownloader(make_image_bytes("JPEG", (320, 240), orientation=6))
        info = await self.run_pipeline(dl)
        self.assertEqual((info.format, info.width, info.height), ("JPEG", 240, 320))
        self.assertEqual(self.leftovers(), [])

    async def test_declared_size_too_large_skips_download(self):
        dl = FakeDownloader(b"x")
        with self.assertRaises(ImageTooLarge):
            await self.run_pipeline(dl, file_size=LIMITS.max_bytes + 1)
        self.assertEqual(dl.calls, 0)
        self.assertEqual(self.leftovers(), [])

    async def test_actual_size_too_large_after_download(self):
        dl = FakeDownloader(b"x" * (LIMITS.max_bytes + 1))
        with self.assertRaises(ImageTooLarge):
            await self.run_pipeline(dl, file_size=None)  # размер заранее неизвестен
        self.assertEqual(self.leftovers(), [])

    async def test_corrupt_image_cleans_up(self):
        data = make_image_bytes("JPEG", (400, 400))
        with self.assertRaises(CorruptImage):
            await self.run_pipeline(FakeDownloader(data[: len(data) // 2]))
        self.assertEqual(self.leftovers(), [])

    async def test_empty_file_is_corrupt(self):
        with self.assertRaises(CorruptImage):
            await self.run_pipeline(FakeDownloader(b""))
        self.assertEqual(self.leftovers(), [])

    async def test_download_error_cleans_up_partial_file(self):
        dl = FakeDownloader(b"partial", error=DownloadFailed())
        with self.assertRaises(DownloadFailed):
            await self.run_pipeline(dl)
        self.assertEqual(self.leftovers(), [])

    async def test_missing_file_after_download(self):
        with self.assertRaises(DownloadFailed):
            await self.run_pipeline(FakeDownloader(write=False))
        self.assertEqual(self.leftovers(), [])

    async def test_cancellation_cleans_up(self):
        dl = FakeDownloader(b"partial", hang=True)
        task = asyncio.create_task(self.run_pipeline(dl))
        await asyncio.wait_for(dl.started.wait(), 2)
        self.assertEqual(len(self.leftovers()), 1)  # временная папка существует
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.leftovers(), [])

    async def test_unexpected_exception_cleans_up(self):
        dl = FakeDownloader(b"x", error=RuntimeError("boom"))
        with self.assertRaises(RuntimeError):
            await self.run_pipeline(dl)
        self.assertEqual(self.leftovers(), [])


class AcquiredImageTests(unittest.IsolatedAsyncioTestCase):
    """Новый контекстный менеджер этапа 3: файл живёт только внутри блока."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def acquire(self, downloader):
        return acquired_image(downloader, IncomingImage("fid", 1000), LIMITS, temp_root=self.root)

    async def test_file_available_inside_and_removed_after(self):
        dl = FakeDownloader(make_image_bytes("JPEG", (320, 240)))
        async with self.acquire(dl) as accepted:
            self.assertTrue(accepted.path.is_file())
            self.assertEqual(accepted.info.format, "JPEG")
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_removed_when_body_raises(self):
        dl = FakeDownloader(make_image_bytes("JPEG", (320, 240)))
        with self.assertRaises(RuntimeError):
            async with self.acquire(dl):
                raise RuntimeError("recognition failed")
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_removed_when_body_cancelled(self):
        dl = FakeDownloader(make_image_bytes("JPEG", (320, 240)))
        started = asyncio.Event()

        async def body():
            async with self.acquire(dl):
                started.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(body())
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_invalid_image_never_enters_block(self):
        entered = False
        with self.assertRaises(CorruptImage):
            async with self.acquire(FakeDownloader(b"junk" * 50)):
                entered = True
        self.assertFalse(entered)
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
