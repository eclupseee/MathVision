import asyncio
import tempfile
import unittest
from pathlib import Path

from services.images import ImageLimits
from services.jobs import ActiveJobs, JobCancelled, run_cancellable
from services.photo_pipeline import IncomingImage, acquired_image
from tests.imgutil import make_image_bytes
from tests.test_photo_pipeline import FakeDownloader

LIMITS = ImageLimits(max_bytes=1024 * 1024)


class JobsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.registry = ActiveJobs()

    def hanging_job(self, started: asyncio.Event):
        async def job():
            dl = FakeDownloader(make_image_bytes("JPEG", (320, 240)))
            async with acquired_image(dl, IncomingImage("f", 1000), LIMITS, temp_root=self.root):
                started.set()
                await asyncio.Event().wait()

        return job()

    async def test_returns_value_and_unregisters(self):
        async def job():
            return 42

        self.assertEqual(await run_cancellable(self.registry, 1, job()), 42)
        self.assertFalse(self.registry.is_active(1))

    async def test_exception_propagates_and_unregisters(self):
        async def job():
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            await run_cancellable(self.registry, 1, job())
        self.assertFalse(self.registry.is_active(1))

    async def test_user_cancel_stops_job_and_cleans_temp_files(self):
        started = asyncio.Event()
        task = asyncio.create_task(run_cancellable(self.registry, 7, self.hanging_job(started)))
        await asyncio.wait_for(started.wait(), 2)
        self.assertEqual(len(list(self.root.iterdir())), 1)  # временная папка есть
        self.assertTrue(self.registry.is_active(7))
        self.assertTrue(self.registry.cancel(7))
        with self.assertRaises(JobCancelled):
            await task
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertFalse(self.registry.is_active(7))

    async def test_outer_cancellation_is_not_swallowed_and_cleans_up(self):
        started = asyncio.Event()
        task = asyncio.create_task(run_cancellable(self.registry, 7, self.hanging_job(started)))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_cancel_when_nothing_running(self):
        self.assertFalse(self.registry.cancel(99))


if __name__ == "__main__":
    unittest.main()
