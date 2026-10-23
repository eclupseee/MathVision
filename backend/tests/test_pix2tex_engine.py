import tempfile
import threading
import unittest
from pathlib import Path

from config import OcrSettings
from services.pix2tex_engine import Pix2TexRecognizer
from services.recognition import (
    ModelDownloadFailed,
    OcrFailed,
    OcrNoText,
    OcrTimeout,
    OcrUnavailable,
)
from tests.imgutil import make_image_bytes


class Pix2TexEngineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.image = Path(self._tmp.name) / "incoming.bin"
        self.image.write_bytes(make_image_bytes("PNG", (300, 100)))

    def engine(self, loader, timeout=5):
        return Pix2TexRecognizer(OcrSettings(engine="pix2tex", timeout_seconds=timeout), loader=loader)

    async def test_success_runs_off_event_loop_and_loads_once(self):
        loads, threads = [], []

        def loader():
            loads.append(1)
            return lambda img: (threads.append(threading.get_ident()) or r"\int_0^1 x^2\,dx")

        eng = self.engine(loader)
        first = await eng.recognize(self.image)
        await eng.recognize(self.image)
        self.assertEqual(first.text, r"\int_0^1 x^2\,dx")
        self.assertEqual(first.engine, "pix2tex")
        self.assertEqual(len(loads), 1)
        self.assertNotIn(threading.get_ident(), threads)

    async def test_import_error_is_unavailable(self):
        def loader():
            raise ImportError("no module named pix2tex")

        with self.assertRaises(OcrUnavailable):
            await self.engine(loader).recognize(self.image)

    async def test_weights_download_network_error(self):
        def loader():
            raise ConnectionError("cannot reach github")

        with self.assertRaises(ModelDownloadFailed):
            await self.engine(loader).recognize(self.image)

    async def test_wrapped_network_error_detected(self):
        def loader():
            try:
                raise TimeoutError("slow")
            except TimeoutError as inner:
                raise RuntimeError("download failed") from inner

        with self.assertRaises(ModelDownloadFailed):
            await self.engine(loader).recognize(self.image)

    async def test_model_error(self):
        with self.assertRaises(OcrFailed):
            await self.engine(lambda: (lambda img: 1 / 0)).recognize(self.image)

    async def test_empty_result(self):
        with self.assertRaises(OcrNoText):
            await self.engine(lambda: (lambda img: "  ")).recognize(self.image)

    async def test_timeout(self):
        release = threading.Event()
        self.addCleanup(release.set)
        with self.assertRaises(OcrTimeout):
            await self.engine(lambda: (lambda img: release.wait(5) and "x"), timeout=0.05).recognize(
                self.image
            )

    def test_diagnose_reports_missing_package(self):
        problem = Pix2TexRecognizer(OcrSettings(engine="pix2tex")).diagnose()
        # В тестовой среде pix2tex не установлен; если установлен — проблем нет.
        self.assertTrue(problem is None or "pix2tex" in problem)


if __name__ == "__main__":
    unittest.main()
