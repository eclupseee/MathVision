import asyncio
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from config import OcrSettings
from services.recognition import OcrFailed, OcrNoText, OcrTimeout, OcrUnavailable
from services.tesseract_engine import TesseractRecognizer, find_tesseract
from tests.imgutil import make_image_bytes

EXE = sys.executable  # существующий исполняемый файл; реально не запускается (spawn подменён)


class FakeProc:
    def __init__(self, stdout=b"", stderr=b"", returncode=0, hang=False):
        self.stdout, self.stderr, self.returncode, self.hang = stdout, stderr, returncode, hang
        self.killed = False

    async def communicate(self):
        if self.hang:
            await asyncio.Event().wait()
        return self.stdout, self.stderr

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


class FakeSpawn:
    def __init__(self, proc=None, error=None):
        self.proc, self.error = proc, error
        self.calls = []
        self.called = asyncio.Event()

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.called.set()
        if self.error:
            raise self.error
        return self.proc


class FindTesseractTests(unittest.TestCase):
    def test_explicit_command_wins(self):
        found = find_tesseract("X:/t.exe", which=lambda c: None, exists=lambda p: p == "X:/t.exe")
        self.assertEqual(found, "X:/t.exe")

    def test_path_lookup(self):
        self.assertEqual(find_tesseract(None, which=lambda c: "/bin/tesseract"), "/bin/tesseract")

    def test_windows_default_folder(self):
        found = find_tesseract(
            None, which=lambda c: None, exists=lambda p: p.endswith("tesseract.exe"),
            candidates=("C:/a/tesseract.exe",),
        )
        self.assertEqual(found, "C:/a/tesseract.exe")

    def test_not_found(self):
        self.assertIsNone(find_tesseract("nope", which=lambda c: None, exists=lambda p: False))
        self.assertIsNone(
            find_tesseract(None, which=lambda c: None, exists=lambda p: False, candidates=())
        )


class TesseractEngineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.image = self.dir / "incoming.bin"
        self.image.write_bytes(make_image_bytes("PNG", (400, 200)))

    def engine(self, spawn, **overrides):
        settings = OcrSettings(
            tesseract_cmd=EXE, langs="rus+eng", timeout_seconds=overrides.pop("timeout", 5),
        )
        return TesseractRecognizer(settings, spawn=spawn)

    async def test_success_builds_safe_command(self):
        spawn = FakeSpawn(FakeProc(stdout="Найти предел функции f(x)\n".encode()))
        result = await self.engine(spawn).recognize(self.image)
        self.assertEqual((result.engine, result.text), ("tesseract", "Найти предел функции f(x)"))
        args, kwargs = spawn.calls[0]
        self.assertEqual(args[0], EXE)
        self.assertIn("rus+eng", args)
        self.assertNotIn("shell", kwargs)  # без shell -> нет инъекции команд
        self.assertTrue(args[1].endswith("ocr_input.png"))

    async def test_executable_not_found(self):
        settings = OcrSettings(tesseract_cmd="/definitely/not/here/tesseract", timeout_seconds=5)
        with self.assertRaises(OcrUnavailable):
            await TesseractRecognizer(settings, spawn=FakeSpawn()).recognize(self.image)

    async def test_spawn_oserror_is_unavailable(self):
        with self.assertRaises(OcrUnavailable):
            await self.engine(FakeSpawn(error=PermissionError("denied"))).recognize(self.image)

    async def test_missing_language_with_zero_exit_code(self):
        # Реальное поведение Tesseract 5: код 0, но в stderr — ошибка языка
        err = b"Error opening data file rus.traineddata\nFailed loading language 'rus'\n"
        spawn = FakeSpawn(FakeProc(stdout=b"garbage text here", stderr=err, returncode=0))
        with self.assertRaises(OcrUnavailable) as ctx:
            await self.engine(spawn).recognize(self.image)
        self.assertIn("языковой пакет", ctx.exception.user_message)

    async def test_nonzero_exit_is_failure(self):
        with self.assertRaises(OcrFailed):
            await self.engine(FakeSpawn(FakeProc(returncode=1, stderr=b"crash"))).recognize(
                self.image
            )

    async def test_empty_output_is_no_text(self):
        with self.assertRaises(OcrNoText):
            await self.engine(FakeSpawn(FakeProc(stdout=b"  \n"))).recognize(self.image)

    async def test_timeout_kills_process(self):
        proc = FakeProc(hang=True)
        with self.assertRaises(OcrTimeout):
            await self.engine(FakeSpawn(proc), timeout=0.05).recognize(self.image)
        self.assertTrue(proc.killed)

    async def test_cancellation_kills_process(self):
        proc, spawn = FakeProc(hang=True), None
        spawn = FakeSpawn(proc)
        task = asyncio.create_task(self.engine(spawn).recognize(self.image))
        await asyncio.wait_for(spawn.called.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(proc.killed)

    async def test_unreadable_image_is_failure_not_crash(self):
        self.image.write_bytes(b"not an image")
        with self.assertRaises(OcrFailed):
            await self.engine(FakeSpawn(FakeProc(stdout=b"text"))).recognize(self.image)


def _tesseract_langs() -> set[str]:
    exe = shutil.which("tesseract")
    if not exe:
        return set()
    out = subprocess.run([exe, "--list-langs"], capture_output=True, text=True, check=False)
    return {line.strip() for line in out.stdout.splitlines()[1:]}


@unittest.skipUnless("eng" in _tesseract_langs(), "Tesseract (eng) не установлен")
class RealTesseractTests(unittest.IsolatedAsyncioTestCase):
    """Интеграционные тесты с настоящим Tesseract (пропускаются, если его нет)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.image = Path(self._tmp.name) / "incoming.bin"
        img = Image.new("RGB", (1000, 160), "white")
        font = ImageFont.load_default(size=40)
        ImageDraw.Draw(img).text((30, 50), "Compute the derivative of the function", font=font, fill="black")
        img.save(self.image, "PNG")

    async def test_reads_printed_text(self):
        engine = TesseractRecognizer(OcrSettings(langs="eng", timeout_seconds=30))
        result = await engine.recognize(self.image)
        self.assertIn("derivative", result.text.lower())

    async def test_missing_language_is_reported(self):
        engine = TesseractRecognizer(OcrSettings(langs="eng+zzz", timeout_seconds=30))
        with self.assertRaises(OcrUnavailable):
            await engine.recognize(self.image)

    async def test_diagnose(self):
        self.assertIsNone(TesseractRecognizer(OcrSettings(langs="eng")).diagnose())
        self.assertIn("zzz", TesseractRecognizer(OcrSettings(langs="eng+zzz")).diagnose())


if __name__ == "__main__":
    unittest.main()
