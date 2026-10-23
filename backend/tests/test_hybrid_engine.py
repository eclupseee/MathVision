import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from config import OcrSettings
from services.hybrid_engine import HybridRecognizer
from services.recognition import RecognizedCondition, OcrFailed
from tests.imgutil import make_image_bytes


class HybridEngineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.image = Path(self.tmp.name) / "input.png"
        self.image.write_bytes(make_image_bytes("PNG", (300, 100)))

    async def test_combines_text_and_formula(self):
        engine = HybridRecognizer(OcrSettings(engine="hybrid"))
        engine._text_engine.recognize = AsyncMock(
            return_value=RecognizedCondition("Найти интеграл", "tesseract")
        )
        engine._recognize_formula = AsyncMock(
            return_value=RecognizedCondition(r"\int x^2\,dx", "pix2tex")
        )

        result = await engine.recognize(self.image)

        self.assertEqual(result.engine, "hybrid")
        self.assertIn("Найти интеграл", result.text)
        self.assertIn(r"\int x^2\,dx", result.text)

    async def test_keeps_text_if_formula_fails(self):
        engine = HybridRecognizer(OcrSettings(engine="hybrid"))
        engine._text_engine.recognize = AsyncMock(
            return_value=RecognizedCondition("Найти интеграл", "tesseract")
        )
        engine._recognize_formula = AsyncMock(side_effect=OcrFailed())

        result = await engine.recognize(self.image)

        self.assertIn("Найти интеграл", result.text)

    async def test_fails_if_both_engines_fail(self):
        engine = HybridRecognizer(OcrSettings(engine="hybrid"))
        engine._text_engine.recognize = AsyncMock(side_effect=OcrFailed())
        engine._recognize_formula = AsyncMock(side_effect=OcrFailed())

        with self.assertRaises(OcrFailed):
            await engine.recognize(self.image)


if __name__ == "__main__":
    unittest.main()
