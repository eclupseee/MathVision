import os
import unittest
from unittest.mock import patch

from config import Config, ConfigError, OcrSettings, load_config, read_ocr_settings

TOKEN = "123456789:AAEhBOweik6ad2K8eoZxQ3pTq9zYb-dE_fg"  # вымышленный, не настоящий


class OcrConfigTests(unittest.TestCase):
    def env(self, **extra):
        return patch.dict(os.environ, {"BOT_TOKEN": TOKEN, **extra}, clear=True)

    def test_defaults(self):
        with self.env():
            s = read_ocr_settings()
        self.assertEqual(s, OcrSettings("tesseract", None, "rus+eng", 60.0))

    def test_load_config_includes_ocr_and_hides_nothing_secret(self):
        with self.env(OCR_LANGS="eng", TESSERACT_CMD='"C:\\Tess\\tesseract.exe"'):
            cfg = load_config(env_file=None)
        self.assertEqual(cfg.ocr.langs, "eng")
        self.assertEqual(cfg.ocr.tesseract_cmd, "C:\\Tess\\tesseract.exe")  # кавычки сняты
        self.assertNotIn(TOKEN, repr(cfg))

    def test_config_still_constructible_without_ocr(self):
        self.assertEqual(Config(bot_token="x", max_photo_bytes=1).ocr.engine, "tesseract")

    def test_invalid_values(self):
        cases = {
            "OCR_ENGINE": "magic",
            "OCR_LANGS": "rus eng; rm -rf",
            "OCR_TIMEOUT_SECONDS": "abc",
        }
        for key, value in cases.items():
            with self.subTest(key=key), self.env(**{key: value}):
                with self.assertRaises(ConfigError) as ctx:
                    read_ocr_settings()
                self.assertIn(key, str(ctx.exception))
        for value in ("1", "999"):
            with self.subTest(timeout=value), self.env(OCR_TIMEOUT_SECONDS=value):
                with self.assertRaises(ConfigError):
                    read_ocr_settings()

    def test_engine_is_case_insensitive(self):
        with self.env(OCR_ENGINE="PIX2TEX"):
            self.assertEqual(read_ocr_settings().engine, "pix2tex")

    def test_missing_token_still_clear_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigError) as ctx:
                load_config(env_file=None)
        self.assertIn("BOT_TOKEN", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
