import os
import unittest
from unittest.mock import AsyncMock, patch
from config import read_ocr_settings, ConfigError, OcrSettings
from services.solver import solve_condition

class TestModelSeparation(unittest.IsolatedAsyncioTestCase):
    def test_solver_model_config_default(self):
        """Проверка значения по умолчанию для модели решателя."""
        with patch.dict(os.environ, {}, clear=True):
            settings = read_ocr_settings()
            self.assertEqual(settings.ollama_solver_model, "gpt-oss:120b-cloud")

    def test_solver_model_config_custom(self):
        """Проверка загрузки модели решателя из окружения."""
        with patch.dict(os.environ, {"OLLAMA_SOLVER_MODEL": "my-custom-solver"}):
            settings = read_ocr_settings()
            self.assertEqual(settings.ollama_solver_model, "my-custom-solver")

    def test_solver_model_validation(self):
        """Проверка валидации имени модели решателя."""
        with patch.dict(os.environ, {"OLLAMA_SOLVER_MODEL": "invalid;model!"}):
            with self.assertRaises(ConfigError, msg="OLLAMA_SOLVER_MODEL содержит недопустимые символы"):
                read_ocr_settings()

    async def test_solve_condition_uses_solver_model(self):
        """Проверка того, что solve_condition использует именно модель решателя, а не OCR."""
        settings = OcrSettings(
            ollama_model="ocr-model",
            ollama_solver_model="solver-model",
            ollama_timeout_seconds=60.0
        )

        # Мокаем run_ollama, чтобы проверить какой аргумент передан
        with patch("services.solver.run_ollama", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = '{"steps_markdown": "Step 1", "final_answer": "42"}'

            await solve_condition("x + 1 = 43", settings=settings)

            # Первый аргумент run_ollama должен быть моделью решателя
            args, _ = mock_run.call_args
            self.assertEqual(args[0], "solver-model")
            self.assertNotEqual(args[0], "ocr-model")

    async def test_recognize_uses_ocr_model(self):
        """Проверка того, что OllamaCloudRecognizer использует именно модель OCR, а не решателя."""
        settings = OcrSettings(
            ollama_model="ocr-model",
            ollama_solver_model="solver-model",
            ollama_timeout_seconds=60.0
        )
        from services.ollama_cloud_engine import OllamaCloudRecognizer
        recognizer = OllamaCloudRecognizer(settings)

        with patch("services.ollama_cloud_engine.run_ollama", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Recognized text"

            with patch("pathlib.Path.is_file", return_value=True), \
                 patch("pathlib.Path.read_bytes", return_value=b"fake-image"):
                from pathlib import Path
                await recognizer.recognize(Path("fake.png"))

                args, _ = mock_run.call_args
                self.assertEqual(args[0], "ocr-model")
                self.assertNotEqual(args[0], "solver-model")

if __name__ == "__main__":
    unittest.main()
