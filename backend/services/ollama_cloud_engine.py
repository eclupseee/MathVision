"""Распознавание печатных математических заданий через Ollama Cloud."""
from __future__ import annotations

from pathlib import Path

from config import OcrSettings
from .ollama_cli import image_prompt, run_ollama
from .recognition import RecognizedCondition, build_condition


class OllamaCloudRecognizer:
    name = "ollama_cloud"

    def __init__(self, settings: OcrSettings) -> None:
        self._settings = settings

    def diagnose(self) -> str | None:
        import os
        mode = os.environ.get("OLLAMA_MODE", "local").strip().lower()
        if mode == "cloud":
            if not os.environ.get("OLLAMA_API_KEY", "").strip():
                return "Облачный режим выбран, но OLLAMA_API_KEY не задан"
            return None

        import shutil
        return None if shutil.which("ollama") else "команда ollama не найдена в PATH"

    async def recognize(self, image_path: Path) -> RecognizedCondition:

        raw = await run_ollama(
            self._settings.ollama_model,
            image_prompt(image_path),
            self._settings.ollama_timeout_seconds,
            image_path=image_path,
        )
        return build_condition(raw, self.name)
