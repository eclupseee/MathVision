"""Выбор движка распознавания по настройкам."""
from __future__ import annotations

from config import OcrSettings

from .recognition import Recognizer


def build_recognizer(settings: OcrSettings) -> Recognizer:
    if settings.engine == "ollama_cloud":
        from .ollama_cloud_engine import OllamaCloudRecognizer

        return OllamaCloudRecognizer(settings)
    if settings.engine == "hybrid":
        from .hybrid_engine import HybridRecognizer

        return HybridRecognizer(settings)
    if settings.engine == "pix2tex":
        from .pix2tex_engine import Pix2TexRecognizer

        return Pix2TexRecognizer(settings)
    from .tesseract_engine import TesseractRecognizer

    return TesseractRecognizer(settings)
