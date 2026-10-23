"""Экспериментальный гибридный OCR: текст через Tesseract, формула через pix2tex."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from PIL import Image, ImageOps

from config import OcrSettings
from .recognition import (
    OcrFailed,
    OcrNoText,
    RecognizedCondition,
    RecognitionError,
    build_condition,
)
from .tesseract_engine import TesseractRecognizer
from .pix2tex_engine import Pix2TexRecognizer

logger = logging.getLogger(__name__)

def _clean_formula_output(raw: str) -> str:
    """Убирает очевидный посторонний текст и разделители по краям формулы."""
    text = raw.strip()

    # Убираем обрамление LaTeX-блока, если модель его добавила.
    for marker in (r"\[", r"\]", "$"):
        text = text.replace(marker, "")

    # Убираем только разделители по краям, не затрагивая выражение внутри.
    return text.strip().strip(";").strip()


class HybridRecognizer:
    name = "hybrid"

    def __init__(self, settings: OcrSettings) -> None:
        self._settings = settings
        self._text_engine = TesseractRecognizer(settings)
        self._formula_engine = Pix2TexRecognizer(settings)

    def diagnose(self) -> str | None:
        text_problem = self._text_engine.diagnose()
        formula_problem = self._formula_engine.diagnose()
        problems = []
        if text_problem:
            problems.append(f"Tesseract: {text_problem}")
        if formula_problem:
            problems.append(f"pix2tex: {formula_problem}")
        return "; ".join(problems) if problems else None

    async def recognize(self, image_path: Path) -> RecognizedCondition:
        text_result: RecognizedCondition | None = None
        formula_result: RecognizedCondition | None = None

        text_crop = image_path.with_name("ocr_text_crop.png")
        try:
            await asyncio.to_thread(self._crop_image, image_path, text_crop, "top")
            text_result = await self._text_engine.recognize(text_crop)
        except RecognitionError as exc:
            logger.info("Tesseract в гибридном режиме: %s", type(exc).__name__)
        finally:
            text_crop.unlink(missing_ok=True)

        try:
            formula_result = await self._recognize_formula(image_path)
        except RecognitionError as exc:
            logger.info("pix2tex в гибридном режиме: %s", type(exc).__name__)

        parts: list[str] = []
        if text_result:
            parts.append(text_result.text)
        if formula_result:
            parts.append(f"Формула (LaTeX): {_clean_formula_output(formula_result.text)}")

        if not parts:
            raise OcrFailed()

        return build_condition("\n\n".join(parts), self.name)

    @staticmethod
    def _crop_image(source_path: Path, target_path: Path, region: str) -> None:
        with Image.open(source_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            if region == "top":
                crop = image.crop((0, 0, width, max(1, height * 2 // 5)))
            else:
                crop = image.crop((0, height // 2, width, height))
            crop.save(target_path, format="PNG")

    async def _recognize_formula(self, image_path: Path) -> RecognizedCondition:
        crop_path = image_path.with_name("ocr_formula_crop.png")
        try:
            await asyncio.to_thread(
                self._crop_image, image_path, crop_path, "bottom"
            )
            return await self._formula_engine.recognize(crop_path)
        finally:
            crop_path.unlink(missing_ok=True)
