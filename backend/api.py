"""Локальный веб-API MathVision."""
import asyncio
import json
import urllib.error
import urllib.request
from pydantic import BaseModel, Field
import logging
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import read_ocr_settings
from services.recognizers import build_recognizer
from services.recognition import RecognitionError
from services.solver import solve_condition

load_dotenv()

app = FastAPI(title="MathVision API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://eclupseee.github.io",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
logger = logging.getLogger(__name__)
class SolveRequest(BaseModel):
    condition: str = Field(min_length=1, max_length=12000)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "MathVision API"}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Загрузите фотографию в формате JPEG, PNG или WebP.",
        )

    data = await file.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Файл пустой.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Размер фотографии не должен превышать 10 МБ.",
        )

    suffix = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }[file.content_type]

    try:
        with tempfile.TemporaryDirectory(prefix="mathvision_") as temp_dir:
            image_path = Path(temp_dir) / f"upload{suffix}"
            image_path.write_bytes(data)

            settings = read_ocr_settings()
            recognizer = build_recognizer(settings)
            problem = recognizer.diagnose()
            if problem:
                raise HTTPException(
                    status_code=503,
                    detail=f"Распознавание пока недоступно: {problem}",
                )

            result = await recognizer.recognize(image_path)

        return {
            "text": result.text,
            "engine": result.engine,
            "truncated": result.truncated,
        }
    except HTTPException:
        raise
    except RecognitionError as exc:
        raise HTTPException(status_code=422, detail=exc.user_message) from None
    except Exception:
        logger.exception("Ошибка обработки изображения в API")
        raise HTTPException(
            status_code=500,
            detail="Не удалось обработать фотографию. Попробуйте другое изображение.",
        ) from None
    finally:
        await file.close()


@app.post("/solve")
async def solve(request: SolveRequest):
    condition = request.condition.strip()
    if not condition:
        raise HTTPException(
            status_code=422,
            detail="Введите условие математической задачи.",
        )

    try:
        settings = read_ocr_settings()
        solution = await solve_condition(condition, settings)

        return {
            "answer": solution.text,
            "verified": solution.verified,
            "model": settings.ollama_model,
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Ошибка решения задачи через MathVision")
        raise HTTPException(
            status_code=502,
            detail="Не удалось получить решение. Проверьте доступность модели и повторите попытку.",
        ) from None
