"""Конфигурация приложения. Токен берётся только из окружения / .env."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_ENV_FILE = Path(__file__).resolve().parent / ".env"
DEFAULT_MAX_PHOTO_MB = 10.0
TELEGRAM_FILE_LIMIT_MB = 20.0  # лимит скачивания файлов ботом в Telegram
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}
_TOKEN_RE = re.compile(r"^\d{5,}:[A-Za-z0-9_-]{20,}$")
_OCR_ENGINES = ("tesseract", "pix2tex", "hybrid", "ollama_cloud")
_OCR_LANGS_RE = re.compile(r"^[A-Za-z0-9_]+(\+[A-Za-z0-9_]+)*$")


class ConfigError(RuntimeError):
    """Ошибка конфигурации. Сообщения никогда не содержат значение токена."""


@dataclass(frozen=True, slots=True)
class OcrSettings:
    engine: str = "tesseract"
    tesseract_cmd: str | None = None  # None -> искать в PATH / стандартных папках
    langs: str = "rus+eng"
    timeout_seconds: float = 60.0
    ollama_model: str = "gemma4:cloud"
    ollama_solver_model: str = "gpt-oss:120b-cloud"
    ollama_timeout_seconds: float = 180.0


@dataclass(frozen=True, slots=True)
class Config:
    # repr=False: токен не попадёт в логи при печати объекта
    bot_token: str = field(repr=False)
    max_photo_bytes: int
    log_level: str = "INFO"
    ocr: OcrSettings = field(default_factory=OcrSettings)


def read_ocr_settings() -> OcrSettings:
    """Настройки распознавания из окружения (токен не требуется)."""
    engine = os.environ.get("OCR_ENGINE", "tesseract").strip().lower() or "tesseract"
    if engine not in _OCR_ENGINES:
        raise ConfigError(f"OCR_ENGINE должен быть одним из: {', '.join(_OCR_ENGINES)}.")

    langs = os.environ.get("OCR_LANGS", "rus+eng").strip() or "rus+eng"
    if not _OCR_LANGS_RE.match(langs):
        raise ConfigError("OCR_LANGS должен иметь вид rus+eng (коды языков через +).")

    raw_timeout = os.environ.get("OCR_TIMEOUT_SECONDS", "").strip()
    try:
        timeout = float(raw_timeout) if raw_timeout else 60.0
    except ValueError:
        raise ConfigError("OCR_TIMEOUT_SECONDS должен быть числом, например 60.") from None
    if not 5 <= timeout <= 300:
        raise ConfigError("OCR_TIMEOUT_SECONDS должен быть от 5 до 300.")

    cmd = os.environ.get("TESSERACT_CMD", "").strip().strip('"').strip("'") or None
    model = os.environ.get("OLLAMA_MODEL", "gemma4:cloud").strip() or "gemma4:cloud"
    solver_model = os.environ.get("OLLAMA_SOLVER_MODEL", "gpt-oss:120b-cloud").strip() or "gpt-oss:120b-cloud"
    raw_ollama_timeout = os.environ.get("OLLAMA_TIMEOUT_SECONDS", "").strip()
    try:
        ollama_timeout = float(raw_ollama_timeout) if raw_ollama_timeout else 180.0
    except ValueError:
        raise ConfigError("OLLAMA_TIMEOUT_SECONDS должен быть числом, например 180.") from None
    if not 10 <= ollama_timeout <= 600:
        raise ConfigError("OLLAMA_TIMEOUT_SECONDS должен быть от 10 до 600.")
    if not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,120}", model):
        raise ConfigError("OLLAMA_MODEL содержит недопустимые символы.")
    if not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,120}", solver_model):
        raise ConfigError("OLLAMA_SOLVER_MODEL содержит недопустимые символы.")
    return OcrSettings(
        engine=engine, tesseract_cmd=cmd, langs=langs, timeout_seconds=timeout,
        ollama_model=model, ollama_solver_model=solver_model, ollama_timeout_seconds=ollama_timeout,
    )


def load_config(env_file: str | Path | None = DEFAULT_ENV_FILE) -> Config:
    """Читает .env (если файл есть) и переменные окружения."""
    if env_file is not None and Path(env_file).is_file():
        load_dotenv(env_file, override=False)

    token = os.environ.get("BOT_TOKEN", "").strip()
    if not token:
        raise ConfigError(
            "Не задана переменная BOT_TOKEN. Скопируйте .env.example в .env "
            "и впишите токен от @BotFather."
        )
    if not _TOKEN_RE.match(token):
        raise ConfigError(
            "BOT_TOKEN имеет неверный формат (ожидается вид 123456789:AAA...). "
            "Проверьте токен в .env."
        )

    raw_mb = os.environ.get("MAX_PHOTO_MB", "").strip()
    try:
        max_mb = float(raw_mb) if raw_mb else DEFAULT_MAX_PHOTO_MB
    except ValueError:
        raise ConfigError("MAX_PHOTO_MB должен быть числом, например 10.") from None
    if not 0 < max_mb <= TELEGRAM_FILE_LIMIT_MB:
        raise ConfigError(
            f"MAX_PHOTO_MB должен быть больше 0 и не больше {TELEGRAM_FILE_LIMIT_MB:g}."
        )

    level = os.environ.get("LOG_LEVEL", "INFO").strip().upper() or "INFO"
    if level not in _VALID_LOG_LEVELS:
        raise ConfigError(
            f"LOG_LEVEL должен быть одним из: {', '.join(sorted(_VALID_LOG_LEVELS))}."
        )

    return Config(
        bot_token=token,
        max_photo_bytes=int(max_mb * 1024 * 1024),
        log_level=level,
        ocr=read_ocr_settings(),
    )
