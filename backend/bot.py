"""Точка входа MathVision."""
from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher

from config import Config, ConfigError, load_config
from handlers import build_root_router
from logging_setup import setup_logging
from services.recognizers import build_recognizer

logger = logging.getLogger(__name__)


def build_dispatcher(config: Config) -> Dispatcher:
    dp = Dispatcher()
    dp["config"] = config  # внедряется в обработчики по имени параметра
    dp["recognizer"] = build_recognizer(config.ocr)
    dp.include_router(build_root_router())
    return dp


async def main(config: Config) -> None:
    bot = Bot(token=config.bot_token)
    dp = build_dispatcher(config)
    problem = dp["recognizer"].diagnose()
    if problem:
        logger.warning("Распознавание не готово: %s. Бот запустится, но ответит ошибкой.", problem)
    else:
        logger.info("Движок распознавания готов: %s", dp["recognizer"].name)
    logger.info("Запуск polling...")
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await bot.session.close()
        logger.info("Бот остановлен.")


def _force_utf8_streams() -> None:
    """Чтобы русский текст не превращался в кракозябры в консоли Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def run() -> None:
    _force_utf8_streams()
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    setup_logging(config.log_level, secrets=[config.bot_token])
    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        logger.info("Остановлено пользователем (Ctrl+C).")


if __name__ == "__main__":
    run()
