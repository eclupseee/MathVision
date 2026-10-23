import logging
from contextlib import suppress

from aiogram import Router
from aiogram.types import ErrorEvent

logger = logging.getLogger(__name__)
router = Router(name="errors")


@router.errors()
async def on_error(event: ErrorEvent) -> bool:
    # Трассировка проходит через SecretRedactor.
    logger.error("Необработанная ошибка в обработчике", exc_info=event.exception)
    message = event.update.message
    if message is not None:
        with suppress(Exception):
            await message.answer("Произошла внутренняя ошибка. Попробуйте позже.")
    return True
