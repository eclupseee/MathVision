"""Обработчики. Порядок роутеров важен:
ошибки -> команды -> подтверждение (по состояниям) -> контент -> запасной."""
from aiogram import Router

from . import commands, confirmation, content, errors, fallback


def build_root_router() -> Router:
    root = Router(name="root")
    root.include_routers(
        errors.router,
        commands.router,
        confirmation.router,
        content.router,
        fallback.router,
    )
    return root
