from aiogram import F, Router
from aiogram.types import Message

router = Router(name="fallback")


@router.message(F.text.startswith("/"))
async def on_unknown_command(message: Message) -> None:
    await message.answer("Неизвестная команда. Список возможностей — /help.")


@router.message()
async def on_other(message: Message) -> None:
    await message.answer("Этот тип сообщений не поддерживается. Отправьте фото или /help.")
