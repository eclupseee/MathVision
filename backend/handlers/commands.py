from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from services.jobs import ACTIVE_JOBS

router = Router(name="commands")

START_TEXT = (
    "Привет! Я MathVision — помощник по высшей математике.\n\n"
    "Как это работает:\n"
    "1. Вы присылаете фото печатного задания.\n"
    "2. Я распознаю условие и показываю его вам.\n"
    "3. Вы подтверждаете или исправляете текст — и только потом я смогу решать.\n\n"
    "После подтверждения условия бот может подготовить пошаговое решение через Ollama Cloud. "
    "Перед отправкой фото или текста во внешний облачный сервис бот запросит отдельное согласие.\n\n"
    "Команды: /help, /cancel"
)

HELP_TEXT = (
    "Что умею сейчас:\n"
    "• принимать фото (и изображения JPEG/PNG/WebP файлом);\n"
    "• проверять размер, целостность и ориентацию (EXIF);\n"
    "• распознавать условие и показывать его для подтверждения или исправления.\n\n"
    "Ограничения:\n"
    "• задачи пока не решаются автоматически без подтверждения условия;\n"
    "• сложные формулы могут распознаваться неточно — всегда проверяйте текст;\n"
    "• для решения текст условия отправляется в Ollama Cloud только после вашего согласия;\n"
    "• при OCR_ENGINE=ollama_cloud фото также отправляется в облако только после отдельного согласия;\n"
    "• временный локальный файл фото удаляется после обработки, но данные, отправленные в облако, "
    "обрабатываются по правилам Ollama.\n\n"
    "/cancel — отменить текущий сценарий или распознавание."
)


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(START_TEXT)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT)


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    stopped = ACTIVE_JOBS.cancel(message.chat.id)  # останавливает идущее распознавание
    if await state.get_state() is None and not stopped:
        await message.answer("Сейчас нечего отменять.")
        return
    await state.clear()
    await message.answer("Сценарий отменён. Можете прислать новое фото.")
