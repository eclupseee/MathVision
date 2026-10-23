import asyncio
import logging
from contextlib import suppress

from aiogram import Bot, F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from config import Config
from services.images import ALLOWED_MIME_TYPES, ImageLimits
from services.jobs import ACTIVE_JOBS, JobCancelled, run_cancellable
from services.photo_errors import PhotoError
from services.photo_pipeline import IncomingImage, acquired_image
from services.recognition import RecognitionError, RecognizedCondition, Recognizer
from services.telegram_files import TelegramDownloader
from services.validation import pick_largest_photo

from .confirmation import show_preview
from .states import Scenario, in_state

logger = logging.getLogger(__name__)
router = Router(name="content")

# Не даём обрабатывать слишком много файлов одновременно (диск/CPU).
_CONCURRENCY = asyncio.Semaphore(3)

BUSY_TEXT = "Предыдущее фото ещё обрабатывается. Подождите немного или нажмите /cancel."
TRUNCATED_NOTE = "\n\n(Текст был слишком длинным и обрезан — при необходимости исправьте его.)"
CB_CLOUD_ALLOW = "mv:cloud:allow"
CB_CLOUD_DECLINE = "mv:cloud:decline"


async def _ask_cloud_consent(message: Message, state: FSMContext, incoming: IncomingImage) -> None:
    """Не отправляем изображение в облако, пока пользователь явно не согласился."""
    if ACTIVE_JOBS.is_active(message.chat.id):
        await message.answer(BUSY_TEXT)
        return
    await state.clear()
    await state.set_state(Scenario.awaiting_cloud_consent)
    await state.update_data(
        cloud_file_id=incoming.file_id,
        cloud_file_size=incoming.file_size,
    )
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="☁️ Отправить в Ollama Cloud", callback_data=CB_CLOUD_ALLOW),
        InlineKeyboardButton(text="Отмена", callback_data=CB_CLOUD_DECLINE),
    ]])
    await message.answer(
        "Для распознавания это фото будет отправлено через интернет в Ollama Cloud "
        "(внешний облачный сервис). Продолжайте только если согласны. "
        "Если не согласны, нажмите «Отмена»; можно переключить OCR_ENGINE на локальный движок.",
        reply_markup=keyboard,
    )


@router.callback_query(F.data == CB_CLOUD_ALLOW)
async def on_cloud_consent(
    callback: CallbackQuery, state: FSMContext, bot: Bot, config: Config, recognizer: Recognizer
) -> None:
    if not await in_state(state, Scenario.awaiting_cloud_consent):
        await callback.answer("Запрос устарел. Отправьте фото ещё раз.", show_alert=True)
        return
    data = await state.get_data()
    file_id = data.get("cloud_file_id")
    if not file_id:
        await callback.answer("Запрос устарел. Отправьте фото ещё раз.", show_alert=True)
        await state.clear()
        return
    await callback.answer("Отправляю фото на распознавание…")
    if callback.message is not None:
        await _intake(
            callback.message,
            state,
            bot,
            config,
            recognizer,
            IncomingImage(str(file_id), data.get("cloud_file_size")),
        )


@router.callback_query(F.data == CB_CLOUD_DECLINE)
async def on_cloud_decline(callback: CallbackQuery, state: FSMContext) -> None:
    if not await in_state(state, Scenario.awaiting_cloud_consent):
        await callback.answer("Запрос устарел.", show_alert=True)
        return
    await state.clear()
    await callback.answer()
    if callback.message is not None:
        with suppress(Exception):
            await callback.message.edit_reply_markup(reply_markup=None)
        await callback.message.answer("Отправка в облако отменена. Фото не отправлено.")



async def _recognize_job(
    bot: Bot, incoming: IncomingImage, limits: ImageLimits, recognizer: Recognizer
) -> RecognizedCondition:
    async with _CONCURRENCY:
        async with acquired_image(TelegramDownloader(bot), incoming, limits) as accepted:
            info = accepted.info
            logger.info(
                "Фото принято: %s %sx%s, %s байт", info.format, info.width, info.height,
                info.size_bytes,
            )
            # Временная папка живёт, пока идёт распознавание; затем удаляется.
            return await recognizer.recognize(accepted.path)


async def _intake(
    message: Message,
    state: FSMContext,
    bot: Bot,
    config: Config,
    recognizer: Recognizer,
    incoming: IncomingImage,
) -> None:
    chat_id = message.chat.id
    if ACTIVE_JOBS.is_active(chat_id):
        await message.answer(BUSY_TEXT)
        return

    await state.clear()
    await state.set_state(Scenario.processing)
    limits = ImageLimits(max_bytes=config.max_photo_bytes)
    try:
        condition = await run_cancellable(
            ACTIVE_JOBS, chat_id, _recognize_job(bot, incoming, limits, recognizer)
        )
    except JobCancelled:
        return  # /cancel уже сбросил состояние и ответил пользователю
    except (PhotoError, RecognitionError) as exc:
        logger.info("Фото не обработано: %s", type(exc).__name__)
        await state.clear()
        await message.answer(exc.user_message)
        return
    except Exception:
        await state.clear()
        raise  # общий обработчик ошибок залогирует и ответит пользователю

    if not await in_state(state, Scenario.processing):
        return  # пользователь отменил сценарий, пока шло распознавание
    # Текст условия намеренно не пишем в лог (содержимое пользователя).
    logger.info("Распознано (%s): %d симв.", condition.engine, len(condition.text))
    await show_preview(message, state, condition.text, engine=condition.engine)
    if condition.truncated:
        await message.answer(TRUNCATED_NOTE.strip())


@router.message(F.photo)
async def on_photo(
    message: Message, state: FSMContext, bot: Bot, config: Config, recognizer: Recognizer
) -> None:
    photo = pick_largest_photo(message.photo or [])
    if photo is None:
        await message.answer("Не удалось прочитать фото. Попробуйте ещё раз.")
        return
    incoming = IncomingImage(photo.file_id, photo.file_size)
    if getattr(recognizer, "name", "") == "ollama_cloud":
        await _ask_cloud_consent(message, state, incoming)
        return
    await _intake(message, state, bot, config, recognizer, incoming)


@router.message(F.document)
async def on_document(
    message: Message, state: FSMContext, bot: Bot, config: Config, recognizer: Recognizer
) -> None:
    doc = message.document
    if doc is None or doc.mime_type not in ALLOWED_MIME_TYPES:
        await message.answer(
            "Этот тип файла не поддерживается. Отправьте фото или изображение "
            "в формате JPEG, PNG или WebP."
        )
        return
    incoming = IncomingImage(doc.file_id, doc.file_size)
    if getattr(recognizer, "name", "") == "ollama_cloud":
        await _ask_cloud_consent(message, state, incoming)
        return
    await _intake(message, state, bot, config, recognizer, incoming)


@router.message(F.text & ~F.text.startswith("/"))
async def on_text(message: Message) -> None:
    await message.answer(
        "Я принимаю фотографии печатных заданий. Отправьте фото условия "
        "или нажмите /help."
    )
