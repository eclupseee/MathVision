"""Подтверждение и исправление распознанного условия. Решение здесь НЕ запускается."""
import logging
import secrets
from contextlib import suppress

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from config import read_ocr_settings
from services.recognition import MAX_CONDITION_CHARS, RecognitionError, clean_text
from services.solver import solve_condition

from .states import Scenario, in_state

logger = logging.getLogger(__name__)
router = Router(name="confirmation")

CB_CONFIRM = "mv:confirm:"
CB_EDIT = "mv:edit:"
CB_CANCEL = "mv:cancel:"
CB_SOLVE_ALLOW = "mv:solve:allow:"
CB_SOLVE_DECLINE = "mv:solve:decline:"

PREVIEW_TEXT = (
    "Распознанное условие:\n\n{text}\n\n"
    "Проверьте его, особенно формулы, степени, индексы и знаки: распознавание "
    "может ошибаться. Всё верно?"
)
STALE_TEXT = "Эта кнопка устарела. Отправьте фото заново."


def new_pending_id() -> str:
    return secrets.token_hex(4)


def confirmation_keyboard(pending_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Верно", callback_data=CB_CONFIRM + pending_id),
                InlineKeyboardButton(text="✏️ Исправить", callback_data=CB_EDIT + pending_id),
            ],
            [InlineKeyboardButton(text="✖ Отмена", callback_data=CB_CANCEL + pending_id)],
        ]
    )


async def show_preview(message: Message, state: FSMContext, text: str, *, engine: str) -> None:
    """Показывает условие и просит подтвердить/исправить (общая точка для OCR и правок)."""
    pending_id = new_pending_id()
    await state.set_state(Scenario.awaiting_confirmation)
    await state.update_data(condition_text=text, engine=engine, pending_id=pending_id)
    await message.answer(
        PREVIEW_TEXT.format(text=text), reply_markup=confirmation_keyboard(pending_id)
    )


async def _valid_pending(callback: CallbackQuery, state: FSMContext, prefix: str, *states) -> bool:
    """Кнопка действительна, только если она от текущего условия и состояние подходит."""
    data = await state.get_data()
    token = (callback.data or "")[len(prefix):]
    ok = bool(token) and token == data.get("pending_id")
    if ok:
        ok = False
        for st in states:
            if await in_state(state, st):
                ok = True
                break
    if not ok:
        await callback.answer(STALE_TEXT, show_alert=True)
    return ok


async def _remove_keyboard(callback: CallbackQuery) -> None:
    edit = getattr(callback.message, "edit_reply_markup", None)
    if edit is not None:
        with suppress(Exception):  # сообщение могло устареть — не критично
            await edit(reply_markup=None)


async def _reply(callback: CallbackQuery, text: str) -> None:
    answer = getattr(callback.message, "answer", None)
    if answer is not None:
        await answer(text)


@router.callback_query(F.data.startswith(CB_CONFIRM))
async def on_confirm(callback: CallbackQuery, state: FSMContext) -> None:
    if not await _valid_pending(callback, state, CB_CONFIRM, Scenario.awaiting_confirmation):
        return
    data = await state.get_data()
    solve_pending_id = new_pending_id()
    await state.set_state(Scenario.awaiting_solution_consent)
    await state.update_data(
        confirmed_text=data.get("condition_text", ""),
        solve_pending_id=solve_pending_id,
    )
    await callback.answer()
    await _remove_keyboard(callback)
    logger.info("Условие подтверждено пользователем")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="☁️ Решить через Ollama Cloud", callback_data=CB_SOLVE_ALLOW + solve_pending_id),
        InlineKeyboardButton(text="Отмена", callback_data=CB_SOLVE_DECLINE + solve_pending_id),
    ]])
    await _reply(
        callback,
        "✅ Условие подтверждено. Для пошагового решения текст условия будет отправлен "
        "через интернет в Ollama Cloud. Продолжить?",
    )
    answer = getattr(callback.message, "answer", None)
    if answer is not None:
        await answer(
            "Отправить текст условия облачной модели?",
            reply_markup=keyboard,
        )


@router.callback_query(F.data.startswith(CB_SOLVE_ALLOW))
async def on_solve_consent(callback: CallbackQuery, state: FSMContext) -> None:
    data = await state.get_data()
    token = (callback.data or "")[len(CB_SOLVE_ALLOW):]
    if (
        not token
        or token != data.get("solve_pending_id")
        or not await in_state(state, Scenario.awaiting_solution_consent)
    ):
        await callback.answer("Запрос устарел. Подтвердите условие заново.", show_alert=True)
        return
    condition = data.get("confirmed_text", "")
    if not condition:
        await callback.answer("Условие не найдено. Отправьте фото заново.", show_alert=True)
        await state.clear()
        return
    await state.set_state(Scenario.confirmed)
    await callback.answer("Запускаю решение…")
    await _remove_keyboard(callback)
    await _reply(callback, "Решаю задачу. Это может занять до нескольких минут…")
    try:
        solution = await solve_condition(condition, read_ocr_settings())
        await _reply(callback, solution.text[:3900])
    except RecognitionError as exc:
        await _reply(callback, exc.user_message)
    except Exception:
        logger.exception("Не удалось обработать подтверждённую задачу")
        await _reply(
            callback,
            "Не удалось получить или обработать решение. Проверьте доступность Ollama Cloud "
            "и попробуйте позже. Условие осталось подтверждённым; отправьте фото заново "
            "или повторите попытку позже.",
        )


@router.callback_query(F.data.startswith(CB_SOLVE_DECLINE))
async def on_solve_decline(callback: CallbackQuery, state: FSMContext) -> None:
    data = await state.get_data()
    token = (callback.data or "")[len(CB_SOLVE_DECLINE):]
    if (
        not token
        or token != data.get("solve_pending_id")
        or not await in_state(state, Scenario.awaiting_solution_consent)
    ):
        await callback.answer("Запрос устарел.", show_alert=True)
        return
    await state.set_state(Scenario.confirmed)
    await callback.answer()
    await _remove_keyboard(callback)
    await _reply(callback, "Облачное решение отменено. Условие осталось подтверждённым; текст не отправлен.")


@router.callback_query(F.data.startswith(CB_EDIT))
async def on_edit(callback: CallbackQuery, state: FSMContext) -> None:
    if not await _valid_pending(callback, state, CB_EDIT, Scenario.awaiting_confirmation):
        return
    await state.set_state(Scenario.editing)
    await callback.answer()
    await _remove_keyboard(callback)
    await _reply(
        callback,
        "Отправьте исправленное условие одним сообщением "
        f"(до {MAX_CONDITION_CHARS} символов). Для отмены — /cancel.",
    )


@router.callback_query(F.data.startswith(CB_CANCEL))
async def on_cancel_button(callback: CallbackQuery, state: FSMContext) -> None:
    if not await _valid_pending(
        callback, state, CB_CANCEL, Scenario.awaiting_confirmation, Scenario.editing
    ):
        return
    await state.clear()
    await callback.answer()
    await _remove_keyboard(callback)
    await _reply(callback, "Отменено. Можете прислать новое фото.")


@router.message(Scenario.editing, F.text & ~F.text.startswith("/"))
async def on_corrected_text(message: Message, state: FSMContext) -> None:
    text = clean_text(message.text or "")
    if not text:
        await message.answer("Сообщение пустое. Отправьте исправленное условие текстом.")
        return
    if len(text) > MAX_CONDITION_CHARS:
        await message.answer(
            f"Текст слишком длинный ({len(text)} символов). "
            f"Сократите его до {MAX_CONDITION_CHARS} и отправьте снова."
        )
        return
    data = await state.get_data()
    await show_preview(message, state, text, engine=f"{data.get('engine', 'ocr')}+правка")


@router.message(Scenario.awaiting_confirmation, F.text & ~F.text.startswith("/"))
async def on_text_while_confirming(message: Message) -> None:
    await message.answer(
        "Нажмите «✅ Верно» или «✏️ Исправить» под сообщением с условием. "
        "Чтобы начать заново — /cancel или новое фото."
    )
