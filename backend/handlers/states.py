from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup


class Scenario(StatesGroup):
    awaiting_cloud_consent = State()  # ждём согласие на отправку фото в облако
    processing = State()  # фото скачивается / распознаётся
    awaiting_confirmation = State()  # показано распознанное условие, ждём ответ
    editing = State()  # пользователь исправляет текст
    awaiting_solution_consent = State()  # ждём согласие отправить текст условия в облачную модель
    confirmed = State()  # условие подтверждено


async def in_state(fsm: FSMContext, target: State) -> bool:
    return await fsm.get_state() == target.state
