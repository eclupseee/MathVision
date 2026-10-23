"""Подтверждение / исправление распознанного условия. Решение не запускается."""
import unittest

try:
    from handlers import confirmation
    from handlers.states import Scenario

    HAS_AIOGRAM = True
except ImportError:
    HAS_AIOGRAM = False

from services.recognition import MAX_CONDITION_CHARS
from tests.handler_fakes import FakeCallback, FakeMessage, FakeState

PID = "abc123ef"
TEXT = "Вычислить ∫ x^2 dx от 0 до 1"


def awaiting_state(text=TEXT):
    return FakeState(Scenario.awaiting_confirmation,
                     {"condition_text": text, "pending_id": PID, "engine": "tesseract"})


@unittest.skipUnless(HAS_AIOGRAM, "aiogram не установлен")
class ConfirmationTests(unittest.IsolatedAsyncioTestCase):
    async def test_confirm_saves_text_and_asks_separate_cloud_consent(self):
        state, cb = awaiting_state(), FakeCallback(confirmation.CB_CONFIRM + PID)
        await confirmation.on_confirm(cb, state)
        self.assertEqual(state.state, Scenario.awaiting_solution_consent.state)
        self.assertEqual(state.data["confirmed_text"], TEXT)
        self.assertTrue(cb.message.keyboard_removed)
        self.assertIn("подтверждено", cb.message.answers[0].lower())
        self.assertIn("ollama cloud", cb.message.answers[0].lower())
        self.assertIn("облачной модели", cb.message.last.lower())
        buttons = [b for row in cb.message.markups[-1].inline_keyboard for b in row]
        callbacks = [b.callback_data for b in buttons]
        self.assertTrue(any(v.startswith(confirmation.CB_SOLVE_ALLOW) for v in callbacks))
        self.assertTrue(any(v.startswith(confirmation.CB_SOLVE_DECLINE) for v in callbacks))
        self.assertEqual(len({v.split(":")[-1] for v in callbacks}), 1)

    async def test_stale_or_foreign_buttons_are_ignored(self):
        cases = {
            "wrong token": (awaiting_state(), confirmation.CB_CONFIRM + "deadbeef"),
            "no token": (awaiting_state(), confirmation.CB_CONFIRM),
            "wrong state": (FakeState(Scenario.confirmed, {"pending_id": PID}),
                            confirmation.CB_CONFIRM + PID),
            "no state": (FakeState(None, {}), confirmation.CB_CONFIRM + PID),
        }
        for name, (state, data) in cases.items():
            with self.subTest(name):
                before = state.state
                cb = FakeCallback(data)
                await confirmation.on_confirm(cb, state)
                self.assertEqual(state.state, before)
                self.assertTrue(cb.alerts and cb.alerts[0][1])  # показан alert «устарела»
                self.assertEqual(cb.message.answers, [])

    async def test_confirm_twice_second_press_is_stale(self):
        state = awaiting_state()
        await confirmation.on_confirm(FakeCallback(confirmation.CB_CONFIRM + PID), state)
        cb2 = FakeCallback(confirmation.CB_CONFIRM + PID)
        await confirmation.on_confirm(cb2, state)
        self.assertTrue(cb2.alerts)

    async def test_edit_flow(self):
        state, cb = awaiting_state(), FakeCallback(confirmation.CB_EDIT + PID)
        await confirmation.on_edit(cb, state)
        self.assertEqual(state.state, Scenario.editing.state)
        self.assertIn("исправленное условие", cb.message.last.lower())

        msg = FakeMessage(text="  Вычислить  ∫ x^3 dx\r\n\r\n\r\n\r\nот 0 до 2 ")
        await confirmation.on_corrected_text(msg, state)
        self.assertEqual(state.state, Scenario.awaiting_confirmation.state)
        self.assertEqual(state.data["condition_text"], "Вычислить  ∫ x^3 dx\n\nот 0 до 2")
        self.assertNotEqual(state.data["pending_id"], PID)  # новые кнопки — старые устарели
        self.assertIn("Всё верно?", msg.last)
        self.assertTrue(state.data["engine"].endswith("+правка"))

        old = FakeCallback(confirmation.CB_CONFIRM + PID)
        await confirmation.on_confirm(old, state)
        self.assertEqual(state.state, Scenario.awaiting_confirmation.state)  # не подтвердилось

    async def test_corrected_text_validation(self):
        for text, fragment in (("   \n ", "пустое"), ("а" * (MAX_CONDITION_CHARS + 1), "длинный")):
            with self.subTest(fragment=fragment):
                state = FakeState(Scenario.editing, {"pending_id": PID, "engine": "t"})
                msg = FakeMessage(text=text)
                await confirmation.on_corrected_text(msg, state)
                self.assertIn(fragment, msg.last)
                self.assertEqual(state.state, Scenario.editing.state)  # остаёмся в правке

    async def test_cancel_button(self):
        state, cb = awaiting_state(), FakeCallback(confirmation.CB_CANCEL + PID)
        await confirmation.on_cancel_button(cb, state)
        self.assertIsNone(state.state)
        self.assertEqual(state.data, {})

    async def test_text_while_waiting_gets_hint(self):
        msg = FakeMessage(text="да")
        await confirmation.on_text_while_confirming(msg)
        self.assertIn("Верно", msg.last)

    async def test_callback_data_fits_telegram_limit(self):
        kb = confirmation.confirmation_keyboard(confirmation.new_pending_id())
        for row in kb.inline_keyboard:
            for button in row:
                self.assertLessEqual(len(button.callback_data.encode()), 64)


if __name__ == "__main__":
    unittest.main()
