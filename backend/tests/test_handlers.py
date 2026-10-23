"""Команды этапов 1-2 и сценарий «фото -> распознавание -> подтверждение» (без Telegram)."""
import asyncio
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from handlers import commands, confirmation, content, fallback
    from handlers.states import Scenario

    HAS_AIOGRAM = True
except ImportError:
    HAS_AIOGRAM = False

from config import Config
from services.images import ImageInfo
from services.jobs import ACTIVE_JOBS
from services.photo_errors import CorruptImage, DownloadFailed, ImageTooLarge
from services.photo_pipeline import AcceptedImage
from services.recognition import (
    ModelDownloadFailed,
    OcrFailed,
    OcrNoText,
    OcrTimeout,
    OcrUnavailable,
    RecognizedCondition,
)
from tests.handler_fakes import FakeMessage, FakeState
from tests.imgutil import make_image_bytes
from tests.test_photo_pipeline import FakeDownloader

CONFIG = Config(bot_token="x", max_photo_bytes=1024 * 1024)
INFO = ImageInfo("JPEG", 1280, 960, 150_000, None)
CONDITION_TEXT = "Найти предел lim x->0 sin(x)/x"


class FakeRecognizer:
    name = "fake"

    def __init__(self, text=CONDITION_TEXT, error=None, hang=False, truncated=False):
        self.text, self.error, self.hang, self.truncated = text, error, hang, truncated
        self.started = asyncio.Event()
        self.paths = []

    def diagnose(self):
        return None

    async def recognize(self, image_path):
        self.paths.append(image_path)
        self.started.set()
        if self.hang:
            await asyncio.Event().wait()
        if self.error:
            raise self.error
        return RecognizedCondition(self.text, self.name, self.truncated)


def photo_msg(chat_id=1):
    msg = FakeMessage(chat_id=chat_id)
    msg.photo = [
        SimpleNamespace(file_id="small", width=90, height=60, file_size=1000),
        SimpleNamespace(file_id="big", width=1280, height=960, file_size=150_000),
    ]
    return msg


def fake_acquired(captured=None):
    @asynccontextmanager
    async def cm(downloader, incoming, limits, **kwargs):
        if captured is not None:
            captured.append(incoming)
        yield AcceptedImage(Path("fake-path"), INFO)

    return cm


@unittest.skipUnless(HAS_AIOGRAM, "aiogram не установлен")
class ExistingCommandsTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_clears_state_and_greets(self):
        msg, state = FakeMessage(), FakeState(Scenario.confirmed, {"x": 1})
        await commands.cmd_start(msg, state)
        self.assertIsNone(state.state)
        self.assertIn("MathVision", msg.last)

    async def test_help_describes_limits(self):
        msg = FakeMessage()
        await commands.cmd_help(msg)
        self.assertIn("/cancel", msg.last)
        self.assertIn("не решаются", msg.last)
        self.assertIn("формулы", msg.last)

    async def test_cancel_without_scenario(self):
        msg, state = FakeMessage(), FakeState()
        await commands.cmd_cancel(msg, state)
        self.assertIn("нечего отменять", msg.last)

    async def test_cancel_resets_every_scenario_state(self):
        for st in (Scenario.processing, Scenario.awaiting_confirmation, Scenario.editing,
                   Scenario.confirmed):
            with self.subTest(state=st.state):
                msg, state = FakeMessage(), FakeState(st, {"condition_text": "t"})
                await commands.cmd_cancel(msg, state)
                self.assertIsNone(state.state)
                self.assertEqual(state.data, {})
                self.assertIn("отменён", msg.last)

    async def test_unknown_command_and_text_still_handled(self):
        msg = FakeMessage(text="/foo")
        await fallback.on_unknown_command(msg)
        self.assertIn("Неизвестная команда", msg.last)
        msg = FakeMessage(text="привет")
        await content.on_text(msg)
        self.assertIn("фото", msg.last)


@unittest.skipUnless(HAS_AIOGRAM, "aiogram не установлен")
class RecognitionFlowTests(unittest.IsolatedAsyncioTestCase):
    async def run_photo(self, recognizer, msg=None, state=None, acquired=None):
        msg, state = msg or photo_msg(), state or FakeState()
        with patch.object(content, "acquired_image", acquired or fake_acquired()):
            await content.on_photo(msg, state, bot=object(), config=CONFIG, recognizer=recognizer)
        return msg, state

    async def test_success_shows_text_and_asks_confirmation_without_solving(self):
        captured = []
        msg, state = await self.run_photo(FakeRecognizer(), acquired=fake_acquired(captured))
        self.assertEqual(captured[0].file_id, "big")  # выбран самый большой размер
        self.assertEqual(state.state, Scenario.awaiting_confirmation.state)
        self.assertEqual(state.data["condition_text"], CONDITION_TEXT)
        self.assertIn(CONDITION_TEXT, msg.last)
        self.assertIn("Всё верно?", msg.last)
        self.assertNotIn("решени", msg.last.lower())  # ничего не решаем
        buttons = [b for row in msg.markups[-1].inline_keyboard for b in row]
        self.assertEqual(len(buttons), 3)
        self.assertTrue(all(state.data["pending_id"] in b.callback_data for b in buttons))

    async def test_cloud_recognizer_requires_explicit_consent_before_download(self):
        recognizer = FakeRecognizer()
        recognizer.name = "ollama_cloud"
        msg, state = photo_msg(), FakeState()
        with patch.object(content, "acquired_image", side_effect=AssertionError("must not download")):
            await content.on_photo(msg, state, bot=object(), config=CONFIG, recognizer=recognizer)
        self.assertEqual(state.state, Scenario.awaiting_cloud_consent.state)
        self.assertEqual(state.data["cloud_file_id"], "big")
        self.assertIn("Ollama Cloud", msg.last)
        buttons = [b for row in msg.markups[-1].inline_keyboard for b in row]
        self.assertEqual({b.callback_data for b in buttons},
                         {content.CB_CLOUD_ALLOW, content.CB_CLOUD_DECLINE})

    async def test_expected_errors_are_reported_and_state_reset(self):
        errors = (
            OcrNoText(), OcrUnavailable(), OcrTimeout(), OcrFailed(), ModelDownloadFailed(),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__):
                msg, state = await self.run_photo(FakeRecognizer(error=error))
                self.assertEqual(msg.last, error.user_message)
                self.assertIsNone(state.state)
                self.assertEqual(len(msg.answers), 1)  # никакого «условия» не показано

    async def test_photo_errors_still_reported(self):
        for error in (ImageTooLarge(5_000_000, 1_000_000), CorruptImage(), DownloadFailed()):
            with self.subTest(error=type(error).__name__):
                @asynccontextmanager
                async def failing(*a, _e=error, **k):
                    raise _e
                    yield  # pragma: no cover

                msg, state = await self.run_photo(FakeRecognizer(), acquired=failing)
                self.assertEqual(msg.last, error.user_message)
                self.assertIsNone(state.state)

    async def test_unexpected_error_resets_state_and_propagates(self):
        state = FakeState()
        with self.assertRaises(RuntimeError):
            await self.run_photo(FakeRecognizer(error=RuntimeError("bug")), state=state)
        self.assertIsNone(state.state)

    async def test_result_is_discarded_if_user_cancelled_just_before_finish(self):
        state = FakeState()

        class RacingRecognizer(FakeRecognizer):
            async def recognize(self, image_path):
                await state.clear()  # /cancel успел сбросить состояние, а задача уже завершается
                return await super().recognize(image_path)

        msg, state = await self.run_photo(RacingRecognizer(), state=state)
        self.assertEqual(msg.answers, [])  # «воскресшего» условия нет
        self.assertIsNone(state.state)

    async def test_truncated_text_is_flagged(self):
        msg, _ = await self.run_photo(FakeRecognizer(truncated=True))
        self.assertIn("обрезан", msg.last)

    async def test_new_photo_replaces_previous_confirmed_condition(self):
        state = FakeState(Scenario.confirmed, {"confirmed_text": "старое"})
        _, state = await self.run_photo(FakeRecognizer(text="Новое условие номер два"), state=state)
        self.assertNotIn("confirmed_text", state.data)
        self.assertEqual(state.state, Scenario.awaiting_confirmation.state)

    async def test_document_flow_and_unsupported_type(self):
        doc = SimpleNamespace(file_id="d1", file_size=2000, mime_type="image/png")
        msg, state = FakeMessage(document=doc), FakeState()
        with patch.object(content, "acquired_image", fake_acquired()):
            await content.on_document(msg, state, bot=object(), config=CONFIG,
                                      recognizer=FakeRecognizer())
        self.assertEqual(state.state, Scenario.awaiting_confirmation.state)

        for mime in ("application/pdf", None):
            bad = FakeMessage(document=SimpleNamespace(file_id="x", file_size=1, mime_type=mime))
            await content.on_document(bad, FakeState(), bot=object(), config=CONFIG,
                                      recognizer=FakeRecognizer())
            self.assertIn("не поддерживается", bad.last)


@unittest.skipUnless(HAS_AIOGRAM, "aiogram не установлен")
class CancelDuringRecognitionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        patcher = patch("tempfile.tempdir", str(self.root))  # временные папки -> наша папка
        patcher.start()
        self.addCleanup(patcher.stop)
        dl_patch = patch.object(
            content, "TelegramDownloader",
            lambda bot: FakeDownloader(make_image_bytes("JPEG", (320, 240))),
        )
        dl_patch.start()
        self.addCleanup(dl_patch.stop)

    async def test_cancel_stops_recognition_and_removes_temp_files(self):
        recognizer = FakeRecognizer(hang=True)
        msg, state = photo_msg(), FakeState()
        task = asyncio.create_task(
            content.on_photo(msg, state, bot=object(), config=CONFIG, recognizer=recognizer)
        )
        await asyncio.wait_for(recognizer.started.wait(), 2)
        self.assertEqual(state.state, Scenario.processing.state)
        self.assertEqual(len(list(self.root.iterdir())), 1)  # файл существует во время OCR
        self.assertTrue(recognizer.paths[0].is_file())

        cancel_msg = FakeMessage()
        await commands.cmd_cancel(cancel_msg, state)
        await asyncio.wait_for(task, 2)  # обработчик завершился без исключения

        self.assertIn("отменён", cancel_msg.last)
        self.assertIsNone(state.state)
        self.assertEqual(msg.answers, [])  # условие НЕ показано
        self.assertEqual(list(self.root.iterdir()), [])  # временные файлы удалены
        self.assertFalse(ACTIVE_JOBS.is_active(1))

    async def test_second_photo_while_busy_is_rejected(self):
        recognizer = FakeRecognizer(hang=True)
        first = asyncio.create_task(
            content.on_photo(photo_msg(), FakeState(), bot=object(), config=CONFIG,
                             recognizer=recognizer)
        )
        await asyncio.wait_for(recognizer.started.wait(), 2)
        second = photo_msg()
        await content.on_photo(second, FakeState(), bot=object(), config=CONFIG,
                               recognizer=FakeRecognizer())
        self.assertEqual(second.last, content.BUSY_TEXT)
        ACTIVE_JOBS.cancel(1)
        await asyncio.wait_for(first, 2)
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_temp_files_removed_after_successful_recognition(self):
        msg, state = photo_msg(), FakeState()
        await content.on_photo(msg, state, bot=object(), config=CONFIG, recognizer=FakeRecognizer())
        self.assertEqual(state.state, Scenario.awaiting_confirmation.state)
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_temp_files_removed_after_recognition_error(self):
        msg, state = photo_msg(), FakeState()
        await content.on_photo(msg, state, bot=object(), config=CONFIG,
                               recognizer=FakeRecognizer(error=OcrFailed()))
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
