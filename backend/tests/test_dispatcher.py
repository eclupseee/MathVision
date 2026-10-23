import pytest

pytest.importorskip("aiogram")

from bot import build_dispatcher  # noqa: E402
from config import Config  # noqa: E402


def test_dispatcher_builds():
    dp = build_dispatcher(Config(bot_token="x", max_photo_bytes=1024))
    assert dp["config"].max_photo_bytes == 1024
