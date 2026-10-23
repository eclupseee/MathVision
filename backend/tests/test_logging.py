import logging
import sys

from logging_setup import SecretRedactor

TOKEN = "123456789:AAEhBOweik6ad2K8eoZxQ3pTq9zYb-dE_fg"


def make_record(msg, exc_info=None):
    return logging.LogRecord("t", logging.ERROR, __file__, 1, msg, None, exc_info)


def test_redacts_message_and_url():
    rec = make_record(f"GET https://api.telegram.org/bot{TOKEN}/getMe")
    SecretRedactor([TOKEN]).filter(rec)
    assert TOKEN not in rec.msg


def test_redacts_unknown_token_by_pattern():
    rec = make_record(f"token {TOKEN}")
    SecretRedactor().filter(rec)
    assert TOKEN not in rec.msg


def test_redacts_traceback():
    try:
        raise RuntimeError(f"failed {TOKEN}")
    except RuntimeError:
        rec = make_record("boom", exc_info=sys.exc_info())
    SecretRedactor([TOKEN]).filter(rec)
    assert TOKEN not in (rec.exc_text or "")
