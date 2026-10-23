import pytest

from config import ConfigError, load_config

VALID = "123456789:AAEhBOweik6ad2K8eoZxQ3pTq9zYb-dE_fg"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("BOT_TOKEN", "MAX_PHOTO_MB", "LOG_LEVEL", "OCR_ENGINE", "OLLAMA_MODEL", "OLLAMA_TIMEOUT_SECONDS"):
        monkeypatch.delenv(key, raising=False)


def test_missing_token_raises_clear_error():
    with pytest.raises(ConfigError, match="BOT_TOKEN"):
        load_config(env_file=None)


def test_invalid_token_format_does_not_leak_value(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "secret-garbage")
    with pytest.raises(ConfigError) as exc:
        load_config(env_file=None)
    assert "secret-garbage" not in str(exc.value)


def test_valid_config_defaults(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", VALID)
    cfg = load_config(env_file=None)
    assert cfg.bot_token == VALID
    assert cfg.max_photo_bytes == 10 * 1024 * 1024
    assert cfg.log_level == "INFO"


def test_repr_hides_token(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", VALID)
    assert VALID not in repr(load_config(env_file=None))


@pytest.mark.parametrize("value", ["0", "-1", "21", "abc"])
def test_bad_max_photo_mb(monkeypatch, value):
    monkeypatch.setenv("BOT_TOKEN", VALID)
    monkeypatch.setenv("MAX_PHOTO_MB", value)
    with pytest.raises(ConfigError, match="MAX_PHOTO_MB"):
        load_config(env_file=None)


def test_bad_log_level(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", VALID)
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    with pytest.raises(ConfigError, match="LOG_LEVEL"):
        load_config(env_file=None)


def test_reads_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text(f"BOT_TOKEN={VALID}\nMAX_PHOTO_MB=5\n", encoding="utf-8")
    cfg = load_config(env_file=env)
    assert cfg.max_photo_bytes == 5 * 1024 * 1024


def test_ollama_cloud_settings(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", VALID)
    monkeypatch.setenv("OCR_ENGINE", "ollama_cloud")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:cloud")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "90")
    cfg = load_config(env_file=None)
    assert cfg.ocr.engine == "ollama_cloud"
    assert cfg.ocr.ollama_model == "gemma4:cloud"
    assert cfg.ocr.ollama_timeout_seconds == 90


def test_invalid_ollama_timeout_is_rejected(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", VALID)
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "1")
    with pytest.raises(ConfigError, match="OLLAMA_TIMEOUT_SECONDS"):
        load_config(env_file=None)
