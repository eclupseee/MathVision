import base64
import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services import ollama_cli
from services.recognition import OllamaFailed, OllamaTimeout, OllamaUnavailable


def make_response(body):
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(body).encode("utf-8")
    return response


def test_run_ollama_uses_http_api():
    response_body = {"response": '{"steps_markdown": "Решение", "final_answer": "5"}'}
    with patch.object(
        ollama_cli.urllib.request, "urlopen", return_value=make_response(response_body)
    ) as urlopen:
        result = ollama_cli._run_ollama("gemma4:cloud", "prompt", 30)

    assert result == response_body["response"]
    args, kwargs = urlopen.call_args
    request = args[0]
    assert request.full_url == ollama_cli.OLLAMA_API_URL
    assert request.method == "POST"
    assert kwargs["timeout"] == 30

    payload = json.loads(request.data.decode("utf-8"))
    assert payload["model"] == "gemma4:cloud"
    assert payload["prompt"] == "prompt"
    assert payload["format"] == "json"
    assert payload["stream"] is False


def test_run_ollama_passes_image_as_base64(tmp_path):
    image = tmp_path / "photo with spaces.jpg"
    image.write_bytes(b"test image")

    with patch.object(
        ollama_cli.urllib.request,
        "urlopen",
        return_value=make_response({"response": "recognized text"}),
    ) as urlopen:
        result = ollama_cli._run_ollama(
            "gemma4:cloud", "recognize this image", 30, image_path=image
        )

    assert result == "recognized text"
    request = urlopen.call_args.args[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert payload["images"] == [base64.b64encode(b"test image").decode("ascii")]


def test_run_ollama_rejects_missing_image(tmp_path):
    missing_image = tmp_path / "missing.jpg"
    with pytest.raises(OllamaFailed):
        ollama_cli._run_ollama(
            "gemma4:cloud", "prompt", 30, image_path=missing_image
        )


def test_run_ollama_timeout():
    with patch.object(
        ollama_cli.urllib.request, "urlopen", side_effect=TimeoutError
    ):
        with pytest.raises(OllamaTimeout):
            ollama_cli._run_ollama("gemma4:cloud", "prompt", 1)


def test_run_ollama_url_timeout():
    with patch.object(
        ollama_cli.urllib.request,
        "urlopen",
        side_effect=urllib.error.URLError(TimeoutError()),
    ):
        with pytest.raises(OllamaTimeout):
            ollama_cli._run_ollama("gemma4:cloud", "prompt", 1)


def test_run_ollama_unavailable():
    with patch.object(
        ollama_cli.urllib.request,
        "urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        with pytest.raises(OllamaUnavailable):
            ollama_cli._run_ollama("gemma4:cloud", "prompt", 30)


def test_run_ollama_http_404():
    error = urllib.error.HTTPError(
        ollama_cli.OLLAMA_API_URL, 404, "Not Found", {}, None
    )
    with patch.object(ollama_cli.urllib.request, "urlopen", side_effect=error):
        with pytest.raises(OllamaUnavailable):
            ollama_cli._run_ollama("gemma4:cloud", "prompt", 30)


def test_run_ollama_http_error():
    error = urllib.error.HTTPError(
        ollama_cli.OLLAMA_API_URL, 500, "Server Error", {}, None
    )
    with patch.object(ollama_cli.urllib.request, "urlopen", side_effect=error):
        with pytest.raises(OllamaFailed):
            ollama_cli._run_ollama("gemma4:cloud", "prompt", 30)


def test_run_ollama_invalid_json():
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b"not json"

    with patch.object(ollama_cli.urllib.request, "urlopen", return_value=response):
        with pytest.raises(OllamaFailed):
            ollama_cli._run_ollama("gemma4:cloud", "prompt", 30)


@pytest.mark.parametrize("body", [{}, {"response": ""}, {"response": None}, {"response": 123}])
def test_run_ollama_rejects_invalid_response(body):
    with patch.object(
        ollama_cli.urllib.request, "urlopen", return_value=make_response(body)
    ):
        with pytest.raises(OllamaFailed):
            ollama_cli._run_ollama("gemma4:cloud", "prompt", 30)


def test_image_prompt_is_about_transcription():
    prompt = ollama_cli.image_prompt(Path("example.jpg"))
    assert "математические формулы" in prompt
    assert "Ничего не решай" in prompt
    assert "example.jpg" not in prompt
