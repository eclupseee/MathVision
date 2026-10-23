import os
import asyncio
import base64
import json
import urllib.error
import urllib.request
from pathlib import Path

from .recognition import OllamaFailed, OllamaTimeout, OllamaUnavailable

LOCAL_API_URL = "http://127.0.0.1:11434/api/generate"
CLOUD_API_URL = "https://ollama.com/api/generate"
OLLAMA_API_URL = LOCAL_API_URL  # For backward compatibility with old tests

def _run_ollama(
    model: str,
    prompt: str,
    timeout: float,
    image_path: Path | None = None,
) -> str:

    mode = os.environ.get("OLLAMA_MODE", "local").strip().lower()
    api_key = os.environ.get("OLLAMA_API_KEY", "").strip()

    url = CLOUD_API_URL if mode == "cloud" else LOCAL_API_URL

    payload: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "num_predict": 16384,
        },
    }

    if image_path is not None:
        path = image_path.resolve()
        if not path.is_file():
            raise OllamaFailed()

        try:
            payload["images"] = [base64.b64encode(path.read_bytes()).decode("ascii")]
        except OSError:
            raise OllamaFailed() from None

    headers = {"Content-Type": "application/json"}
    if mode == "cloud":
        if not api_key:
            raise OllamaUnavailable("OLLAMA_API_KEY is missing for cloud mode")
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except TimeoutError:
        raise OllamaTimeout() from None
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise OllamaUnavailable("Invalid or missing API key for Ollama Cloud") from None
        if exc.code == 404:
            raise OllamaUnavailable("Ollama API endpoint or model not found") from None
        raise OllamaFailed() from None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise OllamaTimeout() from None
        raise OllamaUnavailable() from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise OllamaFailed() from None

    result = data.get("response") if isinstance(data, dict) else None
    if not isinstance(result, str) or not result.strip():
        raise OllamaFailed()

    return result.strip()

async def run_ollama(
    model: str,
    prompt: str,
    timeout: float,
    image_path: Path | None = None,
) -> str:
    """Вызывает Ollama вне event loop."""
    return await asyncio.to_thread(_run_ollama, model, prompt, timeout, image_path)

def image_prompt(image_path: Path) -> str:
    """Инструкция для распознавания печатного математического задания."""
    return (
        "Распознай текст и математические формулы на прикреплённом изображении. "
        "Перепиши ВСЕ задания по порядку, сохрани вводный текст, обозначения "
        "пунктов, числа и формулы. Ничего не решай и не пропускай задания. "
        "Если символ неразборчив, явно отметь это. Ответь на русском языке."
    )
