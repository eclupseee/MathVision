import unittest
from unittest.mock import patch, MagicMock
import json
import os
from pathlib import Path
from services.ollama_cli import run_ollama, _run_ollama
from services.recognition import OllamaFailed, OllamaTimeout, OllamaUnavailable

class TestOllamaHttp(unittest.TestCase):
    def setUp(self):
        self.model = "gemma4:cloud"
        self.prompt = "Hello"
        self.timeout = 10.0
        self.image_path = Path("fake.png")

    def test_local_mode_no_auth(self):
        """Verify local mode uses local URL and no Auth header."""
        with patch.dict(os.environ, {"OLLAMA_MODE": "local"}), \
             patch("urllib.request.urlopen") as mock_url:

            # Mock response
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"response": "local ok"}).encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp
            mock_url.return_value = mock_resp

            _run_ollama(self.model, self.prompt, self.timeout)

            args, _ = mock_url.call_args
            request = args[0]
            self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/generate")
            self.assertNotIn("Authorization", request.headers)

    def test_cloud_mode_with_auth(self):
        """Verify cloud mode uses cloud URL and Bearer token."""
        with patch.dict(os.environ, {"OLLAMA_MODE": "cloud", "OLLAMA_API_KEY": "secret_key"}), \
             patch("urllib.request.urlopen") as mock_url:

            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"response": "cloud ok"}).encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp
            mock_url.return_value = mock_resp

            _run_ollama(self.model, self.prompt, self.timeout)

            args, _ = mock_url.call_args
            request = args[0]
            self.assertEqual(request.full_url, "https://ollama.com/api/generate")
            self.assertEqual(request.headers["Authorization"], "Bearer secret_key")

    def test_cloud_mode_missing_key(self):
        """Cloud mode should raise OllamaUnavailable if key is missing."""
        with patch.dict(os.environ, {"OLLAMA_MODE": "cloud", "OLLAMA_API_KEY": ""}), \
             patch("urllib.request.urlopen") as mock_url:

            with self.assertRaises(OllamaUnavailable) as cm:
                _run_ollama(self.model, self.prompt, self.timeout)
            self.assertIn("OLLAMA_API_KEY is missing", str(cm.exception))

    def test_image_transmission(self):
        """Verify image is base64 encoded and sent in payload."""
        with patch.dict(os.environ, {"OLLAMA_MODE": "local"}), \
             patch("urllib.request.urlopen") as mock_url, \
             patch("pathlib.Path.read_bytes") as mock_read:

            mock_read.return_value = b"fake-image-bytes"
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"response": "image ok"}).encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp
            mock_url.return_value = mock_resp

            # Mocking Path.is_file to return True
            with patch("pathlib.Path.is_file", return_value=True):
                _run_ollama(self.model, self.prompt, self.timeout, image_path=self.image_path)

            args, _ = mock_url.call_args
            request = args[0]
            payload = json.loads(request.data.decode("utf-8"))
            self.assertIn("images", payload)
            self.assertTrue(len(payload["images"]) > 0)

    def test_http_401_auth_error(self):
        """Test 401/403 responses."""
        with patch.dict(os.environ, {"OLLAMA_MODE": "cloud", "OLLAMA_API_KEY": "bad_key"}), \
             patch("urllib.request.urlopen") as mock_url:

            import urllib.error
            mock_url.side_effect = urllib.error.HTTPError(
                "https://ollama.com/api/generate", 401, "Unauthorized", None, None
            )

            with self.assertRaises(OllamaUnavailable) as cm:
                _run_ollama(self.model, self.prompt, self.timeout)
            self.assertIn("Invalid or missing API key", str(cm.exception))

    def test_http_404_not_found(self):
        """Test 404 response."""
        with patch.dict(os.environ, {"OLLAMA_MODE": "local"}), \
             patch("urllib.request.urlopen") as mock_url:

            import urllib.error
            mock_url.side_effect = urllib.error.HTTPError(
                "http://127.0.0.1:11434/api/generate", 404, "Not Found", None, None
            )

            with self.assertRaises(OllamaUnavailable) as cm:
                _run_ollama(self.model, self.prompt, self.timeout)
            self.assertIn("endpoint or model not found", str(cm.exception))

    def test_timeout_handling(self):
        """Test timeout handling."""
        with patch.dict(os.environ, {"OLLAMA_MODE": "local"}), \
             patch("urllib.request.urlopen") as mock_url:

            mock_url.side_effect = TimeoutError()

            with self.assertRaises(OllamaTimeout):
                _run_ollama(self.model, self.prompt, self.timeout)

if __name__ == "__main__":
    unittest.main()
