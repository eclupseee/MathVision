import unittest

from services.recognition import (
    MAX_CONDITION_CHARS,
    OcrNoText,
    build_condition,
    clean_text,
)


class CleanTextTests(unittest.TestCase):
    def test_normalizes_newlines_and_blank_lines(self):
        raw = "Найти предел\r\nlim x->0   \r\n\r\n\r\n\r\nОтвет\x0c"
        self.assertEqual(clean_text(raw), "Найти предел\nlim x->0\n\nОтвет")

    def test_removes_control_chars_but_keeps_math_unicode(self):
        self.assertEqual(clean_text("\x00∫ x² dx ≤ π\x07"), "∫ x² dx ≤ π")

    def test_empty(self):
        self.assertEqual(clean_text("  \n\n "), "")

    def test_removes_terminal_escape_sequences(self):
        self.assertEqual(clean_text("интеграл\x1b[3D\x1b[K"), "интеграл")

    def test_removes_ollama_thinking_preamble(self):
        raw = "Thinking...\nРазмышления модели\nОтвет:\nНайти интеграл ∫ x dx"
        self.assertEqual(clean_text(raw), "Найти интеграл ∫ x dx")

class BuildConditionTests(unittest.TestCase):
    def test_success(self):
        cond = build_condition("Вычислить интеграл ∫ x dx\n", "tesseract")
        self.assertEqual((cond.engine, cond.truncated), ("tesseract", False))
        self.assertTrue(cond.text.startswith("Вычислить"))

    def test_no_text_raises(self):
        for raw in ("", "   \n", "..", "~ ~ |"):
            with self.subTest(raw=raw), self.assertRaises(OcrNoText):
                build_condition(raw, "tesseract")

    def test_long_text_is_truncated(self):
        cond = build_condition("слово " * 2000, "tesseract")
        self.assertTrue(cond.truncated)
        self.assertLessEqual(len(cond.text), MAX_CONDITION_CHARS)


if __name__ == "__main__":
    unittest.main()
