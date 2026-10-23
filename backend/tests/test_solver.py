import json
import unittest
from unittest.mock import AsyncMock, patch

from config import OcrSettings
from services.solver import solve_condition


class SolverTests(unittest.IsolatedAsyncioTestCase):
    async def test_solver_formats_steps_and_verified_integral(self):
        response = json.dumps(
            {
                "steps_markdown": "Используем правило степени.",
                "final_answer": "x**7/7 + C",
                "integrand_sympy": "x**6",
                "antiderivative_sympy": "x**7/7",
                "variable": "x",
            },
            ensure_ascii=False,
        )
        with patch(
            "services.solver.run_ollama", new=AsyncMock(return_value=response)
        ) as run:
            result = await solve_condition("Найти ∫x^6 dx", OcrSettings())
        self.assertIs(result.verified, True)
        self.assertIn("правило степени", result.text.lower())
        self.assertIn("Итоговый ответ", result.text)
        self.assertIn("Независимая проверка", result.text)
        run.assert_awaited_once()

    async def test_solver_does_not_claim_verification_for_unstructured_answer(self):
        with patch(
            "services.solver.run_ollama", new=AsyncMock(return_value="Ответ: x^7/7 + C")
        ):
            result = await solve_condition("Найти ∫x^6 dx", OcrSettings())
        self.assertIsNone(result.verified)
        self.assertIn("не удалось надёжно разобрать", result.text.lower())

    async def test_solver_verifies_equation(self):
        response = json.dumps(
            {
                "steps_markdown": "Перенесём 3 и разделим на 2.",
                "final_answer": "x = 2",
                "equation_left_sympy": "2*x + 3",
                "equation_right_sympy": "7",
                "solution_sympy": "2",
                "variable": "x",
                "integrand_sympy": "",
                "antiderivative_sympy": "",
            },
            ensure_ascii=False,
        )

        with patch(
            "services.solver.run_ollama",
            new=AsyncMock(return_value=response),
        ):
            result = await solve_condition("Решить 2*x + 3 = 7", OcrSettings())

        self.assertIs(result.verified, True)
        self.assertIn("Итоговый ответ", result.text)

    async def test_solver_does_not_use_integral_verification_for_other_tasks(self):
        response = json.dumps(
            {
                "steps_markdown": "Вычислим значение выражения.",
                "final_answer": "5",
                "equation_left_sympy": "",
                "equation_right_sympy": "",
                "solution_sympy": "",
                "integrand_sympy": "",
                "antiderivative_sympy": "",
                "variable": "x",
            },
            ensure_ascii=False,
        )

        with patch(
            "services.solver.run_ollama",
            new=AsyncMock(return_value=response),
        ):
            result = await solve_condition("Вычислить 2 + 3", OcrSettings())

        self.assertIsNone(result.verified)
        self.assertIn(
            "для этого типа задачи пока не реализована",
            result.text.lower(),
        )


if __name__ == "__main__":
    unittest.main()
