import unittest
from unittest.mock import patch, MagicMock
from services.verification import verify_equation, VerificationResult
from pathlib import Path

class TestEquationDetailed(unittest.TestCase):
    def test_correct_single_root(self):
        """x + 1 = 2 => x = 1 (Unique)"""
        payload = {
            "equation_left_sympy": "x + 1",
            "equation_right_sympy": "2",
            "variable": "x",
            "solution_sympy": "1",
        }
        result = verify_equation(payload)
        self.assertTrue(result.verified)
        self.assertIn("единственный корень", result.message.lower())

    def test_correct_one_of_many(self):
        """x^2 - 5x + 6 = 0 => x = 2 (Not unique)"""
        payload = {
            "equation_left_sympy": "x**2 - 5*x + 6",
            "equation_right_sympy": "0",
            "variable": "x",
            "solution_sympy": "2",
        }
        result = verify_equation(payload)
        self.assertTrue(result.verified)
        self.assertIn("может иметь другие решения", result.message.lower())

    def test_incorrect_root(self):
        """x + 1 = 2 => x = 3 (Wrong)"""
        payload = {
            "equation_left_sympy": "x + 1",
            "equation_right_sympy": "2",
            "variable": "x",
            "solution_sympy": "3",
        }
        result = verify_equation(payload)
        self.assertFalse(result.verified)
        self.assertIn("не удовлетворяет уравнению", result.message.lower())

    def test_complex_zero_expression(self):
        """Expression that simplifies to zero"""
        payload = {
            "equation_left_sympy": "(x-1)*(x+1)",
            "equation_right_sympy": "x**2 - 1",
            "variable": "x",
            "solution_sympy": "5",
        }
        result = verify_equation(payload)
        self.assertTrue(result.verified)

    def test_undetermined_result(self):
        """Case where is_zero might be None (e.g. very complex expression)"""
        # We mock sympy.simplify to return an object where .is_zero is None.
        # This targets the call: is_zero = sp.simplify(diff).is_zero
        with patch("sympy.simplify") as mock_simplify:
            mock_expr = MagicMock()
            mock_expr.is_zero = None
            mock_simplify.return_value = mock_expr

            payload = {
                "equation_left_sympy": "x",
                "equation_right_sympy": "x",
                "variable": "x",
                "solution_sympy": "1",
            }
            result = verify_equation(payload)
            self.assertIsNone(result.verified)
            self.assertIn("не смог однозначно определить", result.message.lower())

    def test_completeness_failure(self):
        """Equation where solve fails or returns nothing but root is correct"""
        payload = {
            "equation_left_sympy": "x - 1",
            "equation_right_sympy": "0",
            "variable": "x",
            "solution_sympy": "1",
        }
        with patch("sympy.solve", side_effect=Exception("Solve failed")):
            result = verify_equation(payload)
            self.assertTrue(result.verified)
            self.assertIn("Полноту множества решений подтвердить не удалось", result.message)

if __name__ == "__main__":
    unittest.main()
