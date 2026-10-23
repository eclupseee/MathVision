import json

import pytest

from services.verification import extract_solution_payload, verify_antiderivative


def test_extract_json_with_cli_preamble():
    raw = "Thinking...\\n" + json.dumps(
        {
            "steps_markdown": "Правило степени.",
            "final_answer": "x**7/7 + C",
            "integrand_sympy": "x**6",
            "antiderivative_sympy": "x**7/7",
            "variable": "x",
        },
        ensure_ascii=False,
    )
    payload = extract_solution_payload(raw)
    assert payload["final_answer"] == "x**7/7 + C"


def test_verify_correct_antiderivative():
    result = verify_antiderivative(
        {
            "integrand_sympy": "x**6",
            "antiderivative_sympy": "x**7/7",
            "variable": "x",
        }
    )
    assert result.verified is True
    assert "Проверена только первообразная" in result.message



def test_verify_incorrect_antiderivative():
    result = verify_antiderivative(
        {
            "integrand_sympy": "x**6",
            "antiderivative_sympy": "x**6/6",
            "variable": "x",
        }
    )
    assert result.verified is False


def test_verification_skips_if_expression_missing():
    result = verify_antiderivative({"integrand_sympy": "", "antiderivative_sympy": ""})
    assert result.verified is None


def test_payload_requires_steps_and_answer():
    assert extract_solution_payload('{"foo":"bar"}') is None


def test_verifier_rejects_python_code_in_model_expression():
    result = verify_antiderivative(
        {
            "integrand_sympy": "__import__('os').system('echo bad')",
            "antiderivative_sympy": "x",
            "variable": "x",
        }
    )
    assert result.verified is None


def test_extract_json_with_literal_newlines_inside_steps():
    raw = """{
        "steps_markdown": "Шаг 1.
Находим первообразную.
Шаг 2. Проверяем результат.",
        "final_answer": "x^3 + x^2 + x + C",
        "integrand_sympy": "3*x**2 + 2*x + 1",
        "antiderivative_sympy": "x**3 + x**2 + x",
        "variable": "x"
    }"""

    payload = extract_solution_payload(raw)

    assert payload is not None
    assert "Находим первообразную." in payload["steps_markdown"]
    assert "Шаг 2. Проверяем результат." in payload["steps_markdown"]
    assert payload["final_answer"] == "x^3 + x^2 + x + C"


def test_verify_incorrect_answer_from_mathvision_example():
    result = verify_antiderivative(
        {
            "integrand_sympy": "3*x**2 + 2*x + 1",
            "antiderivative_sympy": "7*x**3/3 + x**2/2 + x",
            "variable": "x",
        }
    )

    assert result.verified is False


def test_verify_correct_definite_integral():
    result = verify_antiderivative(
        {
            "integrand_sympy": "3*x**2 - 2*x + 1",
            "antiderivative_sympy": "x**3 - x**2 + x",
            "variable": "x",
            "lower_limit": "1",
            "upper_limit": "2",
            "final_answer": "5",
        }
    )

    assert result.verified is True


def test_verify_incorrect_definite_integral_answer():
    result = verify_antiderivative(
        {
            "integrand_sympy": "3*x**2 - 2*x + 1",
            "antiderivative_sympy": "x**3 - x**2 + x",
            "variable": "x",
            "lower_limit": "1",
            "upper_limit": "2",
            "final_answer": "6",
        }
    )

    assert result.verified is False


def test_verify_definite_integral_missing_one_limit():
    result = verify_antiderivative(
        {
            "integrand_sympy": "x",
            "antiderivative_sympy": "x**2/2",
            "variable": "x",
            "lower_limit": "1",
            "final_answer": "1",
        }
    )

    assert result.verified is None


def test_extract_json_with_unescaped_latex_commands():
    raw = r'{"steps_markdown":"Решение","final_answer":"\frac{\pi}{4}"}'
    payload = extract_solution_payload(raw)
    assert payload is not None
    assert payload["final_answer"] == r"\frac{\pi}{4}"


def test_extract_json_with_sqrt_and_pm():
    raw = r'{"steps_markdown":"Решение","final_answer":"\pm \sqrt{2}"}'
    payload = extract_solution_payload(raw)
    assert payload is not None
    assert payload["final_answer"] == r"\pm \sqrt{2}"


def test_extract_json_preserves_valid_escaped_backslashes():
    raw = r'{"steps_markdown":"Решение","final_answer":"\\frac{\\pi}{4}"}'
    payload = extract_solution_payload(raw)
    assert payload is not None
    assert payload["final_answer"] == r"\frac{\pi}{4}"


def test_equation_verifier_accepts_correct_solution():
    from services.verification import verify_equation

    result = verify_equation(
        {
            "equation_left_sympy": "2*x + 3",
            "equation_right_sympy": "7",
            "variable": "x",
            "solution_sympy": "2",
        }
    )

    assert result.verified is True


def test_equation_verifier_rejects_incorrect_solution():
    from services.verification import verify_equation

    result = verify_equation(
        {
            "equation_left_sympy": "2*x + 3",
            "equation_right_sympy": "7",
            "variable": "x",
            "solution_sympy": "3",
        }
    )

    assert result.verified is False


def test_equation_verifier_rejects_unsafe_expression():
    from services.verification import verify_equation

    result = verify_equation(
        {
            "equation_left_sympy": "__import__('os')",
            "equation_right_sympy": "7",
            "variable": "x",
            "solution_sympy": "2",
        }
    )

    assert result.verified is None
