"""Решение подтверждённого условия через Ollama Cloud."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import OcrSettings
from .ollama_cli import run_ollama
from .verification import (
    VerificationResult,
    extract_solution_payload,
    verify_antiderivative,
    verify_equation,
)
import re
import unicodedata


@dataclass(frozen=True, slots=True)
class Solution:
    text: str
    verified: bool | None


async def solve_condition(
    condition: str, settings: OcrSettings | None = None
) -> Solution:
    settings = settings or OcrSettings()

    prompt = (
        "Реши задачу по высшей математике, условие которой приведено ниже. "
        "Дай математически корректное пошаговое решение на русском языке. "
        "Сначала определи тип задачи и выбери самый прямой подходящий метод. "
        "Не используй интегрирование по частям, замену переменной или другие "
        "специальные методы, если они не нужны для решения. "
        "Для интегралов от многочленов используй линейность интеграла и правило "
        "степени: интеграл x**n равен x**(n+1)/(n+1) при n != -1. "
        "Для определённого интеграла сначала найди первообразную, затем вычисли "
        "F(b)-F(a), внимательно сохраняя знаки и пределы интегрирования. "
        "Объясняй именно те действия, которые реально выполняешь. "
        "Не называй обычное интегрирование степенной функции интегрированием "
        "по частям. Не добавляй фиктивные шаги и формулы. "
        "СТРОГИЕ ТРЕБОВАНИЯ К МАТЕМАТИЧЕСКОЙ ЗАПИСИ: "
        "все математические выражения записывай обычными ASCII-символами и LaTeX. "
        "Не используй математические Unicode-символы, например 𝑥, 𝑛, 𝑑, 𝐶, ², ⁷. "
        "Используй x, n, dx, C, x^2 и x^7. "
        "Каждую формулу внутри steps_markdown заключай в $...$ для inline-формул "
        "или в $$...$$ для отдельной формулы. "
        "Не оставляй формулы без разделителей. "
        "Не используй LaTeX-разделители \\(...\\) и \\[...\\] внутри JSON-ответа. "
        "Внутри $...$ записывай дроби через \\frac{числитель}{знаменатель}, "
        "интегралы через \\int, производные через корректные математические выражения. "
        "Не переноси строку внутри формулы или команды LaTeX. "
        "Не разбивай числитель и знаменатель на отдельные строки. "
        "Не заменяй дробь визуальным расположением чисел друг над другом. "
        "Например, пиши $\\int x^6\\,dx=\\frac{x^7}{7}+C$, "
        "а не разрывай интеграл, дробь и результат на несколько строк. "
        "Для неопределённого интеграла обязательно добавляй произвольную постоянную C "
        "в итоговый ответ. "
        "Перед возвратом JSON проверь, что все формулы завершены, "
        "содержат разделители и не используют математические Unicode-символы. "
        "Для простых вычислений предпочитай короткие шаги и простые формулы. "
        "Проверь вычисления и согласованность итогового ответа с решением. "
        "Если условие неоднозначно или данных недостаточно, прямо укажи это. "
        "Не утверждай, что компьютерная проверка выполнена, если ты её не выполнял. "
        "Верни ТОЛЬКО один JSON-объект без Markdown-обрамления со следующими ключами: "
        "steps_markdown (строка с пошаговым объяснением в Markdown), "
        "final_answer (строка с итоговым ответом), "
        "integrand_sympy (строка с подынтегральным выражением в синтаксисе Python/SymPy "
        "для неопределённого или определённого интеграла, иначе пустая строка), "
        "antiderivative_sympy (строка с первообразной в синтаксисе Python/SymPy без +C, "
        "иначе пустая строка), variable (имя переменной, обычно x), "
        "equation_left_sympy (левая часть простого уравнения в синтаксисе Python/SymPy, "
        "иначе пустая строка), "
        "equation_right_sympy (правая часть простого уравнения в синтаксисе Python/SymPy, "
        "иначе пустая строка), "
        "solution_sympy (одно конкретное решение простого уравнения в синтаксисе "
        "Python/SymPy, иначе пустая строка). "
        "Поля equation_left_sympy, equation_right_sympy и solution_sympy заполняй "
        "только для простых уравнений с одной переменной и одним конкретным решением. "
        "Не выдавай проверку одного корня за доказательство полноты множества решений. "
        "Для определённого интеграла дополнительно укажи lower_limit и upper_limit "
        "как строки с пределами в синтаксисе Python/SymPy. Если задача не является "
        "определённым интегралом, верни эти поля пустыми строками. "
        "Для выражений используй, например, x**6, x**7/7 и 3*x**2 - 2*x + 1. "
        f"Условие пользователя:\n{condition}"
    )

    raw = await run_ollama(
        settings.ollama_solver_model, prompt, settings.ollama_timeout_seconds
    )

    payload = extract_solution_payload(raw)

    if payload is None:
        # Не выдаём непроверяемый ответ как структурированный/проверенный.
        safe_text = raw[:3000]
        return Solution(
            "Не удалось надёжно разобрать структурированный ответ модели. "
            "Проверьте доступность Ollama Cloud и повторите попытку.\n\n"
            "Фрагмент ответа модели:\n" + safe_text,
            None,
        )

    equation_keys = (
        "equation_left_sympy",
        "equation_right_sympy",
        "solution_sympy",
    )
    integral_keys = (
        "integrand_sympy",
        "antiderivative_sympy",
    )

    if all(
        isinstance(payload.get(key), str) and payload[key].strip()
        for key in equation_keys
    ):
        verification = verify_equation(payload)
    elif all(
        isinstance(payload.get(key), str) and payload[key].strip()
        for key in integral_keys
    ):
        verification = verify_antiderivative(payload)
    else:
        verification = VerificationResult(
            None,
            "Независимая проверка для этого типа задачи пока не реализована.",
        )

    def normalize_math_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)

        # Превращаем распространённые записи степеней в x^7,
        # а не в x7. NFKC уже преобразовал надстрочные цифры.
        text = re.sub(
            r"\b([a-zA-Z])\s*(\d+)\b",
            r"\1^\2",
            text,
        )

        # Убираем случайные пробелы перед пунктуацией.
        text = re.sub(r"[ \t]+([,.;])", r"\1", text)

        # Склеиваем переносы после операторов и внутри дробей.
        lines = text.splitlines()
        result = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                if result and result[-1] != "":
                    result.append("")
                continue

            if (
                result
                and result[-1]
                and not re.match(r"^\s*(?:\d+[.)]\s+|[-*]\s+)", stripped)
                and (
                    re.search(r"[=+\-*/^∫√]\s*$", result[-1])
                    or result[-1].endswith(("(", r"\frac{"))
                )
            ):
                result[-1] += stripped
            else:
                result.append(stripped)

        return "\n".join(result).strip()

    steps = normalize_math_text(payload["steps_markdown"].strip())
    final = normalize_math_text(payload["final_answer"].strip())
    parts = []

    if steps:
        parts.append(steps)

    if final:
        parts.append("**Итоговый ответ:**\n" + final)

    parts.append("**Независимая проверка:**\n" + verification.message)
    return Solution("\n\n".join(parts), verification.verified)
