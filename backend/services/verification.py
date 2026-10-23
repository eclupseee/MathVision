"""Независимая символьная проверка результата через SymPy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class VerificationResult:
    verified: bool | None
    message: str


def _json_object(raw: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()

    def parse_candidate(candidate: str) -> dict[str, Any] | None:
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if (
                isinstance(value, dict)
                and isinstance(value.get("steps_markdown"), str)
                and isinstance(value.get("final_answer"), str)
            ):
                return value
        return None

    # Сначала принимаем корректный JSON без изменений.
    result = parse_candidate(raw)
    if result is not None:
        return result

    # Восстанавливаем распространённые ошибки JSON в ответе модели:
    # буквальные переносы строк и неэкранированные команды LaTeX.
    repaired: list[str] = []
    in_string = False
    escaped = False
    i = 0

    while i < len(raw):
        char = raw[i]

        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            i += 1
            continue

        if escaped:
            repaired.append(char)
            escaped = False
            i += 1
            continue

        if char == '"':
            repaired.append(char)
            in_string = False
            i += 1
            continue

        if char == "\\":
            # Уже экранированный обратный слеш оставляем как есть.
            if i + 1 < len(raw):
                next_char = raw[i + 1]

                # Команды LaTeX, которые совпадают с допустимыми
                # JSON escape-последовательностями, например \frac.
                latex_commands = (
                    "frac",
                    "sqrt",
                    "sin",
                    "cos",
                    "tan",
                    "log",
                    "ln",
                    "lim",
                    "pi",
                    "pm",
                    "mp",
                    "in",
                    "notin",
                    "mathbb",
                    "text",
                    "left",
                    "right",
                    "cdot",
                    "times",
                    "int",
                    "sum",
                    "prod",
                    "alpha",
                    "beta",
                    "gamma",
                    "theta",
                    "lambda",
                    "infty",
                )
                command = next(
                    (name for name in latex_commands if raw.startswith(name, i + 1)),
                    None,
                )

                if command is not None:
                    repaired.append("\\\\")
                    repaired.append(command)
                    i += 1 + len(command)
                    continue

                # Неизвестные JSON escape-последовательности
                # превращаем в буквальный обратный слеш.
                if next_char not in '"\\/bfnrtu':
                    repaired.append("\\\\")
                    i += 1
                    continue

            repaired.append(char)
            escaped = True
            i += 1
            continue

        if char == "\n":
            repaired.append("\\n")
        elif char == "\r":
            repaired.append("\\r")
        elif char == "\t":
            repaired.append("\\t")
        elif ord(char) < 0x20:
            repaired.append(" ")
        else:
            repaired.append(char)

        i += 1

    return parse_candidate("".join(repaired))


def extract_solution_payload(raw: str) -> dict[str, Any] | None:
    """Достаёт JSON-объект из ответа модели, даже если CLI добавил префикс."""
    data = _json_object(raw)
    if not data:
        return None
    if not isinstance(data.get("steps_markdown"), str) or not isinstance(
        data.get("final_answer"), str
    ):
        return None
    return data


def _safe_sympy_expr(expression: str, variable: str, sp):
    """Parse a tiny arithmetic grammar without eval/sympify on model-controlled text."""
    import ast

    tree = ast.parse(expression, mode="eval")
    symbol = sp.Symbol(variable)
    names = {
        variable: symbol,
        "pi": sp.pi,
        "E": sp.E,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
    }
    binary = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a**b,
    }

    def convert(node):
        if isinstance(node, ast.Expression):
            return convert(node.body)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            return (
                sp.Integer(node.value)
                if isinstance(node.value, int)
                else sp.Float(node.value)
            )
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](convert(node.left), convert(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return convert(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -convert(node.operand)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in names
        ):
            function = names[node.func.id]
            if (
                not callable(function)
                or node.keywords
                or not (1 <= len(node.args) <= 2)
            ):
                raise ValueError("unsupported function call")
            return function(*(convert(arg) for arg in node.args))
        raise ValueError("unsupported expression syntax")

    if not expression.strip() or len(expression) > 500:
        raise ValueError("empty or oversized expression")
    return convert(tree)


def verify_antiderivative(payload: dict[str, Any]) -> VerificationResult:
    """Проверяет первообразную, а для определённого интеграла — ещё и численный ответ."""
    integrand = payload.get("integrand_sympy")
    antiderivative = payload.get("antiderivative_sympy")
    variable = payload.get("variable", "x")

    if not all(
        isinstance(v, str) and v.strip() for v in (integrand, antiderivative, variable)
    ):
        return VerificationResult(
            None,
            "SymPy не смог проверить ответ: отсутствуют выражения для проверки.",
        )

    has_lower = bool(str(payload.get("lower_limit") or "").strip())
    has_upper = bool(str(payload.get("upper_limit") or "").strip())

    # Нельзя считать определённый интеграл проверенным,
    # если модель передала только один предел.
    if has_lower != has_upper:
        return VerificationResult(
            None,
            "Проверка определённого интеграла пропущена: указан только один предел.",
        )

    try:
        import sympy as sp

        if (
            not variable.isidentifier()
            or len(variable) > 32
            or len(integrand) > 500
            or len(antiderivative) > 500
        ):
            return VerificationResult(
                None,
                "Проверка пропущена: некорректная переменная или слишком длинное выражение.",
            )

        f_expr = _safe_sympy_expr(integrand, variable, sp)
        F_expr = _safe_sympy_expr(antiderivative, variable, sp)
        symbol = sp.Symbol(variable)

        difference = sp.simplify(sp.diff(F_expr, symbol) - f_expr)
        if difference != 0:
            return VerificationResult(
                False,
                "Проверка не пройдена: производная первообразной не совпадает с подынтегральной функцией.",
            )

        # Для неопределённого интеграла достаточно проверки производной.
        if not has_lower:
            return VerificationResult(
                True,
                "Проверена только первообразная: её производная совпадает с подынтегральной функцией. "
                "Корректность всего пошагового решения не подтверждена.",
            )

        lower = payload.get("lower_limit")
        upper = payload.get("upper_limit")
        answer = payload.get("final_answer")

        if not all(isinstance(v, str) and v.strip() for v in (lower, upper, answer)):
            return VerificationResult(
                None,
                "Первообразная проверена, но пределы или итоговый ответ отсутствуют для проверки определённого интеграла.",
            )

        if any(len(v) > 100 for v in (lower, upper, answer)):
            return VerificationResult(
                None,
                "Проверка значения интеграла пропущена: слишком длинные пределы или ответ.",
            )

        a = _safe_sympy_expr(lower, variable, sp)
        b = _safe_sympy_expr(upper, variable, sp)

        # Итоговый ответ должен быть именно числовым выражением.
        # Текст, произвольный код и выражения с переменными не принимаются.
        answer_expr = _safe_sympy_expr(answer, variable, sp)
        if answer_expr.free_symbols:
            return VerificationResult(
                None,
                "Первообразная проверена, но итоговый ответ не распознан как число.",
            )

        expected = sp.simplify(F_expr.subs(symbol, b) - F_expr.subs(symbol, a))
        difference_from_answer = sp.simplify(expected - answer_expr)

        if difference_from_answer == 0:
            return VerificationResult(
                True,
                "Ответ проверен через SymPy: первообразная верна, а значение определённого интеграла совпадает с итоговым ответом.",
            )

        return VerificationResult(
            False,
            "Проверка не пройдена: значение F(b) − F(a) не совпадает с итоговым ответом модели.",
        )

    except ImportError:
        return VerificationResult(
            None,
            "SymPy не установлен; независимая проверка недоступна.",
        )
    except Exception:
        return VerificationResult(
            None,
            "SymPy не смог разобрать выражения, пределы или итоговый ответ для проверки.",
        )


def verify_equation(payload: dict[str, Any]) -> VerificationResult:
    """Проверяет, удовлетворяет ли предложенное значение заданному уравнению и полноту решений."""
    left = payload.get("equation_left_sympy")
    right = payload.get("equation_right_sympy")
    variable = payload.get("variable", "x")
    solution = payload.get("solution_sympy")

    if not all(
        isinstance(value, str) and value.strip()
        for value in (left, right, variable, solution)
    ):
        return VerificationResult(
            None,
            "Проверка уравнения пропущена: отсутствуют выражения или решение.",
        )

    if (
        not variable.isidentifier()
        or len(variable) > 32
        or len(left) > 500
        or len(right) > 500
        or len(solution) > 500
    ):
        return VerificationResult(
            None,
            "Проверка уравнения пропущена: некорректная переменная или слишком длинное выражение.",
        )

    try:
        import sympy as sp

        symbol = sp.Symbol(variable)
        left_expr = _safe_sympy_expr(left, variable, sp)
        right_expr = _safe_sympy_expr(right, variable, sp)
        solution_expr = _safe_sympy_expr(solution, variable, sp)

        # Значение решения должно быть конкретным, без свободных переменных.
        if solution_expr.free_symbols:
            return VerificationResult(
                None,
                "Проверка уравнения пропущена: решение не является конкретным значением.",
            )

        # Безопасная проверка равенства: simplify -> is_zero
        diff = sp.simplify(left_expr - right_expr).subs(symbol, solution_expr)
        simplified_diff = sp.simplify(diff)
        is_zero = simplified_diff.is_zero

        if is_zero is False:
            return VerificationResult(
                False,
                "Проверка не пройдена: предложенное значение не удовлетворяет уравнению.",
            )
        if is_zero is None:
            return VerificationResult(
                None,
                "SymPy не смог однозначно определить, удовлетворяет ли решение уравнению.",
            )


        # Проверка полноты множества решений
        try:
            # Пытаемся найти все корни уравнения
            equation = sp.Eq(left_expr, right_expr)
            all_solutions = sp.solve(equation, symbol)

            # Превращаем всё в set для сравнения
            all_solutions_set = {sp.simplify(s) for s in all_solutions}
            proposed_solution = sp.simplify(solution_expr)

            if not all_solutions_set:
                fullness_status = "Полноту множества решений подтвердить не удалось."
            elif proposed_solution in all_solutions_set:
                if len(all_solutions_set) == 1:
                    fullness_status = "Полнота множества решений подтверждена: это единственный корень."
                else:
                    fullness_status = (
                        f"Предложенный корень верен, но уравнение может иметь другие решения. "
                        f"SymPy нашёл всего {len(all_solutions_set)} корня."
                    )
            else:
                fullness_status = "Полноту множества решений подтвердить не удалось."
        except Exception:
            fullness_status = "Полноту множества решений подтвердить не удалось."

        return VerificationResult(
            True,
            f"Ответ проверен через SymPy: предложенное значение удовлетворяет уравнению. {fullness_status}",
        )

    except ImportError:
        return VerificationResult(
            None,
            "SymPy не установлен; независимая проверка недоступна.",
        )
    except Exception:
        return VerificationResult(
            None,
            "SymPy не смог разобрать выражения уравнения или решения.",
        )
