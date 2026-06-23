"""
Eval evaluates a simple math expression and returns the result.
Supports: +, -, *, / operators and numbers.
"""

from dataclasses import dataclass
from typing import Union


class EvalError(Exception):
    """Error raised during expression evaluation."""
    pass


@dataclass
class Token:
    is_number: bool
    number: float = 0.0
    operator: str = ""


def eval_expr(expr: str) -> float:
    """
    Evaluate a simple math expression and return the result.
    """
    expr = expr.replace(" ", "")
    if not expr:
        raise EvalError("empty expression")

    tokens = tokenize(expr)
    return evaluate(tokens)


def tokenize(expr: str) -> list[Token]:
    tokens = []
    i = 0

    while i < len(expr):
        ch = expr[i]

        if ch.isdigit() or ch == '.':
            # Parse number
            j = i
            while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                j += 1
            num_str = expr[i:j]
            try:
                num = float(num_str)
            except ValueError:
                raise EvalError(f"invalid number: {num_str}")
            tokens.append(Token(is_number=True, number=num))
            i = j
        elif ch in "+-*/":
            tokens.append(Token(is_number=False, operator=ch))
            i += 1
        else:
            raise EvalError(f"invalid character: {ch}")

    return tokens


def evaluate(tokens: list[Token]) -> float:
    if not tokens:
        raise EvalError("no tokens to evaluate")

    if not tokens[0].is_number:
        raise EvalError("expression must start with a number")

    result = tokens[0].number

    i = 1
    while i < len(tokens):
        op = tokens[i]
        if op.is_number:
            raise EvalError(f"expected operator at position {i}")

        if i + 1 >= len(tokens):
            raise EvalError("missing number after operator")

        num = tokens[i + 1]
        if not num.is_number:
            raise EvalError(f"expected number at position {i + 1}")

        if op.operator == '+':
            result += num.number
        elif op.operator == '-':
            result -= num.number
        elif op.operator == '*':
            result *= num.number
        elif op.operator == '/':
            if num.number == 0:
                raise EvalError("division by zero")
            result /= num.number

        i += 2

    return result
