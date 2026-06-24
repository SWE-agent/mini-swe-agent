from minisweagent.run.expr.errors import EvalError
from minisweagent.run.expr.tokenize import Token


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

        if op.operator == "+":
            result += num.number
        elif op.operator == "-":
            result -= num.number
        elif op.operator == "*":
            result *= num.number
        elif op.operator == "/":
            if num.number == 0:
                raise EvalError("division by zero")
            result /= num.number

        i += 2

    return result
