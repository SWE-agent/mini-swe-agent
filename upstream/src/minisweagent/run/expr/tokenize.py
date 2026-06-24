from dataclasses import dataclass

from minisweagent.run.expr.errors import EvalError


@dataclass
class Token:
    is_number: bool
    number: float = 0.0
    operator: str = ""


def tokenize(expr: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0

    while i < len(expr):
        ch = expr[i]

        if ch.isdigit() or ch == ".":
            j = i
            while j < len(expr) and (expr[j].isdigit() or expr[j] == "."):
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
