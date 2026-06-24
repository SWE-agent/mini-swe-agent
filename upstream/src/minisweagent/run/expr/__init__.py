"""Simple left-to-right math expression evaluator (multi-module layout)."""

from minisweagent.run.expr.errors import EvalError
from minisweagent.run.expr.evaluate import evaluate
from minisweagent.run.expr.tokenize import Token, tokenize


def eval_expr(expr: str) -> float:
    """Evaluate a simple math expression and return the result."""
    expr = expr.replace(" ", "")
    if not expr:
        raise EvalError("empty expression")

    tokens = tokenize(expr)
    return evaluate(tokens)


__all__ = ["EvalError", "Token", "eval_expr", "evaluate", "tokenize"]
