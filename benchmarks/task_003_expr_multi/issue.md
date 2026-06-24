# Fix expr package multiplication bug

The simple expression evaluator lives under `upstream/src/minisweagent/run/expr/` as a multi-file package (`tokenize.py`, `evaluate.py`, etc.).

Run `python -m pytest upstream/tests/run/test_expr.py -v` — at least one test should fail.

Your job:

1. Read the failing test(s) and trace through the expr package source files
2. Fix the bug so all tests in `upstream/tests/run/test_expr.py` pass
3. Run pytest again to verify
4. Run `git diff` and submit when done

Notes:

- Start from `minisweagent.run.expr` (`__init__.py` re-exports `eval_expr`).
- The evaluator tokenizes an expression string and evaluates left-to-right (no operator precedence).
- Supported operators: `+`, `-`, `*`, `/`.
- Division by zero should raise `EvalError`.
