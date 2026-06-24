# Fix expr package operator contract bug

The expression evaluator under `upstream/src/minisweagent/run/expr/` is split across multiple modules (`tokenize.py`, `evaluate.py`, `__init__.py`, …).

Run `python -m pytest upstream/tests/run/test_expr.py -v` — multiplication-related tests should fail.

Your job:

1. Read the failing test(s) and trace through the expr package modules
2. Fix the contract mismatch so tokenizer output matches what the evaluator expects
3. Run pytest again to verify
4. Run `git diff` and submit when done

Notes:

- Entry point: `minisweagent.run.expr.eval_expr`
- Operators are represented as strings passed from tokenization to evaluation
- The evaluator applies operators left-to-right (no operator precedence)
- Division by zero should raise `EvalError`
