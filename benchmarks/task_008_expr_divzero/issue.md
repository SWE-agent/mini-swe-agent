# Fix division-by-zero handling

The expression evaluator in `upstream/src/minisweagent/run/expr/` mishandles division by zero.

Run `python -m pytest upstream/tests/run/test_expr.py -v` — at least one test should fail.

Your job:

1. Read the failing test(s) and the evaluator source
2. Fix the bug so division by zero raises `EvalError` as expected
3. Run pytest again to verify
4. Run `git diff` and submit when done
