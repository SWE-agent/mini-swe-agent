# Fix five expr package bugs

The expression evaluator package has **five independent bugs** across multiple files.

Run `python -m pytest upstream/tests/run/test_expr.py -v` — multiple tests should fail.

Your job:

1. Inspect failures and read all relevant expr modules
2. Fix **every** injected bug (whitespace handling, operators, division guard, token contract, etc.)
3. Re-run pytest until the full suite passes
4. Run `git diff` and submit when done

This task requires multiple edits; do not submit after a partial fix.
