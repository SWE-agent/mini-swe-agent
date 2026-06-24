# Fix expression whitespace handling

The expression evaluator in `upstream/src/minisweagent/run/expr/` fails on expressions with whitespace (e.g. `" 1 + 2 "`).

Run `python -m pytest upstream/tests/run/test_expr.py -v` — at least one test should fail.

Your job:

1. Read the failing test(s) and trace how input is normalized before tokenization
2. Fix whitespace handling so spaced expressions evaluate correctly
3. Run pytest again to verify
4. Run `git diff` and submit when done
