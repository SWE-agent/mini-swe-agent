# Fix multiple expr package bugs

The expression evaluator under `upstream/src/minisweagent/run/expr/` has **multiple independent bugs** spread across more than one module.

Run `python -m pytest upstream/tests/run/test_expr.py -v` — several tests should fail.

Your job:

1. Trace failures across the expr package (`__init__.py`, `tokenize.py`, `evaluate.py`, etc.)
2. Fix **every** bug so all expr tests pass
3. Re-run pytest to verify
4. Run `git diff` and submit when done

Partial fixes that leave other bugs in place are not sufficient.
