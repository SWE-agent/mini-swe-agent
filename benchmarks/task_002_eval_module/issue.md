# Fix eval_module multiplication bug

The simple expression evaluator in `upstream/src/minisweagent/run/eval_module.py` has a bug.

Run `python -m pytest upstream/tests/run/test_eval_module.py -v` — at least one test should fail.

Your job:

1. Read the failing test(s) and the eval_module source code
2. Fix the bug so all tests in `upstream/tests/run/test_eval_module.py` pass
3. Run pytest again to verify
4. Run `git diff` and submit when done

Notes:

- The evaluator tokenizes an expression string and evaluates left-to-right (no operator precedence).
- Supported operators: `+`, `-`, `*`, `/`.
- Division by zero should raise `EvalError`.
