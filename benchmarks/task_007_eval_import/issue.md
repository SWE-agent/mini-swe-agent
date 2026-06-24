# Fix eval_module import

The expression evaluator was split into the `upstream/src/minisweagent/run/expr/` package, with `eval_module.py` kept as a backward-compatible entry point.

Tests import from `minisweagent.run.eval_module`, but something is wrong with the import wiring.

Run `python -m pytest upstream/tests/run/test_eval_module.py -v` — tests should fail to import or fail at runtime.

Your job:

1. Trace how `test_eval_module.py` imports the evaluator and locate the broken import path
2. Fix the facade so `from minisweagent.run.eval_module import eval_expr` works again
3. Run pytest again to verify
4. Run `git diff` and submit when done
