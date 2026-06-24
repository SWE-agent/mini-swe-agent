# Fix multiple sudoku bugs

The sudoku implementation in `upstream/src/minisweagent/run/sudoku.py` has **multiple independent bugs**. Fixing only one issue may not be enough.

Run `python -m pytest upstream/tests/run/test_sudoku.py -v` — several tests should fail.

Your job:

1. Read all failing tests and inspect the sudoku source thoroughly
2. Fix **every** bug so the full test file passes
3. Re-run pytest to confirm all tests pass
4. Run `git diff` and submit when done

Do not stop after the first partial fix.
