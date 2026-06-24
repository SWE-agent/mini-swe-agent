# Fix five sudoku bugs

The sudoku game in `upstream/src/minisweagent/run/sudoku.py` contains **five separate logic bugs**. A single-line fix will not be enough.

Run `python -m pytest upstream/tests/run/test_sudoku.py -v` — tests should fail.

Your job:

1. Systematically read failing tests and the full `SudokuGame` implementation
2. Fix **all five** issues before submitting
3. Confirm the entire test file passes
4. Run `git diff` and submit when done

Expect to iterate: fixing one bug may reveal others still failing.
