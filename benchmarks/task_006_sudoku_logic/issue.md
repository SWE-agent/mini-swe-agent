# Fix sudoku validation logic

The sudoku game in `upstream/src/minisweagent/run/sudoku.py` has a bug in move validation logic.

Run `python -m pytest upstream/tests/run/test_sudoku.py -v` — at least one test should fail.

Your job:

1. Read the failing test(s) and the sudoku source code
2. Fix the bug so all tests in `upstream/tests/run/test_sudoku.py` pass
3. Run pytest again to verify
4. Run `git diff` and submit when done
