from minisweagent.run.sudoku import PRESETS, SudokuGame


def test_sudoku_rejects_fixed_and_conflicting_moves():
    game = SudokuGame(PRESETS["easy"])
    assert not game.set_value(0, 0, 1)
    assert not game.set_value(0, 2, 5)
    assert game.set_value(0, 2, 4)


def test_sudoku_clear_and_hint():
    game = SudokuGame(PRESETS["easy"])
    assert game.set_value(0, 2, 4)
    assert game.clear_value(0, 2)
    hint = game.hint()
    assert hint is not None
    row, col, value = hint
    assert game.is_valid_move(row, col, value)


def test_sudoku_detects_complete_board():
    game = SudokuGame([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ])
    assert game.is_complete()
