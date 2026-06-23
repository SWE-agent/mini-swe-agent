from __future__ import annotations

import random
from dataclasses import dataclass

import typer

app = typer.Typer(help="Play Sudoku in the terminal.")

PRESETS = {
    "easy": [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ],
    "medium": [
        [0, 0, 0, 2, 6, 0, 7, 0, 1],
        [6, 8, 0, 0, 7, 0, 0, 9, 0],
        [1, 9, 0, 0, 0, 4, 5, 0, 0],
        [8, 2, 0, 1, 0, 0, 0, 4, 0],
        [0, 0, 4, 6, 0, 2, 9, 0, 0],
        [0, 5, 0, 0, 0, 3, 0, 2, 8],
        [0, 0, 9, 3, 0, 0, 0, 7, 4],
        [0, 4, 0, 0, 5, 0, 0, 3, 6],
        [7, 0, 3, 0, 1, 8, 0, 0, 0],
    ],
    "hard": [
        [0, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 8, 0, 0, 0, 7, 0, 9, 0],
        [6, 0, 2, 0, 0, 0, 5, 0, 0],
        [0, 7, 0, 0, 6, 0, 0, 0, 0],
        [0, 0, 0, 9, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 4, 0],
        [0, 0, 5, 0, 0, 0, 6, 0, 3],
        [0, 9, 0, 4, 0, 0, 0, 7, 0],
        [0, 0, 6, 0, 0, 0, 0, 0, 0],
    ],
}


@dataclass
class SudokuGame:
    puzzle: list[list[int]]

    def __post_init__(self) -> None:
        self.board = [row[:] for row in self.puzzle]
        self.fixed = {(r, c) for r in range(9) for c in range(9) if self.puzzle[r][c] != 0}

    def render(self) -> str:
        lines: list[str] = []
        for r, row in enumerate(self.board):
            if r % 3 == 0:
                lines.append("+-------+-------+-------+")
            values = [str(value) if value else "." for value in row]
            lines.append(f"| {' '.join(values[0:3])} | {' '.join(values[3:6])} | {' '.join(values[6:9])} |")
        lines.append("+-------+-------+-------+")
        return "\n".join(lines)

    def is_valid_move(self, row: int, col: int, value: int) -> bool:
        if not (0 <= row < 9 and 0 <= col < 9 and 1 <= value <= 9):
            return False
        if (row, col) in self.fixed:
            return False
        current = self.board[row][col]
        self.board[row][col] = 0
        valid = all(self.board[row][c] != value for c in range(9))
        valid = valid and all(self.board[r][col] != value for r in range(9))
        start_row, start_col = (row // 3) * 3, (col // 3) * 3
        valid = valid and all(self.board[r][c] != value for r in range(start_row, start_row + 3) for c in range(start_col, start_col + 3))
        self.board[row][col] = current
        return valid

    def set_value(self, row: int, col: int, value: int) -> bool:
        if not self.is_valid_move(row, col, value):
            return False
        self.board[row][col] = value
        return True

    def clear_value(self, row: int, col: int) -> bool:
        if (row, col) in self.fixed:
            return False
        self.board[row][col] = 0
        return True

    def is_complete(self) -> bool:
        return all(0 not in self.board[r] and self.is_valid_group(self.board[r]) for r in range(9)) and all(
            self.is_valid_group([self.board[r][c] for r in range(9)]) for c in range(9)
        ) and all(
            self.is_valid_group([
                self.board[r][c]
                for r in range(start_row, start_row + 3)
                for c in range(start_col, start_col + 3)
            ])
            for start_row in range(0, 9, 3)
            for start_col in range(0, 9, 3)
        )

    @staticmethod
    def is_valid_group(values: list[int]) -> bool:
        return sorted(values) == list(range(1, 10))

    def hint(self) -> tuple[int, int, int] | None:
        empties = [(r, c) for r in range(9) for c in range(9) if self.board[r][c] == 0]
        random.shuffle(empties)
        for row, col in empties:
            for value in range(1, 10):
                if self.is_valid_move(row, col, value):
                    return row, col, value
        return None


def _parse_position(token: str) -> tuple[int, int]:
    row, col = (int(part) - 1 for part in token.split(",", 1))
    return row, col


@app.command()
def main(difficulty: str = typer.Option("easy", "-d", "--difficulty", case_sensitive=False, help="easy, medium, or hard")) -> None:
    game = SudokuGame(PRESETS[difficulty.lower()])
    typer.echo("Sudoku")
    typer.echo("Commands: set ROW,COL VALUE | clear ROW,COL | hint | show | quit")
    while True:
        typer.echo(game.render())
        if game.is_complete():
            typer.echo("You solved it! 🎉")
            raise typer.Exit()
        command = typer.prompt("move").strip()
        if command in {"quit", "exit"}:
            raise typer.Exit()
        if command == "show":
            continue
        if command == "hint":
            if hint := game.hint():
                row, col, value = hint
                typer.echo(f"Try row {row + 1}, col {col + 1} = {value}")
            else:
                typer.echo("No hint available.")
            continue
        parts = command.split()
        if len(parts) == 3 and parts[0] == "set":
            row, col = _parse_position(parts[1])
            if game.set_value(row, col, int(parts[2])):
                typer.echo("Move accepted.")
            else:
                typer.echo("Invalid move.")
            continue
        if len(parts) == 2 and parts[0] == "clear":
            row, col = _parse_position(parts[1])
            if game.clear_value(row, col):
                typer.echo("Cell cleared.")
            else:
                typer.echo("Cannot clear that cell.")
            continue
        typer.echo("Unknown command.")


if __name__ == "__main__":
    app()
