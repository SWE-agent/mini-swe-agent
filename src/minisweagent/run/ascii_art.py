import typer

app = typer.Typer(help="Simple ASCII art demo command.")


ASCII_ART = r"""
          /\\
         /  \\
        /++++\\
       /  ()  \\
       /      \\
      /~`~`~`~`\\
     /  ()  ()  \\
     /          \\
    /------------\\
          ||
          ||
          ||
"""


@app.command()
def main() -> None:
    """Print a small ASCII-art tree."""
    print(ASCII_ART)


if __name__ == "__main__":
    app()
