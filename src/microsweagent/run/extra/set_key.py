"""Utility to set a key in the global config file.

You can also directly edit the `.env` file in the config directory.
The location is printed when you run `micro --help`.

Find all configuration options at https://mellow-pegasus-562d44.netlify.app/advanced/configuration/

Common examples:

[bold green]micro-extra set-key ANTHROPIC_API_KEY <your-api-key>[/bold green]  (set LM API key)
[bold green]micro-extra set-key MSWEA_MODEL_NAME claude-sonnet-4-20250514[/bold green]  (set default model)
"""

from dotenv import set_key
from rich.console import Console
from typer import Argument, Typer

from microsweagent import global_config_file

app = Typer(
    help=__doc__,
)
console = Console()


@app.command()
def set_global_key(
    key: str = Argument(..., help="The key to set"), value: str = Argument(..., help="The value to set")
):
    """Set a key in the global config file."""
    set_key(global_config_file, key, value)
    console.print(
        f"Key [bold green]{key}[/bold green] set to [bold green]{value}[/bold green] in [bold green]{global_config_file}[/bold green]"
    )


if __name__ == "__main__":
    app()
