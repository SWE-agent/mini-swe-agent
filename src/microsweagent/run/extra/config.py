"""Utility to manage the global config file.

You can also directly edit the `.env` file in the config directory.

It is located at [bold green]{global_config_file}[/bold green].
"""

import os
import subprocess

from dotenv import set_key, unset_key
from prompt_toolkit import prompt
from rich.console import Console
from rich.rule import Rule
from typer import Option, Typer

from microsweagent import global_config_file

app = Typer(
    help=__doc__.format(global_config_file=global_config_file),  # type: ignore
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
)
console = Console(highlight=False)


_SETUP_HELP = """Welcome to Micro!

To get started, we need to set up your global config file.

You can edit it manually or use the [bold green]micro-extra config set[/bold green] or [bold green]micro-extra config edit[/bold green] commands.

This setup will ask you for your model and an API key.

Here's a few popular models and the required API keys:

[bold green]claude-sonnet-4-20250514[/bold green] ([bold green]ANTHROPIC_API_KEY[/bold green])
[bold green]o3[/bold green] ([bold green]OPENAI_API_KEY[/bold green])

[bold yellow]You can leave any setting blank to skip it.[/bold yellow]
"""


def configure_if_first_time():
    if not os.getenv("MSWEA_CONFIGURED"):
        console.print(Rule())
        setup()
        console.print(Rule())


@app.command()
def setup():
    """Setup the global config file."""
    console.print(_SETUP_HELP.format(global_config_file=global_config_file))
    default_model = prompt(
        "Enter your default model (e.g., claude-sonnet-4-20250514): ", default=os.getenv("MSWEA_MODEL_NAME", "")
    ).strip()
    if default_model:
        set_key(global_config_file, "MSWEA_MODEL_NAME", default_model)
    key_name = prompt("Enter your API key name (e.g., ANTHROPIC_API_KEY): ").strip()
    key_value = None
    if key_name:
        key_value = prompt("Enter your API key value (e.g., sk-1234567890): ", default=os.getenv(key_name, "")).strip()
        if key_value:
            set_key(global_config_file, key_name, key_value)
    if not key_value:
        console.print(
            "[bold red]API key setup not completed.[/bold red] Totally fine if you have your keys as environment variables."
        )
    set_key(global_config_file, "MSWEA_CONFIGURED", "true")
    console.print(
        "\n[bold yellow]Config finished.[/bold yellow] If you want to revisit it, run [bold green]micro-extra config setup[/bold green]."
    )


@app.command()
def set(
    key: str = Option(..., help="The key to set", prompt=True),
    value: str = Option(..., help="The value to set", prompt=True),
):
    """Set a key in the global config file."""
    set_key(global_config_file, key, value)


@app.command()
def unset(key: str = Option(..., help="The key to unset")):
    """Unset a key in the global config file."""
    unset_key(global_config_file, key)


@app.command()
def edit():
    """Edit the global config file."""
    editor = os.getenv("EDITOR", "nano")
    subprocess.run([editor, global_config_file])


if __name__ == "__main__":
    app()
