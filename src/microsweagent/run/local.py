#!/usr/bin/env python3
import os
from pathlib import Path

import typer
import yaml
from rich.console import Console

from microsweagent.agents.interactive import InteractiveAgent
from microsweagent.config import builtin_config_dir, get_config_path
from microsweagent.environments.local import LocalEnvironment
from microsweagent.models import get_model
from microsweagent.run.utils.save import save_traj

DEFAULT_CONFIG = Path(os.getenv("MSWEA_LOCAL_CONFIG_PATH", builtin_config_dir / "local.yaml"))
console = Console(highlight=False)
app = typer.Typer()


def get_multiline_task() -> str:
    """Get a multiline task/problem statement from the user."""
    console.print(
        "[bold yellow]What do you want to do?[/bold yellow] (press [bold red]Ctrl+D[/bold red] in a [red]new line[/red] to finish)"
    )
    lines = []
    while True:
        try:
            lines.append(console.input("[bold green]>[/bold green] "))
        except EOFError:
            break

    print()
    return "\n".join(lines).strip()


@app.command()
def main(
    config: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
    model: str | None = typer.Option(
        None,
        "-m",
        "--model",
        help="Model to use",
    ),
    task: str | None = typer.Option(None, "-t", "--task", help="Task/problem statement", show_default=False),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
    output: Path | None = typer.Option(None, "-o", "--output", help="Output file"),
) -> InteractiveAgent:
    """Run micro-SWE-agent right here, right now."""
    _config = yaml.safe_load(get_config_path(config).read_text())

    if not task:
        task = get_multiline_task()

    mode = "confirm" if not yolo else "yolo"

    # Use get_model to defer model imports (can take a while), but also to switch in
    # some optimized models (especially for anthropic)
    agent = InteractiveAgent(
        get_model(model, _config.get("model", {})),
        LocalEnvironment(),
        **(_config.get("agent", {}) | {"mode": mode}),
    )

    exit_status, result = None, None
    try:
        exit_status, result = agent.run(task)
    except KeyboardInterrupt:
        console.print("\n[bold red]KeyboardInterrupt -- goodbye[/bold red]")
    finally:
        if output:
            save_traj(agent, output, exit_status=exit_status, result=result)
    return agent


if __name__ == "__main__":
    app()
