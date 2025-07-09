#!/usr/bin/env python3
import json
import os
from pathlib import Path

import typer
import yaml
from rich.console import Console

from microswea import package_dir
from microswea.agents.interactive import InteractiveAgent
from microswea.environments.local import LocalEnvironment
from microswea.models import get_model

DEFAULT_CONFIG = Path(os.getenv("MSWEA_LOCAL_CONFIG_PATH", package_dir / "config" / "local.yaml"))
console = Console(highlight=False)
app = typer.Typer()


def get_multiline_task() -> str:
    """Get a multiline problem statement from the user."""
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
    config: str = typer.Option(str(DEFAULT_CONFIG), "-c", "--config", help="Path to config file"),
    model: str | None = typer.Option(
        None,
        "-m",
        "--model",
        help="Model to use",
    ),
    problem: str | None = typer.Option(None, "-p", "--problem", help="Problem statement", show_default=False),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
) -> InteractiveAgent:
    """Run micro-SWE-agent right here, right now."""
    _config = yaml.safe_load(Path(config).read_text())

    if not problem:
        problem = get_multiline_task()

    # Use get_model to defer model imports (can take a while), but also to switch in
    # some optimized models (especially for anthropic)
    agent = InteractiveAgent(
        get_model(model, _config.get("model", {})),
        LocalEnvironment(),
        **(_config.get("agent", {}) | {"confirm_actions": not yolo}),
    )

    try:
        exit_status, result = agent.run(problem)
    except KeyboardInterrupt:
        console.print("\n[bold red]KeyboardInterrupt -- goodbye[/bold red]")
    else:
        Path("patch.txt").write_text(result)
    finally:
        Path("traj.json").write_text(
            json.dumps(agent.messages, indent=2),
        )
        console.print(f"Total cost: [bold green]${agent.model.cost:.2f}[/bold green]")
        console.print(f"Total steps: [bold green]{agent.model.n_calls}[/bold green]")
    return agent


if __name__ == "__main__":
    app()
