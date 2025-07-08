#!/usr/bin/env python3
import json
import os
from pathlib import Path

import typer
import yaml

from microswea import package_dir
from microswea.agents.interactive_textual import AgentApp
from microswea.environments.local import LocalEnvironment
from microswea.models import get_model
from microswea.run.local import get_multiline_problem_statement

DEFAULT_CONFIG = Path(os.getenv("MSWEA_LOCAL2_CONFIG_PATH", package_dir / "config" / "local.yaml"))
app = typer.Typer()


@app.command()
def main(
    config: str = typer.Option(str(DEFAULT_CONFIG), "--config", help="Path to config file"),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Model to use",
    ),
    problem: str | None = typer.Option(None, "--problem", help="Problem statement", show_default=False),
    yolo: bool = typer.Option(False, "--yolo", help="Run without confirmation"),
) -> None:
    """Run micro-SWE-agent with textual interface."""
    _config = yaml.safe_load(Path(config).read_text())

    if not problem:
        problem = get_multiline_problem_statement()
        if not problem:
            typer.echo("No problem statement provided.")
            raise typer.Exit(1)

    # Create and run the agent app
    agent_app = AgentApp(
        model=get_model(model, _config.get("model", {})),
        env=LocalEnvironment(),
        problem_statement=problem,
        **_config.get("agent", {}),
    )

    try:
        agent_app.run()
    except KeyboardInterrupt:
        typer.echo("\nKeyboardInterrupt -- goodbye")
    finally:
        # Save results
        if agent_app.agent.messages:
            Path("traj.json").write_text(json.dumps(agent_app.agent.messages, indent=2))
            typer.echo(f"Total cost: ${agent_app.agent.model.cost:.2f}")
            typer.echo(f"Total steps: {agent_app.agent.model.n_calls}")


def cli() -> None:
    """CLI entry point for script execution."""
    app()


if __name__ == "__main__":
    app()
