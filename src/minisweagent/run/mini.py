#!/usr/bin/env python3

"""Run mini-SWE-agent in your local environment. This is the default executable `mini`."""
# Read this first: https://mini-swe-agent.com/latest/usage/mini/  (usage)

import os
from pathlib import Path
from typing import Any

import typer
import yaml
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console

from minisweagent import Environment, Model, global_config_dir
from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.agents.interactive_textual import AgentApp
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.run.extra.config import configure_if_first_time
from minisweagent.run.utils.save import save_traj

DEFAULT_CONFIG = Path(os.getenv("MSWEA_MINI_CONFIG_PATH", builtin_config_dir / "mini.yaml"))
DEFAULT_OUTPUT = global_config_dir / "last_mini_run.traj.json"
console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich")
prompt_session = PromptSession(history=FileHistory(global_config_dir / "mini_task_history.txt"))
_HELP_TEXT = """Run mini-SWE-agent in your local environment.

[not dim]
There are two different user interfaces:

[bold green]mini[/bold green] Simple REPL-style interface
[bold green]mini -v[/bold green] Pager-style interface (Textual)

More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/mini/[/bold green]
[/not dim]
"""


def run_interactive(model: Model, env: Environment, agent_config: dict, task: str, output: Path | None = None) -> Any:
    agent = InteractiveAgent(
        model,
        env,
        **agent_config,
    )

    exit_status, result = None, None
    try:
        exit_status, result = agent.run(task)
    finally:
        if output:
            save_traj(agent, output, exit_status=exit_status, result=result)
    return agent


def run_textual(model: Model, env: Environment, agent_config: dict, task: str, output: Path | None = None) -> Any:
    agent_app = AgentApp(
        model,
        env,
        task,
        **agent_config,
    )
    try:
        agent_app.run()
    finally:
        if output:
            save_traj(agent_app.agent, output, exit_status=agent_app.exit_status, result=agent_app.result)


@app.command(help=_HELP_TEXT)
def main(
    visual: bool = typer.Option(
        False,
        "-v",
        "--visual",
        help="Toggle (pager-style) UI (Textual) depending on the MSWEA_VISUAL_MODE_DEFAULT environment setting",
    ),
    model_name: str | None = typer.Option(
        None,
        "-m",
        "--model",
        help="Model to use",
    ),
    task: str | None = typer.Option(None, "-t", "--task", help="Task/problem statement", show_default=False),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
    config_spec: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
    output: Path | None = typer.Option(DEFAULT_OUTPUT, "-o", "--output", help="Output trajectory file"),
    exit_immediately: bool = typer.Option(
        False, "--exit-immediately", help="Exit immediately when the agent wants to finish instead of prompting."
    ),
) -> Any:
    configure_if_first_time()
    config = yaml.safe_load(get_config_path(config_spec).read_text())

    if not task:
        console.print("[bold yellow]What do you want to do?")
        task = prompt_session.prompt(
            "",
            multiline=True,
            bottom_toolbar=HTML(
                "Submit task: <b fg='yellow' bg='black'>Esc+Enter</b> | "
                "Navigate history: <b fg='yellow' bg='black'>Arrow Up/Down</b> | "
                "Search history: <b fg='yellow' bg='black'>Ctrl+R</b>"
            ),
        )
        console.print("[bold green]Got that, thanks![/bold green]")

    config["agent"]["mode"] = "confirm" if not yolo else "yolo"
    if cost_limit:
        config["agent"]["cost_limit"] = cost_limit
    if exit_immediately:
        config["agent"]["confirm_exit"] = False
    model = get_model(model_name, config.get("model", {}))
    env = LocalEnvironment(**config.get("env", {}))

    # Both visual flag and the MSWEA_VISUAL_MODE_DEFAULT flip the mode, so it's essentially a XOR
    if visual == (os.getenv("MSWEA_VISUAL_MODE_DEFAULT", "false") == "false"):
        return run_textual(model, env, config["agent"], task, output)  # type: ignore[arg-type]
    else:
        return run_interactive(model, env, config["agent"], task, output)  # type: ignore[arg-type]


if __name__ == "__main__":
    app()
