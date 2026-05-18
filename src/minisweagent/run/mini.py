#!/usr/bin/env python3

"""Run mini-SWE-agent in your local environment. This is the default executable `mini`."""
# Read this first: https://mini-swe-agent.com/latest/usage/mini/  (usage)

import os
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from minisweagent import global_config_dir
from minisweagent.agents import get_agent
from minisweagent.agents.utils.prompt_user import _multiline_prompt
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models import get_model
from minisweagent.run.gitlab_utils import create_merge_request, fetch_issue_details, parse_issue_url
from minisweagent.run.utilities.config import configure_if_first_time
from minisweagent.utils.serialize import UNSET, recursive_merge

DEFAULT_CONFIG_FILE = Path(os.getenv("MSWEA_MINI_CONFIG_PATH", builtin_config_dir / "mini.yaml"))
DEFAULT_OUTPUT_FILE = global_config_dir / "/last_mini_run.traj.json"


_HELP_TEXT = """Run mini-SWE-agent in your local environment.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/mini/[/bold green]
[/not dim]
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c mini.yaml <other options>[/bold green]

Multiple configs will be recursively merged.

Examples:

[bold red]-c model.model_kwargs.temperature=0[/bold red] [red]You forgot to add the default config file! See above.[/red]

[bold green]-c mini.yaml -c model.model_kwargs.temperature=0.5[/bold green]

[bold green]-c swebench.yaml agent.mode=yolo[/bold green]
"""

console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich")


# fmt: off
@app.callback(invoke_without_command=True, help=_HELP_TEXT)
def main(
    ctx: typer.Context,
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use",),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'litellm' or 'minisweagent.models.litellm_model.LitellmModel')", rich_help_panel="Advanced"),
    agent_class: str | None = typer.Option(None, "--agent-class", help="Agent class to use (e.g., 'interactive' or 'minisweagent.agents.interactive.InteractiveAgent')", rich_help_panel="Advanced"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment class to use (e.g., 'local' or 'minisweagent.environments.local.LocalEnvironment')", rich_help_panel="Advanced"),
    task: str | None = typer.Option(None, "-t", "--task", help="Task/problem statement", show_default=False),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT),
    output: Path | None = typer.Option(DEFAULT_OUTPUT_FILE, "-o", "--output", help="Output trajectory file"),
    exit_immediately: bool = typer.Option(False, "--exit-immediately", help="Exit immediately when the agent wants to finish instead of prompting.", rich_help_panel="Advanced"),
) -> Any:
    # fmt: on
    if ctx.invoked_subcommand is not None:
        return None
    configure_if_first_time()
    console.print(f"DEBUG: Using DEFAULT_CONFIG_FILE: [bold cyan]{DEFAULT_CONFIG_FILE}[/bold cyan]")

    # Build the config from the command line arguments
    console.print(f"Building agent config from specs: [bold green]{config_spec}[/bold green]")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "run": {
            "task": task or UNSET,
        },
        "agent": {
            "agent_class": agent_class or UNSET,
            "mode": "yolo" if yolo else UNSET,
            "cost_limit": cost_limit or UNSET,
            "confirm_exit": False if exit_immediately else UNSET,
            "output_path": output or UNSET,
        },
        "model": {
            "model_class": model_class or UNSET,
            "model_name": model_name or UNSET,
        },
        "environment": {
            "environment_class": environment_class or UNSET,
        },
    })
    config = recursive_merge(*configs)

    if (run_task := config.get("run", {}).get("task", UNSET)) is UNSET:
        console.print("[bold yellow]What do you want to do?")
        run_task = _multiline_prompt()
        console.print("[bold green]Got that, thanks![/bold green]")

    model = get_model(config=config.get("model", {}))
    env = get_environment(config.get("environment", {}), default_type="local")
    agent = get_agent(model, env, config.get("agent", {}), default_type="interactive")
    agent.run(run_task)
    if (output_path := config.get("agent", {}).get("output_path")):
        console.print(f"Saved trajectory to [bold green]'{output_path}'[/bold green]")
    return agent


@app.command(help="Run agent to resolve a GitLab issue")
def gitlab(
    issue_url: str = typer.Argument(..., help="GitLab issue URL"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help="Config spec"),
):
    configure_if_first_time()
    # 1. Ingest issue
    console.print(f"Fetching GitLab issue: [bold green]{issue_url}[/bold green]")
    try:
        base_url, project_id, issue_iid = parse_issue_url(issue_url)
        issue = fetch_issue_details(base_url, project_id, issue_iid)
    except Exception as e:
        console.print(f"[bold red]Error parsing or fetching issue: {e}[/bold red]")
        raise typer.Exit(1)

    task_prompt = f"Fix the following issue:\nTitle: {issue['title']}\nDescription: {issue['description']}"

    # 2. Setup Docker Sandbox
    token = os.getenv("GITLAB_TOKEN")
    if not token:
        console.print("[bold red]Error: GITLAB_TOKEN environment variable is not set[/bold red]")
        raise typer.Exit(1)

    console.print("Initializing Docker sandbox...")
    # Decode project path to get repo path
    from urllib.parse import unquote_plus

    repo_path = unquote_plus(project_id)
    # Use base_url from the input URL for cloning
    # If base_url is localhost, replace it with host.docker.internal for docker access
    effective_base_url = base_url.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")
    clone_url = f"{effective_base_url.replace('://', f'://oauth2:{token}@')}/{repo_path}.git"

    env = DockerEnvironment(image="python:3.11-slim")

    # Execute clone securely
    clone_cmd = [
        "bash",
        "-c",
        f"apt-get update && apt-get install -y git && git clone {clone_url} /workspace && cd /workspace && git config user.email 'agent@swe.com' && git config user.name 'SWE Agent'",
    ]
    env.logger.info("Cloning repository inside sandbox")
    import subprocess

    subprocess.run([env.config.executable, "exec", env.container_id, *clone_cmd], check=True)
    env.config.cwd = "/workspace"  # Update cwd for subsequent commands

    # 3. Run Agent
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "run": {"task": task_prompt},
        "model": {"model_name": model_name or UNSET},
    })
    config = recursive_merge(*configs)

    model = get_model(config=config.get("model", {}))
    agent = get_agent(model, env, config.get("agent", {}), default_type="interactive")

    console.print("[bold green]Starting autonomous execution...[/bold green]")
    agent.run(task_prompt)

    # 4. Commit and MR
    branch_name = f"swe-agent/issue-{issue_iid}"
    commit_cmds = [
        f"git checkout -b {branch_name}",
        "git add -A",
        f"git commit -m 'Fix issue #{issue_iid}'",
        f"git push origin {branch_name}",
    ]
    for cmd in commit_cmds:
        subprocess.run(
            [env.config.executable, "exec", "-w", "/workspace", env.container_id, "bash", "-c", cmd], check=True
        )

    console.print("[bold green]Pushed branch successfully. Creating Merge Request...[/bold green]")
    try:
        mr_url = create_merge_request(
            base_url, project_id, branch_name, issue_iid, issue["title"], "Automated fix generated by SWE Agent."
        )
        console.print(f"🎉 [bold green]Success! Merge Request created:[/bold green] {mr_url}")
    except Exception as e:
        console.print(f"[bold red]Failed to create Merge Request: {e}[/bold red]")


if __name__ == "__main__":
    app()
