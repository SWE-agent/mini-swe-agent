#!/usr/bin/env python3
import json
import os
from pathlib import Path

import requests
import typer
import yaml
from rich.console import Console

from microswea import package_dir
from microswea.agents.interactive import InteractiveAgent
from microswea.environments.docker import DockerEnvironment
from microswea.models import get_model

DEFAULT_CONFIG = Path(os.getenv("MSWEA_GITHUB_CONFIG_PATH", package_dir / "config" / "github_issue.yaml"))
console = Console(highlight=False)
app = typer.Typer()


def fetch_github_issue(issue_url: str) -> str:
    """Fetch GitHub issue text from the URL."""
    # Convert GitHub issue URL to API URL
    api_url = issue_url.replace("github.com", "api.github.com/repos").replace("/issues/", "/issues/")

    headers = {}
    if github_token := os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"token {github_token}"

    response = requests.get(api_url, headers=headers)
    issue_data = response.json()

    title = issue_data["title"]
    body = issue_data["body"] or ""

    return f"GitHub Issue: {title}\n\n{body}"


@app.command()
def main(
    issue_url: str = typer.Option(prompt="Enter GitHub issue URL", help="GitHub issue URL"),
    config: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use"),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
) -> InteractiveAgent:
    """Run micro-SWE-agent on a GitHub issue"""

    _config = yaml.safe_load(Path(config).read_text())
    _agent_config = _config.get("agent", {})
    if yolo:
        _agent_config["confirm_actions"] = False

    problem_statement = fetch_github_issue(issue_url)

    agent = InteractiveAgent(
        get_model(model, _config.get("model", {})),
        DockerEnvironment(**_config.get("environment", {})),
        problem_statement,
        **_agent_config,
    )

    repo_url = issue_url.split("/issues/")[0]
    if github_token := os.getenv("GITHUB_TOKEN"):
        repo_url = repo_url.replace("https://github.com/", f"https://{github_token}@github.com/") + ".git"

    agent.env.execute(f"git clone {repo_url} /testbed", cwd="/")

    try:
        exit_status, result = agent.run()
    except KeyboardInterrupt:
        console.print("\n[bold red]KeyboardInterrupt -- goodbye[/bold red]")
    else:
        console.print(f"\nAgent finished with {exit_status}: {result}")
        Path("patch.txt").write_text(result)
    finally:
        Path("traj.json").write_text(
            json.dumps(agent.messages, indent=2),
        )
        console.print(f"Total cost: [bold green]${agent.model.cost:.4f}[/bold green]")
        console.print(f"Total steps: [bold green]{agent.model.n_calls}[/bold green]")
    return agent


if __name__ == "__main__":
    app()
