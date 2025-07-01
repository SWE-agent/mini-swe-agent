#!/usr/bin/env python3
import argparse
from pathlib import Path

import requests
import yaml

from nanoswea.agent import Agent, AgentConfig
from nanoswea.environment import DockerEnvironment
from nanoswea.model import LitellmModel, ModelConfig


def fetch_github_issue(issue_url: str) -> str:
    """Fetch GitHub issue text from the URL."""
    # Convert GitHub issue URL to API URL
    api_url = issue_url.replace("github.com", "api.github.com/repos").replace("/issues/", "/issues/")

    response = requests.get(api_url)
    issue_data = response.json()

    title = issue_data["title"]
    body = issue_data["body"] or ""

    return f"GitHub Issue: {title}\n\n{body}"


def run_github_issue(issue_url: str, repo_url: str) -> Agent:
    problem_statement = fetch_github_issue(issue_url)

    config = yaml.safe_load((Path(__file__).parent / "config" / "github_issue.yaml").read_text())
    agent_config = AgentConfig(**config["agent"])
    model_config = ModelConfig(**config["model"])

    model = LitellmModel(model_config)
    env = DockerEnvironment(config["image"])
    agent = Agent(agent_config, model, env, problem_statement)

    print(f"Cloning {repo_url} to /testbed...")
    agent.env.execute(f"git clone {repo_url} /testbed", cwd="/")

    result = agent.run()
    print(f"\nFinal result: {result}")
    print(f"Total cost: ${agent.model.cost:.4f}")
    print(f"Total steps: {agent.model.n_calls}")

    print("Saving patch to patch.txt...")
    Path("patch.txt").write_text(result)
    return agent


def run_from_cli(cli_args: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run nano-SWE-agent on a GitHub issue")
    parser.add_argument("issue_url", help="GitHub issue URL")

    args = parser.parse_args(args=cli_args)
    repo_url = args.issue_url.split("/issues/")[0]
    run_github_issue(args.issue_url, repo_url)


if __name__ == "__main__":
    run_from_cli()
