#!/usr/bin/env python3

"""
This is the [yellow]central entry point for all extra commands[/yellow] from micro-swe-agent.

Available sub-commands:

  [bold green]gh[/bold green] or [bold green]github-issue[/bold green]: Run on a GitHub issue
  [bold green]i[/bold green] or [bold green]inspect[/bold green]: Run in inspector mode
  [bold green]set-key[/bold green]: Set a key in the config file
"""

import sys

from dotenv import set_key
from rich.console import Console

from microsweagent import global_config_file


def main() -> None:
    """Main entry point for pipx run commands."""
    args = sys.argv[1:]

    if len(args) > 0 and args[0] in ["gh", "github-issue"]:
        from microsweagent.run.github_issue import app as github_app

        sys.argv = sys.argv[:1] + args[1:]  # Keep script name, remove "gh"
        github_app()
    elif len(args) > 0 and args[0] in ["i", "inspect", "inspector"]:
        from microsweagent.run.inspector import app as inspect_app

        sys.argv = sys.argv[:1] + args[1:]  # Keep script name, remove "i"
        inspect_app()
    elif len(args) > 0 and args[0] == "set-key":
        if len(args) != 3:
            raise ValueError("Usage: micro-extra set-key <key> <value>")

        set_key(global_config_file, args[1], args[2])
    else:
        Console().print(__doc__)


if __name__ == "__main__":
    main()
