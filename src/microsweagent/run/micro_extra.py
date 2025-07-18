#!/usr/bin/env python3

"""
This is the [yellow]central entry point for all extra commands[/yellow] from micro-swe-agent.

Available sub-commands:

  [bold green]gh[/bold green] or [bold green]github-issue[/bold green]: Run on a GitHub issue
  [bold green]i[/bold green] or [bold green]inspect[/bold green]: Run in inspector mode
  [bold green]set-key[/bold green]: Set a key in the config file
"""

import sys

from rich.console import Console


def main() -> None:
    """Main entry point for pipx run commands."""
    args = sys.argv[1:]

    if len(args) > 0 and args[0] in ["gh", "github-issue"]:
        from microsweagent.run.github_issue import app as github_app

        github_app(args[1:], prog_name="micro-extra gh")
    elif len(args) > 0 and args[0] in ["i", "inspect", "inspector"]:
        from microsweagent.run.inspector import app as inspect_app

        inspect_app(args[1:], prog_name="micro-extra inspect")
    elif len(args) > 0 and args[0] == "set-key":
        from microsweagent.run.extra.set_key import app as set_key_app

        set_key_app(args[1:], prog_name="micro-extra set-key")
    else:
        Console().print(__doc__)


if __name__ == "__main__":
    main()
