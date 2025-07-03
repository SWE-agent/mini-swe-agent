#!/usr/bin/env python3
"""Main entry point for pipx run micro-swe-agent commands."""

import sys

from microswea.run.github_issue import app as github_app
from microswea.run.local import app as local_app


def main() -> None:
    """Main entry point for pipx run commands."""
    args = sys.argv[1:]

    if len(args) > 0 and args[0] == "gh":
        # Remove "gh" from args and run GitHub issue CLI
        sys.argv = sys.argv[:1] + args[1:]  # Keep script name, remove "gh"
        github_app()
    else:
        # Run local CLI with all args
        sys.argv = sys.argv[:1] + args  # Keep script name, pass all args
        local_app()


if __name__ == "__main__":
    main()
