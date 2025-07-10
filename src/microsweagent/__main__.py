#!/usr/bin/env python3
"""Main entry point for pipx run micro-swe-agent commands."""

import importlib
import os
import sys


def main() -> None:
    """Main entry point for pipx run commands."""
    args = sys.argv[1:]

    if len(args) > 0 and args[0] == "gh":
        from microsweagent.run.github_issue import app as github_app

        sys.argv = sys.argv[:1] + args[1:]  # Keep script name, remove "gh"
        github_app()
    elif len(args) > 0 and args[0] == "i":
        from microsweagent.run.inspector import app as inspect_app

        sys.argv = sys.argv[:1] + args[1:]  # Keep script name, remove "i"
        inspect_app()
    else:
        default_module = os.getenv("MSWEA_DEFAULT_RUN", "microsweagent.run.local")
        module = importlib.import_module(default_module)
        sys.argv = sys.argv[:1] + args  # Keep script name, pass all args
        module.app()


if __name__ == "__main__":
    main()
