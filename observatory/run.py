# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Connect any agent repository to the Agent Observatory: Claude Code wires the logger in and drafts the architecture diagram.

Put this `observatory/` folder in the root of the agent repository, then:

    uv run observatory/run.py                  # wire the agent to the logger, draw the architecture diagram, check
    uv run observatory/run.py --serve          # ... and start the dashboard afterwards
    uv run observatory/run.py --serve-only     # later: just start the dashboard
    uv run observatory/run.py --archify-only   # after committing: re-pin the diagram spec to HEAD and regenerate the HTML

The dashboard (dashboard.py) and the logger (observatory_log.py) are the same for every agent. What differs per agent is
done by Claude Code: where the logger calls go, and the Archify spec of the agent's loop (the left panel). The diagram needs
Archify (--archify-dir, $ARCHIFY_DIR, or an `archify/` repository next to this checkout); without it only the logging is wired.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

KIT = Path(__file__).resolve().parent
REPO = KIT.parent
LOGGER = KIT / "observatory_log.py"
SPEC = KIT / "diagram/agent-architecture.json"
DIAGRAM = KIT / "diagram/agent-architecture.html"
RUNS = KIT / "runs"
# The Claude Code slot: a fresh, memoryless session that may edit files and run uv/git diff. The prompt arrives on stdin.
CLAUDE_ARGS = [
    "-p",
    "--no-session-persistence",
    "--setting-sources",
    "project",
    "--strict-mcp-config",
    "--disable-slash-commands",
    "--permission-mode",
    "acceptEdits",
]
CLAUDE_TOOLS = ["Read", "Edit", "Write", "Grep", "Glob", "Bash(uv run:*)", "Bash(git diff:*)"]


def step(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def find_claude() -> str:
    vscode = sorted(
        Path.home().glob(".vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude"), reverse=True
    )
    if found := next(
        (c for c in [os.environ.get("CLAUDE_BIN"), shutil.which("claude"), *vscode] if c and Path(c).exists()), None
    ):
        return str(found)
    raise SystemExit("Claude Code CLI not found. Put `claude` on PATH or set CLAUDE_BIN.")


def claude(prompt: str, model: str, budget: float, archify_dir: Path | None) -> None:
    tools = [*CLAUDE_TOOLS, f"Bash(node {archify_dir}/bin/archify.mjs validate:*)"] if archify_dir else CLAUDE_TOOLS
    allow_archify = ["--add-dir", str(archify_dir)] if archify_dir else []
    result = subprocess.run(
        [
            find_claude(),
            *CLAUDE_ARGS,
            *allow_archify,
            "--allowedTools",
            *tools,
            "--model",
            model,
            "--max-budget-usd",
            str(budget),
        ],
        input=prompt,
        cwd=REPO,
        text=True,
        capture_output=True,
        env={**os.environ, "OBSERVATORY_RUNS": str(RUNS)},
    )
    print(result.stdout.strip())
    if result.returncode:
        raise SystemExit(f"Claude Code failed:\n{result.stderr}")


def prompt_text(archify_dir: Path | None) -> str:
    wiring, diagram = (KIT / "prompt/task.md").read_text().split("<!-- diagram -->")
    return (
        (wiring + (diagram if archify_dir else ""))
        .replace("{kit}", str(KIT.relative_to(REPO)))
        .replace("{archify_dir}", str(archify_dir))
    )


def git(*args: str, env: dict | None = None) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, env=env).stdout.strip()


def check_wiring() -> None:
    """Restore any edited copy of the logger, then report what is wired."""
    exclude = f":!{KIT.relative_to(REPO)}"
    copies = [
        REPO / p
        for p in git("ls-files", "-co", "--exclude-standard", "--", "*observatory_log.py", exclude).splitlines()
    ]
    for copy in copies:
        if copy.read_text() != LOGGER.read_text():
            shutil.copy(LOGGER, copy)
            print(f"restored {copy.relative_to(REPO)}: the logger must stay identical to {LOGGER.relative_to(REPO)}")
    users = [
        f
        for f in git("grep", "--untracked", "-l", "observatory_log", "--", ".", exclude).splitlines()
        if REPO / f not in copies
    ]
    if copies and users:
        print(f"logger: {', '.join(str(c.relative_to(REPO)) for c in copies)}\ncalled from: {', '.join(users)}")
    else:
        print(
            "warning: nothing calls the logger yet"
            if copies
            else "warning: Claude did not copy observatory_log.py into the agent package"
        )


def check_trace() -> None:
    kinds = Counter(json.loads(line)["kind"] for run in RUNS.glob("*.jsonl") for line in run.read_text().splitlines())
    print(
        f"events from Claude's own check run: {dict(kinds)}"
        if kinds
        else f"warning: no events in {RUNS.relative_to(REPO)}: run your agent with OBSERVATORY_RUNS to see the history"
    )


def check_spec_sources() -> list[str]:
    """Sources of the architecture spec that point to a missing file or a line past its end."""
    problems = []
    for component in json.loads(SPEC.read_text())["components"]:
        for source in component.get("sources", []):
            path = REPO / source["path"]
            lines = len(path.read_text().splitlines()) if path.exists() else 0
            if max(source["line"], source.get("end_line", 0)) > lines:
                problems.append(f"{component['id']}: {source['path']}:{source['line']} ({lines} lines)")
    return problems


def current_revision() -> str:
    """HEAD, or when the work tree has uncommitted changes a dangling commit of it, so that Archify's source check sees the new code.

    The snapshot commit is built with a temporary index: HEAD, the branches and the real index stay untouched.
    """
    if not git("status", "--porcelain"):
        return git("rev-parse", "HEAD")
    with tempfile.TemporaryDirectory() as tmp:
        env = {
            **os.environ,
            "GIT_INDEX_FILE": f"{tmp}/index",
            "GIT_AUTHOR_NAME": "observatory",
            "GIT_AUTHOR_EMAIL": "o@o",
            "GIT_COMMITTER_NAME": "observatory",
            "GIT_COMMITTER_EMAIL": "o@o",
        }
        git("read-tree", "HEAD", env=env)
        git("add", "-A", env=env)
        return git(
            "commit-tree", git("write-tree", env=env), "-p", "HEAD", "-m", "observatory work tree snapshot", env=env
        )


def pin_repository() -> None:
    """Point the spec's source links to this checkout: its origin URL and the current revision."""
    text = SPEC.read_text()
    text = re.sub(r'("revision":\s*")[0-9a-f]{40}(")', rf"\g<1>{current_revision()}\2", text)
    if match := re.match(
        r"(?:git@github\.com:|https://github\.com/)(.+?)(?:\.git)?$", git("remote", "get-url", "origin")
    ):
        text = re.sub(
            r'("url":\s*")https://github\.com/[^"]+(")', rf"\g<1>https://github.com/{match[1]}\2", text, count=1
        )
    SPEC.write_text(text)


def archify(archify_dir: Path, model: str, budget: float) -> None:
    node = lambda *args, **kwargs: subprocess.run(
        ["node", "bin/archify.mjs", *args, "--quality", "showcase", "--repo-root", REPO],
        cwd=archify_dir,
        text=True,
        **kwargs,
    )
    for attempt in range(3):
        pin_repository()
        validation = node("validate", "architecture", SPEC, capture_output=True)
        if validation.returncode == 0:
            break
        if attempt == 2:
            raise SystemExit(f"Archify validation failed:\n{validation.stdout}{validation.stderr}")
        claude(
            f"`{SPEC.relative_to(REPO)}` fails Archify validation. Fix only what the errors point to; keep everything else and the "
            "file formatting as it is. Do not create HTML. The `line` values must match the files in the work tree as they are now: "
            f"never move them to fit another revision.\n\n{validation.stdout}{validation.stderr}",
            model,
            budget,
            archify_dir,
        )
    delivery = node("deliver", "architecture", SPEC, DIAGRAM, "--json", capture_output=True)
    if delivery.returncode:
        raise SystemExit(f"Archify failed to generate the diagram:\n{delivery.stdout}{delivery.stderr}")
    print(f"diagram regenerated: {DIAGRAM.relative_to(REPO)}")
    for receipt in (
        SPEC.parent / "agent-architecture.delivery.json",
        DIAGRAM.parent / "agent-architecture.delivery.json",
    ):
        receipt.unlink(missing_ok=True)


def serve(port: int) -> int:
    return subprocess.run([sys.executable, KIT / "dashboard.py", "--runs", RUNS, "--port", str(port)]).returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="sonnet", help="Claude Code model")
    parser.add_argument("--budget", type=float, default=3.0, help="max USD per Claude Code session")
    parser.add_argument(
        "--archify-dir",
        default=os.environ.get("ARCHIFY_DIR"),
        help="Archify's `archify/` directory (default: ../archify/archify)",
    )
    parser.add_argument("--serve", action="store_true", help="start the dashboard when done")
    parser.add_argument("--serve-only", action="store_true", help="only start the dashboard")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--no-claude", action="store_true", help="skip the Claude Code session (the agent is already wired)"
    )
    parser.add_argument("--archify-only", action="store_true", help="only re-pin the spec and regenerate the diagram")
    parser.add_argument("--test", help="a command to run in the repository at the end, for example 'uv run pytest -q'")
    args = parser.parse_args()
    archify_dir = Path(args.archify_dir).expanduser() if args.archify_dir else REPO.parent / "archify/archify"
    archify_dir = archify_dir if (archify_dir / "bin/archify.mjs").exists() else None
    print(f"Repository: {REPO}")

    if args.serve_only:
        raise SystemExit(serve(args.port))
    if not args.archify_only:
        step("Checking the kit")
        subprocess.run([sys.executable, KIT / "selftest.py"], check=True, capture_output=True)
        print("logger and dashboard self-test passed")
        if not args.no_claude:
            step(
                "Claude Code: connecting the agent to the logger"
                + (" and drawing the diagram" if archify_dir else "")
                + " (takes a few minutes)"
            )
            claude(prompt_text(archify_dir), args.model, args.budget, archify_dir)
        step("Checking the result")
        check_wiring()
        check_trace()
        if archify_dir and SPEC.exists() and (problems := check_spec_sources()):
            print("spec sources that do not match the code:\n  " + "\n  ".join(problems))
    if archify_dir and SPEC.exists():
        step("Archify: validating the spec and generating the diagram")
        archify(archify_dir, args.model, args.budget)
    else:
        print(
            "\n(no diagram: "
            + (
                "the spec was not written"
                if archify_dir
                else "Archify not found. Put the archify repo next to this checkout or set --archify-dir / $ARCHIFY_DIR"
            )
            + ")"
        )
    tests = subprocess.run(shlex.split(args.test), cwd=REPO).returncode if args.test and not args.archify_only else 0
    print(
        f"\nDashboard:  uv run {KIT.relative_to(REPO)}/dashboard.py\nRun your agent with:  OBSERVATORY_RUNS={RUNS} <your agent command>"
    )
    raise SystemExit((serve(args.port) if args.serve else 0) or tests)
