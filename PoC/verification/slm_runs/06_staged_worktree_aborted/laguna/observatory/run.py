# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Connect any agent repository to the Agent Observatory in small steps, one Claude Code session each (prompt/*.md):
the anchors, the LLM logs, the tool logs, the user input logs, a scripted run, and the architecture diagram drawn from the
data-flow graph of the static checker (flow_verify.py). The checker runs between the steps: it gives each step the places
to log and sends what it still finds back to a fix session.

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
ANCHORS = KIT / "diagram/verify-anchors.json"
VERIFY = KIT / "verify"
SPEC = KIT / "diagram/agent-architecture.json"
DIAGRAM = KIT / "diagram/agent-architecture.html"
RUNS = KIT / "runs"
# The Claude Code slot: a fresh, memoryless session that may edit files and run uv/git diff. The prompt arrives on stdin.
CLAUDE_ARGS = [
    "-p",
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
    tools = (
        [*CLAUDE_TOOLS, f"Bash(node {archify_dir.as_posix()}/bin/archify.mjs validate:*)"]
        if archify_dir
        else CLAUDE_TOOLS
    )
    allow_archify = ["--add-dir", archify_dir.as_posix()] if archify_dir else []
    result = subprocess.run(
        [
            find_claude(),
            *CLAUDE_ARGS,
            *allow_archify,
            "--allowedTools",
            *tools,
            "--model",
            model,
            *(["--max-budget-usd", str(budget)] if budget else []),
        ],
        input=prompt,
        cwd=REPO,
        text=True,
        capture_output=True,
        # a smaller output budget leaves room for the input in models with a 128k context
        env={"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "16000", **os.environ, "OBSERVATORY_RUNS": str(RUNS)},
    )
    print(result.stdout.strip())
    if result.returncode:
        raise SystemExit(f"Claude Code failed:\n{result.stderr}")


def prompt(name: str, **fields: str) -> str:
    """One step's prompt: the common preamble and prompt/<name>.md, with {kit}, {anchors} and the step's fields filled in."""
    text = (KIT / "prompt/common.md").read_text() + (KIT / f"prompt/{name}.md").read_text()
    for key, value in {
        "kit": KIT.relative_to(REPO).as_posix(),
        "anchors": ANCHORS.relative_to(REPO).as_posix(),
    }.items():
        text = text.replace(f"{{{key}}}", value)
    for key, value in fields.items():
        text = text.replace(f"{{{key}}}", value)
    return text


def agent_files() -> list[str]:
    return git("ls-files", "-co", "--exclude-standard", "--", "*.py", f":!{KIT.relative_to(REPO)}").splitlines()


def verify() -> tuple[bool, str | None]:
    """Run the static checker. (whether it could follow the anchors, None when every value is logged whole else its output)"""
    if not ANCHORS.exists():
        return False, f"`{ANCHORS.relative_to(REPO).as_posix()}` was not written."
    result = subprocess.run(
        [sys.executable, KIT / "flow_verify.py", REPO, ANCHORS, VERIFY, "--only", *agent_files()],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode:
        print(output.splitlines()[-1])
        return False, output
    rows = json.loads((VERIFY / "report.json").read_text(encoding="utf-8"))
    print(", ".join(f"{s} {n}" for s, n in Counter(r["status"] for r in rows).items()) or "no flows found")
    return True, None if rows and all(r["status"] == "covered" for r in rows) else output


def sites(*roles: str) -> str:
    """The graph's calls of these roles, one line each, for a step's prompt."""
    nodes = json.loads((VERIFY / "graph.json").read_text(encoding="utf-8"))["nodes"] if VERIFY.exists() else []
    return (
        "\n".join(
            f"- `{n['label']}` at {n['path']}:{n['line']}" + (f" (in `{n['in']}`)" if n.get("in") else "")
            for n in nodes
            if n["role"] in roles
        )
        or "- (the checker found none: read the code)"
    )


def copy_logger() -> str:
    """Copy the logger next to the entry function's file, the agent's own code, and return where it is."""
    target = REPO / Path(json.loads(ANCHORS.read_text())["entry"]["path"]).parent / LOGGER.name
    if not target.exists():
        shutil.copy(LOGGER, target)
    return target.relative_to(REPO).as_posix()


def fix(session, fixes: int, anchors_only: bool = False) -> None:
    """Check, and let a session fix what the checker finds until it passes or `fixes` sessions ran."""
    for attempt in range(fixes + 1):
        followed, problems = verify()
        if not problems or (anchors_only and followed):
            return
        if attempt == fixes:
            print("warning: the checker still finds the problems above")
            return
        step(f"Claude Code: fixing what the checker found ({attempt + 1}/{fixes})")
        session("fix", problems=problems, files=" ".join(agent_files()))


def git(*args: str, env: dict | None = None) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, env=env).stdout.strip()


def check_wiring() -> None:
    """Restore any edited copy of the logger, then report what is wired."""
    exclude = f":!{KIT.relative_to(REPO)}"
    copies = [
        REPO / p
        for p in git("ls-files", "-co", "--exclude-standard", "--", "*observatory_log.py", exclude).splitlines()
        if (REPO / p).exists()
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


def archify_input() -> tuple[Path, Path, str]:
    """The spec as Archify reads it, the Git top level it checks the sources against, and the agent's path below it.

    Archify resolves source paths from the top level. When the agent lives in a subdirectory of its repository, the spec
    keeps paths relative to the agent and a copy for Archify gets the subdirectory in front of each source path.
    """
    top = Path(git("rev-parse", "--show-toplevel")).resolve()
    if top == REPO:
        return SPEC, top, ""
    prefix = REPO.relative_to(top).as_posix()

    def walk(x):
        if isinstance(x, dict):
            if isinstance(x.get("path"), str) and "line" in x:
                x["path"] = f"{prefix}/{x['path']}"
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    spec = json.loads(SPEC.read_text())
    walk(spec)
    staged = SPEC.with_name("agent-architecture.archify.json")
    staged.write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    return staged, top, prefix


def archify(archify_dir: Path, model: str, budget: float, fixes: int) -> None:
    def node(*args, spec: Path, top: Path):
        return subprocess.run(
            ["node", "bin/archify.mjs", args[0], "architecture", spec, *args[1:], "--quality", "showcase"]
            + ["--repo-root", top],
            cwd=archify_dir,
            text=True,
            capture_output=True,
        )

    for attempt in range(fixes + 1):
        pin_repository()
        spec, top, prefix = archify_input()
        validation = node("validate", spec=spec, top=top)
        if validation.returncode == 0:
            break
        if attempt == fixes:
            raise SystemExit(f"Archify validation failed:\n{validation.stdout}{validation.stderr}")
        step(f"Claude Code: fixing the diagram ({attempt + 1}/{fixes})")
        moved = (
            f" The errors show source paths with `{prefix}/` in front: the kit adds it, keep the paths as they are."
            if prefix
            else ""
        )
        claude(
            prompt("diagram_fix", moved=moved, problems=validation.stdout + validation.stderr),
            model,
            budget,
            archify_dir,
        )
    delivery = node("deliver", DIAGRAM, "--json", spec=spec, top=top)
    if delivery.returncode:
        raise SystemExit(f"Archify failed to generate the diagram:\n{delivery.stdout}{delivery.stderr}")
    print(f"diagram regenerated: {DIAGRAM.relative_to(REPO)}")
    for leftover in {*SPEC.parent.glob("*.delivery.json"), *DIAGRAM.parent.glob("*.delivery.json")}:
        leftover.unlink()
    if spec != SPEC:
        spec.unlink()


def serve(port: int) -> int:
    return subprocess.run([sys.executable, KIT / "dashboard.py", "--runs", RUNS, "--port", str(port)]).returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="sonnet", help="Claude Code model")
    parser.add_argument("--budget", type=float, default=3.0, help="max USD per Claude Code session (0: no limit)")
    parser.add_argument("--fixes", type=int, default=2, help="sessions that fix what the checker finds")
    parser.add_argument(
        "--archify-dir",
        default=os.environ.get("ARCHIFY_DIR"),
        help="Archify's `archify/` directory (default: ../archify/archify)",
    )
    parser.add_argument("--serve", action="store_true", help="start the dashboard when done")
    parser.add_argument("--serve-only", action="store_true", help="only start the dashboard")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--no-claude", action="store_true", help="skip the Claude Code sessions (the agent is already wired)"
    )
    parser.add_argument("--archify-only", action="store_true", help="only re-pin the spec and regenerate the diagram")
    parser.add_argument("--test", help="a command to run in the repository at the end, for example 'uv run pytest -q'")
    args = parser.parse_args()
    archify_dir = Path(args.archify_dir).expanduser().resolve() if args.archify_dir else REPO.parent / "archify/archify"
    archify_dir = archify_dir if (archify_dir / "bin/archify.mjs").exists() else None
    print(f"Repository: {REPO}")

    if args.serve_only:
        raise SystemExit(serve(args.port))
    if not args.archify_only:
        step("Checking the kit")
        subprocess.run([sys.executable, KIT / "selftest.py"], check=True, capture_output=True)
        print("logger and dashboard self-test passed")
        fixes = 0 if args.no_claude else args.fixes

        def session(name: str, archify: Path | None = None, **fields: str) -> None:
            claude(prompt(name, **fields), args.model, args.budget, archify)

        if not args.no_claude:
            step("1/6 Claude Code: finding the anchors (where the task, the LLM replies and the user input come in)")
            session("anchors")
        fix(session, fixes, anchors_only=True)
        if not args.no_claude and ANCHORS.exists():
            logger = copy_logger()
            for n, (name, roles, what) in enumerate(
                [
                    ("llm", ["llm_response"], "the LLM calls"),
                    ("tools", ["tool_response"], "the tool executions"),
                    ("user_input", ["user_input"], "the user input"),
                ],
                start=2,
            ):
                verify()  # the graph with the lines as they are now
                step(f"{n}/6 Claude Code: logging {what}")
                session(name, logger=logger, sites=sites(*roles))
        step("Checking that the logging records every value whole")
        fix(session, fixes)
        if not args.no_claude and ANCHORS.exists():
            step("5/6 Claude Code: running the agent once with a scripted model and reading its log")
            session("run_check", entry=json.loads(ANCHORS.read_text())["entry"]["function"])
        step("Checking the result")
        check_wiring()
        check_trace()
        if verify()[0] and archify_dir and not args.no_claude:
            step("6/6 Claude Code: drawing the diagram from the checked graph")
            session("diagram", archify_dir)
        if archify_dir and SPEC.exists() and (problems := check_spec_sources()):
            print("spec sources that do not match the code:\n  " + "\n  ".join(problems))
    if archify_dir and SPEC.exists():
        step("Archify: validating the spec and generating the diagram")
        archify(archify_dir, args.model, args.budget, args.fixes)
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
