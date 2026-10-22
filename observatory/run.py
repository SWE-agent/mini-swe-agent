# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Connect any agent repository to the Agent Observatory in small steps, one Claude Code session each (prompt/*.md):
the anchors, the LLM logs, the tool logs and the user input logs. The static checker (flow_verify.py)
gives each step the places to log, and at the end judges the result on its own: the sessions never see its verdict. It
also writes the agent loop the dashboard draws (verify/loop.json).

Put this `observatory/` folder in the root of the agent repository, then:

    uv run observatory/run.py                  # wire the agent to the logger and check it
    uv run observatory/run.py --serve          # ... and start the dashboard afterwards
    uv run observatory/run.py --serve-only     # later: just start the dashboard
    uv run observatory/run.py --no-claude      # only check the wiring again and redraw the loop

The dashboard (dashboard.py), the logger (observatory_log.py) and the checker are the same for every agent. What
differs per agent is done by Claude Code: where the logger calls go.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

KIT = Path(__file__).resolve().parent
REPO = KIT.parent
LOGGER = KIT / "observatory_log.py"
ANCHORS = KIT / "diagram/verify-anchors.json"
VERIFY = KIT / "verify"
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
STEPS = [
    ("llm", ["llm_response"], "the LLM calls"),
    ("tools", ["tool_response"], "the tool executions"),
    ("user_input", ["user_input"], "the user input"),
]


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


def claude(prompt: str, model: str, budget: float) -> None:
    result = subprocess.run(
        [
            find_claude(),
            *CLAUDE_ARGS,
            "--allowedTools",
            *CLAUDE_TOOLS,
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
        **fields,
    }.items():
        text = text.replace(f"{{{key}}}", value)
    return text


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def agent_files() -> list[str]:
    return git("ls-files", "-co", "--exclude-standard", "--", "*.py", f":!{KIT.relative_to(REPO)}").splitlines()


def verify(quiet: bool = False) -> list[dict] | None:
    """Run the static checker: its verdict per value flow, or None when it cannot follow the anchors."""
    if not ANCHORS.exists():
        print(f"no anchors: {ANCHORS.relative_to(REPO).as_posix()} was not written")
        return None
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
        return None
    if not quiet:
        print(output)
    return json.loads((VERIFY / "report.json").read_text(encoding="utf-8"))


def flows(rows: list[dict] | None) -> set[tuple]:
    """Which value reaches which use, without line numbers (they move when the logging is added)."""
    return {(r["value"], r["source"].split(" @ ")[0], r["use"], r["sink"].split(" @ ")[0]) for r in rows or []}


def removed_lines(before: dict[str, str]) -> list[str]:
    """Lines of the agent's code that are gone. Indentation is ignored (wrapping a call in try/except), and so are the
    two rewrites logging needs: names added to a `from ... import` line, and `return <expr>` turned into
    `<name> = <expr>` so that the value can be logged before it is returned."""
    gone = []
    for file, text in before.items():
        now = [
            s.strip()
            for s in ((REPO / file).read_text(encoding="utf-8") if (REPO / file).exists() else "").splitlines()
        ]
        lost = Counter(s for s in map(str.strip, text.splitlines()) if s) - Counter(now)
        gone += [f"{file}: {line}" for line in lost.elements() if not rewritten(line, now)]
    return gone


def rewritten(line: str, now: list[str]) -> bool:
    """Whether a removed line is still there in one of the two forms logging needs."""
    if line.startswith("from ") and " import " in line:
        module, names = line.split(" import ", 1)
        old = {n.strip() for n in names.strip("()").split(",") if n.strip()}
        return any(
            n.startswith(f"{module} import ")
            and old <= {x.strip() for x in n.split(" import ", 1)[1].strip("()").split(",")}
            for n in now
        )
    if line.startswith("return "):
        return any(re.fullmatch(rf"\w+ = {re.escape(line[7:])}", n) for n in now)
    return False


def code_errors(files: list[str]) -> list[str]:
    """Code that would fail when it runs: a syntax error, or a name that is not defined (ruff F821)."""
    errors = []
    for file in files:
        try:
            compile((REPO / file).read_text(encoding="utf-8"), file, "exec")
        except SyntaxError as error:
            errors.append(f"{file}:{error.lineno}: {error.msg}")
    if shutil.which("uvx") and files:
        check = subprocess.run(
            ["uvx", "ruff", "check", "--select", "F821", "--output-format", "concise", "--no-cache", *files],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        errors += [line for line in check.stdout.splitlines() if "F821" in line]
    return errors


def sites(*roles: str) -> str:
    """The graph's calls of these roles, one line each, for a step's prompt, and the anchors of these kinds the graph
    does not show (a value it cannot follow, such as user text carried by an exception)."""
    graph = VERIFY / "graph.json"
    nodes = json.loads(graph.read_text(encoding="utf-8"))["nodes"] if graph.exists() else []
    lines = [
        f"- `{n['label']}` at {n['path']}:{n['line']}" + (f" (in `{n['in']}`)" if n.get("in") else "")
        for n in nodes
        if n["role"] in roles
    ]
    shown = {(n["path"], n["line"]) for n in nodes if n["role"] in roles}
    anchors = json.loads(ANCHORS.read_text()).get("sources", []) if ANCHORS.exists() else []
    lines += [
        f"- `{a['call']}` at {a['path']}:{a['line']}"
        for a in anchors
        if a["kind"] in roles and (a["path"], a["line"]) not in shown
    ]
    return "\n".join(lines) or "- (the checker found none: read the code)"


def copy_logger() -> str:
    """Copy the logger next to the entry function's file, the agent's own code, and return where it is. The anchor's
    path may be off by a leading directory, as the checker allows: take the agent's file it ends with, else the root."""
    path = json.loads(ANCHORS.read_text())["entry"]["path"]
    entry = next((f for f in agent_files() if path == f or path.endswith(f"/{f}")), ".")
    target = REPO / Path(entry).parent / LOGGER.name
    if not target.exists():
        shutil.copy(LOGGER, target)
    return target.relative_to(REPO).as_posix()


def judge(rows: list[dict] | None, baseline: set[tuple], before: dict[str, str], checker: bytes) -> bool:
    """The checker's verdict, independent of the sessions: the code compiles and defines every name it uses, every flow
    is logged whole, none of the flows found before the logging was added has disappeared, no line of the agent's code
    was removed, and the checker itself is unchanged."""
    problems = []
    if errors := code_errors([f for f in agent_files() if (REPO / f).exists()]):
        problems += ["the agent's code would fail when it runs:"] + [f"  {e}" for e in errors]
    if rows is None:
        problems.append("the checker cannot follow the anchors (see above)")
    elif missing := json.loads((VERIFY / "graph.json").read_text(encoding="utf-8")).get("missing_anchors"):
        problems += ["anchors that do not match the code (left out of the check):"] + [f"  {a}" for a in missing]
    elif gaps := [r for r in rows if r["status"] != "covered"]:
        problems.append(f"{len(gaps)} of {len(rows)} value flows are not logged whole (see the lines above)")
    if lost := baseline - flows(rows):
        problems += ["a value flow disappeared, so the agent's behavior changed:"] + [
            f"  {v}: {s} -> {u}: {k}" for v, s, u, k in sorted(lost)
        ]
    if gone := removed_lines(before):
        problems += ["lines of the agent's code were removed:"] + [f"  {line}" for line in gone]
    if (KIT / "flow_verify.py").read_bytes() != checker:
        problems.append("the checker (flow_verify.py) was modified")
    print("\n".join(problems) or "PASS: every value flow is logged whole and the agent's code is unchanged otherwise")
    return not problems


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
        for f in git("grep", "--untracked", "-I", "-l", "observatory_log", "--", ".", exclude).splitlines()
        if REPO / f not in copies
    ]
    if copies and users:
        print(f"logger: {', '.join(str(c.relative_to(REPO)) for c in copies)}\ncalled from: {', '.join(users)}")
    else:
        print("warning: nothing calls the logger yet" if copies else "warning: the logger is not in the agent's code")


def check_trace() -> None:
    kinds = Counter(json.loads(line)["kind"] for run in RUNS.glob("*.jsonl") for line in run.read_text().splitlines())
    print(
        f"events in {RUNS.relative_to(REPO)}: {dict(kinds)}"
        if kinds
        else f"no run yet: run your agent with OBSERVATORY_RUNS={RUNS} to see its history on the dashboard"
    )


def serve(port: int) -> int:
    return subprocess.run([sys.executable, KIT / "dashboard.py", "--runs", RUNS, "--port", str(port)]).returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="sonnet", help="Claude Code model")
    parser.add_argument("--budget", type=float, default=3.0, help="max USD per Claude Code session (0: no limit)")
    parser.add_argument("--serve", action="store_true", help="start the dashboard when done")
    parser.add_argument("--serve-only", action="store_true", help="only start the dashboard")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--no-claude", action="store_true", help="skip the Claude Code sessions (the agent is already wired)"
    )
    parser.add_argument("--test", help="a command to run in the repository at the end, for example 'uv run pytest -q'")
    args = parser.parse_args()
    print(f"Repository: {REPO}")
    if args.serve_only:
        raise SystemExit(serve(args.port))

    step("Checking the kit")
    subprocess.run([sys.executable, KIT / "selftest.py"], check=True, capture_output=True)
    print("logger and dashboard self-test passed")

    def session(name: str, **fields: str) -> None:
        claude(prompt(name, **fields), args.model, args.budget)

    # The sessions never see the checker's verdict: it judges the result on its own at the end.
    before = {f: (REPO / f).read_text(encoding="utf-8") for f in agent_files()}
    checker, baseline = (KIT / "flow_verify.py").read_bytes(), set()
    if not args.no_claude:
        step("1/4 Claude Code: finding the anchors (where the task, the LLM replies and the user input come in)")
        session("anchors")
        if ANCHORS.exists():
            logger = copy_logger()
            baseline = flows(verify(quiet=True))  # the value flows of the agent before any logging
            for n, (name, roles, what) in enumerate(STEPS, start=2):
                if n > 2:
                    verify(quiet=True)  # only for the places to log: the graph with the lines as they are now
                step(f"{n}/4 Claude Code: logging {what}")
                session(name, logger=logger, sites=sites(*roles))
    step("Checking the result")
    check_wiring()
    check_trace()
    step("The checker's verdict")
    rows = verify()
    passed = judge(rows, baseline, before, checker)
    if rows is not None:
        print(
            f"report: {(VERIFY / 'report.html').relative_to(REPO)}  agent loop: {(VERIFY / 'loop.mmd').relative_to(REPO)}"
        )
    tests = subprocess.run(shlex.split(args.test), cwd=REPO).returncode if args.test else 0
    print(
        f"\nDashboard:  uv run {KIT.relative_to(REPO)}/dashboard.py\nRun your agent with:  OBSERVATORY_RUNS={RUNS} <your agent command>"
    )
    raise SystemExit((serve(args.port) if args.serve else 0) or tests or (0 if passed else 1))
