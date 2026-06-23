"""Parse mini-swe-agent trajectory files into structured steps."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

PYTEST_SESSION = re.compile(r"={5,} test session starts ={5,}.*?(?=\n={5,}|\Z)", re.DOTALL)
FILE_PATH = re.compile(r"(?:^|[\s'\"`])([\w./-]+\.(?:py|yaml|yml|md|toml|json|sh))(?:[\s'\"`]|$)")


@dataclass
class ToolCallRecord:
    command: str
    returncode: int | None = None
    output: str = ""
    output_preview: str = ""

    def __post_init__(self) -> None:
        if not self.output_preview and self.output:
            self.output_preview = _preview(self.output)


@dataclass
class TraceStep:
    step: int
    reasoning: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    step_cost: float | None = None


@dataclass
class PytestRun:
    phase: str
    returncode: int | None
    log: str
    summary: str = ""

    def __post_init__(self) -> None:
        if not self.summary:
            self.summary = _pytest_summary(self.log)


def load_trajectory(path: Path) -> dict:
    return json.loads(path.read_text())


def _preview(text: str, limit: int = 500) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… [{len(text) - limit} chars elided]"


def _pytest_summary(log: str) -> str:
    for line in reversed(log.splitlines()):
        stripped = line.strip()
        if stripped.startswith("=") and ("passed" in stripped or "failed" in stripped):
            return stripped.strip("= ")
    return ""


def extract_pytest_sections(text: str) -> list[str]:
    return [m.group(0).rstrip() for m in PYTEST_SESSION.finditer(text)]


def iter_tool_rows(messages: list[dict]) -> list[tuple[str, str, int | None]]:
    rows: list[tuple[str, str, int | None]] = []
    pending: list[str] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            pending = [a["command"] for a in msg.get("extra", {}).get("actions", [])]
        elif msg.get("role") == "tool":
            extra = msg.get("extra", {})
            cmd = pending.pop(0) if pending else ""
            output = extra.get("raw_output") or extra.get("output") or msg.get("content", "")
            rows.append((cmd, output, extra.get("returncode")))
    return rows


def iter_trace_steps(messages: list[dict]) -> list[TraceStep]:
    steps: list[TraceStep] = []
    pending_calls: list[str] = []
    step_num = 0

    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            pending_calls = [a["command"] for a in msg.get("extra", {}).get("actions", [])]
            if not pending_calls:
                continue
            step_num += 1
            steps.append(
                TraceStep(
                    step=step_num,
                    reasoning=(msg.get("content") or "").strip(),
                    step_cost=msg.get("extra", {}).get("cost"),
                )
            )
        elif role == "tool" and steps:
            extra = msg.get("extra", {})
            cmd = pending_calls.pop(0) if pending_calls else ""
            output = extra.get("raw_output") or extra.get("output") or msg.get("content", "")
            steps[-1].tool_calls.append(
                ToolCallRecord(
                    command=cmd,
                    returncode=extra.get("returncode"),
                    output=output if isinstance(output, str) else str(output),
                )
            )
    return steps


def extract_pytest_runs(rows: list[tuple[str, str, int | None]]) -> list[PytestRun]:
    runs: list[PytestRun] = []
    for cmd, output, rc in rows:
        if "pytest" not in cmd.lower():
            continue
        sections = extract_pytest_sections(output)
        log = sections[-1] if sections else output
        runs.append(PytestRun(phase="", returncode=rc, log=log))
    for i, run in enumerate(runs):
        if i == 0:
            run.phase = "pre_fix"
        elif i == len(runs) - 1:
            run.phase = "post_fix"
        else:
            run.phase = f"intermediate_{i}"
    return runs


def extract_viewed_files(rows: list[tuple[str, str, int | None]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for cmd, output, _rc in rows:
        for text in (cmd, output):
            for match in FILE_PATH.finditer(text):
                path = match.group(1)
                if path.startswith((".", "upstream/", "src/", "tests/")) and path not in seen:
                    seen.add(path)
                    ordered.append(path)
    return ordered


def extract_patch_diff(rows: list[tuple[str, str, int | None]]) -> tuple[str, str]:
    """Return (patch_text, source_description)."""
    for cmd, output, _rc in rows:
        if "git diff" in cmd and output.strip().startswith("diff --git"):
            return output.strip(), "git diff in trajectory"

    for cmd, _output, _rc in rows:
        if "path.write_text" in cmd or "sed -i" in cmd:
            patch = _patch_from_edit_command(cmd)
            if patch:
                return patch, "reconstructed from edit command"

    return "", "not found"


def _patch_from_edit_command(cmd: str) -> str:
    old_match = re.search(r"old\s*=\s*['\"](.+?)['\"]", cmd, re.DOTALL)
    new_match = re.search(r"new\s*=\s*['\"](.+?)['\"]", cmd, re.DOTALL)
    path_match = re.search(r"Path\(['\"]([^'\"]+)['\"]\)", cmd)
    if not (old_match and new_match and path_match):
        return ""

    rel_path = path_match.group(1)
    old_line = old_match.group(1)
    new_line = new_match.group(1)
    if old_line == new_line:
        return ""

    return f"""diff --git a/{rel_path} b/{rel_path}
--- a/{rel_path}
+++ b/{rel_path}
@@ -1,1 +1,1 @@
-{old_line.rstrip()}
+{new_line.rstrip()}
"""
