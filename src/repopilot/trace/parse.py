"""Parse mini-swe-agent trajectory files into structured steps."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

PYTEST_SESSION_START = re.compile(r"={5,}\s*test session starts\s*={5,}", re.IGNORECASE)
PYTEST_SESSION_END = re.compile(
    r"={5,}\s*\d+[^\n]*(?:passed|failed|error|skipped|deselected|selected)[^\n]*={5,}\s*$",
    re.MULTILINE,
)
FILE_PATH = re.compile(r"(?:^|[\s'\"`])([\w./-]+\.(?:py|yaml|yml|md|toml|json|sh))(?:[\s'\"`]|$)")
PYTEST_FAIL_HEADER = re.compile(r"^_{5,}\s+(\S+)\s+_{5,}\s*$")
PYTEST_ASSERT_LINE = re.compile(r"^>\s+(.*)$")
PYTEST_ERROR_LINE = re.compile(r"^E\s+(.*)$")
PYTEST_FILE_LINE = re.compile(r"^(\S+\.py:\d+):\s")
PYTEST_INLINE_FAILED = re.compile(r"^(\S+::(\w+))\s+FAILED(?:\s|\[|$)")
PYTEST_SUMMARY_FAILED = re.compile(r"^FAILED\s+(\S+)")


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
class PytestFailure:
    test: str
    assertion: str = ""
    file_line: str = ""


@dataclass
class PytestRun:
    phase: str
    returncode: int | None
    log: str
    summary: str = ""
    failed_tests: list[PytestFailure] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.summary:
            self.summary = _pytest_summary(self.log)
        if not self.failed_tests:
            self.failed_tests = parse_pytest_failures(self.log)


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


def parse_pytest_failures(log: str) -> list[PytestFailure]:
    """Extract failing test names and assertion lines from pytest output."""
    failures: list[PytestFailure] = []
    seen: set[str] = set()
    lines = log.splitlines()

    i = 0
    while i < len(lines):
        header = PYTEST_FAIL_HEADER.match(lines[i])
        if header:
            test_name = header.group(1)
            assertion = ""
            file_line = ""
            error_line = ""
            for j in range(i + 1, min(i + 40, len(lines))):
                if PYTEST_FAIL_HEADER.match(lines[j]):
                    break
                err = PYTEST_ERROR_LINE.match(lines[j])
                if err and not error_line:
                    error_line = err.group(1).strip()
                ast_line = PYTEST_ASSERT_LINE.match(lines[j])
                if ast_line and not assertion:
                    assertion = ast_line.group(1).strip()
                loc = PYTEST_FILE_LINE.match(lines[j])
                if loc and not file_line:
                    file_line = loc.group(1)
            if error_line:
                assertion = error_line
            if test_name not in seen:
                seen.add(test_name)
                failures.append(PytestFailure(test=test_name, assertion=assertion, file_line=file_line))
        i += 1

    for line in lines:
        stripped = line.strip()
        summary = PYTEST_SUMMARY_FAILED.match(stripped)
        if summary:
            test_id = summary.group(1)
            short_name = test_id.rsplit("::", 1)[-1] if "::" in test_id else test_id
            if short_name and short_name not in seen:
                seen.add(short_name)
                failures.append(PytestFailure(test=short_name))
            continue
        inline = PYTEST_INLINE_FAILED.match(stripped)
        if inline:
            short_name = inline.group(2)
            if short_name not in seen:
                seen.add(short_name)
                failures.append(PytestFailure(test=short_name))
    return failures


def extract_pytest_log(output: str) -> str:
    """Extract a complete pytest run from mixed command output (includes FAILURES)."""
    match = PYTEST_SESSION_START.search(output)
    if not match:
        return output
    tail = output[match.start() :]
    end_match = PYTEST_SESSION_END.search(tail)
    if end_match:
        return tail[: end_match.end()].rstrip()
    return tail.rstrip()


def extract_pytest_sections(text: str) -> list[str]:
    """Legacy helper; prefer extract_pytest_log for full session capture."""
    log = extract_pytest_log(text)
    if PYTEST_SESSION_START.search(log):
        return [log]
    return []


def is_edit_command(command: str) -> bool:
    """Return True if a shell command likely modifies source files."""
    if not command:
        return False
    return any(
        token in command
        for token in (
            "path.write_text",
            "p.write_text",
            ".write_text(",
            ".replace(old",
            "replace(old",
            "sed -i",
            "git apply",
            "patch -p",
        )
    )


def classify_command_stage(command: str) -> str:
    cmd = command.lower()
    if "pytest" in cmd:
        return "test"
    if is_edit_command(command):
        return "edit"
    if "git diff" in cmd or "submit" in cmd:
        return "submit"
    if any(
        token in cmd
        for token in (" cat ", " head ", " tail ", " nl ", "sed -n", " grep ", "printf ", " read ", " find ", "ls ")
    ) or cmd.strip().startswith(("cat ", "head ", "tail ", "nl ", "grep ", "printf ")):
        return "read"
    return "other"


def infer_step_stage(tool_calls: list[ToolCallRecord]) -> str:
    stages = [classify_command_stage(tc.command) for tc in tool_calls]
    for preferred in ("test", "edit", "read", "submit"):
        if preferred in stages:
            return preferred
    return "other"


def extract_files_touched(command: str, output: str = "") -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for text in (command, output):
        for match in FILE_PATH.finditer(text):
            path = match.group(1)
            if path.startswith((".", "upstream/", "src/", "tests/", "benchmarks/")) and path not in seen:
                seen.add(path)
                ordered.append(path)
    path_in_write = re.search(r"Path\(['\"]([^'\"]+)['\"]\)", command)
    if path_in_write:
        path = path_in_write.group(1)
        if path not in seen:
            ordered.insert(0, path)
    return ordered


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
        log = extract_pytest_log(output)
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


def capture_workspace_diff(workspace: Path) -> str:
    """Return unstaged git diff from a task worktree after the agent run."""
    workspace = workspace.resolve()
    if not workspace.is_dir():
        return ""
    if not (workspace / ".git").exists() and not _is_git_worktree(workspace):
        return ""
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _is_git_worktree(path: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=path,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _split_unified_diff_blocks(text: str) -> list[str]:
    if not text.strip():
        return []
    return [
        block.rstrip() + "\n"
        for block in re.split(r"(?=^diff --git )", text.strip(), flags=re.MULTILINE)
        if block.strip()
    ]


def _diff_block_path(block: str) -> str:
    match = re.search(r"^diff --git a/(.+?) b/", block, flags=re.MULTILINE)
    return match.group(1) if match else ""


def merge_unified_diffs(*texts: str) -> str:
    """Merge diff outputs; later blocks win for the same file path."""
    blocks_by_path: dict[str, str] = {}
    order: list[str] = []
    for text in texts:
        for block in _split_unified_diff_blocks(text):
            path = _diff_block_path(block)
            if not path:
                continue
            if path not in blocks_by_path:
                order.append(path)
            blocks_by_path[path] = block
    if not blocks_by_path:
        return ""
    return "".join(blocks_by_path[path] for path in order)


def extract_patch_diff(rows: list[tuple[str, str, int | None]]) -> tuple[str, str]:
    """Return (patch_text, source_description)."""
    git_diffs: list[str] = []
    for cmd, output, _rc in rows:
        if "git diff" in cmd and output.strip().startswith("diff --git"):
            git_diffs.append(output.strip())

    if git_diffs:
        merged = merge_unified_diffs(*git_diffs)
        if merged:
            source = "git diff in trajectory"
            if len(git_diffs) > 1:
                source = "git diff in trajectory (merged)"
            return merged, source

    edit_patches: list[str] = []
    for cmd, _output, _rc in rows:
        if is_edit_command(cmd):
            patch = _patch_from_edit_command(cmd)
            if patch:
                edit_patches.append(patch)
    if edit_patches:
        merged = merge_unified_diffs(*edit_patches)
        if merged:
            source = "reconstructed from edit command"
            if len(edit_patches) > 1:
                source = "reconstructed from edit command(s)"
            return merged, source

    return "", "not found"


def _extract_assign_string(cmd: str, name: str) -> str | None:
    match = re.search(
        rf"\b{name}\s*=\s*('(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")",
        cmd,
        re.DOTALL,
    )
    if not match:
        return None
    try:
        value = ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, str) else None


def _unified_diff_block(rel_path: str, old: str, new: str) -> str:
    old_lines = old.rstrip("\n").splitlines()
    new_lines = new.rstrip("\n").splitlines()
    if old_lines == new_lines:
        return ""
    body = "\n".join(f"-{line}" for line in old_lines) + "\n" + "\n".join(f"+{line}" for line in new_lines)
    return (
        f"diff --git a/{rel_path} b/{rel_path}\n"
        f"--- a/{rel_path}\n"
        f"+++ b/{rel_path}\n"
        f"@@ -1,{len(old_lines)} +1,{len(new_lines)} @@\n"
        f"{body}\n"
    )


def _patch_from_edit_command(cmd: str) -> str:
    path_match = re.search(r"Path\(['\"]([^'\"]+)['\"]\)", cmd)
    if not path_match:
        return ""

    old_text = _extract_assign_string(cmd, "old")
    new_text = _extract_assign_string(cmd, "new")
    if old_text is None or new_text is None or old_text == new_text:
        return ""

    return _unified_diff_block(path_match.group(1), old_text, new_text)
