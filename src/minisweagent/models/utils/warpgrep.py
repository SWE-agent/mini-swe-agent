import json
import logging
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

logger = logging.getLogger("warpgrep")

WARPGREP_TOOL = {
    "type": "function",
    "function": {
        "name": "warpgrep",
        "description": (
            "Search a codebase using an AI-powered code search agent. Returns relevant code spans matching the query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of the code to search for",
                }
            },
            "required": ["query"],
        },
    },
}

_EXCLUDED_DIRS = {".git", "__pycache__", "node_modules", ".venv", "env"}


def generate_repo_tree(root: Path, max_depth: int = 4) -> str:
    lines: list[str] = []

    def _walk(directory: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        entries = sorted(directory.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        dirs = [e for e in entries if e.is_dir() and e.name not in _EXCLUDED_DIRS]
        files = [e for e in entries if e.is_file()]
        items = dirs + files
        for i, entry in enumerate(items):
            connector = "--- " if i == len(items) - 1 else "|-- "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                extension = "    " if i == len(items) - 1 else "|   "
                _walk(entry, prefix + extension, depth + 1)

    lines.append(root.name)
    _walk(root, "", 1)
    return "\n".join(lines)


class WarpGrepClient:
    def __init__(
        self,
        api_key: str,
        endpoint: str = "https://api.morphllm.com/v1/chat/completions",
        model: str = "morph-warp-grep-v2",
        max_turns: int = 4,
        timeout: int = 60,
    ):
        self._api_key = api_key
        self._endpoint = endpoint
        self._model = model
        self._max_turns = max_turns
        self._timeout = timeout

    def search(self, repo_root: str | Path, query: str) -> str:
        repo_root = Path(repo_root).resolve()
        tree = generate_repo_tree(repo_root)
        messages = [
            {
                "role": "user",
                "content": f"<repo_structure>\n{tree}\n</repo_structure>\n\n<search_string>\n{query}\n</search_string>",
            }
        ]
        for turn in range(1, self._max_turns + 1):
            content = self._complete(messages)
            tool_calls = self._parse_tool_calls(content)
            if not tool_calls:
                return content
            results: list[str] = []
            for call in tool_calls:
                if call["name"] == "finish":
                    return self._format_finish(call.get("specs", []), repo_root)
                results.append(self._execute_tool_call(call, repo_root))
            response_text = (
                f"<tool_response>\nTurn {turn}/{self._max_turns}\n" + "\n".join(results) + "\n</tool_response>"
            )
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": response_text})
        return "WarpGrep: max turns reached without finish."

    def _complete(self, messages: list[dict]) -> str:
        payload = json.dumps({"model": self._model, "messages": messages}).encode()
        req = urllib.request.Request(
            self._endpoint,
            data=payload,
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    def _parse_tool_calls(self, content: str) -> list[dict]:
        calls = []
        for match in re.finditer(r"<tool_call><function=(\w+)>(.*?)</function></tool_call>", content, re.DOTALL):
            name, args_str = match.group(1), match.group(2).strip()
            args = json.loads(args_str)
            if name == "finish":
                calls.append({"name": "finish", "specs": args.get("files", [])})
            else:
                calls.append({"name": name, **args})
        return calls

    def _execute_tool_call(self, call: dict, repo_root: Path) -> str:
        name = call["name"]
        if name == "ripgrep":
            if not shutil.which("rg"):
                return "Error: rg (ripgrep) is not installed"
            cmd = ["rg", "--no-heading", "-n"]
            if include := call.get("include"):
                cmd.extend(["--glob", include])
            cmd.append(call.get("pattern", ""))
            cmd.append(str(repo_root))
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.stdout[:20000] or "(no matches)"
        if name == "read":
            path = (repo_root / call.get("path", "")).resolve()
            if not path.is_relative_to(repo_root):
                return "Error: path traversal blocked"
            if not path.is_file():
                return f"Error: {path} not found"
            lines = path.read_text().splitlines()
            start = call.get("start", 1) - 1
            end = call.get("end", len(lines))
            return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines[start:end], start=start))
        if name == "list_directory":
            path = (repo_root / call.get("path", "")).resolve()
            if not path.is_relative_to(repo_root):
                return "Error: path traversal blocked"
            if not path.is_dir():
                return f"Error: {path} is not a directory"
            return "\n".join(sorted(e.name for e in path.iterdir()))
        return f"Unknown tool: {name}"

    def _format_finish(self, specs: list[str], repo_root: Path) -> str:
        if not specs:
            return "WarpGrep: no results."
        parts: list[str] = []
        for spec in specs:
            file_part, *range_parts = spec.split(":")
            path = (repo_root / file_part).resolve()
            if not path.is_relative_to(repo_root) or not path.is_file():
                parts.append(f"# {spec}\n(file not found)")
                continue
            lines = path.read_text().splitlines()
            if not range_parts:
                parts.append(f"# {file_part}\n" + "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines)))
                continue
            for rng in range_parts[0].split(","):
                if "-" in rng:
                    start, end = rng.split("-", 1)
                    s, e = int(start) - 1, int(end)
                else:
                    s, e = int(rng) - 1, int(rng)
                parts.append(
                    f"# {file_part}:{rng}\n"
                    + "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines[s:e], start=s))
                )
        return "\n\n".join(parts)
