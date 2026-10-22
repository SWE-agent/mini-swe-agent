"""Check the Observatory wiring with a scripted model (no API key, no network): uv run python check_observatory.py"""

import io
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.modules["anthropic"] = types.SimpleNamespace(Anthropic=lambda **kwargs: None)
# The real SDK has module file names that cross the Windows 260-char path limit inside this deep checkout, so it cannot
# be imported from here at all; the scripted client below replaces it before any call is made.

from agent import ToyAgent  # noqa: E402


class ScriptedClient:
    """anthropic.Anthropic stand-in: hands out canned replies, one of them an exception, instead of calling the API."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = self

    def create(self, **kwargs):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def reply(text="", calls=()):
    content = [
        SimpleNamespace(type="tool_use", id=f"call_{i}", name=name, input=args) for i, (name, args) in enumerate(calls)
    ]
    if text:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(content=content, usage=SimpleNamespace(input_tokens=11, output_tokens=7))


TASK = "Use the bash and python tools to work out the answer, then tell me."
agent = ToyAgent()
agent.llm.client = ScriptedClient(
    [
        reply(calls=[("bash", {"command": "echo observatory-wiring-check"})]),
        TimeoutError("scripted timeout"),
        reply(calls=[("python", {"code": "result = 6 * 7"})]),
        reply(text="42"),
    ]
)
sys.stdin = io.StringIO("y\n")  # answer the confirmation prompt without a terminal
print(agent.run(TASK))

runs = os.environ.get("OBSERVATORY_RUNS")
if not runs:
    sys.exit("OBSERVATORY_RUNS is not set: the logger stayed silent, nothing to check")
file = max(Path(runs).glob("*.jsonl"), key=lambda path: path.stat().st_mtime)
events = [json.loads(line) for line in file.read_text().splitlines()]

step, groups = None, {}
for event in events:  # same rule as the dashboard: a tool event without step belongs to the latest llm_query
    step = event["step"] if event["step"] is not None else step
    groups.setdefault(step, []).append((event["kind"], event["data"].get("action")))
for step_no, group in groups.items():
    print(f"step {step_no}: " + ", ".join(f"{kind}({action})" if action else kind for kind, action in group))
    kinds = [kind for kind, _ in group]
    if step_no is not None:
        assert kinds[0] == "llm_query" and "llm_response" in kinds, group

data = lambda kind, action=None: [e["data"] for e in events if e["kind"] == kind and e["data"].get("action") == action]
ends = data("tool_end", "user_input")
assert data("tool_start", "user_input")[0]["params"] == TASK and ends[0]["result"] == TASK  # the task, whole
query = next(e for e in data("llm_query") if e["messages"] == [{"role": "user", "content": TASK}])
assert query["model"] == "claude-haiku-5-5"  # the message history, whole
assert "observatory-wiring-check" in data("tool_end", "bash")[0]["result"]
assert data("tool_end", "python")[0]["result"] == "42"
assert next(e for e in data("llm_error") if e["error_type"] == "TimeoutError")
first = data("llm_response")[0]
assert first["actions"] == ["bash"] and first["reply"]["calls"][0]["name"] == "bash" and "response" in first
assert ends[-1]["result"] == "y"  # the confirmation answer
print(f"observatory wiring check passed: {file.name} shows every step's llm_query, llm_response and tool events in order")
