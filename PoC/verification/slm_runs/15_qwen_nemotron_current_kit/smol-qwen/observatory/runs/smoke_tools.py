"""Throwaway smoke test for the tool-execution logging in agents.py (safe to delete)."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = HERE / "smoke_runs"
os.environ["OBSERVATORY_RUNS"] = str(RUNS)
if RUNS.exists():
    for f in RUNS.iterdir():
        f.unlink()
RUNS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))

from smolagents import Tool  # noqa: E402
from smolagents.agents import AgentExecutionError, AgentToolExecutionError, CodeAgent, ToolCallingAgent  # noqa: E402


class AddTool(Tool):
    name = "add"
    description = "add two numbers"
    inputs = {"a": {"type": "number"}, "b": {"type": "number"}}
    output_type = "number"

    def forward(self, a, b):
        return a + b


class BoomTool(Tool):
    name = "boom"
    description = "always fails"
    inputs = {}
    output_type = "string"

    def forward(self):
        raise ValueError("kaboom")


def make_chat_message(tool_name, arguments):
    return SimpleNamespace(
        content="",
        raw=None,
        token_usage=None,
        tool_calls=[
            SimpleNamespace(function=SimpleNamespace(name=tool_name, arguments=arguments), id=f"call_{tool_name}")
        ],
    )


failures = []


def check(label, cond):
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        failures.append(label)


# ---------- ToolCallingAgent: success ----------
agent = ToolCallingAgent(tools=[AddTool()], model=object())
step = SimpleNamespace(step_number=1)
outs = list(agent.process_tool_calls(make_chat_message("add", {"a": 1, "b": 2}), step))
check("tca: ToolOutput produced", outs[-1].output == 3)

# ---------- ToolCallingAgent: error ----------
agent2 = ToolCallingAgent(tools=[BoomTool()], model=object())
step2 = SimpleNamespace(step_number=1)
try:
    list(agent2.process_tool_calls(make_chat_message("boom", {}), step2))
    check("tca: error propagated", False)
except AgentToolExecutionError:
    check("tca: error propagated", True)

# ---------- CodeAgent steps driven directly ----------
code_model_calls = {"n": 0}


class FakeCodeModel:
    def __init__(self, replies):
        self.replies = replies

    def generate(self, messages, **kwargs):
        reply = self.replies[code_model_calls["n"]]
        code_model_calls["n"] += 1
        return SimpleNamespace(content=reply, token_usage=None)


agent3 = CodeAgent(tools=[], model=object())
agent3.write_memory_to_messages = lambda *a, **k: []

# normal execution
agent3.model = FakeCodeModel(["do it:\n<code>\nprint(1 + 2)\n</code>"])
step3 = SimpleNamespace(step_number=1)
list(agent3._step_stream(step3))

# final answer (submit)
agent3.model = FakeCodeModel(["answer:\n<code>\nfinal_answer(3)\n</code>"])
agent3.python_executor.send_tools(agent3.tools)
step4 = SimpleNamespace(step_number=2)
out = list(agent3._step_stream(step4))
check("ca: final answer flagged", out[-1].is_final_answer is True)

# execution error
agent3.model = FakeCodeModel(["oops:\n<code>\n1 / 0\n</code>"])
step5 = SimpleNamespace(step_number=3)
try:
    list(agent3._step_stream(step5))
    check("ca: error propagated", False)
except AgentExecutionError:
    check("ca: error propagated", True)

# ---------- Inspect the log ----------
lines = [
    json.loads(line)
    for line in (sorted(RUNS.glob("*.jsonl"))[0]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
tool_starts = [l for l in lines if l["kind"] == "tool_start"]
tool_ends = [l for l in lines if l["kind"] == "tool_end"]
tool_errors = [l for l in lines if l["kind"] == "tool_error"]

check("log: 5 tool_start (add, boom, 3x python_interpreter)", len(tool_starts) == 5)
check("log: 3 tool_end (add, pi print, pi submit)", len(tool_ends) == 3)
check("log: 2 tool_error (boom, pi division)", len(tool_errors) == 2)

add_start = [t for t in tool_starts if t["data"]["action"] == "add"]
check(
    "log: add start name/params/step",
    len(add_start) == 1 and add_start[0]["data"]["params"] == {"a": 1, "b": 2} and add_start[0]["step"] == 1,
)
check("log: add start at", add_start[0]["at"].endswith("agents.py:1433"))
add_end = [t for t in tool_ends if t["data"]["action"] == "add"]
check(
    "log: add end result=3 + duration",
    len(add_end) == 1 and add_end[0]["data"]["result"] == 3 and "duration_seconds" in add_end[0]["data"],
)
check("log: add end at", add_end[0]["at"].endswith("agents.py:1439"))

boom_error = [t for t in tool_errors if t["data"]["action"] == "boom"]
check(
    "log: boom error logged", len(boom_error) == 1 and boom_error[0]["data"]["error_type"] == "AgentToolExecutionError"
)
check("log: boom error at", boom_error[0]["at"].endswith("agents.py:1437"))

pi_starts = sorted([t for t in tool_starts if t["data"]["action"] == "python_interpreter"], key=lambda t: t["step"])
check("log: 3 python_interpreter starts", len(pi_starts) == 3)
check(
    "log: pi starts params are the code, steps 1-3",
    pi_starts[0]["data"]["params"] == "print(1 + 2)"
    and pi_starts[1]["data"]["params"] == "final_answer(3)"
    and pi_starts[2]["data"]["params"] == "1 / 0"
    and [t["step"] for t in pi_starts] == [1, 2, 3],
)
check("log: pi start at", all(t["at"].endswith("agents.py:1789") for t in pi_starts))

pi_ends = sorted([t for t in tool_ends if t["data"]["action"] == "python_interpreter"], key=lambda t: t["step"])
check("log: 2 python_interpreter ends", len(pi_ends) == 2)
check(
    "log: pi end step1 whole CodeOutput (not final)",
    pi_ends[0]["step"] == 1 and "is_final_answer=False" in str(pi_ends[0]["data"]["result"]),
)
check(
    "log: pi end step2 submit reported as tool_end with whole CodeOutput",
    pi_ends[1]["step"] == 2 and "is_final_answer=True" in str(pi_ends[1]["data"]["result"]),
)
check("log: pi end at", all(t["at"].endswith("agents.py:1796") for t in pi_ends))

pi_error = [t for t in tool_errors if t["data"]["action"] == "python_interpreter"]
check(
    "log: division error logged",
    len(pi_error) == 1 and pi_error[0]["data"]["error_type"] == "InterpreterError" and pi_error[0]["step"] == 3,
)
check("log: pi error at", pi_error[0]["at"].endswith("agents.py:1794"))

# ordering: each tool_start immediately followed by its matching end/error
ok_order = True
for i, l in enumerate(lines):
    if l["kind"] == "tool_start":
        nxt = lines[i + 1]
        ok_order = (
            ok_order and nxt["kind"] in ("tool_end", "tool_error") and nxt["data"]["action"] == l["data"]["action"]
        )
check("log: start immediately followed by matching end/error", ok_order)

llm_resp_1 = [
    l
    for l in lines
    if l["kind"] == "llm_response" and l["step"] == 1 and l["data"]["actions"] == ["python_interpreter"]
]
check("log: ca llm_response links python_interpreter", len(llm_resp_1) == 1)

print()
print("KINDS:", [l["kind"] for l in lines])
if failures:
    print("FAILURES:", failures)
    sys.exit(1)
print("ALL OK")
