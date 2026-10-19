"""Run ToyAgent with a scripted model (no API key, no network) to check the observatory_log events.

usage (from this folder):  OBSERVATORY_RUNS=<dir> uv run --with anthropic python run_scripted.py < answers.txt
Step 1 asks for two tool calls (bash, python), step 2 answers with text; stdin answers the confirm prompt.
"""

import os
from types import SimpleNamespace as NS

os.environ.setdefault("ANTHROPIC_API_KEY", "scripted")

from agent import ToyAgent  # noqa: E402

REPLIES = [
    [("bash", {"command": "echo hello"}), ("python", {"code": "result = 6 * 7"})],
    [],
]


class ScriptedMessages:
    def __init__(self):
        self.n = 0

    def create(self, **_):
        calls = REPLIES[min(self.n, len(REPLIES) - 1)]
        self.n += 1
        content = [NS(type="tool_use", id=f"toolu_{i}", name=n, input=a) for i, (n, a) in enumerate(calls)]
        content = content or [NS(type="text", text="The answer is 42.")]
        stop = "tool_use" if calls else "end_turn"
        return NS(id=f"scripted-{self.n}", content=content, stop_reason=stop, usage=NS(input_tokens=0, output_tokens=0))


agent = ToyAgent()
agent.llm.client = NS(messages=ScriptedMessages())
print("final:", agent.run("What is 6 * 7?"))
