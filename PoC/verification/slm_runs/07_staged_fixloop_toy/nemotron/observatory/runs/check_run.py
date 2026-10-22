"""Run the agent with a scripted model (no API key or network needed) and observe the observatory log.

This script replaces the real Anthropic Claude client with a small scripted object
that returns a reply asking for one tool call, then a final answer. The observatory
logger will record events to observatory/runs/ for inspection.
"""

import importlib.util
import json
import os
from pathlib import Path

# Set the OBSERVATORY_RUNS environment variable first
runs_base = Path(
    r"C:\Users\USER\AppData\Local\Temp\claude\c--Users-USER-Developments-mini-swe-agent\2b6776a6-acde-42c1-9433-2f8ca7ccb7f5\scratchpad\v3\observatory\runs"
)
os.environ["OBSERVATORY_RUNS"] = str(runs_base)
runs_base.mkdir(parents=True, exist_ok=True)

# Clear any cached logger files so we get a fresh log
log_files = Path(
    r"C:\Users\USER\AppData\Local\Temp\claude\c--Users-USER-Developments-mini-swe-agent\2b6776a6-acde-42c1-9433-2f8ca7ccb7f5\scratchpad\v3\observatory\runs"
).glob("*.jsonl")
for f in log_files:
    f.unlink()

# Import the logger module directly
obs_dir = Path(
    r"C:\Users\USER\AppData\Local\Temp\claude\c--Users-USER-Developments-mini-swe-agent\2b6776a6-acde-42c1-9433-2f8ca7ccb7f5\scratchpad\v3\observatory"
)
spec = importlib.util.spec_from_file_location("observatory_log", str(obs_dir / "observatory_log.py"))
log = importlib.util.module_from_spec(spec)
spec.loader.exec_module(log)

# ---------------------------------------------------------------------------
# Scripted model: mimics the Anthropic Claude client without making network calls.
# ---------------------------------------------------------------------------


class ScriptedToolCall:
    """Represents a tool use block from the model response."""

    def __init__(self, id_, name, args):
        self.id = id_
        self.name = name
        self.input = args


class ScriptedModelResponse:
    """A mock response from the model that includes one tool call and some text."""

    def __init__(self, tool_name: str, tool_args: dict, text: str = ""):
        self.content = [
            {"type": "tool_use", "id": f"tool_{id(tool_name)}", "name": tool_name, "input": tool_args},
            {"type": "text", "text": text},
        ]
        self.role = "assistant"


class ScriptedModel:
    """A mock Anthropic client that returns predetermined replies.

    The first call returns a tool call; the second returns a final answer.
    """

    def __init__(self):
        self.call_count = 0
        # First response: ask for a tool call (bash 'echo hello')
        self.first_reply = ScriptedModelResponse(
            tool_name="bash",
            tool_args={"command": "echo hello"},
            text="Please run the following command:",
        )
        # Second response: final answer
        self.second_reply = ScriptedModelResponse(
            tool_name="",
            tool_args={},
            text="Hello! The command output was:",
        )

    def chat(self, messages, tools, step):  # noqa: ARG002
        """Mock the LLM chat call.

        Returns a dict mimicking the real response structure.
        """
        self.call_count += 1
        if self.call_count == 1:
            # Return first reply with a tool call
            return {
                "content": self.first_reply.content,
                "role": self.first_reply.role,
                "calls": [
                    {
                        "id": f"tool_{id(self.first_reply.content[0])}",
                        "name": self.first_reply.content[0]["name"],
                        "args": self.first_reply.content[0]["input"],
                    }
                ],
                "model": self.first_reply.content[0]["name"],
            }
        else:
            # Return second reply with no tool calls (final answer)
            return {
                "content": self.second_reply.content,
                "role": self.second_reply.role,
                "calls": [],
                "model": self.second_reply.content[0]["name"] if self.second_reply.content else "mock",
            }


# ---------------------------------------------------------------------------
# Replace the real LLM client with our scripted one
# ---------------------------------------------------------------------------

# Monkey-patch the LLM class used by ToyAgent to use our scripted model
import llm as llm_module

original_llm_chat = llm_module.LLM.chat


def scripted_chat(self, messages, tools, step):  # noqa: ARG002
    """Replace the real chat with our scripted version."""
    model = ScriptedModel()
    reply = model.chat(messages, tools, step)
    # Parse the reply the same way the real LLM.parse does
    calls = []
    text = ""
    for block in reply["content"]:
        if block.get("type") == "tool_use":
            calls.append({"id": block["id"], "name": block["name"], "args": block["input"]})
        elif block.get("type") == "text":
            text = block.get("text", "")
    return {
        "content": text,
        "calls": calls,
        "message": {"role": "assistant", "content": text},
    }


llm_module.LLM.chat = scripted_chat

# ---------------------------------------------------------------------------
# Run the agent
# ---------------------------------------------------------------------------

from observatory_log import ToyAgent


def main():
    task = "Say hello by running a bash command"
    print(f"Running agent with task: {task}")
    print("Check the log at: observatory/runs/<jsonl-file>")
    print()

    agent = ToyAgent()
    result = agent.run(task, max_steps=5)
    print()
    print(f"Agent result: {result}")

    # List the log files
    runs_dir = Path(os.environ["OBSERVATORY_RUNS"])
    log_files = sorted(runs_dir.glob("*.jsonl"))
    if log_files:
        print(f"\nLog file(s) created: {log_files}")
        # Read and display the events
        for lf in log_files:
            print(f"\n--- {lf.name} ---")
            lines = lf.read_text().splitlines()
            for i, line in enumerate(lines, 1):
                event = json.loads(line)
                kind = event["kind"]
                step_val = event.get("step", "N/A")
                at = event.get("at", "")
                data_summary = {}
                if "data" in event:
                    for k, v in event["data"].items():
                        if isinstance(v, (str, int, float, bool)):
                            data_summary[k] = v
                        else:
                            data_summary[k] = f"<{type(v).__name__}>"
                print(f"  {i}: [{kind}] step={step_val} at={at} data={data_summary}")
    else:
        print("\nNo log files were created - the logger may not have been triggered.")


if __name__ == "__main__":
    main()
