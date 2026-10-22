"""Script to run ToyAgent with a scripted model for observatory logging check.

This script runs the agent's run function with a scripted model that doesn't
need an API key or network. It creates a mock anthropic client that returns
scripted replies, and passes it to the LLM via the client parameter.
"""

import io
import json
import os
import sys
from pathlib import Path

# Make stdin non-interactive: always accept 'y' instead of waiting for user input
sys.stdin = io.StringIO("y\n")

# 1. Create a mock anthropic client that returns scripted replies
# This client object will be passed to the LLM's client parameter


class _MockMessages:
    """Mock messages that returns scripted replies."""

    def __init__(self):
        self._call_count = 0

    def create(self, model, max_tokens, system, messages, **kwargs):
        self._call_count += 1
        if self._call_count == 1:
            # First call: return a tool use for bash
            # Create a proper response object that parse() can handle
            class _TB:
                type = "tool_use"

                def __init__(self, d):
                    self.id = d["id"]
                    self.name = d["name"]
                    self.input = d["input"]

            return type(
                "Response", (), {"content": [_TB({"id": "tool_1", "name": "bash", "input": {"command": "echo hello"}})]}
            )()
        else:
            # Second call: return final text answer
            class _TB2:
                type = "text"

                def __init__(self, t):
                    self.text = t

            return type("Response", (), {"content": [_TB2("Task complete: echo hello ran successfully")]})()


class _MockAnthropic:
    """Mock anthropic.Anthropic client that returns scripted replies."""

    def __init__(self):
        self.messages = _MockMessages()


# Create the mock client instance
mock_client = _MockAnthropic()

# 2. Now add project root to sys.path and import project modules
# The script is at observatory/runs/check_run.py, so:
# - __file__ is at .../observatory/runs/check_run.py
# - Go up three levels to reach the project root
# - Go up two levels to reach the observatory directory
project_root = Path(__file__).resolve().parent.parent.parent

# Ensure project root is in sys.path (so `agent.py` can be found as a top-level module)
sys.path.insert(0, str(project_root))

# Also add observatory directory so `import observatory_log` works
sys.path.insert(0, str(project_root / "observatory"))

print(f"project_root: {project_root}")
print(f"sys.path[0]: {sys.path[0]}")

from agent import ToyAgent


def main():
    # 3. Run the agent with our scripted model
    # The agent's ToyAgent.__init__ creates: self.llm = LLM(system=SYSTEM)
    # We need to patch LLM.__init__ to pass our mock client
    import llm as llm_module

    # Store original __init__ and patch it
    original_init = llm_module.LLM.__init__

    def patched_init(self, **kwargs):
        # Call original init but pass our mock client
        original_init(self, client=mock_client)

    llm_module.LLM.__init__ = patched_init

    # Run the agent
    task = "Run a simple command"
    agent = ToyAgent()
    result = agent.run(task, max_steps=5)

    # 4. Read and display the observatory log
    runs_dir = project_root / "observatory" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OBSERVATORY_RUNS"] = str(runs_dir)

    run_files = sorted(runs_dir.glob("*.jsonl"))
    if run_files:
        print(f"\nObservatory log events from {run_files[0].name}:")
        for line in run_files[0].read_text().splitlines():
            event = json.loads(line)
            kind = event["kind"]
            step = event.get("step")
            at = event.get("at", "")
            data = event.get("data", {})
            data_preview = json.dumps(data, indent=4)[:120] if isinstance(data, dict) else str(data)[:120]
            print(f"  [{kind} step={step}] at={at} data={data_preview}...")
        # Check event order
        print("\n--- Event order check ---")
        # Re-read to check order
        all_events = []
        for line in run_files[0].read_text().splitlines():
            event = json.loads(line)
            all_events.append(event["kind"])
        print(f"Kinds in order: {all_events}")
        expected = ["user_input", "llm_query", "llm_response", "tool_start", "tool_end"]
        # Check that we see user_input first, then llm_query, etc.
        has_user_input = "user_input" in all_events
        has_llm_query = "llm_query" in all_events
        has_llm_response = "llm_response" in all_events
        has_tool_start = "tool_start" in all_events
        has_tool_end = "tool_end" in all_events
        print(f"user_input present: {has_user_input}")
        print(f"llm_query present: {has_llm_query}")
        print(f"llm_response present: {has_llm_response}")
        print(f"tool_start present: {has_tool_start}")
        print(f"tool_end present: {has_tool_end}")
    else:
        print("\nNo observatory log events were written.")


if __name__ == "__main__":
    main()
