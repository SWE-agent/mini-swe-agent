#!/usr/bin/env python3
"""Check observatory logging by running ToyAgent with a scripted model.

This script replaces the model client with a small object that returns
a reply asking for one tool call, then a final answer — no API key or
network needed.
"""

import json
import os
import sys
from pathlib import Path

# Ensure the project root is on the path so we can import agent/llm.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# OBSERVATORY_RUNS is expected to be set by the environment;
# if not, default to observatory/runs relative to the project root.
if "OBSERVATORY_RUNS" not in os.environ:
    os.environ["OBSERVATORY_RUNS"] = str(PROJECT_ROOT / "observatory/runs")

import llm
from agent import ToyAgent


class _MockToolUseBlock:
    """Minimal mock of an Anthropic tool_use content block."""

    def __init__(self, name: str, args: dict, block_id: str = "tool_1"):
        self.type = "tool_use"
        self.name = name
        self.input = args
        self.id = block_id


class _MockTextBlock:
    """Minimal mock of an Anthropic text content block."""

    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _MockResponse:
    """Mock returned by :meth:`anthropic.Anthropic.messages.create`."""

    def __init__(self, blocks):
        self.content = blocks  # list of _MockToolUseBlock or _MockTextBlock


class _MockClient:
    """Mock Anthropic client that returns scripted responses."""

    def __init__(self, responses):
        """responses: list of (tool_calls, text_answer) tuples, one per LLM call.

        * *tool_calls* — list of ``{"name": str, "args": dict}`` or ``[]``.
        * *text_answer* — ``str`` returned as the final text after tool execution.
        """
        self.responses = responses
        self._i = 0

    def messages_create(self, **kwargs):
        if self._i >= len(self.responses):
            return _MockResponse([_MockTextBlock("")])

        tool_calls, text_answer = self.responses[self._i]
        self._i += 1

        blocks = []
        for call in tool_calls:
            blocks.append(_MockToolUseBlock(name=call["name"], args=call["args"]))
        blocks.append(_MockTextBlock(text_answer))
        return _MockResponse(blocks)


def run_check():
    """Run ``ToyAgent.run`` with a scripted model and read the observatory log."""

    # Scripted responses: one round of tool call, then final answer.
    # The agent loop will then stop because there are no more tool calls.
    responses = [
        # First LLM reply: ask for a bash tool call.
        (
            [{"name": "bash", "args": {"command": "dir"}}],
            "I've run the command",
        ),
        # Second LLM reply: final answer with no tool calls → agent stops.
        ([], "Task finished"),
    ]

    # Patch the LLM client before creating the agent.
    mock_client = _MockClient(responses)
    llm.LLM.__init__ = lambda self, **kw: None  # suppress real init
    # Replace the client after the LLM is constructed by the agent.
    # We'll do this per-agent below.

    # Run the agent
    task = "list the files in the current directory"
    print(f"Running agent with task: {task}")

    agent = ToyAgent()
    # Replace the client on the agent's LLM instance so actual API calls are avoided.
    agent.llm.client = mock_client

    result = agent.run(task, max_steps=5)

    print(f"Agent result: {result}")

    # Read the observatory log
    runs_dir = Path(os.environ["OBSERVATORY_RUNS"])

    # Ensure the directory exists (the logger creates it on first write).
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Find the JSONL file that was just written.
    jsonl_files = sorted(runs_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not jsonl_files:
        print(f"⚠ No JSONL files found in {runs_dir}")
        print("Available files:")
        for f in sorted(runs_dir.iterdir()):
            print(f"  {f}")
        return

    latest_file = jsonl_files[0]
    print(f"\nReading: {latest_file}")

    with open(latest_file, encoding="utf-8") as f:
        lines = f.read().splitlines()

    events = []
    for idx, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"⚠ Skipping malformed line {idx + 1}: {exc}")

    print(f"\nTotal events: {len(events)}")

    # Print a compact summary of each event.
    print("\nEvent sequence (kind, step, at):")
    for i, ev in enumerate(events):
        kind = ev.get("kind", "?")
        step = ev.get("step", "?")
        at = ev.get("at", "?")
        data = ev.get("data", {})
        # Show only the most relevant data fields.
        relevant = {}
        for k in ("action", "result", "duration_seconds", "model", "cost"):
            if k in data:
                relevant[k] = data[k]
        extra = ", ".join(f"{k}={v}" for k, v in relevant.items()) if relevant else ""
        print(f"  {i}: kind={kind:12s} step={str(step):>3s} at={at}  [{extra}]")

    # Verify the expected order.
    print("\n--- Verifying expected event order ---")

    # Extract the kind of each event.
    kinds = [ev["kind"] for ev in events]

    # The expected sequence (based on ToyAgent.run + observatory_log):
    #   tool_start("user_input")   → kind="user_input" with step=None implicitly
    #   tool_end(..., result=task) → kind="user_input" again (step from context)
    #   llm_query                  → kind="llm_query"
    #   llm_response               → kind="llm_response"
    #   tool_start(tool)           → kind="tool_start"
    #   tool_end(tool)             → kind="tool_end"
    #   llm_query                  → next cycle's kind="llm_query"
    #
    # Because the logger records ``user_input`` as two separate events
    # (tool_start / tool_end) around the task, and then the agent loop
    # produces the pattern llm_query / llm_response / tool_start / tool_end / llm_query ...

    # Find first occurrences.
    try:
        i_user_input = next(i for i, k in enumerate(kinds) if k == "user_input")
        i_llm_query = next(i for i, k in enumerate(kinds) if k == "llm_query")
        i_llm_response = next(i for i, k in enumerate(kinds) if k == "llm_response")
        i_tool_start = next(i for i, k in enumerate(kinds) if k == "tool_start")
        i_tool_end = next(i for i, k in enumerate(kinds) if k == "tool_end")
    except StopIteration:
        print("⚠ Could not find all expected event kinds in the log.")
        print(f"Kinds found: {set(kinds)}")
        return

    print(f"First user_input at index {i_user_input}")
    print(f"First llm_query at index {i_llm_query}")
    print(f"First llm_response at index {i_llm_response}")
    print(f"First tool_start at index {i_tool_start}")
    print(f"First tool_end at index {i_tool_end}")

    # Check the ordering.
    expected = [
        "user_input",  # tool_start
        "user_input",  # tool_end (second user_input event, or same kind logged twice)
        "llm_query",
        "llm_response",
        "tool_start",
        "tool_end",
        "llm_query",  # next cycle
    ]

    actual_prefix = kinds[: len(expected)]
    if actual_prefix == expected:
        print("✓ Event order matches the expected pattern exactly.")
    else:
        print("✗ Event order mismatch:")
        print(f"  Expected: {expected}")
        print(f"  Actual:   {actual_prefix}")

    # Print a quick kind-only trace for debugging.
    print(f"\nAll kinds in order: {kinds}")


if __name__ == "__main__":
    run_check()
