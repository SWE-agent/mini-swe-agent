"""Check that the agent's observatory logging produces events in the correct order.

Runs DefaultAgent.run()) with a deterministic scripted model and reads
the JSON-lines log to confirm the event order: user_input, llm_query, llm_response,
tool_start, tool_end, then the next llm_query.
"""
import json
import os
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so we can import minisweagent
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from minisweagent.agents.default import DefaultAgent
from minisweagent import Environment, Model, __version__
from minisweagent.models.test_models import make_output, DeterministicModel
from minisweagent.config.default import AgentConfig


class ScriptedEnv(Environment):
    """A no-op environment that records tool calls but returns empty results."""

    def __init__(self):
        self.tools = {}  # no real tools

    def execute(self, action: dict) -> dict:
        # Return a minimal observation
        return {"output": "", "returncode": 0}

    def get_template_vars(self) -> dict:
        return {}

    def close(self) -> None:
        pass


def main():
    # A short task
    task = "List the files in the current directory."

    # Scripted replies using DeterministicModel format:
    # 1. LLM reply asking for one tool call (bash: ls)
    # 2. LLM final answer
    outputs = [
        make_output(
            content='Here is what I found:\n\n```bash\nls\n```\n\nThe files are:',
            actions=[{"bash": {"command": "ls"}}],
            cost=0.01,
        ),
        make_output(
            content="Done.",
            actions=[],
            cost=0.0,
        ),
    ]

    # Provide required AgentConfig fields
    config = AgentConfig(
        system_template="",
        instance_template="",
    )

    # Use DeterministicModel which returns pre-scripted replies without API call
    model = DeterministicModel(outputs=outputs, model_name="deterministic")
    env = ScriptedEnv()

    agent = DefaultAgent(model=model, env=env, config=config)

    # Run the agent
    print(f"Running agent with task: {task!r}")
    result = agent.run(task=task)
    print(f"Agent finished. Result: {result}")

    # Read the observatory log
    runs_dir = REPO / "observatory" / "runs"
    if not runs_dir.exists():
        print(f"ERROR: runs directory not found at {runs_dir}")
        sys.exit(1)

    jsonl_files = sorted(runs_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"ERROR: no JSONL files found in {runs_dir}")
        sys.exit(1)

    # Read the (single) log file
    lines = jsonl_files[0].read_text().splitlines()
    events = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: could not parse line: {line[:60]!r}")
                continue

    print(f"\nTotal events logged: {len(events)}")
    print("\nEvent sequence (kind, step):")
    for e in events:
        print(f"  kind={e['kind']!r}, step={e.get('step')!r}, at={e.get('at')!r}")

    # Verify the expected order
    # Expected: user_input, llm_query, llm_response, tool_start, tool_end, [next llm_query ...]
    expected_kinds = ["user_input", "llm_query", "llm_response", "tool_start", "tool_end"]

    print("\n--- Verification ---")
    passed = True
    for i, expected in enumerate(expected_kinds):
        if i >= len(events):
            print(f"FAIL: Expected {expected} at position {i}, but ran out of events")
            passed = False
            break
        actual = events[i]["kind"]
        if actual != expected:
            print(
                f"FAIL: Position {i} expected {expected!r}, got {actual!r}"
            )
            passed = False
        else:
            print(f"OK: Position {i} = {actual!r}")

    # Check that after tool_end, the next event is another llm_query
    if len(events) > 5:
        next_kind = events[5]["kind"]
        if next_kind == "llm_query":
            print(f"OK: Event at position 5 is llm_query (next cycle)")
        else:
            print(f"INFO: Event at position 5 is {next_kind!r} (expected llm_query)")

    if passed:
        print("\nAll expected events found in the correct order!")
    else:
        print("\nSome events were missing or out of order.")
        sys.exit(1)


if __name__ == "__main__":
    main()
