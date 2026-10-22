"""Run ToyAgent once with a scripted model, no API key needed.

This runs the agent's run function with a scripted model reply
(no API key or network needed) and verifies the observatory logger
events come in the expected order: user_input, llm_query, llm_response,
tool_start, tool_end, then the next llm_query.
"""

import json
import os
import sys
from pathlib import Path

# Ensure observability_log module is importable FIRST
# The script is at observatory/runs/check_run.py
# observability_log.py is at observatory/observability_log.py
# Add the observatory directory to sys.path so the import works

OBSERVATORY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(OBSERVATORY_DIR))

# Import observability_log module FIRST - this ensures the name 'observability_log'
# is in sys.modules and will be available when the agent's llm.py does
# "from observability_log import ...". Without this, the agent's LLM calls
# would fail because observability_log wouldn't be importable.
import observability_log

# Also import the specific functions that the agent's llm.py also imports
from observability_log import llm_query, llm_response, tool_start, tool_end, tool_error

# Verify the module is accessible
print(f"observability_log imported from: {observability_log.__file__}")
print(f"'observability_log' in sys.modules: {'observability_log' in sys.modules}")

# Verify we can call observability_log.llm_query
try:
    test_result = observability_log.llm_query(1, model="m", messages=[])
    print(f"Can call observability_log.llm_query: {test_result}")
except Exception as e:
    print(f"Cannot call observability_log.llm_query: {type(e).__name__}: {e}")

# ------------------------------------------------------------
# Now stub the anthropic module in sys.modules
# We do this AFTER observability_log is imported
# ------------------------------------------------------------

# Build a minimal anthropic module stub from scratch
_mock_anthropic = type(sys)("anthropic")

class _MockAnthropic:
    """Mock Anthropic client that returns scripted replies."""
    def __init__(self, *args, **kwargs):
        pass

    def messages_create(self, *args, **kwargs):
        class FakeContent:
            def __init__(self):
                self.content = [
                    {"type": "text", "text": "Let me run a command.\n"},
                    {"type": "tool_use", "name": "bash", "id": "toolu_bash", "input": {"command": "echo hello"}},
                ]

        class FakeResponse:
            def __init__(self):
                self.id = "msg_1"
                self.type = "message"
                self.content = self._get_content()

            def _get_content(self):
                return [
                    {"type": "text", "text": "Let me run a command.\n"},
                    {"type": "tool_use", "name": "bash", "id": "toolu_bash", "input": {"command": "echo hello"}},
                ]

        return FakeResponse()

# Wire up the mock
_mock_anthropic.Anthropic = _MockAnthropic
_mock_anthropic.Client = _MockAnthropic
_mock_anthropic.__version__ = "0.0.0"
_mock_anthropic.__file__ = "<mock>"

# Register this stub in sys.modules FIRST, before any code does "import anthropic"
# Since we already imported observability_log above, it's safe
sys.modules["anthropic"] = _mock_anthropic

# Also register submodules to prevent AttributeError when the real package
# code tries to access ._client, ._base_client etc.
sys.modules["anthropic._client"] = type(sys)("anthropic._client")
sys.modules["anthropic._client"].Anthropic = _MockAnthropic
sys.modules["anthropic._client"].Client = _MockAnthropic

sys.modules["anthropic._base_client"] = type(sys)("anthropic._base_client")
sys.modules["anthropic._base_client"].Client = _MockAnthropic

print("anthropic module stub registered in sys.modules")

# ------------------------------------------------------------
# Now import the repo code
# The project root is two levels up from observability/runs/
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent import ToyAgent

# ------------------------------------------------------------
# Run the agent with the mock anthropic client
# ------------------------------------------------------------

def main():
    task = "Say hello world"

    observability_dir = Path(__file__).resolve().parent
    os.environ["OBSERVATORY_RUNS"] = str(observability_dir)

    agent = ToyAgent()
    result = agent.run(task, max_steps=5)

    print(f"\nAgent result: {result!r}")

    # List the log file and print events
    runs_dir = observability_dir / "runs"
    if runs_dir.exists():
        files = sorted(runs_dir.glob("*.jsonl"))
        if files:
            print(f"\nLog file: {files[0]}")
            print("\nEvents (kind, step, at):")
            for line in files[0].read_text().splitlines():
                if line.strip():
                    event = json.loads(line)
                    at_basename = event["at"].rsplit("/", 1)[-1]
                    print(f"  {event['kind']:12s} step={event['step']:>2s}  at={at_basename}")
        else:
            print(f"\nNo JSONL files found in {runs_dir}")
    else:
        print(f"\nRuns directory not found: {runs_dir}")


if __name__ == "__main__":
    main()
