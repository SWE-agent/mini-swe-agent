"""Test: does pre-importing observatory_log work when llm.py is imported after?"""

import json
import sys
from pathlib import Path

# ------------------------------------------------------------
# STUB the anthropic module BEFORE any other code imports it
# ------------------------------------------------------------

# Build a minimal anthropic module stub
_minimal_anthropic = type(sys)("anthropic")


class _MinimalAnthropicClient:
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
                self.content = [
                    {"type": "text", "text": "Let me run a command.\n"},
                    {"type": "tool_use", "name": "bash", "id": "toolu_bash", "input": {"command": "echo hello"}},
                ]

        return FakeResponse()


_minimal_anthropic.Anthropic = _MinimalAnthropicClient
_minimal_anthropic.Client = _MinimalAnthropicClient
_minimal_anthropic.__version__ = "0.0.0"
_minimal_anthropic.__file__ = "<mock>"

# Register in sys.modules FIRST
sys.modules["anthropic"] = _minimal_anthropic
sys.modules["anthropic._client"] = type(sys)("anthropic._client")
sys.modules["anthropic._client"].Anthropic = _MinimalAnthropicClient
sys.modules["anthropic._client"].Client = _MinimalAnthropicClient
sys.modules["anthropic._base_client"] = type(sys)("anthropic._base_client")
sys.modules["anthropic._base_client"].Client = _MinimalAnthropicClient

print("anthropic stub registered in sys.modules")

# ------------------------------------------------------------
# NOW: Pre-import observatory_log BEFORE importing agent/llm
# ------------------------------------------------------------

# The script is at observatory/runs/check_run.py
# observatory_log.py is at observatory/observatory_log.py
_OBSERVATORY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_OBSERVATORY_DIR))

# Import observatory_log module FIRST - this ensures the name 'observatory_log'
# is in sys.modules and will be available when llm.py does "from observatory_log import ..."
# AND when it references "observatory_log.llm_query(...)"

# Also import the names that llm.py will import

print("observatory_log imported and names extracted")
print("  'observatory_log' in globals:", "observatory_log" in globals())
print("  'observatory_log' in sys.modules:", "observatory_log" in sys.modules)

# ------------------------------------------------------------
# NOW: Import the repo code
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent import ToyAgent

print("Agent imported successfully")

# ------------------------------------------------------------
# NOW: Run the agent
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
    import os

    main()
