"""Run ToyAgent with scripted model to verify observatory logging."""
import builtins
import sys
from pathlib import Path

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Mock anthropic to avoid import errors
import types
anthropic_mock = types.ModuleType("anthropic")
class _DummyAnthropic:
    def __init__(self, *a, **kw): pass
anthropic_mock.Anthropic = _DummyAnthropic
sys.modules["anthropic"] = anthropic_mock

import observatory_log
from agent import ToyAgent

# Patch input to automatically answer prompts
_original_input = builtins.input
def _fake_input(prompt=""):
    # Always accept
    print(prompt, end="")
    return "y"
builtins.input = _fake_input

# Scripted LLM that logs like the real one
class ScriptedLLM:
    def __init__(self, system=""):
        self.model = "scripted"
        self.system = system

    def chat(self, messages, tools, step):
        observatory_log.llm_query(step, model=self.model, messages=messages)
        # Step 1: request a tool call
        if step == 1:
            reply = {
                "content": "I will run a command.",
                "calls": [{"id": "call_1", "name": "bash", "args": {"command": "echo hello"}}],
                "message": {"role": "assistant", "content": []},
            }
        else:
            # Step 2+: final answer, no calls
            reply = {
                "content": "Done",
                "calls": [],
                "message": {"role": "assistant", "content": []},
            }
        observatory_log.llm_response(step, actions=[c["name"] for c in reply["calls"]], reply=reply)
        return reply

    @staticmethod
    def results_message(results):
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": i, "content": out} for i, out in results],
        }

# Create agent with scripted model
agent = ToyAgent()
agent.llm = ScriptedLLM(system="You are a small coding agent.")
# Run on a short task
result = agent.run("test task", max_steps=5)
print("Agent result:", result)

# Restore input
builtins.input = _original_input
