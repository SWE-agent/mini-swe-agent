"""Check run script for ToyAgent with scripted model."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parents[0]))


# Fake anthropic module to avoid real SDK import and network calls
class FakeResponse:
    def __init__(self, content):
        self.content = content


class MockMessages:
    def __init__(self):
        self.call_count = 0

    def create(self, model, max_tokens, system, messages, tools=None):
        self.call_count += 1
        if self.call_count == 1:
            # First LLM call: request a tool call
            block = SimpleNamespace(type="tool_use", id="call_1", name="bash", input={"command": "echo hello"})
            return FakeResponse([block])
        else:
            # Second LLM call: final answer
            block = SimpleNamespace(type="text", text="Task complete.")
            return FakeResponse([block])


class FakeAnthropic:
    def __init__(self, *args, **kwargs):
        self.messages = MockMessages()


fake_anthropic = ModuleType("anthropic")
fake_anthropic.Anthropic = FakeAnthropic
sys.modules["anthropic"] = fake_anthropic

from agent import ToyAgent

# Avoid interactive confirmation prompts
ToyAgent.confirm = lambda self, answer: True

if __name__ == "__main__":
    agent = ToyAgent()
    result = agent.run("Test task for observatory logging", max_steps=5)
    print("Result:", result)
