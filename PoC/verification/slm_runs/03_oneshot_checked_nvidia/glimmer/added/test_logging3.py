import os

os.environ.setdefault("OBSERVATORY_RUNS", "observatory/runs")

import sys
import types

# Fake anthropic module
fake_anthropic = types.ModuleType("anthropic")


class FakeMessage:
    def __init__(self, type_, text=None, id=None, name=None, input=None):
        self.type = type_
        self.text = text
        self.id = id
        self.name = name
        self.input = input


class FakeResponse:
    def __init__(self):
        self.content = [
            FakeMessage("text", text="Test answer"),
            FakeMessage("tool_use", id="1", name="bash", input={"command": "echo hi"}),
        ]


class FakeMessages:
    @staticmethod
    def create(**kwargs):
        return FakeResponse()


class DummyAnthropic:
    def __init__(self, *args, **kwargs):
        self.messages = FakeMessages()


fake_anthropic.Anthropic = DummyAnthropic
sys.modules["anthropic"] = fake_anthropic

import agent

ag = agent.ToyAgent()
result = ag.run("test task", max_steps=2)
print("Result:", result)
