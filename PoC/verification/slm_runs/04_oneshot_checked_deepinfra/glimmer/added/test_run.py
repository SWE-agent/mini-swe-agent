import sys
import types


class FakeMessage:
    def __init__(self, msg_type, text="", id=None, name=None, input=None):
        self.type = msg_type
        self.text = text
        self.id = id
        self.name = name
        self.input = input or {}


class FakeMessages:
    @staticmethod
    def create(model, max_tokens, system, messages, tools=None):
        if not hasattr(FakeMessages.create, "call_count"):
            FakeMessages.create.call_count = 0
        FakeMessages.create.call_count += 1
        if FakeMessages.create.call_count == 1:
            msg1 = FakeMessage("tool_use", id="call_1", name="bash", input={"command": "echo hi"})
            msg2 = FakeMessage("text", text="I will run bash")
            return type("Resp", (), {"content": [msg1, msg2]})()
        else:
            msg = FakeMessage("text", text="done")
            return type("Resp", (), {"content": [msg]})()


class FakeAnthropic:
    def __init__(self, *args, **kwargs):
        self.messages = FakeMessages


# inject fake anthropic module before any import
fake_anthropic = types.ModuleType("anthropic")
fake_anthropic.Anthropic = FakeAnthropic
sys.modules["anthropic"] = fake_anthropic

from agent import ToyAgent

# Run with a task
agent = ToyAgent()
# mock ask_user to avoid real input
agent.ask_user = lambda q: "y"
result = agent.run("test task", max_steps=3)
print("Result:", result)
