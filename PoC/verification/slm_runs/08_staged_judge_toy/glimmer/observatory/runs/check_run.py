import sys
import types
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Create fake anthropic module to avoid real network/API
anthropic_mod = types.ModuleType("anthropic")


class _Msg:
    def __init__(self, type_, id=None, name=None, input=None, text=None):
        self.type = type_
        self.id = id
        self.name = name
        self.input = input
        self.text = text


class MockResponse:
    def __init__(self, content):
        self.content = content


class MockMessages:
    def __init__(self):
        self._count = 0

    def create(self, model=None, max_tokens=None, system=None, messages=None, tools=None):
        self._count += 1
        if self._count == 1:
            # First call: ask for one tool call
            content = [
                _Msg(type_="text", text="I will run a bash command."),
                _Msg(type_="tool_use", id="call-1", name="bash", input={"command": "echo hello"}),
            ]
        else:
            # Subsequent call: final answer
            content = [_Msg(type_="text", text="Task completed.")]
        return MockResponse(content)


class Anthropic:
    def __init__(self, *args, **kwargs):
        self.messages = MockMessages()


anthropic_mod.Anthropic = Anthropic
sys.modules["anthropic"] = anthropic_mod

# Now import agent normally – it will use the fake anthropic
import agent


# Patch ask_user to avoid interactive input
def fake_ask_user(self, question):
    # Always accept confirmation prompts
    return "y"


agent.ToyAgent.ask_user = fake_ask_user

if __name__ == "__main__":
    agent_instance = agent.ToyAgent()
    result = agent_instance.run("Say hello")
    print("Agent result:", result)
