"""Test logging wiring with deterministic mock LLM."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.makedirs("observatory/runs", exist_ok=True)


class MockBlock:
    def __init__(self, typ, text="", id="1", name="bash", input=None):
        self.type = typ
        self.text = text
        self.id = id
        self.name = name
        self.input = input or {}


class MockResponse:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = SimpleNamespace(input_tokens=10, output_tokens=20)


def make_response(with_tool=False):
    if with_tool:
        blocks = [
            MockBlock("text", text="I will run bash"),
            MockBlock("tool_use", id="call_1", name="bash", input={"command": "echo hello"}),
        ]
    else:
        blocks = [MockBlock("text", text="Done")]
    return MockResponse(blocks)


def run_test():
    import sys
    from io import StringIO

    # Provide dummy stdin to avoid EOF in confirm
    sys.stdin = StringIO("y\n")
    from agent import ToyAgent

    with patch("llm.anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        # Two steps with tool calls to avoid user prompt
        mock_client.messages.create.side_effect = [make_response(with_tool=True), make_response(with_tool=True)]
        agent = ToyAgent()
        result = agent.run("test task", max_steps=2)
        print("Agent result:", repr(result))


if __name__ == "__main__":
    run_test()
