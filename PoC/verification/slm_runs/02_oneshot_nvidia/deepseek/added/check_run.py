"""Run the agent on a scripted model and print its answer: uv run python check_run.py"""

import io
import sys
import types


class Text:
    type = "text"

    def __init__(self, text):
        self.text = text


class ToolUse:
    type = "tool_use"

    def __init__(self, id, name, input):
        self.id, self.name, self.input = id, name, input


class Response:
    def __init__(self, content):
        self.content = content


class FakeMessages:
    def __init__(self, script):
        self.script = script

    def create(self, **kwargs):
        result = self.script.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class FakeClient:
    def __init__(self, script):
        self.messages = FakeMessages(script)


class Anthropic:
    def __init__(self, *args, **kwargs):
        self.messages = FakeMessages([])


fake = types.ModuleType("anthropic")
fake.Anthropic = Anthropic
sys.modules["anthropic"] = fake

import agent  # noqa: E402


def toy(script: list) -> agent.ToyAgent:
    agent_ = agent.ToyAgent()
    agent_.llm.client = FakeClient(script)
    return agent_


def main() -> None:
    sys.stdin = io.StringIO("y\n")
    # step 1: the model call times out and the fallback asks for two tools; step 2: a plain answer, confirmed
    print(
        "run 1:",
        toy(
            [
                TimeoutError("model timed out"),
                Response([ToolUse("t1", "bash", {"command": "echo hi"}), ToolUse("t2", "python", {"code": "1 + 1"})]),
                Response([Text("done")]),
            ]
        ).run("list the files"),
    )
    # a tool that raises is reported with tool_error and travels on unchanged
    try:
        toy([Response([ToolUse("t3", "python", {"code": "raise ValueError('boom')"})])]).run("break something")
    except ValueError as error:
        print("run 2 raised:", error)


main()
