"""Check the observatory wiring: runs ToyAgent with a scripted model, no API key or network. uv run python scripted_run.py"""

import io
import os
import sys

from agent import ToyAgent


class Block:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def model_dump(self, mode="json"):
        return dict(self.__dict__)


class Reply:
    def __init__(self, blocks):
        self.content = blocks

    def model_dump(self, mode="json"):
        return {"content": [block.model_dump() for block in self.content]}


class ScriptedModel:
    """Stand-in for the Anthropic client: pops one canned reply per create() call, raising the ones that are exceptions."""

    def __init__(self, replies):
        self.messages = self
        self.replies = list(replies)

    def create(self, **kwargs):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def main() -> None:
    os.environ.setdefault("ANTHROPIC_API_KEY", "stub")  # the client refuses to construct without one; no call is made
    agent = ToyAgent()
    agent.llm.client = ScriptedModel(
        [
            TimeoutError("scripted: network down"),  # exercises the llm_error path and the agent's fallback query
            Reply([Block(type="tool_use", id="t1", name="bash", input={"command": "echo hello from bash"})]),
            Reply([Block(type="tool_use", id="t2", name="python", input={"code": "result = 6 * 7"})]),
            Reply([Block(type="text", text="Both tools work: bash echoed and python computed 42.")]),
        ]
    )
    sys.stdin = io.StringIO("y\n")  # answers the agent's confirm prompt
    print(agent.run("Check that the bash and python tools work"))


if __name__ == "__main__":
    main()
