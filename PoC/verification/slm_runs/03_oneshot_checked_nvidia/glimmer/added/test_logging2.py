import os

os.environ.setdefault("OBSERVATORY_RUNS", "observatory/runs")

import sys
import types

# Fake anthropic module to avoid broken import
fake_anthropic = types.ModuleType("anthropic")


class DummyAnthropic:
    def __init__(self, *args, **kwargs):
        pass


fake_anthropic.Anthropic = DummyAnthropic
sys.modules["anthropic"] = fake_anthropic

from llm import LLM


def fake_chat(self, messages, tools, step):
    import observatory_log

    observatory_log.llm_query(step, model=self.model, messages=messages)
    reply = {
        "content": "Test answer",
        "calls": [{"id": "1", "name": "bash", "args": {"command": "echo hi"}}],
        "message": {"role": "assistant", "content": "fake"},
    }

    class DummyResp:
        content = []

    observatory_log.llm_response(step, actions=["bash"], reply=reply, response=DummyResp())
    return reply


LLM.chat = fake_chat

import agent

ag = agent.ToyAgent()
result = ag.run("test task", max_steps=2)
print("Result:", result)
