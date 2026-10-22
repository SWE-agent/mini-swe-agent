import os

os.environ.setdefault("OBSERVATORY_RUNS", "observatory/runs")

# Import after env set
from llm import LLM


# Deterministic fake chat
def fake_chat(self, messages, tools, step):
    # Simulate llm_query logging already done in LLM.chat, but we are replacing chat
    # We'll call logger manually to mimic real flow
    import observatory_log

    observatory_log.llm_query(step, model=self.model, messages=messages)
    # Fake response
    reply = {
        "content": "Test answer",
        "calls": [{"id": "1", "name": "bash", "args": {"command": "echo hi"}}],
        "message": {"role": "assistant", "content": "fake"},
    }

    # Fake response object for logging
    class DummyResp:
        content = []

    observatory_log.llm_response(step, actions=["bash"], reply=reply, response=DummyResp())
    return reply


LLM.chat = fake_chat

import agent

ag = agent.ToyAgent()
result = ag.run("test task", max_steps=2)
print("Result:", result)
