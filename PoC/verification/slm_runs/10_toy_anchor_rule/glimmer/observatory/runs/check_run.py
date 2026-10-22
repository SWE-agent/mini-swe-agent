"""Check run script for Agent Observatory logging."""

import sys
import types
from pathlib import Path

# Ensure runs directory exists
Path("observatory/runs").mkdir(parents=True, exist_ok=True)

# Add project root to sys.path so imports work from observatory/runs
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Mock llm module to avoid importing anthropic
llm_module = types.ModuleType("llm")


class DummyLLM:
    def __init__(self, model="dummy", system=""):
        self.model = model
        self.system = system


llm_module.LLM = DummyLLM
sys.modules["llm"] = llm_module

import observatory_log
from agent import ToyAgent


class FakeLLM:
    def __init__(self, system=""):
        self.model = "fake-model"
        self.system = system

    def chat(self, messages, tools, step):
        # Log LLM query and response to mimic real LLM behavior
        observatory_log.llm_query(step, model=self.model, messages=len(messages))
        if step == 1:
            reply = {
                "content": "I need to run a tool to proceed.",
                "calls": [{"id": "call-1", "name": "bash", "args": {"command": "echo hello"}}],
                "message": {"role": "assistant", "content": "tool call"},
            }
        else:
            reply = {"content": "Final answer.", "calls": [], "message": {"role": "assistant", "content": "final"}}
        observatory_log.llm_response(step, actions=[c["name"] for c in reply["calls"]], reply=reply)
        return reply

    def results_message(self, results):
        return {"role": "user", "content": results}


# Create agent with fake LLM
agent = ToyAgent()
agent.llm = FakeLLM(system=agent.llm.system)

# Avoid interactive confirmation prompts
agent.confirm = lambda answer: True
agent.ask_user = lambda question: "y"

# Run agent
result = agent.run("Test task for logging verification", max_steps=3)
print("Agent result:", result)
