"""Toy agent with KNOWN logging coverage, to test io_audit.py against a ground truth.

Entry: ToyAgent.run
LLM: Claude through the Anthropic SDK (ANTHROPIC_API_KEY, e.g. `uv run --env-file <repo>/.env ...`).
Logging: agent_observatory_kit's observatory_log (llm_* in LLM.chat, tool_* in Registry.run, user input reported as
the tool action "user_input" in ask_user). The three former BUG paths now go through those wrappers.

Expected verdicts:
  covered   LLM      llm.py      LLM.chat                 client.messages.create           (inside the wrapper)
  covered   EXTRACT  llm.py      LLM.parse                "tool_use" blocks                 (below LLM.chat)
  covered   TOOL     tools.py    bash                     subprocess.run                    (below Registry.run)
  covered   TOOL     tools.py    python                   exec                              (below Registry.run)
  covered   TOOL     tools.py    Registry.run             self.tools[name](...)             (inside the wrapper)
  covered   INPUT    agent.py    ToyAgent.ask_user        input()                           (tool_* "user_input")
  (fixed)   fallback_query -> LLM.chat, confirm -> ask_user
"""

import observatory_log
from llm import LLM
from tools import Registry

SYSTEM = "You are a small coding agent on Windows. The bash tool runs commands with cmd.exe. Keep answers short."
TOOLS = [
    {
        "name": name,
        "description": desc,
        "input_schema": {"type": "object", "properties": {arg: {"type": "string"}}, "required": [arg]},
    }
    for name, arg, desc in [
        ("bash", "command", "Run a shell command and return its stdout."),
        ("python", "code", "Run Python code; assign the answer to `result`."),
    ]
]


class ToyAgent:
    def __init__(self):
        self.llm = LLM(system=SYSTEM)
        self.registry = Registry()
        self.messages: list[dict] = []

    def run(self, task: str, max_steps: int = 10) -> str:
        start = observatory_log.tool_start("user_input", {"task": task})
        observatory_log.tool_end(start, "user_input", result=task)
        self.messages = [{"role": "user", "content": task}]
        for n in range(1, max_steps + 1):
            reply = self.step(n)
            if not reply["calls"]:
                return reply["content"] if self.confirm(reply["content"]) else self.ask_user("What next?")
        return ""

    def step(self, n: int) -> dict:
        try:
            reply = self.llm.chat(self.messages, TOOLS, n)
        except TimeoutError:
            reply = self.fallback_query(n)
        self.messages.append(reply["message"])
        results = [(call["id"], self.registry.run(call["name"], call["args"])) for call in reply["calls"]]
        if results:
            self.messages.append(self.llm.results_message(results))
        return reply

    def fallback_query(self, n: int) -> dict:
        return self.llm.chat(self.messages, [], n)

    def ask_user(self, question: str) -> str:
        start = observatory_log.tool_start("user_input", {"question": question})
        answer = input(f"{question} ")
        observatory_log.tool_end(start, "user_input", result=answer)
        return answer

    def confirm(self, answer: str) -> bool:
        return self.ask_user(f"Accept '{answer[:40]}'? [y/N]").lower() == "y"
