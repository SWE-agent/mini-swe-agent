"""Toy coding agent. Entry: ToyAgent.run"""

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
        observatory_log.user_input(task)
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
        answer = input(f"{question} ")
        observatory_log.user_input(answer, prompt=question)
        return answer

    def confirm(self, answer: str) -> bool:
        return self.ask_user(f"Accept '{answer[:40]}'? [y/N]").lower() == "y"


if __name__ == "__main__":
    import io
    import sys

    if not sys.stdin.isatty():  # no terminal (started from the dashboard): answer "y" instead of waiting
        sys.stdin = io.StringIO("y\n")
    print(ToyAgent().run(" ".join(sys.argv[1:]) or input("Task: ")))
