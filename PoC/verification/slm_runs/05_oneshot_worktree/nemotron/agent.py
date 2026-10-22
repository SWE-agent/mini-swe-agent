"""Toy coding agent. Entry: ToyAgent.run"""
from llm import LLM
from tools import Registry
from observatory_log import (
    llm_query,
    llm_response,
    llm_error,
    tool_start,
    tool_end,
    tool_error,
)

SYSTEM = "You are a small coding agent on Windows. The bash tool runs commands with cmd.exe. Keep answers short."


class ToyAgent:
    def __init__(self):
        self.llm = LLM(system=SYSTEM)
        self.registry = Registry()
        self.messages: list[dict] = []

    def run(self, task: str, max_steps: int = 10) -> str:
        self.messages = [{"role": "user", "content": task}]
        for n in range(1, max_steps + 1):
            reply = self.step(n)
            if not reply["calls"]:
                return reply["content"] if self.confirm(reply["content"], n) else self.ask_user("What next?", n)
        return ""

    def step(self, n: int) -> dict:
        try:
            llm_query(n, model=self.llm.model, messages=self.messages)
            reply = self.llm.chat(self.messages, TOOLS, n)
        except Exception as e:
            llm_error(n, error=e)
            reply = self.fallback_query(n)
        self.messages.append(reply["message"])
        if reply["calls"]:
            llm_response(n, actions=[c["name"] for c in reply["calls"]], reply=reply["message"], response=reply["message"])
        else:
            llm_response(n, actions=[], reply=reply["message"], response=reply["message"])
        results = []
        for call in reply["calls"]:
            start = tool_start(call["name"], call["args"], step=n)
            out = self.registry.run(call["name"], call["args"])
            tool_end(start, call["name"], result=out, step=n)
            results.append((call["id"], out))
        if results:
            self.messages.append(self.llm.results_message(results))
        return reply

    def fallback_query(self, n: int) -> dict:
        return self.llm.chat(self.messages, [], n)

    def ask_user(self, question: str, n: int) -> str:
        start = tool_start("user_input", question, step=n)
        answer = input(f"{question} ")
        tool_end(start, "user_input", result=answer, step=n)
        return answer

    def confirm(self, answer: str, n: int) -> bool:
        return self.ask_user(f"Accept '{answer[:40]}'? [y/N]", n)


if __name__ == "__main__":
    import io
    import os
    import sys
    from pathlib import Path

    if not sys.stdin.isatty():  # no terminal (started from the dashboard): answer "y" instead of waiting
        sys.stdin = io.StringIO("y\n")
    print(ToyAgent().run(" ".join(sys.argv[1:]) or input("Task: ")))