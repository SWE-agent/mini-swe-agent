"""LLM client wrapper (Claude)."""

import anthropic


class LLM:
    def __init__(self, model: str = "claude-haiku-5-5", system: str = ""):
        self.client = anthropic.Anthropic()
        self.model = model
        self.system = system

    def chat(self, messages: list[dict], tools: list[dict], step: int) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=self.system,
            messages=messages,
            **({"tools": tools} if tools else {}),
        )
        reply = self.parse(response)
        return reply

    def parse(self, response) -> dict:
        calls = [{"id": b.id, "name": b.name, "args": b.input} for b in response.content if b.type == "tool_use"]
        text = "".join(b.text for b in response.content if b.type == "text")
        return {"content": text, "calls": calls, "message": {"role": "assistant", "content": response.content}}

    @staticmethod
    def results_message(results: list[tuple[str, str]]) -> dict:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": i, "content": out} for i, out in results],
        }
