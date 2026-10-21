"""LLM client wrapper (Claude). `LLM.chat` is the logging wrapper for every LLM call."""

import anthropic
import observatory_log


class LLM:
    def __init__(self, model: str = "claude-haiku-5-5", system: str = ""):
        self.client = anthropic.Anthropic()
        self.model = model
        self.system = system

    def chat(self, messages: list[dict], tools: list[dict], step: int) -> dict:
        observatory_log.llm_query(step, model=self.model, messages=messages)
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=self.system,
                messages=messages,
                **({"tools": tools} if tools else {}),
            )
        except Exception as e:
            observatory_log.llm_error(step, e)
            raise
        reply = self.parse(response)
        observatory_log.llm_response(
            step,
            reply=reply,
            actions=[c["name"] for c in reply["calls"]],
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            response=response,
        )
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
