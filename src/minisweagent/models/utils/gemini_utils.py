"""Utilities for Google Gemini and LiteLLM provider compatibility."""


def _deduplicate_gemini_thought_signatures(messages: list[dict]) -> list[dict]:
    """Deduplicate provider_specific_fields when tool_calls already contain thought signatures.

    When LiteLLM returns a response with function calling from a Gemini reasoning/thinking model
    (such as Gemini Earhart or Flash Thinking), it populates thought signatures in both message-level
    `provider_specific_fields` and `tool_calls[*].provider_specific_fields`. Passing both locations back
    into LiteLLM causes the translation layer to duplicate the thought signature across both text and
    function call content parts, doubling the prompt token consumption of reasoning turns.
    """
    result = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls") and "provider_specific_fields" in msg:
            tool_calls = msg["tool_calls"]
            has_tool_thought = any(isinstance(tc, dict) and "provider_specific_fields" in tc for tc in tool_calls)
            if has_tool_thought:
                msg = {k: v for k, v in msg.items() if k != "provider_specific_fields"}
        result.append(msg)
    return result
