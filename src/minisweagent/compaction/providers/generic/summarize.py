"""Generic LLM summarize compaction: keeps anchors + summary of old steps + last N steps."""

import litellm

from minisweagent.compaction.providers.generic.sliding_window import _group_into_steps

_SUMMARY_PROMPT = (
    "You are a coding assistant. The following are past conversation steps between an agent and a tool environment. "
    "Summarize what was attempted, what succeeded, and what failed — keep it concise and factual.\n\n"
    "{history}"
)


def _steps_to_text(steps: list[list[dict]]) -> str:
    parts = []
    for i, step in enumerate(steps, 1):
        for msg in step:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            parts.append(f"[Step {i} / {role}]\n{content}")
    return "\n\n".join(parts)


class SummarizeCompaction:
    """Summarize older steps via an LLM call, keep the last `keep_last_n_steps` verbatim.

    Args:
        keep_last_n_steps: Number of recent steps to keep as-is.
        summary_model: LiteLLM model string for summarization. Defaults to the first
            available cheap model via litellm.
    """

    def __init__(self, keep_last_n_steps: int = 5, summary_model: str = ""):
        self.keep_last_n_steps = keep_last_n_steps
        self.summary_model = summary_model

    def compact(self, messages: list[dict]) -> list[dict]:
        if len(messages) <= 2:
            return messages
        anchors = messages[:2]
        steps = _group_into_steps(messages[2:])
        if len(steps) <= self.keep_last_n_steps:
            return messages
        old_steps, recent_steps = steps[: -self.keep_last_n_steps], steps[-self.keep_last_n_steps :]
        summary_text = self._summarize(old_steps)
        summary_msg = {"role": "user", "content": f"[Summary of earlier steps]\n{summary_text}"}
        recent_messages = [msg for step in recent_steps for msg in step]
        return anchors + [summary_msg] + recent_messages

    def _summarize(self, steps: list[list[dict]]) -> str:
        history = _steps_to_text(steps)
        prompt = _SUMMARY_PROMPT.format(history=history)
        response = litellm.completion(
            model=self.summary_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
