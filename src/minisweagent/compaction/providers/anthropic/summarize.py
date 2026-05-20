"""Anthropic-flavored summarize compaction — defaults to claude-haiku for summaries."""

from minisweagent.compaction.providers.generic.summarize import SummarizeCompaction

_DEFAULT_ANTHROPIC_SUMMARY_MODEL = "anthropic/claude-haiku-4-5-20251001"

_ANTHROPIC_SUMMARY_PROMPT = (
    "Human: You are helping a coding agent manage its context window. "
    "Below are earlier conversation steps. Summarize what the agent tried, "
    "what worked, what failed, and any important file paths or error messages. "
    "Be concise.\n\n{history}\n\nAssistant:"
)


class AnthropicSummarizeCompaction(SummarizeCompaction):
    """Summarize older steps using a claude-haiku model by default."""

    def __init__(self, keep_last_n_steps: int = 5, summary_model: str = ""):
        super().__init__(
            keep_last_n_steps=keep_last_n_steps,
            summary_model=summary_model or _DEFAULT_ANTHROPIC_SUMMARY_MODEL,
        )
