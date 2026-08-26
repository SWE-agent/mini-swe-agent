import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import litellm
from jinja2 import StrictUndefined, Template
from pydantic import BaseModel, Field

from minisweagent.models.utils.content_string import get_content_string

DEFAULT_SUMMARY_TEMPLATE = """Create a concise checkpoint of the conversation below.
Preserve the objective, important technical details, files changed or inspected, commands and results,
decisions, failures, blockers, and the next actions. Do not add information that is not present.

Previous checkpoint:
{{ previous_summary or "(none)" }}

Conversation to incorporate:
{{ transcript }}
"""


class CompactionConfig(BaseModel):
    enabled: bool = False
    auto: bool = True
    context_limit: int = Field(default=0, ge=0)
    keep_tokens: int = Field(default=15_000, ge=0)
    buffer: int = Field(default=20_000, ge=0)
    summary_max_tokens: int = Field(default=4_096, gt=0)
    summary_template: str = DEFAULT_SUMMARY_TEMPLATE


@dataclass
class CompactionRecord:
    summary: str
    compacted_until: int
    reason: str
    estimated_tokens_before: int
    estimated_tokens_after: int
    cost: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class CompactionState:
    summary: str = ""
    compacted_until: int = 1
    records: list[CompactionRecord] = field(default_factory=list)

    def serialize(self) -> dict:
        return {
            "summary": self.summary,
            "compacted_until": self.compacted_until,
            "records": [asdict(record) for record in self.records],
        }


def without_extra(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: without_extra(item) for key, item in value.items() if key != "extra"}
    if isinstance(value, list):
        return [without_extra(item) for item in value]
    return value


def estimate_tokens(messages: list[dict]) -> int:
    return (len(json.dumps(without_extra(messages), default=str, ensure_ascii=False)) + 3) // 4


def resolve_context_limit(model_name: str, configured_limit: int) -> int:
    if configured_limit:
        return configured_limit
    try:
        info = litellm.get_model_info(model_name)
    except Exception as e:
        raise ValueError(
            f"Could not determine the context limit for {model_name!r}; set agent.compaction.context_limit."
        ) from e
    limit = info.get("max_input_tokens") or info.get("max_tokens")
    if not limit:
        raise ValueError(f"No context limit is registered for {model_name!r}; set agent.compaction.context_limit.")
    return int(limit)


def requested_output_tokens(model: Any) -> int:
    kwargs = getattr(getattr(model, "config", None), "model_kwargs", {})
    return int(kwargs.get("max_output_tokens") or kwargs.get("max_completion_tokens") or kwargs.get("max_tokens") or 0)


def compaction_threshold(context_limit: int, buffer: int, output_tokens: int) -> int:
    return max(0, context_limit - max(buffer, output_tokens))


def _is_assistant(message: dict) -> bool:
    return message.get("role") == "assistant" or message.get("object") == "response"


def complete_groups(messages: list[dict], start: int) -> list[tuple[int, int]]:
    groups = []
    group_start = start
    i = start
    while i < len(messages):
        message = messages[i]
        if not _is_assistant(message):
            i += 1
            continue
        n_outputs = len(message.get("extra", {}).get("actions", []))
        group_end = min(len(messages), i + 1 + n_outputs)
        if group_end == i + 1 + n_outputs:
            groups.append((group_start, group_end))
            group_start = group_end
        i = group_end
    if group_start < len(messages):
        groups.append((group_start, len(messages)))
    return groups


def compactable_boundary(messages: list[dict], start: int, keep_tokens: int) -> int | None:
    groups = complete_groups(messages, start)
    if len(groups) < 2:
        return None
    kept_tokens = 0
    first_kept = len(groups) - 1
    for i in range(len(groups) - 1, -1, -1):
        group_tokens = estimate_tokens(messages[groups[i][0] : groups[i][1]])
        if i < len(groups) - 1 and kept_tokens + group_tokens > keep_tokens:
            break
        kept_tokens += group_tokens
        first_kept = i
    if first_kept == 0:
        first_kept = 1
    return groups[first_kept][0]


def fitting_summary_boundary(
    messages: list[dict],
    start: int,
    target: int,
    *,
    previous_summary: str,
    template: str,
    input_limit: int,
) -> int | None:
    boundary = None
    for _, group_end in complete_groups(messages, start):
        if group_end > target:
            break
        prompt = render_summary_prompt(template, previous_summary, messages[start:group_end])
        if estimate_tokens([{"role": "user", "content": prompt}]) > input_limit:
            break
        boundary = group_end
    return boundary


def transcript(messages: list[dict]) -> str:
    parts = []
    for message in messages:
        role = message.get("role") or message.get("type") or message.get("object") or "unknown"
        content = get_content_string(without_extra(message))
        if content:
            parts.append(f"<{role}>\n{content}\n</{role}>")
    return "\n\n".join(parts)


def render_summary_prompt(template: str, previous_summary: str, messages: list[dict]) -> str:
    return Template(template, undefined=StrictUndefined).render(
        previous_summary=previous_summary, transcript=transcript(messages)
    )


def summary_message(model: Any, summary: str) -> dict:
    return model.format_message(
        role="user",
        content=f"<conversation_summary>\n{summary}\n</conversation_summary>\nContinue from this checkpoint.",
    )
