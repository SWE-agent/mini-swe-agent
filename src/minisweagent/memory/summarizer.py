"""Conversation summarizer using A-mem's LLM for 50% compression."""

from typing import List, Dict
from minisweagent.memory.prompts import SUMMARIZATION_PROMPT


class ConversationSummarizer:
    """Summarizes conversation history using LLM to achieve ~50% compression."""

    def __init__(self, llm_controller):
        """Initialize with A-mem's LLM controller.

        Args:
            llm_controller: LLM controller from A-mem system
        """
        self.llm_controller = llm_controller
        self.summaries_cache = {}  # Cache summaries by message range

    def summarize(self, messages: List[Dict], target_ratio: float = 0.5) -> str:
        """Summarize a list of messages to achieve target compression ratio.

        Args:
            messages: List of message dicts with 'role' and 'content'
            target_ratio: Target compression ratio (0.5 = 50% of original)

        Returns:
            Summarized text
        """
        if not messages:
            return ""

        # Format messages for summarization
        history_text = self._format_messages(messages)

        # Check cache
        cache_key = hash(history_text)
        if cache_key in self.summaries_cache:
            return self.summaries_cache[cache_key]

        # Generate summary using LLM
        prompt = SUMMARIZATION_PROMPT.format(history=history_text)

        # Define response format for structured JSON output
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "summary_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"}
                    },
                    "required": ["summary"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }

        try:
            import json
            result = self.llm_controller.get_completion(
                prompt,
                response_format=response_format
            )

            # Parse JSON response and extract summary
            parsed = json.loads(result)
            summary = parsed.get("summary", "")

            # Cache the summary
            self.summaries_cache[cache_key] = summary

            return summary
        except Exception as e:
            # Fallback: return truncated history if LLM fails
            print(f"Warning: Summarization failed ({e}), using truncation fallback")
            return self._fallback_truncate(history_text, target_ratio)

    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages into readable text for summarization.

        Args:
            messages: List of message dicts

        Returns:
            Formatted text
        """
        formatted_parts = []

        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Format based on role
            if role == "system":
                formatted_parts.append(f"[SYSTEM PROMPT]\n{content}\n")
            elif role == "user":
                formatted_parts.append(f"[USER #{i}]\n{content}\n")
            elif role == "assistant":
                formatted_parts.append(f"[ASSISTANT #{i}]\n{content}\n")
            else:
                formatted_parts.append(f"[{role.upper()} #{i}]\n{content}\n")

        return "\n".join(formatted_parts)

    def _fallback_truncate(self, text: str, target_ratio: float) -> str:
        """Fallback truncation method if LLM summarization fails.

        Args:
            text: Text to truncate
            target_ratio: Target compression ratio

        Returns:
            Truncated text
        """
        lines = text.split('\n')
        target_lines = max(1, int(len(lines) * target_ratio))

        # Keep beginning and end, skip middle
        if len(lines) <= target_lines:
            return text

        keep_start = target_lines // 2
        keep_end = target_lines - keep_start

        truncated_lines = (
            lines[:keep_start] +
            [f"\n... [Truncated {len(lines) - target_lines} lines] ...\n"] +
            lines[-keep_end:]
        )

        return '\n'.join(truncated_lines)

    def summarize_with_split(self, messages: List[Dict], chunk_size: int = 10) -> str:
        """Summarize long conversations by splitting into chunks.

        Useful for very long conversations (>30 messages).

        Args:
            messages: List of message dicts
            chunk_size: Number of messages per chunk

        Returns:
            Combined summary
        """
        if len(messages) <= chunk_size:
            return self.summarize(messages)

        summaries = []

        # Process in chunks
        for i in range(0, len(messages), chunk_size):
            chunk = messages[i:i + chunk_size]
            chunk_summary = self.summarize(chunk, target_ratio=0.5)
            summaries.append(f"**Phase {i//chunk_size + 1}:**\n{chunk_summary}")

        # Combine chunk summaries
        combined = "\n\n".join(summaries)

        # If combined is still too long, summarize again
        if len(combined) > len(self._format_messages(messages)) * 0.5:
            return self.summarize([{"role": "user", "content": combined}], target_ratio=0.7)

        return combined

    def clear_cache(self):
        """Clear the summary cache."""
        self.summaries_cache.clear()
