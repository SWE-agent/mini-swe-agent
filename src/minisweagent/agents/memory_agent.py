"""Memory-enhanced agent using A-mem for message compression."""

from minisweagent.agents.default import DefaultAgent, LimitsExceeded
from minisweagent.memory.amem_wrapper import AMemWrapper


class MemoryAgent(DefaultAgent):
    """Agent with memory-based message compression.

    Only overrides query() to compress message history before sending to LLM.
    """

    def __init__(self, model, env, *, memory_config=None, **kwargs):
        """Initialize MemoryAgent with memory system.

        Args:
            model: LLM model
            env: Environment
            memory_config: Config dict with keys: embedding_model, llm_backend, llm_model
            **kwargs: Additional args for DefaultAgent
        """
        super().__init__(model, env, **kwargs)

        # Initialize memory system
        memory_config = memory_config or {}
        self.memory = AMemWrapper(
            embedding_model=memory_config.get("embedding_model", "all-MiniLM-L6-v2"),
            llm_backend=memory_config.get("llm_backend", "openai"),
            llm_model=memory_config.get("llm_model", "gpt-5-nano-2025-08-07"),
            persist_directory=memory_config.get("persist_directory", "./memory_db")
        )

    def query(self) -> dict:
        """Query model with compressed messages.

        Compresses message history to 50% size using memory system,
        then queries the model with compacted context.
        """
        # Check limits
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()

        # Compress messages using memory system
        compact_messages = self._compress_messages()

        # Query model with compact context
        response = self.model.query(compact_messages)

        # Add response to message history
        self.add_message("assistant", **response)

        return response

    def _compress_messages(self):
        """Compress message history to 50% size.

        Strategy:
        - Keep system message (first message)
        - Keep last 3 messages (recent context)
        - Summarize everything in between to 50% size

        Returns:
            List of compressed messages
        """
        if len(self.messages) <= 5:
            # Too short to compress
            return self.messages

        # Split messages: system, middle, recent
        system_msg = self.messages[0] if self.messages[0]["role"] == "system" else None
        recent_msgs = self.messages[-3:]  # Keep last 3 messages
        middle_msgs = self.messages[1:-3] if system_msg else self.messages[:-3]

        if not middle_msgs:
            return self.messages

        # Use memory system to summarize middle messages
        try:
            summary = self.memory.summarize_conversation(middle_msgs, target_ratio=0.5)

            # Build compact message list
            compact = []
            if system_msg:
                compact.append(system_msg)

            # Add summary as a user message
            compact.append({
                "role": "user",
                "content": f"[Previous conversation summary]\n{summary}\n[End summary]"
            })

            # Add recent messages
            compact.extend(recent_msgs)

            return compact

        except Exception as e:
            # If compression fails, use original messages
            print(f"Warning: Message compression failed: {e}")
            return self.messages
