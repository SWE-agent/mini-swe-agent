"""Memory-enhanced agent using A-mem for context management."""

from typing import Dict, List
from minisweagent.agents.default import DefaultAgent
from minisweagent.memory.amem_wrapper import AMemWrapper
from minisweagent.memory.summarizer import ConversationSummarizer


class MemoryAgent(DefaultAgent):
    """Agent with A-mem memory system for intelligent context management.

    This agent extends DefaultAgent by:
    1. Storing experiences in A-mem after each step
    2. Retrieving relevant past experiences before querying LLM
    3. Summarizing conversation history to reduce token usage
    4. Maintaining hybrid context: recent + summary + memories
    """

    def __init__(
        self,
        model,
        env,
        *,
        memory_config: Dict = None,
        summarization_threshold: int = 20,
        recent_messages_keep: int = 5,
        memory_retrieval_k: int = 3,
        **kwargs
    ):
        """Initialize MemoryAgent.

        Args:
            model: LLM model
            env: Environment
            memory_config: Config for A-mem system
            summarization_threshold: Trigger summary when messages exceed this
            recent_messages_keep: Number of recent messages to keep raw
            memory_retrieval_k: Number of memories to retrieve
            **kwargs: Additional args for DefaultAgent
        """
        super().__init__(model, env, **kwargs)

        # Memory configuration
        memory_config = memory_config or {}
        self.memory = AMemWrapper(
            embedding_model=memory_config.get("embedding_model", "all-MiniLM-L6-v2"),
            llm_backend=memory_config.get("llm_backend", "openai"),
            llm_model=memory_config.get("llm_model", "gpt-4o-mini"),
            persist_directory=memory_config.get("persist_directory", "./memory_db")
        )

        # Summarizer (uses A-mem's LLM)
        self.summarizer = ConversationSummarizer(
            self.memory.memory_system.llm_controller
        )

        # Configuration
        self.summarization_threshold = summarization_threshold
        self.recent_messages_keep = recent_messages_keep
        self.memory_retrieval_k = memory_retrieval_k

        # State tracking
        self.last_summary_at = 0
        self.current_summary = None
        self.original_messages = []  # Keep full history for reference

    def query(self) -> dict:
        """Query model with memory-enhanced context.

        Overrides DefaultAgent.query() to:
        1. Retrieve relevant memories
        2. Compress message history if needed
        3. Build hybrid context
        """
        # Check limits before querying
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            from minisweagent.agents.default import LimitsExceeded
            raise LimitsExceeded()

        # Build compressed context
        compressed_messages = self._build_hybrid_context()

        # Query model with compressed context
        response = self.model.query(compressed_messages)

        # Store in both original and compressed
        self.add_message("assistant", **response)
        self.original_messages.append({"role": "assistant", **response})

        return response

    def execute_action(self, action: dict) -> dict:
        """Execute action and store experience in memory.

        Overrides DefaultAgent.execute_action() to store results.
        """
        # Execute using parent's method
        try:
            output = self.env.execute(action["action"])
            status = "success"
        except Exception as e:
            # Handle errors
            output = {"output": str(e), "returncode": -1}
            status = "error"
            # Re-raise to let parent handle
            raise

        # Store in memory
        self._store_experience(
            command=action["action"],
            output=output.get("output", ""),
            status=status
        )

        return output

    def get_observation(self, response: dict) -> dict:
        """Get observation and handle memory storage.

        Overrides to ensure memory storage happens correctly.
        """
        try:
            output = self.execute_action(self.parse_action(response))
            observation = self.render_template(
                self.config.action_observation_template,
                output=output
            )
            self.add_message("user", observation)
            self.original_messages.append({"role": "user", "content": observation})
            return output

        except Exception as e:
            # Store failed attempt
            action = self.parse_action(response)
            self._store_experience(
                command=action.get("action", "unknown"),
                output=str(e),
                status=type(e).__name__
            )
            raise

    def _build_hybrid_context(self) -> List[Dict]:
        """Build hybrid context: system + summary + memories + recent.

        Returns:
            List of messages for LLM
        """
        compressed = []

        # 1. Always keep system message
        if self.messages and self.messages[0].get("role") == "system":
            compressed.append(self.messages[0])

        # 2. Add summarized history if conversation is long
        if len(self.messages) > self.summarization_threshold:
            summary = self._get_or_create_summary()
            if summary:
                compressed.append({
                    "role": "user",
                    "content": f"[CONVERSATION SUMMARY]\n{summary}\n[END SUMMARY]"
                })

        # 3. Retrieve and inject relevant memories
        memories = self._retrieve_relevant_memories()
        if memories:
            memory_context = self.memory.format_memories_for_injection(memories)
            compressed.append({
                "role": "user",
                "content": memory_context
            })

        # 4. Add recent messages (last N)
        recent_start = max(1, len(self.messages) - self.recent_messages_keep)
        compressed.extend(self.messages[recent_start:])

        return compressed

    def _get_or_create_summary(self) -> str:
        """Get existing summary or create new one.

        Returns:
            Summary text
        """
        # Check if we need to update summary
        messages_since_summary = len(self.messages) - self.last_summary_at

        if messages_since_summary < 10 and self.current_summary:
            # Use cached summary
            return self.current_summary

        # Create new summary
        # Summarize messages between last summary and recent messages
        end_idx = len(self.messages) - self.recent_messages_keep
        start_idx = 1  # Skip system message

        if end_idx <= start_idx:
            return ""

        messages_to_summarize = self.messages[start_idx:end_idx]

        if not messages_to_summarize:
            return ""

        # Generate summary
        self.current_summary = self.summarizer.summarize(
            messages_to_summarize,
            target_ratio=0.5
        )
        self.last_summary_at = end_idx

        return self.current_summary

    def _retrieve_relevant_memories(self) -> List[Dict]:
        """Retrieve relevant memories based on current context.

        Returns:
            List of memory dicts
        """
        if not self.messages:
            return []

        # Build query from recent messages
        recent_content = []
        for msg in self.messages[-3:]:  # Last 3 messages
            if msg.get("role") in ["user", "assistant"]:
                recent_content.append(msg.get("content", ""))

        query = " ".join(recent_content)[-500:]  # Limit query length

        if not query.strip():
            return []

        # Determine scenario
        scenario = "general"
        if "error" in query.lower() or "exception" in query.lower():
            scenario = "error_encountered"
        elif "test" in query.lower():
            scenario = "testing"
        elif "file" in query.lower() or "find" in query.lower():
            scenario = "file_search"

        # Retrieve memories
        try:
            memories = self.memory.retrieve_relevant(
                query=query,
                k=self.memory_retrieval_k,
                scenario=scenario
            )
            return memories
        except Exception as e:
            print(f"Warning: Memory retrieval failed: {e}")
            return []

    def _store_experience(self, command: str, output: str, status: str):
        """Store experience in memory.

        Args:
            command: Command executed
            output: Command output
            status: Status (success/error/timeout)
        """
        try:
            # Get context from recent messages
            context = self._extract_current_context()

            # Store in memory
            self.memory.store_interaction(
                command=command,
                output=output,
                status=status,
                context=context,
                auto_decide=True  # Let memory system decide if worth storing
            )
        except Exception as e:
            print(f"Warning: Failed to store experience: {e}")

    def _extract_current_context(self) -> str:
        """Extract current task context from recent messages.

        Returns:
            Context string
        """
        # Look at recent user messages for context
        for msg in reversed(self.messages[-5:]):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Extract first line as context
                first_line = content.split('\n')[0]
                if len(first_line) > 10:
                    return first_line[:200]

        return "Software engineering task"

    def get_memory_stats(self) -> Dict:
        """Get memory system statistics.

        Returns:
            Dict with stats
        """
        return {
            **self.memory.get_stats(),
            "total_messages": len(self.messages),
            "original_messages": len(self.original_messages),
            "last_summary_at": self.last_summary_at,
            "has_summary": bool(self.current_summary)
        }
