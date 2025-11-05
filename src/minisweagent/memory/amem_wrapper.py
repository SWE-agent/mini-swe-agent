"""Wrapper for A-mem system tailored for software engineering tasks."""

import os
from typing import List, Dict, Optional
from agentic_memory.memory_system import AgenticMemorySystem
from minisweagent.memory.prompts import (
    CODE_MEMORY_SYSTEM_PROMPT,
    RETRIEVAL_PROMPTS,
    STORAGE_DECISION_PROMPT,
    MEMORY_INJECTION_TEMPLATE,
    MEMORY_FORMAT_TEMPLATE
)


class AMemWrapper:
    """Wrapper around A-mem system for software engineering context."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_backend: str = "openai",
        llm_model: str = "gpt-4o-mini",
        persist_directory: str = "./memory_db"
    ):
        """Initialize A-mem wrapper.

        Args:
            embedding_model: Sentence transformer model for embeddings
            llm_backend: LLM backend (openai or ollama)
            llm_model: LLM model name
            persist_directory: Directory for ChromaDB persistence
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize A-mem system
        self.memory_system = AgenticMemorySystem(
            model_name=embedding_model,
            llm_backend=llm_backend,
            llm_model=llm_model,
            persist_directory=persist_directory
        )

        # Set code-specific system prompt for A-mem
        self._configure_for_code()

        # Track stored memories
        self.memory_count = 0

    def _configure_for_code(self):
        """Configure A-mem with code-specific prompts."""
        # This is a conceptual override - A-mem will use our code-specific
        # prompts when we provide explicit metadata
        pass

    def store_interaction(
        self,
        command: str,
        output: str,
        status: str,
        context: str = "",
        auto_decide: bool = True
    ) -> Optional[str]:
        """Store an agent interaction in memory.

        Args:
            command: Command executed
            output: Command output
            status: Execution status (success/error/timeout)
            context: Additional context about the step
            auto_decide: Whether to auto-decide if worth storing

        Returns:
            Memory ID if stored, None otherwise
        """
        # Prepare content
        content = f"Command: {command}\n"
        if output:
            # Truncate very long outputs
            output_preview = output[:1000] + "..." if len(output) > 1000 else output
            content += f"Output: {output_preview}\n"
        content += f"Status: {status}"

        # Auto-decide if worth storing (skip routine successful operations)
        if auto_decide and not self._should_store(command, output, status):
            return None

        # Extract metadata
        keywords = self._extract_keywords(command, output, status)
        tags = self._extract_tags(command, output, status)
        context_desc = context or self._generate_context(command, output, status)

        try:
            # Store in A-mem
            memory_id = self.memory_system.add_note(
                content=content,
                keywords=keywords,
                context=context_desc,
                tags=tags
            )

            self.memory_count += 1
            return memory_id

        except Exception as e:
            print(f"Warning: Failed to store memory: {e}")
            return None

    def retrieve_relevant(
        self,
        query: str,
        k: int = 3,
        scenario: str = "general"
    ) -> List[Dict]:
        """Retrieve relevant memories.

        Args:
            query: Search query
            k: Number of results
            scenario: Scenario type (error_encountered, file_search, testing, etc.)

        Returns:
            List of memory dicts
        """
        # Format query based on scenario
        if scenario in RETRIEVAL_PROMPTS and scenario != "general":
            # Use scenario-specific prompt template
            formatted_query = query  # Can be enhanced with RETRIEVAL_PROMPTS
        else:
            formatted_query = query

        try:
            # Search using A-mem
            results = self.memory_system.search_agentic(formatted_query, k=k)
            return results if results else []
        except Exception as e:
            print(f"Warning: Memory retrieval failed: {e}")
            return []

    def format_memories_for_injection(self, memories: List[Dict]) -> str:
        """Format retrieved memories for injection into agent context.

        Args:
            memories: List of memory dicts from A-mem

        Returns:
            Formatted string ready for injection
        """
        if not memories:
            return ""

        formatted_memories = []
        for i, mem in enumerate(memories, 1):
            # Extract fields with defaults
            content = mem.get('content', 'N/A')
            context = mem.get('context', 'N/A')
            keywords = ', '.join(mem.get('keywords', []))
            tags = ', '.join(mem.get('tags', []))
            score = mem.get('score', 0.0)

            # Determine outcome from content
            outcome = "Success" if "Status: success" in content else "Error/Attempt"

            formatted_mem = MEMORY_FORMAT_TEMPLATE.format(
                index=i,
                score=score,
                context=context,
                content=content,
                keywords=keywords,
                tags=tags,
                outcome=outcome
            )
            formatted_memories.append(formatted_mem)

        return MEMORY_INJECTION_TEMPLATE.format(
            formatted_memories="\n".join(formatted_memories)
        )

    def _should_store(self, command: str, output: str, status: str) -> bool:
        """Decide if interaction is worth storing.

        Args:
            command: Command executed
            output: Command output
            status: Execution status

        Returns:
            True if should store, False otherwise
        """
        # Always store errors
        if status in ['error', 'timeout', 'ExecutionTimeoutError', 'FormatError']:
            return True

        # Always store if output contains error indicators
        error_indicators = ['error', 'exception', 'failed', 'traceback', 'warning']
        if any(indicator in output.lower() for indicator in error_indicators):
            return True

        # Store novel/complex commands
        complex_commands = ['grep', 'find', 'sed', 'awk', 'pytest', 'git']
        if any(cmd in command.lower() for cmd in complex_commands):
            return True

        # Skip simple commands like 'ls', 'pwd', 'cd'
        simple_commands = ['ls', 'pwd', 'cd', 'echo']
        if any(command.strip().startswith(cmd) for cmd in simple_commands):
            return False

        # Store by default for learning
        return True

    def _extract_keywords(self, command: str, output: str, status: str) -> List[str]:
        """Extract keywords from interaction.

        Args:
            command: Command executed
            output: Command output
            status: Execution status

        Returns:
            List of keywords
        """
        keywords = set()

        # Extract from command
        keywords.add(command.split()[0] if command else "unknown")

        # Extract file paths (simple heuristic)
        import re
        file_patterns = re.findall(r'[\w\-_/\.]+\.[\w]+', command + " " + output[:500])
        keywords.update(file_patterns[:5])  # Limit to 5 files

        # Extract error types
        error_patterns = re.findall(r'(\w+Error|\w+Exception)', output[:500])
        keywords.update(error_patterns[:3])

        # Add status
        keywords.add(status)

        return list(keywords)[:7]  # Limit to 7 keywords

    def _extract_tags(self, command: str, output: str, status: str) -> List[str]:
        """Extract tags for categorization.

        Args:
            command: Command executed
            output: Command output
            status: Execution status

        Returns:
            List of tags
        """
        tags = []

        # Status-based tags
        if status == "success":
            tags.append("successful_pattern")
        elif status in ["error", "timeout"]:
            tags.append("failed_attempt")

        # Command-based tags
        if any(cmd in command.lower() for cmd in ['grep', 'find', 'ls']):
            tags.append("file_exploration")
        if any(cmd in command.lower() for cmd in ['pytest', 'test', 'unittest']):
            tags.append("testing")
        if any(cmd in command.lower() for cmd in ['git']):
            tags.append("version_control")
        if any(cmd in command.lower() for cmd in ['vim', 'nano', 'sed', 'awk']):
            tags.append("code_modification")

        # Error-based tags
        if 'error' in output.lower() or status == 'error':
            tags.append("error_resolution")

        # Default tag
        if not tags:
            tags.append("general")

        return tags[:5]  # Limit to 5 tags

    def _generate_context(self, command: str, output: str, status: str) -> str:
        """Generate context description.

        Args:
            command: Command executed
            output: Command output
            status: Execution status

        Returns:
            Context string
        """
        if status == "success":
            return f"Successfully executed: {command.split()[0]}"
        elif status == "error":
            # Try to extract error type
            import re
            errors = re.findall(r'(\w+Error|\w+Exception)', output[:200])
            if errors:
                return f"Encountered {errors[0]} while executing {command.split()[0]}"
            return f"Error executing {command.split()[0]}"
        elif status == "timeout":
            return f"Command timed out: {command.split()[0]}"
        else:
            return f"Executed {command.split()[0]} with status {status}"

    def get_stats(self) -> Dict:
        """Get memory system statistics.

        Returns:
            Dict with statistics
        """
        return {
            "total_memories": self.memory_count,
            "persist_directory": self.persist_directory
        }

    def clear_all(self):
        """Clear all memories (use with caution!)."""
        # This would require A-mem API support
        print("Warning: Clear all not implemented - would require A-mem API extension")
