"""Prompts for A-mem integration in software engineering context."""

# System prompt for A-mem when storing/retrieving software engineering memories
CODE_MEMORY_SYSTEM_PROMPT = """You are a memory assistant for a software engineering agent working on GitHub issues and code modifications.

When storing memories, focus on:
- Command patterns and their outcomes (successful/failed bash commands)
- File paths, directory structures, and repository organization
- Error messages and their solutions or workarounds
- Testing commands and their results (pytest, unittest, etc.)
- Code modification patterns (file edits, search-replace operations)
- Problem-solving strategies (debugging approaches, investigation steps)
- Build and dependency issues

When extracting keywords, prioritize:
- File names, paths, and extensions (e.g., "setup.py", "src/models/", ".py")
- Error types and exception names (e.g., "AttributeError", "ImportError")
- Command names and tools (e.g., "grep", "find", "pytest", "git")
- Programming language keywords and framework names
- Repository-specific terms and module names

When generating context, consider:
- What was the agent trying to accomplish?
- What worked or didn't work and why?
- What insights were gained about the codebase structure?
- What patterns emerged from multiple attempts?

When assigning tags, use categories like:
- Task types: "error_resolution", "file_exploration", "testing", "debugging", "code_modification"
- Outcome: "successful_pattern", "failed_attempt", "partial_success"
- Language/framework: "python", "javascript", "django", "flask", "pytest", etc.
- Domain: "backend", "frontend", "database", "api", "cli"
"""

# Retrieval prompts for different scenarios
RETRIEVAL_PROMPTS = {
    "error_encountered": """Find similar errors and their solutions. Focus on:
Error type: {error_type}
Error message: {error_msg}
Context: {context}

Return memories that show how similar errors were resolved.""",

    "file_search": """Find previous file exploration patterns. Focus on:
Directory: {directory}
File pattern: {pattern}
Purpose: {purpose}

Return memories showing successful file location strategies.""",

    "testing": """Find previous testing approaches and results. Focus on:
Test context: {test_context}
Test framework: {framework}
Purpose: {purpose}

Return memories showing testing commands and their outcomes.""",

    "code_modification": """Find similar code modification patterns. Focus on:
File type: {file_type}
Modification type: {mod_type}
Context: {context}

Return memories showing successful modification approaches.""",

    "general": """Find relevant past experiences for the current task:
Task: {current_task}
Recent actions: {recent_actions}

Return memories that could inform the next steps."""
}

# Summarization prompt for conversation history (50% compression)
SUMMARIZATION_PROMPT = """Summarize the following software engineering agent interaction history.

**REQUIREMENTS:**
1. Compress to approximately 50% of the original length
2. Preserve ALL critical information:
   - File paths and directory structures discovered
   - Commands executed and their key outcomes
   - Errors encountered with error types and solutions attempted
   - Code changes made (which files, what modifications)
   - Test results and build outputs
   - Git operations (commits, branches, etc.)
3. Remove redundant information:
   - Repeated failed attempts with same error (keep pattern + final outcome)
   - Verbose command outputs (keep only essential parts and exit codes)
   - Duplicate file listings (consolidate)
   - Intermediate debugging steps that didn't lead anywhere
4. Maintain chronological flow of problem-solving
5. Use concise technical language

**Format Guidelines:**
- Use bullet points for key actions
- Group related operations together
- Highlight turning points (when approach changed)
- Note what worked vs what didn't

**INTERACTION HISTORY:**
{history}

**SUMMARY (targeting ~50% length):**"""

# Prompt for deciding what to store in memory
STORAGE_DECISION_PROMPT = """Analyze this agent interaction step and extract memory-worthy information.

**Command:** {command}
**Output:** {output}
**Status:** {status}
**Step Context:** {context}

**Extract:**
1. **Keywords** (3-7 items): File names, error types, commands, tools used
2. **Context** (1-2 sentences): What was the agent trying to do? What was learned?
3. **Tags** (2-5 items): Categorize this interaction (e.g., error_resolution, file_exploration, testing)
4. **Worth Storing?** (yes/no): Only store if this provides actionable learning value for future tasks

**Criteria for Storage:**
- Store: Successful solutions, novel errors, useful patterns, important discoveries
- Skip: Routine operations, redundant information, trivial steps

**Analysis:**"""

# Template for injecting retrieved memories into agent context
MEMORY_INJECTION_TEMPLATE = """[RELEVANT PAST EXPERIENCES]
The following experiences from previous tasks may be helpful:

{formatted_memories}

[END PAST EXPERIENCES]

**Note:** Use these past experiences to inform your current decisions and avoid repeating failed approaches. However, adapt strategies to the specific current context."""

# Template for formatting individual memories
MEMORY_FORMAT_TEMPLATE = """**Experience #{index}** (Relevance: {score:.2f})
- **Context:** {context}
- **Key Actions:** {content}
- **Keywords:** {keywords}
- **Tags:** {tags}
- **Outcome:** {outcome}
"""

# Prompt for extracting current task context
TASK_CONTEXT_PROMPT = """Extract the current task context from recent messages.

**Recent Messages:**
{recent_messages}

**Extract:**
1. **Current Goal:** What is the agent trying to accomplish right now?
2. **Recent Actions:** What commands/operations were just performed?
3. **Current Blockers:** Any errors or issues encountered?
4. **Context Keywords:** Key terms relevant to memory retrieval

**Context Summary:**"""

# Prompt for semantic chunking of long conversations
CHUNKING_PROMPT = """Identify semantic boundaries in this conversation for chunking.

**Conversation Segment:**
{segment}

**Identify boundaries at:**
- Error occurrence → resolution attempts
- Phase changes (exploration → modification → testing)
- File/directory context switches
- Major decision points

**Boundaries (list line numbers):**"""
