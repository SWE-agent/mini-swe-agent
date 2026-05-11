"""Heuristic phase detection from action commands and thought text.

All detection is regex-based: zero LLM cost per step.
"""

import re

from minisweagent.agents.planmem.types import TaskPhase

# ── Compiled patterns ───────────────────────────────────────────────────────

# Commands that indicate exploration
EXPLORATION_CMDS = re.compile(
    r"^\s*(find|grep|rg|ag|ack|ls|tree|cat|head|tail|less|more|wc|file|"
    r"nl|bat|fd|locate|git\s+(log|show|diff|blame))\b",
    re.IGNORECASE,
)
EXPLORATION_THOUGHT = re.compile(
    r"\b(understand|look\s+at|explore|check|examine|inspect|read|browse|"
    r"search|find|investigate|navigate|locate|overview)\b",
    re.IGNORECASE,
)

# Commands that indicate implementation
IMPLEMENTATION_CMDS = re.compile(
    r"^\s*(sed|awk|perl|ed|ex|tee|cp|mv|truncate|apply_patch|patch|"
    r"git\s+(apply|checkout|restore|mv|rm))\b",
    re.IGNORECASE,
)
IMPLEMENTATION_REDIRECT = re.compile(r"(^|\s)(>|>>)\s*\S")
IMPLEMENTATION_HEREDOC = re.compile(r"cat\s+<<")

# Commands that indicate verification
VERIFICATION_CMDS = re.compile(
    r"^\s*(python|python3|pytest|py\.test|tox|nox|make\s+test|"
    r"npm\s+test|yarn\s+test|go\s+test|cargo\s+test|"
    r"\.\/test|bash\s+test|unittest)\b",
    re.IGNORECASE,
)
VERIFICATION_THOUGHT = re.compile(
    r"\b(test|verify|check\s+if|confirm|validate|reproduce|run)\b",
    re.IGNORECASE,
)

# Commands/thought patterns indicating hypothesis formation
# (reasoning about *why* something fails — not editing, not blindly exploring)
HYPOTHESIS_CMDS = re.compile(
    r"^\s*(python\d?\s+-c|python\d?\s+-m\s+pdb|pdb|ipython|"
    r"git\s+blame|git\s+log\s+-p)\b",
    re.IGNORECASE,
)
HYPOTHESIS_THOUGHT = re.compile(
    r"\b(root\s+cause|because|hypothes(is|ize)|suspect|"
    r"reason\s+(why|for)|likely\s+(due|caused)|trace|culprit|"
    r"why\s+(is|does|the|this))\b",
    re.IGNORECASE,
)


def detect_phase(action: str, thought: str, current_phase: TaskPhase) -> TaskPhase:
    """Detect task phase from a bash command and optional THOUGHT text.

    Priority: IMPLEMENTATION > VERIFICATION > HYPOTHESIS > EXPLORATION > maintain current.

    HYPOTHESIS is detected when the agent is reasoning about a cause
    (via THOUGHT text or via diagnostic-style commands like `python -c`,
    `git blame`, `pdb`) without yet editing files.

    Args:
        action: The bash command string.
        thought: The THOUGHT section from the LLM response.
        current_phase: The current phase (returned if no pattern matches).

    Returns:
        Detected TaskPhase.
    """
    first_line = action.split("\n")[0].strip() if action else ""

    if _is_implementation(first_line):
        return TaskPhase.IMPLEMENTATION

    # Diagnostic-shaped commands (python -c, pdb, git blame) take precedence
    # over verification — `python -c` is reasoning, not test-running.
    if HYPOTHESIS_CMDS.search(first_line):
        return TaskPhase.HYPOTHESIS

    if _is_verification(first_line, thought):
        return TaskPhase.VERIFICATION

    if _is_hypothesis(first_line, thought):
        return TaskPhase.HYPOTHESIS

    if _is_exploration(first_line, thought):
        return TaskPhase.EXPLORATION

    return current_phase


def is_edit_action(action: str) -> bool:
    """Check if an action modifies files (for file edit tracking)."""
    first_line = action.split("\n")[0].strip() if action else ""
    return _is_implementation(first_line)


def _is_implementation(first_line: str) -> bool:
    return bool(
        IMPLEMENTATION_CMDS.search(first_line)
        or IMPLEMENTATION_REDIRECT.search(first_line)
        or IMPLEMENTATION_HEREDOC.search(first_line)
    )


def _is_verification(first_line: str, thought: str) -> bool:
    if VERIFICATION_CMDS.search(first_line):
        return True
    if thought and VERIFICATION_THOUGHT.search(thought):
        if re.search(r"\b(python|pytest|test|run)\b", first_line, re.IGNORECASE):
            return True
    return False


def _is_exploration(first_line: str, thought: str) -> bool:
    if EXPLORATION_CMDS.search(first_line):
        return True
    if thought and EXPLORATION_THOUGHT.search(thought):
        return True
    return False


def _is_hypothesis(first_line: str, thought: str) -> bool:
    """HYPOTHESIS = reasoning about cause without editing.

    Triggers on diagnostic commands OR thought-text signaling
    causal reasoning. To avoid swallowing every exploration step,
    we require either a diagnostic command, or *both* hypothesis
    thought-text AND no exploration command.
    """
    if HYPOTHESIS_CMDS.search(first_line):
        return True
    if thought and HYPOTHESIS_THOUGHT.search(thought):
        if not EXPLORATION_CMDS.search(first_line):
            return True
    return False
