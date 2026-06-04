"""Reviewer prompt templates and patch extraction for the Coder-Reviewer repair loop."""

import re

# ---------------------------------------------------------------------------
# Truncation / formatting helpers
# ---------------------------------------------------------------------------

TRACE_MAX_CHARS = 4000
DIFF_HEADER_RE = re.compile(r"^--- ", re.MULTILINE)


def _truncate(text: str, max_chars: int = TRACE_MAX_CHARS) -> str:
    """Truncate *text* to *max_chars*, keeping head + tail when possible."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n\n... [{len(text) - max_chars} chars elided] ...\n\n" + text[-half:]


# ---------------------------------------------------------------------------
# Reviewer prompts
# ---------------------------------------------------------------------------

REVIEWER_SYSTEM_PROMPT = """\
You are an expert code reviewer. Your task is to fix a broken git patch.

You will be given:
1. The original task / PR description
2. A patch (unified diff) that was submitted as the fix
3. The failure trace: output from applying the patch and running the tests

The patch either failed to apply (`git apply` error) or caused test failures.
Analyze the failure carefully and output a **corrected unified diff** that fixes the issue.

**Output format:**
- Output ONLY the corrected unified diff, starting with `--- ` on the first line.
- The diff must be applicable with `git apply` in the repository root.
- Preserve the original file paths (`--- a/...` / `+++ b/...`).
- Do NOT include any explanation, markdown fences, or commentary — just the diff."""


def build_reviewer_prompt(task: str, patch: str, trace: str) -> str:
    """Build the user message for the reviewer LM.

    Parameters
    ----------
    task:
        The original task / PR description.
    patch:
        The rejected unified diff.
    trace:
        Failure trace from apply + test evaluation (truncated to ~4k chars).
    """
    return f"""## Task / PR description

{task}

## Rejected patch

{patch}

## Failure trace

{_truncate(trace)}"""


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_patch(text: str) -> str:
    """Extract a unified-diff patch from a free-text model response.

    Looks for the first ``--- `` line and returns everything from there
    to the end of *text*.  Falls back to returning *text* unchanged if no
    diff header is found.
    """
    m = DIFF_HEADER_RE.search(text)
    if m:
        return text[m.start() :].strip()
    # Fallback: try to strip markdown fences at string boundaries
    cleaned = re.sub(r"^```(?:diff)?\s*\n", "", text)
    cleaned = re.sub(r"\n```\s*$", "", cleaned)
    return cleaned.strip()
