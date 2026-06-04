"""Tests for minisweagent.patch_repair.prompts — pure functions, no mocking needed."""

import pytest

from minisweagent.patch_repair.prompts import (
    REVIEWER_SYSTEM_PROMPT,
    _truncate,
    build_reviewer_prompt,
    extract_patch,
)


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------

SHORT = "hello world"


def test_truncate_short_text_unchanged():
    assert _truncate(SHORT, max_chars=100) == SHORT


def test_truncate_exact_boundary():
    assert _truncate("abcde", max_chars=5) == "abcde"


def test_truncate_long_text_contains_elision_marker():
    result = _truncate("x" * 5000, max_chars=4000)
    assert "elided" in result


def test_truncate_long_text_keeps_head_and_tail():
    head = "AAAAA"
    tail = "ZZZZZ"
    long = head + ("x" * 5000) + tail
    result = _truncate(long, max_chars=100)
    assert head in result
    assert tail in result


def test_truncate_empty_string():
    assert _truncate("", max_chars=10) == ""


# ---------------------------------------------------------------------------
# build_reviewer_prompt
# ---------------------------------------------------------------------------


def test_build_reviewer_prompt_includes_all_inputs():
    task = "Fix the null-pointer bug in UserService"
    patch = "--- a/UserService.java\n+++ b/UserService.java\n@@ -1 +1 @@ null check"
    trace = "ERROR: NullPointerException at line 42"
    result = build_reviewer_prompt(task, patch, trace)
    assert task in result
    assert patch in result
    assert trace in result


def test_build_reviewer_prompt_truncates_long_trace():
    task = "task"
    patch = "patch"
    trace = "ERR:" + ("x" * 8000)
    result = build_reviewer_prompt(task, patch, trace)
    assert len(result) < 11000  # task + patch + truncated trace
    assert "elided" in result


def test_build_reviewer_prompt_has_section_headers():
    result = build_reviewer_prompt("task", "patch", "trace")
    assert "## Task" in result
    assert "## Rejected patch" in result
    assert "## Failure trace" in result


# ---------------------------------------------------------------------------
# extract_patch
# ---------------------------------------------------------------------------


def test_extract_from_plain_diff():
    text = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new"
    result = extract_patch(text)
    assert result.startswith("--- a/foo.py")
    assert "old" in result


def test_extract_from_text_with_preamble():
    text = "Here is the corrected patch:\n\n--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-x\n+y"
    result = extract_patch(text)
    assert result.startswith("--- a/file.py")
    assert "x" not in result[:40]  # preamble stripped


def test_extract_from_markdown_fenced_diff():
    text = "```diff\n--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n```"
    result = extract_patch(text)
    assert "--- a/file.py" in result
    assert not result.endswith("```")


def test_extract_from_markdown_fenced_no_lang():
    text = "```\n--- a/file.py\n+++ b/file.py\n```"
    result = extract_patch(text)
    assert "--- a/file.py" in result


def test_extract_no_diff_header_returns_empty():
    text = "Just some plain text without any diff markers"
    result = extract_patch(text)
    assert result == ""


def test_extract_empty_string():
    assert extract_patch("") == ""


# ---------------------------------------------------------------------------
# REVIEWER_SYSTEM_PROMPT
# ---------------------------------------------------------------------------


def test_reviewer_system_prompt_is_non_empty():
    assert len(REVIEWER_SYSTEM_PROMPT) > 0


def test_reviewer_system_prompt_mentions_diff():
    assert "unified diff" in REVIEWER_SYSTEM_PROMPT.lower() or "diff" in REVIEWER_SYSTEM_PROMPT


def test_reviewer_system_prompt_explicitly_forbids_markdown():
    assert "fences" in REVIEWER_SYSTEM_PROMPT or "--- " in REVIEWER_SYSTEM_PROMPT
