"""Tests for trace v2 parsing and classification (Phase 3)."""

from repopilot.trace.classify import build_failure_reason_md, classify_trace
from repopilot.trace.parse import (
    classify_command_stage,
    extract_files_touched,
    extract_patch_diff,
    extract_pytest_log,
    infer_step_stage,
    is_edit_command,
    merge_unified_diffs,
    parse_pytest_failures,
)
from repopilot.trace.parse import ToolCallRecord


SAMPLE_PYTEST_LOG = """\
============================= test session starts ==============================
collected 5 items

upstream/tests/run/test_eval_module.py::test_eval_left_to_right_no_precedence FAILED [ 40%]

=================================== FAILURES ===================================
__________________________ test_eval_basic_arithmetic __________________________

    def test_eval_basic_arithmetic():
>       assert eval_expr("4*5") == 20
E       AssertionError: assert 9.0 == 20

upstream/tests/run/test_eval_module.py:9: AssertionError
========================= 2 failed, 3 passed in 0.02s ==========================
"""

TASK_003_FIX_CMD = """\
python - <<'PY'
from pathlib import Path
p = Path('upstream/src/minisweagent/run/expr/evaluate.py')
text = p.read_text()
old = '        elif op.operator == \"*\":\\n            result += num.number  # BUG: should multiply, not add\\n'
new = '        elif op.operator == \"*\":\\n            result *= num.number\\n'
p.write_text(text.replace(old, new))
PY
"""


def test_classify_command_stage():
    assert classify_command_stage("python -m pytest tests/ -v") == "test"
    assert classify_command_stage("path.write_text('x')") == "edit"
    assert classify_command_stage(TASK_003_FIX_CMD) == "edit"
    assert classify_command_stage("cat upstream/src/foo.py") == "read"
    assert classify_command_stage("git diff") == "submit"


def test_is_edit_command_detects_heredoc_replace():
    assert is_edit_command(TASK_003_FIX_CMD)
    assert not is_edit_command("pytest upstream/tests/run/test_expr.py -v")
    assert not is_edit_command("git diff")


def test_infer_step_stage():
    calls = [
        ToolCallRecord(command="pytest -v", returncode=1),
        ToolCallRecord(command="cat foo.py", returncode=0),
    ]
    assert infer_step_stage(calls) == "test"


def test_extract_files_touched():
    cmd = "path = Path('upstream/src/minisweagent/run/eval_module.py')"
    paths = extract_files_touched(cmd)
    assert "upstream/src/minisweagent/run/eval_module.py" in paths


def test_parse_pytest_failures():
    failures = parse_pytest_failures(SAMPLE_PYTEST_LOG)
    names = {f.test for f in failures}
    assert "test_eval_basic_arithmetic" in names
    arith = next(f for f in failures if f.test == "test_eval_basic_arithmetic")
    assert "9.0" in arith.assertion or "4*5" in arith.assertion
    assert arith.file_line.endswith(":9")


def test_classify_trace_success():
    result = classify_trace(
        steps=[{"step": 1, "tool_calls": [{"command": "pytest", "returncode": 0}]}],
        pytest_runs=[{"phase": "post_fix", "returncode": 0, "summary": "3 passed"}],
        patch={"text": "diff --git", "source": "test"},
        exit_status="Submitted",
        tests_passed=True,
    )
    assert result["outcome"] == "success"
    assert result["failure_category"] is None


def test_classify_trace_tests_still_failing():
    result = classify_trace(
        steps=[
            {"step": 1, "tool_calls": [{"command": "pytest -v", "returncode": 1}]},
        ],
        pytest_runs=[
            {
                "phase": "pre_fix",
                "returncode": 1,
                "summary": "1 failed",
                "failed_tests": [{"test": "test_foo", "assertion": "AssertionError"}],
            }
        ],
        patch={"text": "", "source": "none"},
        exit_status="Submitted",
        tests_passed=False,
    )
    assert result["outcome"] == "failure"
    assert result["failure_category"] == "tests_still_failing"
    assert result["failure_stage"] == "test"


def test_classify_trace_wrong_file_edited():
    result = classify_trace(
        steps=[
            {"step": 1, "tool_calls": [{"command": "pytest -v", "returncode": 1}]},
            {
                "step": 2,
                "tool_calls": [
                    {
                        "command": "p = Path('upstream/src/minisweagent/run/expr/tokenize.py'); p.write_text('x')",
                        "returncode": 0,
                    }
                ],
                "files_touched": ["upstream/src/minisweagent/run/expr/tokenize.py"],
            },
            {"step": 3, "tool_calls": [{"command": "pytest -v", "returncode": 1}]},
        ],
        pytest_runs=[
            {
                "phase": "pre_fix",
                "returncode": 1,
                "summary": "1 failed",
                "log": SAMPLE_PYTEST_LOG,
                "failed_tests": [
                    {
                        "test": "test_eval_basic_arithmetic",
                        "file_line": "upstream/tests/run/test_eval_module.py:9",
                    }
                ],
            },
            {"phase": "post_fix", "returncode": 1, "summary": "1 failed", "log": SAMPLE_PYTEST_LOG},
        ],
        patch={"text": "diff --git", "source": "test"},
        exit_status="Submitted",
        tests_passed=False,
    )
    assert result["failure_category"] == "wrong_file_edited"
    assert result["failure_stage"] == "edit"
    assert "tokenize.py" in result["failure_message"]


def test_classify_trace_replace_edit_not_exited_early():
    """Heredoc replace edits should not be classified as agent_exited_early."""
    result = classify_trace(
        steps=[
            {"step": 1, "tool_calls": [{"command": "pytest -v", "returncode": 1}]},
            {"step": 2, "tool_calls": [{"command": TASK_003_FIX_CMD, "returncode": 0}]},
            {"step": 3, "tool_calls": [{"command": "git diff", "returncode": 0}]},
            {"step": 4, "tool_calls": [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "returncode": -1}]},
        ],
        pytest_runs=[
            {"phase": "pre_fix", "returncode": 1, "summary": "2 failed"},
            {"phase": "post_fix", "returncode": 0, "summary": "5 passed"},
        ],
        patch={"text": "diff --git a/upstream/foo.py", "source": "test"},
        exit_status="Submitted",
        tests_passed=True,
    )
    assert result["outcome"] == "success"
    assert result["failure_category"] is None


def test_classify_trace_submitted_with_replace_edit_unknown_not_early():
    """Without verify context, replace edits must not trigger agent_exited_early."""
    result = classify_trace(
        steps=[
            {"step": 2, "tool_calls": [{"command": TASK_003_FIX_CMD, "returncode": 0}]},
            {"step": 4, "tool_calls": [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "returncode": -1}]},
        ],
        pytest_runs=[
            {"phase": "pre_fix", "returncode": 1, "summary": "2 failed"},
            {"phase": "post_fix", "returncode": 0, "summary": "5 passed"},
        ],
        patch={"text": "", "source": "not found"},
        exit_status="Submitted",
        tests_passed=None,
    )
    assert result["failure_category"] != "agent_exited_early"


def test_build_failure_reason_md():
    trace = {
        "task_id": "task_test",
        "outcome": "failure",
        "failure_category": "tests_still_failing",
        "failure_stage": "test",
        "failed_step": 1,
        "failure_message": "1 failed",
        "metrics": {"tests_passed": False},
        "pytest_runs": [
            {
                "phase": "pre_fix",
                "returncode": 1,
                "failed_tests": [{"test": "test_foo", "assertion": "AssertionError: x"}],
            }
        ],
        "steps": [{"step": 1, "stage": "test", "files_touched": ["upstream/foo.py"], "reasoning": "run tests"}],
    }
    md = build_failure_reason_md(trace)
    assert "# Failure Reason" in md
    assert "test_foo" in md
    assert "AssertionError" in md


TRUNCATED_PROGRESS_LOG = """\
============================= test session starts ==============================
collected 5 items

upstream/tests/run/test_eval_module.py::test_eval_basic_arithmetic FAILED [ 20%]
upstream/tests/run/test_eval_module.py::test_eval_left_to_right_no_precedence FAILED [ 40%]
upstream/tests/run/test_eval_module.py::test_eval_whitespace_and_decimals PASSED [ 60%]
"""


def test_parse_pytest_failures_from_progress_lines_only():
    failures = parse_pytest_failures(TRUNCATED_PROGRESS_LOG)
    names = {f.test for f in failures}
    assert names == {"test_eval_basic_arithmetic", "test_eval_left_to_right_no_precedence"}
    assert "20%]" not in names


def test_extract_pytest_log_includes_failures_section():
    mixed = "patched upstream/src/foo.py\n" + SAMPLE_PYTEST_LOG
    log = extract_pytest_log(mixed)
    assert "FAILURES" in log
    assert "test_eval_basic_arithmetic" in log
    assert "2 failed, 3 passed" in log
    assert not log.startswith("patched")


def test_extract_patch_from_replace_command():
    patch, source = extract_patch_diff([(TASK_003_FIX_CMD, "", 0)])
    assert source == "reconstructed from edit command"
    assert "upstream/src/minisweagent/run/expr/evaluate.py" in patch
    assert "result += num.number" in patch
    assert "result *= num.number" in patch
    assert "sudoku" not in patch.lower()


def test_merge_unified_diffs_keeps_latest_block_per_file():
    first = (
        "diff --git a/upstream/src/foo.py b/upstream/src/foo.py\n"
        "--- a/upstream/src/foo.py\n"
        "+++ b/upstream/src/foo.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+first\n"
    )
    second = (
        "diff --git a/upstream/src/foo.py b/upstream/src/foo.py\n"
        "--- a/upstream/src/foo.py\n"
        "+++ b/upstream/src/foo.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+second\n"
    )
    other = (
        "diff --git a/upstream/src/bar.py b/upstream/src/bar.py\n"
        "--- a/upstream/src/bar.py\n"
        "+++ b/upstream/src/bar.py\n"
        "@@ -1 +1 @@\n"
        "-x\n"
        "+y\n"
    )
    merged = merge_unified_diffs(first, second, other)
    assert merged.count("diff --git") == 2
    assert "+second" in merged
    assert "+first" not in merged
    assert "upstream/src/bar.py" in merged


def test_extract_patch_merges_multiple_git_diff_outputs():
    tokenize = (
        "diff --git a/upstream/src/minisweagent/run/expr/tokenize.py "
        "b/upstream/src/minisweagent/run/expr/tokenize.py\n"
        "--- a/upstream/src/minisweagent/run/expr/tokenize.py\n"
        "+++ b/upstream/src/minisweagent/run/expr/tokenize.py\n"
        "@@ -1 +1 @@\n"
        "-bug\n"
        "+fix-tokenize\n"
    )
    evaluate = (
        "diff --git a/upstream/src/minisweagent/run/expr/evaluate.py "
        "b/upstream/src/minisweagent/run/expr/evaluate.py\n"
        "--- a/upstream/src/minisweagent/run/expr/evaluate.py\n"
        "+++ b/upstream/src/minisweagent/run/expr/evaluate.py\n"
        "@@ -1 +1 @@\n"
        "-bug\n"
        "+fix-evaluate\n"
    )
    rows = [
        ("git diff -- tokenize.py", tokenize, 0),
        ("git diff -- evaluate.py", evaluate, 0),
    ]
    patch, source = extract_patch_diff(rows)
    assert source == "git diff in trajectory (merged)"
    assert "fix-tokenize" in patch
    assert "fix-evaluate" in patch
