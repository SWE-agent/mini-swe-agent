"""Tests for minisweagent.patch_repair.repair — requires mocking env + model."""

import base64
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.patch_repair.repair import (
    _call_reviewer_lm,
    attempt_patch_repair,
    evaluate_patch,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_env(side_effect=None):
    """Return a mock Environment whose ``execute()`` returns *side_effect* items."""
    env = MagicMock()
    env.execute.side_effect = side_effect or []
    return env


def _make_model():
    """Return a minimal mock Model for reviewer calls."""
    model = MagicMock()
    model.format_message.side_effect = lambda **kw: dict(kw)
    model._prepare_messages_for_api.side_effect = lambda msgs: [
        {k: v for k, v in m.items() if k != "extra"} for m in msgs
    ]
    model.config.model_name = "test-model"
    model.config.model_kwargs = {"temperature": 0.0}
    model.abort_exceptions = []
    return model


def _litellm_response(content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
    """Build a mock litellm completion response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


# ---------------------------------------------------------------------------
# evaluate_patch
# ---------------------------------------------------------------------------


SAMPLE_PATCH = "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"
SAMPLE_B64 = base64.b64encode(SAMPLE_PATCH.encode()).decode()
RESET_CMD = "git reset --hard HEAD"
WRITE_CMD = f"echo '{SAMPLE_B64}' | base64 -d > /tmp/repair.diff"
CHECK_CMD = "git apply --check /tmp/repair.diff"
APPLY_CMD = "git apply /tmp/repair.diff"


def _patch_instance(**overrides) -> dict:
    """Return a minimal SWE-bench instance dict for evaluate_patch."""
    inst = {"FAIL_TO_PASS": []}
    inst.update(overrides)
    return inst


def _ok_output(output: str = "") -> dict:
    return {"output": output, "returncode": 0, "exception_info": ""}


def _fail_output(output: str = "error") -> dict:
    return {"output": output, "returncode": 1, "exception_info": ""}


class TestEvaluatePatch:
    def test_git_reset_fails(self):
        env = _make_env([_fail_output("fatal: unable to reset")])
        apply_ok, test_ok, trace = evaluate_patch(SAMPLE_PATCH, env, _patch_instance())
        assert not apply_ok
        assert not test_ok
        assert "fatal" in trace

    def test_write_patch_fails(self):
        env = _make_env([_ok_output(), _fail_output("no space left")])
        apply_ok, test_ok, trace = evaluate_patch(SAMPLE_PATCH, env, _patch_instance())
        assert not apply_ok
        assert "no space" in trace

    def test_apply_check_fails(self):
        env = _make_env([_ok_output(), _ok_output(), _fail_output("patch does not apply")])
        apply_ok, test_ok, trace = evaluate_patch(SAMPLE_PATCH, env, _patch_instance())
        assert not apply_ok
        assert "patch does not apply" in trace

    def test_apply_fails_but_check_passes(self):
        """git apply --check passes but git apply fails → apply_ok=True, test_ok=False."""
        env = _make_env([
            _ok_output(),  # reset
            _ok_output(),  # write patch
            _ok_output(),  # check
            _fail_output("error: patch failed"),  # apply
        ])
        apply_ok, test_ok, trace = evaluate_patch(SAMPLE_PATCH, env, _patch_instance())
        assert apply_ok
        assert not test_ok

    def test_apply_ok_no_tests(self):
        env = _make_env([
            _ok_output(),  # reset
            _ok_output(),  # write
            _ok_output(),  # check
            _ok_output(),  # apply
        ])
        apply_ok, test_ok, _ = evaluate_patch(SAMPLE_PATCH, env, _patch_instance())
        assert apply_ok
        assert test_ok  # no tests → automatic pass

    def test_apply_ok_tests_pass(self):
        env = _make_env([
            _ok_output(),  # reset
            _ok_output(),  # write
            _ok_output(),  # check
            _ok_output(),  # apply
            _ok_output("4 passed"),  # pytest
        ])
        apply_ok, test_ok, _ = evaluate_patch(
            SAMPLE_PATCH, env, _patch_instance(FAIL_TO_PASS=["tests/test_a.py::test_x"])
        )
        assert apply_ok
        assert test_ok

    def test_apply_ok_tests_fail(self):
        env = _make_env([
            _ok_output(),  # reset
            _ok_output(),  # write
            _ok_output(),  # check
            _ok_output(),  # apply
            _fail_output("1 failed, 3 passed"),  # pytest
        ])
        apply_ok, test_ok, trace = evaluate_patch(
            SAMPLE_PATCH, env, _patch_instance(FAIL_TO_PASS=["tests/test_x.py::test_fail"])
        )
        assert apply_ok
        assert not test_ok
        assert "1 failed" in trace

    def test_multiple_fail_to_pass_tests(self):
        """FAIL_TO_PASS with multiple entries are joined into one pytest invocation."""
        cmds_seen = []

        def _record(cmd: dict) -> dict:
            cmds_seen.append(cmd["command"])
            return _ok_output("all passed")

        env = _make_env()
        env.execute.side_effect = _record
        evaluate_patch(
            SAMPLE_PATCH,
            env,
            _patch_instance(FAIL_TO_PASS=["tests/a.py::t1", "tests/b.py::t2"]),
        )
        pytest_calls = [c for c in cmds_seen if "pytest" in c]
        assert len(pytest_calls) == 1
        assert "tests/a.py::t1" in pytest_calls[0]
        assert "tests/b.py::t2" in pytest_calls[0]

    def test_trace_includes_exception_info(self):
        env = _make_env([{"output": "", "returncode": -1, "exception_info": "timeout"}])
        apply_ok, test_ok, trace = evaluate_patch(SAMPLE_PATCH, env, _patch_instance())
        assert not apply_ok
        assert "timeout" in trace


# ---------------------------------------------------------------------------
# _call_reviewer_lm
# ---------------------------------------------------------------------------


class TestCallReviewerLM:
    def test_returns_content_on_success(self):
        model = _make_model()
        mock_resp = _litellm_response("--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new")
        with patch("minisweagent.patch_repair.repair.litellm.completion", return_value=mock_resp):
            with patch("minisweagent.patch_repair.repair.litellm.cost_calculator.completion_cost", return_value=0.01):
                result = _call_reviewer_lm(model, "task", "patch", "trace")
        assert result is not None
        assert "--- a/file.py" in result

    def test_returns_none_on_auth_error(self):
        model = _make_model()
        from litellm.exceptions import AuthenticationError

        with patch("minisweagent.patch_repair.repair.litellm.completion", side_effect=AuthenticationError("bad key", "test-provider", "test-model")):
            result = _call_reviewer_lm(model, "task", "patch", "trace")
        assert result is None

    def test_returns_none_on_generic_error(self):
        model = _make_model()
        with patch("minisweagent.patch_repair.repair.litellm.completion", side_effect=RuntimeError("boom")):
            result = _call_reviewer_lm(model, "task", "patch", "trace")
        assert result is None

    def test_empty_content_returns_empty_string(self):
        model = _make_model()
        mock_resp = _litellm_response("")
        with patch("minisweagent.patch_repair.repair.litellm.completion", return_value=mock_resp):
            with patch("minisweagent.patch_repair.repair.litellm.cost_calculator.completion_cost", return_value=0.0):
                result = _call_reviewer_lm(model, "task", "patch", "trace")
        assert result == ""

    def test_filters_tool_only_kwargs(self):
        """Verify parallel_tool_calls and tool_choice are stripped from the API call."""
        model = _make_model()
        model.config.model_kwargs = {"temperature": 0.0, "parallel_tool_calls": True, "tool_choice": "auto"}
        mock_resp = _litellm_response("ok")
        with patch("minisweagent.patch_repair.repair.litellm.completion", return_value=mock_resp) as mock_completion:
            with patch("minisweagent.patch_repair.repair.litellm.cost_calculator.completion_cost", return_value=0.01):
                _call_reviewer_lm(model, "task", "patch", "trace")
            call_kwargs = mock_completion.call_args.kwargs
        assert "parallel_tool_calls" not in call_kwargs
        assert "tool_choice" not in call_kwargs
        assert call_kwargs["temperature"] == 0.0

    def test_cost_tracking_adds_to_global_stats(self):
        model = _make_model()
        mock_resp = _litellm_response("fix")
        with patch("minisweagent.patch_repair.repair.litellm.completion", return_value=mock_resp):
            with patch("minisweagent.patch_repair.repair.litellm.cost_calculator.completion_cost", return_value=0.05):
                with patch("minisweagent.patch_repair.repair.GLOBAL_MODEL_STATS") as mock_stats:
                    _call_reviewer_lm(model, "task", "patch", "trace")
                    mock_stats.add.assert_called_once_with(0.05)


# ---------------------------------------------------------------------------
# attempt_patch_repair
# ---------------------------------------------------------------------------


class TestAttemptPatchRepair:
    def test_patch_passes_first_round(self):
        """Patch applies + tests pass → returns immediately with success."""
        with patch("minisweagent.patch_repair.repair.evaluate_patch", return_value=(True, True, "all good")):
            result = attempt_patch_repair("task", "patch", MagicMock(), MagicMock(), {})
        assert result["success"]
        assert result["rounds_used"] == 1
        assert result["patch"] == "patch"
        assert len(result["trace"]) == 1

    def test_reviewer_fixes_patch_on_second_round(self):
        """Patch fails first round, reviewer fixes it, second round passes."""
        eval_results = [(False, False, "apply error"), (True, True, "fixed")]
        with patch(
            "minisweagent.patch_repair.repair.evaluate_patch",
            side_effect=eval_results,
        ):
            with patch(
                "minisweagent.patch_repair.repair._call_reviewer_lm",
                return_value="--- a/fixed.py\n+++ b/fixed.py\n-old\n+new",
            ):
                result = attempt_patch_repair("task", "bad patch", MagicMock(), MagicMock(), {})
        assert result["success"]
        assert result["rounds_used"] == 2
        assert result["patch"] == "--- a/fixed.py\n+++ b/fixed.py\n-old\n+new"

    def test_max_rounds_exhausted(self):
        """Patch keeps failing → returns failure after max_rounds."""
        with patch(
            "minisweagent.patch_repair.repair.evaluate_patch",
            return_value=(False, False, "nope"),
        ):
            with patch(
                "minisweagent.patch_repair.repair._call_reviewer_lm",
                return_value="--- a/another.py\n+++ b/another.py",
            ):
                result = attempt_patch_repair("task", "bad", MagicMock(), MagicMock(), {}, max_rounds=2)
        assert not result["success"]
        assert result["rounds_used"] == 2
        assert len(result["trace"]) == 2

    def test_reviewer_call_fails_stops_loop(self):
        """When reviewer LM returns None, repair stops."""
        eval_results = [(False, False, "error")]  # only first round runs
        with patch(
            "minisweagent.patch_repair.repair.evaluate_patch",
            side_effect=eval_results,
        ):
            with patch("minisweagent.patch_repair.repair._call_reviewer_lm", return_value=None):
                result = attempt_patch_repair("task", "bad", MagicMock(), MagicMock(), {}, max_rounds=2)
        assert not result["success"]
        assert result["rounds_used"] == 1
        assert result["patch"] == "bad"  # original patch returned

    def test_extraction_fails_stops_loop(self):
        """Reviewer output has no diff → stops."""
        with patch(
            "minisweagent.patch_repair.repair.evaluate_patch",
            return_value=(False, False, "error"),
        ):
            with patch(
                "minisweagent.patch_repair.repair._call_reviewer_lm",
                return_value="no diff here, just text",
            ):
                result = attempt_patch_repair("task", "bad", MagicMock(), MagicMock(), {}, max_rounds=2)
        assert not result["success"]
        assert result["trace"][0]["extraction_failed"]

    def test_max_rounds_one_is_evaluate_only(self):
        """max_rounds=1 means evaluate the original patch, never call reviewer."""
        with patch(
            "minisweagent.patch_repair.repair.evaluate_patch",
            return_value=(False, False, "fail"),
        ) as mock_eval:
            reviewer_called = []

            def _fake_reviewer(*args, **kwargs):
                reviewer_called.append(1)
                return "fixed"

            with patch("minisweagent.patch_repair.repair._call_reviewer_lm", side_effect=_fake_reviewer):
                result = attempt_patch_repair("task", "patch", MagicMock(), MagicMock(), {}, max_rounds=1)
        assert not result["success"]
        assert result["rounds_used"] == 1
        assert len(reviewer_called) == 0

    def test_trace_contains_metadata_per_round(self):
        with patch(
            "minisweagent.patch_repair.repair.evaluate_patch",
            side_effect=[(False, False, "err1"), (True, True, "ok")],
        ):
            with patch(
                "minisweagent.patch_repair.repair._call_reviewer_lm",
                return_value="--- a/x.py\n+++ b/x.py",
            ):
                result = attempt_patch_repair("task", "patch", MagicMock(), MagicMock(), {})
        assert len(result["trace"]) == 2
        r1 = result["trace"][0]
        assert r1["round"] == 1
        assert not r1["apply_ok"]
        assert r1["reviewer_called"] is True
        r2 = result["trace"][1]
        assert r2["round"] == 2
        assert r2["apply_ok"]
        assert r2["test_ok"]
        assert r2["reviewer_called"] is False

    def test_empty_patch_is_handled(self):
        """Empty string patch should not crash — evaluate_patch handles it."""
        with patch(
            "minisweagent.patch_repair.repair.evaluate_patch",
            return_value=(False, False, "empty patch"),
        ):
            result = attempt_patch_repair("task", "", MagicMock(), MagicMock(), {}, max_rounds=1)
        assert not result["success"]
        assert result["patch"] == ""
