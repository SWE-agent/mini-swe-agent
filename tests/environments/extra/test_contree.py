import os
import shlex
from unittest.mock import MagicMock, patch

import pytest
from contree_sdk.config import ContreeConfig

from minisweagent.environments.extra.contree import (
    ContreeEnvironment,
    ContreeEnvironmentConfig,
)
from minisweagent.exceptions import Submitted


def _make_env(**kwargs) -> ContreeEnvironment:
    """Create a ContreeEnvironment with mocked __init__ (no ConTree infra)."""
    with patch.object(ContreeEnvironment, "__init__", lambda self, **kw: None):
        env = ContreeEnvironment()
        env.config = ContreeEnvironmentConfig(
            contree_config=ContreeConfig(base_url="http://fake", token="fake-token"),
            image="python:3.11",
            **kwargs,
        )
        env.session = MagicMock()
        return env


def _setup_session(env: ContreeEnvironment, stdout: str = "", stderr: str = "", exit_code: int = 0):
    env.session.stdout = stdout
    env.session.stderr = stderr
    env.session.exit_code = exit_code
    env.session.run.return_value = env.session


def test_execute_passes_correct_args_to_sdk():
    """Test that execute() passes correct shell, cwd, timeout, disposable, and env to session.run()."""
    env = _make_env(cwd="/workspace", timeout=42)
    _setup_session(env, stdout="hello\n", exit_code=0)

    result = env.execute({"command": "echo hello"})

    env.session.run.assert_called_once()
    call_kwargs = env.session.run.call_args.kwargs
    assert call_kwargs["shell"] == f"bash -c {shlex.quote('echo hello')}"
    assert call_kwargs["cwd"] == "/workspace"
    assert call_kwargs["timeout"] == 42
    assert call_kwargs["disposable"] is False
    assert call_kwargs["env"] is None
    assert result["output"] == "hello\n"
    assert result["returncode"] == 0
    assert result["exception_info"] == ""


def test_execute_combines_stdout_and_stderr():
    """Test that execute() combines stdout and stderr into a single output string."""
    env = _make_env()
    _setup_session(env, stdout="out\n", stderr="err\n", exit_code=0)

    result = env.execute({"command": "cmd"})

    assert result["output"] == "out\nerr\n"


def test_execute_exception():
    """Test that SDK exceptions are captured and returned as error output."""
    env = _make_env()
    env.session.run.side_effect = RuntimeError("connection lost")

    result = env.execute({"command": "echo hello"})

    assert result["returncode"] == -1
    assert "connection lost" in result["exception_info"]
    assert result["extra"]["exception_type"] == "RuntimeError"


def test_execute_raises_submitted():
    """Test that execute() raises Submitted when output contains the submission marker."""
    env = _make_env()
    _setup_session(
        env,
        stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndiff --git a/f b/f\n",
        exit_code=0,
    )

    with pytest.raises(Submitted) as exc_info:
        env.execute({"command": "submit"})

    msg = exc_info.value.messages[0]
    assert msg["extra"]["exit_status"] == "Submitted"
    assert "diff --git" in msg["extra"]["submission"]


def test_execute_no_submit_on_nonzero_returncode():
    """Test that the submission marker is ignored when the command fails."""
    env = _make_env()
    _setup_session(env, stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", exit_code=1)

    result = env.execute({"command": "false"})
    assert result["returncode"] == 1


def test_execute_cwd_parameter_overrides_config():
    """Test that the cwd parameter in execute() overrides the config cwd."""
    env = _make_env(cwd="/workspace")
    _setup_session(env)

    env.execute({"command": "pwd"}, cwd="/tmp")

    assert env.session.run.call_args.kwargs["cwd"] == "/tmp"


def test_execute_env_passed_to_sdk():
    """Test that env variables from config are passed to session.run()."""
    env = _make_env(env={"MY_VAR": "my_value"})
    _setup_session(env)

    env.execute({"command": "echo $MY_VAR"})

    assert env.session.run.call_args.kwargs["env"] == {"MY_VAR": "my_value"}


def test_execute_forward_env_passed_to_sdk():
    """Test that forwarded env variables from the host are passed to session.run()."""
    env = _make_env(forward_env=["HOST_VAR"])
    _setup_session(env)

    with patch.dict(os.environ, {"HOST_VAR": "host_value"}):
        env.execute({"command": "echo $HOST_VAR"})

    assert env.session.run.call_args.kwargs["env"] == {"HOST_VAR": "host_value"}


def test_execute_forward_env_missing_var_passes_none():
    """Test that missing host env variables result in env=None passed to session.run()."""
    env = _make_env(forward_env=["NONEXISTENT_VAR"])
    _setup_session(env)

    env.execute({"command": "echo $NONEXISTENT_VAR"})

    assert env.session.run.call_args.kwargs["env"] is None


def test_execute_env_overrides_forward_env():
    """Test that explicitly set env variables take precedence over forwarded host variables."""
    env = _make_env(env={"CONFLICT": "from_config"}, forward_env=["CONFLICT"])
    _setup_session(env)

    with patch.dict(os.environ, {"CONFLICT": "from_host"}):
        env.execute({"command": "echo $CONFLICT"})

    assert env.session.run.call_args.kwargs["env"]["CONFLICT"] == "from_config"
