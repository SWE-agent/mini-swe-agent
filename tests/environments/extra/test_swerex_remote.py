import socket
import subprocess
import sys
from pathlib import Path

import pytest

from minisweagent.environments import get_environment
from minisweagent.exceptions import Submitted

TOKEN = "s3cret-token"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _env(port: int, *, auth_token: str = TOKEN, startup_timeout: float = 30.0, **kwargs):
    return get_environment(
        {
            "environment_class": "swerex_remote",
            "port": port,
            "auth_token": auth_token,
            "startup_timeout": startup_timeout,
            **kwargs,
        }
    )


@pytest.fixture(scope="module")
def port():
    port = _free_port()
    server = subprocess.Popen(
        [
            str(Path(sys.executable).parent / "swerex-remote"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--auth-token",
            TOKEN,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    yield port
    server.terminate()
    server.wait(timeout=10)


def test_execute_runs_on_server_and_honors_cwd(port, tmp_path):
    env = _env(port, cwd=str(tmp_path.resolve()))
    assert env.execute({"command": "pwd"}) == {
        "output": f"{tmp_path.resolve()}\n",
        "returncode": 0,
        "exception_info": "",
    }
    assert env.execute({"command": "pwd"}, cwd="/")["output"] == "/\n"
    assert env.execute({"command": "echo out; echo err >&2; exit 3"}) == {
        "output": "out\nerr\n",
        "returncode": 3,
        "exception_info": "",
    }


def test_submission_marker_raises_submitted(port):
    with pytest.raises(Submitted) as exc_info:
        _env(port).execute({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; echo my-patch"})
    assert exc_info.value.messages[0]["extra"] == {"exit_status": "Submitted", "submission": "my-patch\n"}


def test_auth_token_never_reaches_trajectory_or_templates(port):
    env = _env(port)
    serialized = env.serialize()
    assert TOKEN not in str(serialized) and TOKEN not in str(env.get_template_vars()) and TOKEN not in repr(env.config)
    assert serialized["info"]["config"]["environment"]["port"] == port
    assert (
        serialized["info"]["config"]["environment_type"]
        == "minisweagent.environments.extra.swerex_remote.SwerexRemoteEnvironment"
    )


def test_wrong_token_fails_at_startup(port):
    with pytest.raises(TimeoutError):
        _env(port, auth_token="wrong", startup_timeout=1.0)


def test_unreachable_server_fails_at_startup():
    with pytest.raises(TimeoutError):
        _env(_free_port(), startup_timeout=1.0)
