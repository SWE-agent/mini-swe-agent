import json
import platform
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from minisweagent import package_dir
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments import get_environment
from minisweagent.exceptions import Submitted
from minisweagent.models.test_models import DeterministicModel, make_output

TOKEN = "s3cret-token"
SERVER_ERROR_CALLS = []


def record_server_error(message: str) -> Exception:
    SERVER_ERROR_CALLS.append(message)
    return Exception(message)


def _env(port: int | None = None, **kwargs):
    return get_environment({"environment_class": "swerex_remote", "port": port, "startup_timeout": 30.0, **kwargs})


class _FakeServerHandler(BaseHTTPRequestHandler):
    """Reports being alive after `server.delay` seconds and answers commands with a transferred exception."""

    def _send(self, status: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        time.sleep(self.server.delay)
        self._send(200, {"is_alive": True, "message": ""})

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        class_path = f"{__name__}.{record_server_error.__name__}"
        self._send(511, {"swerexception": {"message": "server error", "class_path": class_path}})

    def log_message(self, *args):
        pass


@pytest.fixture(autouse=True)
def auth_token(monkeypatch):
    monkeypatch.setenv("SWEREX_AUTH_TOKEN", TOKEN)


@pytest.fixture
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeServerHandler)
    server.delay = 0.0
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
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
    deadline = time.time() + 30
    while True:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            break
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(0.1)
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


def test_env_is_added_to_server_environment(port):
    assert _env(port, env={"FOO": "a b$c"}).execute({"command": 'echo "$FOO"; command -v bash >/dev/null; echo $?'})[
        "output"
    ] == ("a b$c\n0\n")


def test_interpreter_runs_the_command(port):
    assert _env(port, interpreter=[sys.executable, "-c"]).execute({"command": "print('from python')"})["output"] == (
        "from python\n"
    )


def test_command_timeout_is_reported_and_server_stays_usable(port):
    env = _env(port)
    output = env.execute({"command": "sleep 5"}, timeout=1)
    assert output["returncode"] == -1 and "Timeout" in output["exception_info"]
    assert env.execute({"command": "echo still-alive"})["output"] == "still-alive\n"


def test_submission_marker_raises_submitted(port):
    with pytest.raises(Submitted) as exc_info:
        _env(port).execute({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; echo my-patch"})
    assert exc_info.value.messages[0]["extra"] == {"exit_status": "Submitted", "submission": "my-patch\n"}


def test_mini_config_runs_end_to_end_without_leaking_token(port, tmp_path):
    traj = tmp_path / "traj.json"
    agent = DefaultAgent(
        DeterministicModel(
            outputs=[make_output("done", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; echo my-patch"}])]
        ),
        _env(port),
        **yaml.safe_load((package_dir / "config" / "mini.yaml").read_text())["agent"],
        output_path=traj,
    )
    assert agent.run("task") == {"exit_status": "Submitted", "submission": "my-patch\n"}
    assert platform.system() in str(agent.messages[1]["content"])
    assert platform.machine() in str(agent.messages[1]["content"])
    assert TOKEN not in traj.read_text()
    assert json.loads(traj.read_text())["info"]["config"]["environment_type"] == (
        "minisweagent.environments.extra.swerex_remote.SwerexRemoteEnvironment"
    )


def test_token_stays_out_of_config_and_template_vars(port):
    env = _env(port)
    assert (
        TOKEN not in str(env.serialize())
        and TOKEN not in str(env.get_template_vars())
        and TOKEN not in repr(env.config)
    )


def test_missing_token_fails_at_startup(monkeypatch):
    monkeypatch.delenv("SWEREX_AUTH_TOKEN")
    with pytest.raises(KeyError, match="SWEREX_AUTH_TOKEN"):
        _env(1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"auth_token": TOKEN},
        {"image": "python:3.11"},
        {"prot": 8000},
        {"timeout": 300},
    ],
)
def test_invalid_config_is_rejected_before_connecting(kwargs):
    with pytest.raises(ValidationError):
        _env(1, **kwargs)


def test_wrong_token_fails_at_startup(port, monkeypatch):
    monkeypatch.setenv("SWEREX_AUTH_TOKEN", "wrong")
    with pytest.raises(RuntimeError, match="401"):
        _env(port)


def test_slow_server_connects_within_startup_timeout(fake_server):
    fake_server.delay = 0.3
    assert _env(fake_server.server_address[1], startup_timeout=5.0).config.startup_timeout == 5.0


def test_slow_server_past_startup_timeout_fails(fake_server):
    fake_server.delay = 1.0
    with pytest.raises(RuntimeError):
        _env(fake_server.server_address[1], startup_timeout=0.2)


def test_unreachable_server_fails_at_startup(fake_server):
    port = fake_server.server_address[1]
    fake_server.shutdown()
    fake_server.server_close()
    with pytest.raises(RuntimeError, match="Failed to connect"):
        _env(port, startup_timeout=1.0)


def test_server_cannot_choose_what_the_client_calls(fake_server):
    SERVER_ERROR_CALLS.clear()
    output = _env(fake_server.server_address[1]).execute({"command": "echo hi"})
    assert SERVER_ERROR_CALLS == []
    assert output["returncode"] == -1 and output["output"] == "server error"


def test_server_dying_mid_run_raises(fake_server):
    env = _env(fake_server.server_address[1])
    fake_server.shutdown()
    fake_server.server_close()
    with pytest.raises(RuntimeError, match="unreachable"):
        env.execute({"command": "echo hi"})
