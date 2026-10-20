import json
import logging
import math
import os
import platform
import shlex
import subprocess
import uuid
from typing import Any

from pydantic import BaseModel

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge

MARKER = "@@minisweagent-exec-exit@@"

# Runs inside the task container, independently of the docker exec client.
# A new session isolates the shell and descendants. Always kill remaining group
# members, even if the shell exits on TERM while a child ignores it.
# Reserve 64 MiB of cgroup headroom via RLIMIT_AS on the child so a runaway
# allocation inside the command raises MemoryError or is killed before the
# container cgroup OOM kills the supervisor itself.
SUPERVISOR = r"""
import json, os, resource, signal, subprocess, sys, time
from pathlib import Path
seconds, grace = float(sys.argv[1]), float(sys.argv[2])
def _child_limits():
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
            if raw and raw != "max":
                limit = int(raw)
                if 128 * 1024 * 1024 <= limit < (1 << 50):
                    cap = limit - 64 * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
                break
        except Exception:
            pass
child = subprocess.Popen(sys.argv[3:], start_new_session=True, preexec_fn=_child_limits)
def send(sig):
    try:
        os.killpg(child.pid, sig)
    except ProcessLookupError:
        pass
timed_out = False
try:
    child.wait(timeout=seconds)
except subprocess.TimeoutExpired:
    timed_out = True
    send(signal.SIGTERM)
    time.sleep(grace)
finally:
    send(signal.SIGKILL)
child.wait(timeout=1)
print("\n@@minisweagent-exec-exit@@" + json.dumps({"returncode": child.returncode, "timed_out": timed_out}), flush=True)
"""


class DockerEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = {}
    """Environment variables to set in the container."""
    forward_env: list[str] = []
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""
    terminate_grace_seconds: float = 5.0
    """Grace period in seconds between SIGTERM and SIGKILL when a command times out."""
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    """Path to the docker/container executable."""
    run_args: list[str] = ["--rm"]
    """Additional arguments to pass to the docker/container executable.
    Default is ["--rm"], which removes the container after it exits.
    """
    container_timeout: str = "2h"
    """Max duration to keep container running. Uses the same format as the sleep command."""
    pull_timeout: int = 120
    """Timeout in seconds for pulling images."""
    interpreter: list[str] = ["bash", "-lc"]
    """Interpreter to use to execute commands. Default is ["bash", "-lc"].
    The actual command will be appended as argument to this. Override this to e.g., modify shell flags
    (e.g., to remove the `-l` flag to disable login shell) or to use python instead of bash to interpret commands.
    """


class DockerEnvironment:
    def __init__(
        self,
        *,
        config_class: type = DockerEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """This class executes bash commands in a Docker container using direct docker commands.
        See `DockerEnvironmentConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.container_id: str | None = None
        self.config = config_class(**kwargs)
        self._start_container()

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def _start_container(self):
        """Start the Docker container and return the container ID."""
        container_name = f"minisweagent-{uuid.uuid4().hex[:8]}"
        cmd = [
            self.config.executable,
            "run",
            "-d",
            "--name",
            container_name,
            "-w",
            self.config.cwd,
            *self.config.run_args,
            self.config.image,
            "sleep",
            self.config.container_timeout,
        ]
        self.logger.debug(f"Starting container with command: {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.pull_timeout,  # docker pull might take a while
            check=True,
        )
        self.logger.info(f"Started container {container_name} with ID {result.stdout.strip()}")
        self.container_id = result.stdout.strip()

    def _is_container_running(self) -> bool:
        if not self.container_id:
            return False
        try:
            probe = subprocess.run(
                [self.config.executable, "inspect", "-f", "{{.State.Running}}", self.container_id],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return probe.returncode == 0 and probe.stdout.strip() == "true"
        except Exception:
            return False

    def execute(self, action: dict, cwd: str = "", *, timeout: int | float | None = None) -> dict[str, Any]:
        """Execute a command in the Docker container and return the result as a dict."""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd
        seconds = self.config.timeout if timeout is None else timeout
        grace = self.config.terminate_grace_seconds
        if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(grace) or not 0 < grace <= 5:
            raise ValueError("Command timeout must be positive and termination grace in (0, 5]")
        if not self.container_id:
            raise RuntimeError("Task container is not running")

        cmd = [self.config.executable, "exec", "-w", cwd]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend(
            [
                self.container_id,
                "python3",
                "-c",
                SUPERVISOR,
                str(seconds),
                str(grace),
                *self.config.interpreter,
                command,
            ]
        )

        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=seconds + grace + 10,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output, marker, encoded = result.stdout.rpartition("\n" + MARKER)
            if not marker:
                if result.returncode == 137 and self._is_container_running():
                    output = result.stdout
                    status = {"returncode": -9, "timed_out": False, "oom_killed": True}
                else:
                    raise RuntimeError(
                        f"Command supervisor did not confirm termination (exit {result.returncode}): {result.stdout[-1000:]}"
                    )
            else:
                status = json.loads(encoded)
                if not isinstance(status.get("returncode"), int) or not isinstance(status.get("timed_out"), bool):
                    raise RuntimeError("Invalid command supervisor result")
        except Exception as cause:
            try:
                self.cleanup()
            except Exception as cleanup_error:
                raise RuntimeError(f"{cause}; {cleanup_error}") from cause
            raise

        timed_out = status["timed_out"]
        rc = status["returncode"]
        if rc == 137:
            rc = -9
        oom_killed = bool(status.get("oom_killed")) or rc == -9
        observation = {
            "output": output,
            "returncode": -1 if timed_out else rc,
            "exception_info": (
                f"Command deadline exceeded after {seconds}s; process group terminated"
                if timed_out
                else ("Command killed by signal 9 (SIGKILL / container memory limit exceeded)" if oom_killed else "")
            ),
        }
        self._check_finished(observation)
        return observation

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        if output.get("returncode") != 0:
            return
        lines = output.get("output", "").splitlines(keepends=True)
        for index, line in enumerate(lines):
            if line.rstrip("\r\n") == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT":
                submission = "".join(lines[index + 1 :])
                raise Submitted(
                    {
                        "role": "exit",
                        "content": submission,
                        "extra": {"exit_status": "Submitted", "submission": submission},
                    }
                )

    def cleanup(self):
        """Stop and remove the Docker container."""
        container = getattr(self, "container_id", None)
        if container is None:
            return
        try:
            result = subprocess.run(
                [self.config.executable, "rm", "-f", container],
                capture_output=True,
                text=True,
                timeout=5,
            )
            err = result.stderr.lower()
            if result.returncode != 0 and "no such container" not in err and "no container with" not in err:
                raise RuntimeError(f"Task container cleanup failed: {result.stderr.strip()}")
            self.container_id = None
        except Exception as error:
            self.logger.error("Task container cleanup failed for %s: %s", container, error)
            raise

    def __del__(self):
        """Cleanup container when object is destroyed."""
        try:
            self.cleanup()
        except Exception:
            pass
