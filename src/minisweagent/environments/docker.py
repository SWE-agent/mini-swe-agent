import logging
import os
import platform
import shlex
import subprocess
import uuid
from typing import Any

from pydantic import BaseModel

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge


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
    terminate_process_group_on_timeout: bool = True
    """Terminate the complete container-side process group after a host-side command timeout.

    This protection currently applies to the default ``bash -lc`` interpreter. Custom interpreters retain
    the previous behavior because mini-swe-agent cannot safely assume how to wrap their command argument.
    """
    termination_grace_period: float = 1.0
    """Seconds to wait between TERM and KILL when cleaning up a timed-out command."""
    cleanup_timeout: int = 15
    """Maximum seconds to wait while stopping or force-removing the task container."""


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

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the Docker container and return the result as a dict."""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd
        assert self.container_id, "Container not started"
        pidfile: str | None = None

        if self.config.terminate_process_group_on_timeout and self.config.interpreter == ["bash", "-lc"]:
            pidfile = f"/tmp/minisweagent-exec-{uuid.uuid4().hex}.pid"
            command = self._wrap_command_with_process_group(command, pidfile)

        cmd = [self.config.executable, "exec", "-w", cwd]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.container_id, *self.config.interpreter, command])

        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
        except Exception as e:
            if isinstance(e, subprocess.TimeoutExpired) and pidfile is not None:
                self._terminate_process_group(pidfile)
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    @staticmethod
    def _wrap_command_with_process_group(command: str, pidfile: str) -> str:
        """Run a command in a new session and persist its process-group ID for timeout cleanup."""
        quoted_command = shlex.quote(command)
        quoted_pidfile = shlex.quote(pidfile)
        return (
            f"setsid bash -lc {quoted_command} & "
            f"child=$!; printf '%s' \"$child\" > {quoted_pidfile}; "
            f"wait \"$child\"; status=$?; rm -f {quoted_pidfile}; exit \"$status\""
        )

    def _terminate_process_group(self, pidfile: str) -> None:
        """Best-effort termination of the process group created for one ``docker exec`` command."""
        if not self.container_id:
            return
        quoted_pidfile = shlex.quote(pidfile)
        grace_period = max(0.0, self.config.termination_grace_period)
        cleanup_script = (
            f"if [ -s {quoted_pidfile} ]; then "
            f"pid=$(cat {quoted_pidfile}); "
            'kill -TERM -- "-$pid" 2>/dev/null || true; '
            f"sleep {grace_period}; "
            'kill -KILL -- "-$pid" 2>/dev/null || true; '
            f"rm -f {quoted_pidfile}; "
            "fi"
        )
        try:
            subprocess.run(
                [self.config.executable, "exec", self.container_id, "bash", "-lc", cleanup_script],
                capture_output=True,
                text=True,
                timeout=max(5.0, grace_period + 3.0),
            )
        except Exception as e:
            self.logger.warning("Failed to terminate timed-out command process group: %s", e)

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        marker = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        marker_index = next((index for index, line in enumerate(lines) if line.strip() == marker), None)
        if marker_index is not None and output["returncode"] == 0:
            submission = "".join(lines[marker_index + 1 :])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def cleanup(self):
        """Stop and remove the Docker container."""
        container_id = getattr(self, "container_id", None)
        if container_id is None:  # if init fails early, container_id might not be set
            return

        # Clear first so an explicit cleanup followed by __del__ is idempotent even if Docker is slow or fails.
        self.container_id = None
        try:
            result = subprocess.run(
                [self.config.executable, "stop", "--time", "10", container_id],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.config.cleanup_timeout,
            )
            if result.returncode == 0:
                return
        except Exception as e:
            self.logger.debug("Graceful container stop failed for %s: %s", container_id, e)

        try:
            subprocess.run(
                [self.config.executable, "rm", "-f", container_id],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.config.cleanup_timeout,
            )
        except Exception as e:
            self.logger.warning("Failed to force-remove container %s: %s", container_id, e)

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()
