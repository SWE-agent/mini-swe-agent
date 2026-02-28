import logging
import os
import shlex
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DockerEnvironmentConfig:
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    """Path to the docker/container executable."""
    run_args: list[str] = field(default_factory=lambda: ["--rm"])
    """Additional arguments to pass to the docker/container executable.
    Default is ["--rm"], which removes the container after it exits.
    """
    container_timeout: str = "2h"
    """Max duration to keep container running. Uses the same format as the sleep command."""
    pull_timeout: int = 120
    """Timeout in seconds for pulling images."""
    container_id: str | None = None
    """Optional container ID to use instead of starting a new container."""


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
        print("Docker Environment Config:", self.config)
        if self.config.container_id is not None:
            self._attach_to_container(self.config.container_id)
        else:
            self._start_container()

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

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

    def _attach_to_container(self, container_id: str):
        """
        Attach to an existing, already-running Docker container
        by its ID or name.
        """
        cmd_check_running = [
            self.config.executable,
            "inspect",
            "-f",
            "{{.State.Running}}",
            container_id
        ]
        
        self.logger.debug(f"Checking status of container {container_id}")
        
        try:
            result_check = subprocess.run(
                cmd_check_running,
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to inspect container {container_id}. Does it exist? Error: {e.stderr}")
            raise RuntimeError(f"Container '{container_id}' not found or docker error.") from e
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout while inspecting container {container_id}.")
            raise RuntimeError(f"Timeout checking container {container_id}.")

        if result_check.stdout.strip() != "true":
            self.logger.error(f"Container {container_id} is not in a 'running' state.")
            raise RuntimeError(f"Container {container_id} is not 'running'. Please start it first.")

        cmd_get_info = [
            self.config.executable,
            "inspect",
            "-f",
            "{{.Id}} {{.Name}}",
            container_id
        ]
        
        result_info = subprocess.run(
            cmd_get_info,
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        
        full_id, name = result_info.stdout.strip().split()
        name = name.lstrip('/')
        self.container_id = full_id
        
        self.logger.info(f"Successfully attached to running container {name} (ID: {self.container_id})")

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the Docker container and return the result as a dict."""
        cwd = cwd or self.config.cwd
        assert self.container_id, "Container not started"

        cmd = [self.config.executable, "exec", "-w", cwd]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.container_id, "bash", "-lc", command])

        result = subprocess.run(
            cmd,
            text=True,
            timeout=timeout or self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}

    def cleanup(self):
        """Stop and remove the Docker container."""
        if getattr(self, "container_id", None) is not None:  # if init fails early, container_id might not be set
            cmd = f"(timeout 60 {self.config.executable} stop {self.container_id} || {self.config.executable} rm -f {self.container_id}) >/dev/null 2>&1 &"
            subprocess.Popen(cmd, shell=True)

    def __del__(self):
        """Cleanup container when object is destroyed."""
        if self.config.container_id is None:
            self.cleanup()