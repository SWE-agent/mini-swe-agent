import asyncio
import os
import shlex
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from swerex.exceptions import SwerexException
from swerex.runtime.remote import RemoteRuntime

from minisweagent.environments.extra.swerex_docker import SwerexDockerEnvironment


class SwerexRemoteEnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "http://127.0.0.1"
    """Host of the `swerex-remote` server, including the scheme."""
    port: int | None = 8000
    """Port of the `swerex-remote` server. Set to `None` to use the default port of the scheme."""
    cwd: str = "/"
    """Working directory in which to execute commands."""
    timeout: int = Field(30, lt=300)
    """Timeout for executing commands. Must be below SWE-ReX's fixed 300s limit for client requests."""
    env: dict[str, str] = {}
    """Environment variables to add when executing commands."""
    interpreter: list[str] = ["bash", "-c"]
    """Interpreter to use to execute commands."""
    startup_timeout: float = 10.0
    """How long a single check whether the server is alive may take."""


class _SafeRemoteRuntime(RemoteRuntime):
    def _handle_transfer_exception(self, exc_transfer) -> None:
        # Only trust the message: the class path comes from the server.
        raise SwerexException(exc_transfer.message) from None


class SwerexRemoteEnvironment(SwerexDockerEnvironment):
    def __init__(self, **kwargs):
        """This class executes bash commands on an already running `swerex-remote` server.

        The server is managed by you and is left running when the agent finishes.
        The token of the server is read from the `SWEREX_AUTH_TOKEN` environment variable.
        """
        self.config = SwerexRemoteEnvironmentConfig(**kwargs)
        self.deployment = SimpleNamespace(
            runtime=_SafeRemoteRuntime(
                auth_token=os.environ["SWEREX_AUTH_TOKEN"],
                host=self.config.host,
                port=self.config.port,
                timeout=self.config.startup_timeout,
            )
        )
        if not (alive := asyncio.run(self.deployment.runtime.is_alive())):
            raise RuntimeError(alive.message)
        uname = super().execute({"command": "uname -s; uname -r; uname -v; uname -m"})
        if uname["returncode"] != 0:
            raise RuntimeError(f"swerex-remote server cannot run commands: {uname['output']}")
        self._uname = dict(zip(("system", "release", "version", "machine"), uname["output"].splitlines()))

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command on the server. Raises if the server can no longer be reached."""
        env = [f"{key}={value}" for key, value in self.config.env.items()]
        command = shlex.join(["env", *env, *self.config.interpreter, action.get("command", "")])
        output = super().execute({"command": command}, cwd, timeout=timeout)
        if "extra" in output and not asyncio.run(self.deployment.runtime.is_alive()):
            raise RuntimeError(f"swerex-remote server unreachable: {output['exception_info']}")
        return output

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return super().get_template_vars(**(self._uname | kwargs))
