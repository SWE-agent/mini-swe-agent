import asyncio

from pydantic import BaseModel, Field
from swerex.deployment.remote import RemoteDeployment

from minisweagent.environments.extra.swerex_docker import SwerexDockerEnvironment


class SwerexRemoteEnvironmentConfig(BaseModel):
    auth_token: str = Field(exclude=True, repr=False)
    """Token of the `swerex-remote` server. Excluded from dumps so that it doesn't end up in trajectories."""
    host: str = "http://127.0.0.1"
    """Host of the `swerex-remote` server, including the scheme."""
    port: int | None = None
    """Port of the `swerex-remote` server."""
    cwd: str = "/"
    """Working directory in which to execute commands."""
    timeout: int = 30
    """Timeout for executing commands on the server."""
    startup_timeout: float = 10.0
    """How long to wait for the server to respond when connecting."""


class SwerexRemoteEnvironment(SwerexDockerEnvironment):
    def __init__(self, **kwargs):
        """This class executes bash commands on an already running `swerex-remote` server.

        The server is managed by you and is left running when the agent finishes.
        """
        self.config = SwerexRemoteEnvironmentConfig(**kwargs)
        self.deployment = RemoteDeployment(
            auth_token=self.config.auth_token,
            host=self.config.host,
            port=self.config.port,
            timeout=self.config.startup_timeout,
        )
        asyncio.run(self.deployment.start())
        asyncio.run(self.deployment.runtime.wait_until_alive(timeout=self.config.startup_timeout))
