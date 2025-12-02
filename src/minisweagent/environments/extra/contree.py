from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict

from contree_sdk import ContreeSync
from contree_sdk.config import ContreeConfig


@dataclass
class ContreeEnvironmentConfig:
    contree_config: ContreeConfig

    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    cwd_auto_create: bool = True
    """Create cwd before running any commands."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""


class ExecutionResult(TypedDict):
    output: str
    returncode: int


class ContreeEnvironment:
    def __init__(self, *, config_class: type[ContreeEnvironmentConfig] = ContreeEnvironmentConfig, **kwargs):
        """This class executes bash commands in a Contree container using contree-sdk"""
        self.config: ContreeEnvironmentConfig = config_class(**kwargs)
        self.client = ContreeSync(config=self.config.contree_config)
        self.session = self.client.images.pull(self.config.image).session()
        if self.config.cwd_auto_create:
            self.execute(
                command=f"mkdir -p {self.config.cwd}",
                cwd="/",
            )

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the environment and return the raw output."""
        self.session.run(
            shell=command,
            cwd=cwd or self.config.cwd,
            timeout=timeout or self.config.timeout,
            disposable=False,
        ).wait()

        return {
            "output": self.session.stdout + self.session.stderr,
            "returncode": self.session.exit_code,
        }

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)
