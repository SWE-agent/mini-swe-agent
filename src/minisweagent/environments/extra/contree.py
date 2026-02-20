import logging
import platform
import shlex
from dataclasses import asdict, is_dataclass, replace
from numbers import Number
from typing import Any, TypedDict
from urllib.parse import urlparse

from contree_sdk import ContreeSync
from contree_sdk.config import ContreeConfig
from contree_sdk.sdk.exceptions import NotFoundError
from contree_sdk.sdk.objects.image import ContreeImageSync
from pydantic import BaseModel, Field

from minisweagent import Environment
from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge

logger = logging.getLogger(__name__)


class ContreeEnvironmentConfig(BaseModel):
    contree_config: ContreeConfig | dict[str, Any]

    image: str
    image_tag: str = None
    """If set, used to pull image by tag. If fails, then it imports by `image` and sets `image_tag` value to image tag"""
    cwd: str = "/"
    """Working directory in which to execute commands."""
    cwd_auto_create: bool = True
    """Create cwd before running any commands."""
    env: dict[str, str] = Field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = Field(default_factory=list)
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    interpreter: list[str] = Field(default_factory=lambda: ["bash", "-c"])
    """Interpreter to execute commands"""
    timeout: int = 30
    """Timeout for executing commands in the container."""


class ExecutionResult(TypedDict):
    output: str
    returncode: int


class ContreeEnvironment(Environment):
    def __init__(self, *, config_class: type[ContreeEnvironmentConfig] = ContreeEnvironmentConfig, **kwargs):
        """This class executes bash commands in a [ConTree](https://contree.dev) container
        using [contree-sdk](https://github.com/nebius/contree-sdk)"""

        self.config: ContreeEnvironmentConfig = config_class(**kwargs)
        self.logger = logging.getLogger("minisweagent.environment")

        if isinstance(self.config.contree_config, dict):
            self.config = replace(self.config, contree_config=ContreeConfig(**self.config.contree_config))

        self.client = ContreeSync(config=self.config.contree_config)
        self.session = self._pull_image().session()
        if self.config.cwd_auto_create:
            self.execute(
                action={"command": f"mkdir -p {self.config.cwd}"},
                cwd="/",
            )

    def _pull_image(self) -> ContreeImageSync:
        image_tag = self.config.image_tag or ContreeEnvironment.get_tag_by_image_url(self.config.image)
        if image_tag:
            try:
                self.logger.info(f"Pulling image by tag: {image_tag}")
                image = self.client.images.pull(image_tag)
                self.logger.info(f"Pulled image by tag: {image_tag}")
                return image
            except NotFoundError:
                self.logger.warning(
                    f"Failed to pull image by tag: {image_tag}, starting to import from: {self.config.image}"
                )

        self.logger.info(f"Pulling image: {self.config.image}")
        return self.client.images.pull(self.config.image, new_tag=image_tag)

    def _shell_command(self, command: str) -> str:
        shell_cmd = " ".join(self.config.interpreter)
        return f"{shell_cmd} {shlex.quote(command)}"

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the environment and return the raw output."""
        command = action.get("command")
        self.session.run(
            shell=self._shell_command(command),
            cwd=cwd or self.config.cwd,
            timeout=timeout or self.config.timeout,
            disposable=False,
        ).wait()

        cwd = cwd or self.config.cwd
        try:
            self.session.run(
                shell=self._shell_command(command),
                cwd=cwd or self.config.cwd,
                timeout=timeout or self.config.timeout,
                disposable=False,
            ).wait()
            output = {
                "output": self.session.stdout + self.session.stderr,
                "returncode": self.session.exit_code,
                "exception_info": "",
            }
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
            )
            extras = {}
            if is_dataclass(e):
                extras = {k: str(v) if not isinstance(v, Number) else v for k, v in asdict(e).items()}

            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e), **extras},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

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

    @staticmethod
    def get_tag_by_image_url(url: str) -> str:
        url_parsed = urlparse(url)
        if url_parsed.netloc:
            url = url_parsed.path

        if ":" not in url:
            url += ":latest"
        parts = url.split("/", 1)
        if len(parts) == 1:
            return parts[0]
        domain, url_path = parts
        if "." in domain and ("docker" in domain or "io" in domain):
            return url_path or domain
        if domain:
            return f"{domain}/{url_path}"
        return url_path
