import logging
from dataclasses import asdict, dataclass, field, replace
from typing import Any, TypedDict

from contree_sdk import ContreeSync
from contree_sdk.config import ContreeConfig
from contree_sdk.sdk.exceptions import NotFoundError
from contree_sdk.sdk.objects.image import ContreeImageSync


@dataclass
class ContreeEnvironmentConfig:
    contree_config: ContreeConfig | dict[str, Any]

    image: str
    image_tag: str = None
    """If set, used to pull image by tag. If fails, then it imports by `image` and sets `image_tag` value to image tag"""
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

        self.logger = logging.getLogger("minisweagent.environment")

        if isinstance(self.config.contree_config, dict):
            self.config = replace(self.config, contree_config=ContreeConfig(**self.config.contree_config))

        self.client = ContreeSync(config=self.config.contree_config)
        self.session = self._pull_image().session()
        if self.config.cwd_auto_create:
            self.execute(
                command=f"mkdir -p {self.config.cwd}",
                cwd="/",
            )

    def _pull_image(self) -> ContreeImageSync:
        image_tag = self.config.image_tag or None
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

    @staticmethod
    def get_tag_by_image_url(url: str) -> str:
        if ":" not in url:
            url += ":latest"
        domain, url_path = url.split("/", 1)
        if "." in domain and ("docker" in domain or "io" in domain):
            return url_path or domain
        return domain + "/" + url_path
