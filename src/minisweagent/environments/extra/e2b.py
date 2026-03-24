"""E2B cloud sandbox environment implementation."""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import re
from typing import Any

from pydantic import BaseModel, Field


class E2BEnvironmentConfig(BaseModel):
    image: str
    """Docker Hub image name to use as the E2B template base.
    Example: ``'swebench/sweb.eval.x86_64.django__django-11099:latest'``
    """
    cwd: str = "/"
    """Working directory in which to execute commands."""
    timeout: int = 30
    """Timeout for executing commands in the sandbox."""
    env: dict[str, str] = Field(default_factory=dict)
    """Environment variables to set when executing commands."""
    sandbox_timeout: int = 3600
    """How long (in seconds) the sandbox is allowed to stay alive."""

    # Template build options (passed to Template.build())
    cpu_count: int = 2
    """Number of vCPUs allocated to the sandbox."""
    memory_mb: int = 2048
    """Memory allocated to the sandbox in MiB. Default is higher than E2B's 1024 MiB default
    to accommodate larger SWE-bench images."""
    skip_cache: bool = False
    """If True, force-rebuild the template even if it already exists."""
    tags: list[str] = Field(default_factory=list)
    """Optional tags to attach to the template."""
    build_timeout: int = 1800
    """Timeout for template builds in seconds (default 30 min to handle large images)."""

    # E2B authentication (can also be set via E2B_API_KEY / E2B_ACCESS_TOKEN env vars)
    api_key: str | None = None
    """E2B API key. Falls back to the E2B_API_KEY environment variable."""
    access_token: str | None = None
    """E2B access token. Falls back to the E2B_ACCESS_TOKEN environment variable."""

    # Private registry credentials (passed to Template().from_image())
    registry_username: str | None = None
    """Username for authenticating against a private Docker registry."""
    registry_password: str | None = None
    """Password for authenticating against a private Docker registry."""


class E2BTemplateManager:
    """Converts Docker images to E2B templates and manages their lifecycle.

    Can be used independently of :class:`E2BEnvironment` for pre-building
    templates in batch scripts.
    """

    def __init__(self, config: E2BEnvironmentConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("minisweagent.environment.e2b")

    @staticmethod
    def _image_to_template_name(docker_image: str) -> str:
        """Deterministically map a Docker image name to a valid E2B template name.

        A sha256 8-character suffix is appended to avoid collisions between
        images that produce the same sanitized prefix. The result is at most
        63 characters and contains only lower-case alphanumerics and hyphens.

        Example::

            'swebench/sweb.eval.x86_64.django__django-11099:latest'
            → 'swebench-sweb-eval-x86-64-django--django-11099-l-a1b2c3d4'
        """
        hash_suffix = hashlib.sha256(docker_image.encode()).hexdigest()[:8]
        name = re.sub(r"[^a-zA-Z0-9-]", "-", docker_image)
        name = re.sub(r"-{3,}", "--", name)
        name = name.lower()
        # Reserve 9 characters for "-" + 8-char hash suffix → prefix max 54 chars
        prefix = name[:54].strip("-")
        if not prefix:
            return hash_suffix
        return f"{prefix}-{hash_suffix}"

    def get_or_build(self, docker_image: str) -> str:
        """Return the E2B template name for *docker_image*, building it if needed."""
        from e2b import Template

        template_name = self._image_to_template_name(docker_image)
        if not Template.exists(template_name, api_key=self.config.api_key):
            self.logger.info(
                "E2B template %s not found. Starting build (up to %d seconds)...",
                template_name,
                self.config.build_timeout,
            )
            self._build_template(docker_image, template_name)
            self.logger.info("E2B template %s built successfully.", template_name)
        else:
            self.logger.debug("E2B template %s already exists.", template_name)
        return template_name

    def _build_template(self, docker_image: str, template_name: str) -> None:
        """Build an E2B template from *docker_image*.

        Uses :class:`concurrent.futures.ThreadPoolExecutor` for timeout
        enforcement because ``signal.alarm`` only works on the main thread
        and this method may be called from worker threads.
        """
        from e2b import Template

        template = Template().from_image(
            docker_image,
            username=self.config.registry_username,
            password=self.config.registry_password,
        )

        def _do_build() -> None:
            Template.build(
                template,
                template_name,
                cpu_count=self.config.cpu_count,
                memory_mb=self.config.memory_mb,
                skip_cache=self.config.skip_cache,
                tags=self.config.tags or None,
                api_key=self.config.api_key,
                access_token=self.config.access_token,
            )

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_do_build)
        try:
            future.result(timeout=self.config.build_timeout)
        except concurrent.futures.TimeoutError as e:
            executor.shutdown(wait=False, cancel_futures=True)
            msg = f"E2B template build timed out after {self.config.build_timeout}s: {template_name}"
            raise TimeoutError(msg) from e
        except Exception:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)


class E2BEnvironment:
    """Executes bash commands inside an E2B cloud sandbox.

    `E2B <https://e2b.dev>`_ provides isolated cloud sandboxes that can run
    arbitrary Docker images without requiring a local Docker daemon. This
    makes it suitable for large-scale, fully-remote SWE-bench evaluations.

    The first time a Docker image is used it is converted into a persistent
    E2B template; subsequent runs reuse the cached template.

    See :class:`E2BEnvironmentConfig` for keyword arguments.
    """

    def __init__(self, **kwargs: Any) -> None:
        from e2b import Sandbox

        self.logger = logging.getLogger("minisweagent.environment.e2b")
        self.config = E2BEnvironmentConfig(**kwargs)
        manager = E2BTemplateManager(self.config)
        template_name = manager.get_or_build(self.config.image)
        self.logger.info("Creating E2B sandbox (template: %s)...", template_name)
        self.sandbox = Sandbox.create(
            template=template_name,
            timeout=self.config.sandbox_timeout,
            api_key=self.config.api_key,
            access_token=self.config.access_token,
        )
        self.logger.info("E2B sandbox ready (id: %s)", self.sandbox.sandbox_id)

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the sandbox and return the output."""
        command = action.get("command", "") if isinstance(action, dict) else action
        try:
            result = self.sandbox.commands.run(
                command,
                cwd=cwd or self.config.cwd,
                timeout=timeout or self.config.timeout,
                envs=self.config.env or None,
            )
            output: dict[str, Any] = {
                "output": result.stdout + result.stderr,
                "returncode": result.exit_code,
                "exception_info": "",
            }
        except Exception as e:
            output = {
                "output": "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict) -> None:
        """Raise :class:`~minisweagent.exceptions.Submitted` when the task-submission marker is detected."""
        from minisweagent.exceptions import Submitted

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

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        from minisweagent.utils.serialize import recursive_merge

        return recursive_merge(self.config.model_dump(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(
                        mode="json",
                        exclude={"api_key", "access_token", "registry_password"},
                    ),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def stop(self) -> None:
        sandbox = getattr(self, "sandbox", None)
        if sandbox is not None:
            try:
                sandbox.kill()
            except Exception:
                pass

    def __del__(self) -> None:
        self.stop()
