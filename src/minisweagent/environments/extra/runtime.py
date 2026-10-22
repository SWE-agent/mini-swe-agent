import atexit
import hashlib
import logging
import os
import platform
import shlex
import weakref
from typing import Any

from pydantic import BaseModel
from withruntime import Runtime

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge

# Part of the tag an image built from `image` is kept under: Runtime's image builder
# version, so an image built by an older builder is not reused.
_IMAGE_RECIPE = "runtime-builder/8"
_TIMEOUT_EXIT_CODE = 124
_LIVE: "weakref.WeakSet[RuntimeEnvironment]" = weakref.WeakSet()


class RuntimeEnvironmentConfig(BaseModel):
    image: str | None = None
    """Docker image to run in, such as a SWE-bench image. Built once into a Runtime image
    and found again by a tag derived from the reference, so later instances start at once."""
    runtime_image: str | None = None
    """An existing Runtime image (id, name or name:tag) to start from instead of `image`.
    With neither, the sandbox uses Runtime's base image (Ubuntu 24.04 with Python, Node.js and git)."""
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = {}
    """Environment variables to set for every command."""
    forward_env: list[str] = []
    """Environment variables to forward to the sandbox.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 60
    """Timeout in seconds for executing each command."""
    interpreter: list[str] = ["bash", "-lc"]
    """Interpreter to execute commands with. The command is appended as its last argument."""
    user: str | None = "root"
    """User to run commands as. Commands start as the sandbox user (uid 1000) and reach any
    other user through its passwordless sudo. `None` runs them as the sandbox user."""
    vcpu: int | None = None
    """vCPUs for the sandbox. Runtime's default when unset."""
    memory_mib: int | None = None
    """Memory for the sandbox in MiB. Runtime's default when unset."""
    create: dict[str, Any] = {}
    """Any other `Sandbox.create` field, such as `region`, `disk_mib` or `network`."""
    keep_alive_margin: int = 600
    """Seconds of lease kept ahead while the environment is open. The lease ends the sandbox
    by itself if this process dies without cleaning up."""
    api_key: str | None = None
    """Runtime API key. Defaults to `RUNTIME_API_KEY`, then the key `withruntime login` saved."""


class RuntimeEnvironment:
    def __init__(
        self,
        *,
        config_class: type = RuntimeEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """Executes bash commands in a [Runtime](https://withruntime.com) sandbox, a
        Firecracker microVM, using the [withruntime](https://pypi.org/project/withruntime/) SDK.
        See `RuntimeEnvironmentConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self.sandbox = None
        self.client = Runtime(api_key=self.config.api_key) if self.config.api_key else Runtime()
        fields = {
            "vcpu": self.config.vcpu,
            "memory_mib": self.config.memory_mib,
            "image": self._image(),
            "timeout_seconds": max(60, self.config.keep_alive_margin),
            "on_lease_end": "stop",
            "labels": {"created_by": "mini-swe-agent"},
            **self.config.create,
        }
        self.sandbox = self.client.sandboxes.create(**{k: v for k, v in fields.items() if v is not None})
        _LIVE.add(self)
        self.sandbox.keep_alive(margin_seconds=self.config.keep_alive_margin)
        self._sudo = self._probe()

    def _image(self) -> str | None:
        if self.config.runtime_image:
            return self.config.runtime_image
        if not self.config.image:
            return None
        reference = self.config.image.removeprefix("docker://")
        name = "mini-swe-agent-" + reference.rsplit("/", 1)[-1].split(":", 1)[0].split("@", 1)[0]
        name = "".join(c if c.isalnum() or c in "-_." else "-" for c in name)[:100]
        tag = hashlib.sha256(f"{_IMAGE_RECIPE}|{reference}".encode()).hexdigest()[:24]
        try:
            found = self.client.images.resolve(f"{name}:{tag}")
            if found.get("state") == "ready":
                return found["id"]
        except Exception:
            pass
        self.logger.info(f"Building Runtime image {name}:{tag} from {reference}")
        return self.client.images.build(image=reference, name=name, tags=[tag], no_start=True)["id"]

    def _probe(self) -> str:
        """The sudo prefix for commands as `user`, or empty when the sandbox user is that user."""
        user = self.config.user
        if user is None:
            return ""
        found = self.sandbox.exec(["sh", "-c", "id -u; id -un; command -v sudo || true"])
        uid, name, sudo = (found.stdout.splitlines() + ["", "", ""])[:3]
        if user == name or (user == "root" and uid.strip() == "0"):
            return ""
        if not sudo.strip() or self.sandbox.exec(["sudo", "-n", "true"]).exit_code != 0:
            self.cleanup()
            msg = (
                f"Runtime sandbox: commands must run as {user}, but the image has no passwordless sudo for "
                "the sandbox user (uid 1000). Add it to the image, for example: RUN apt-get update && "
                "apt-get install -y sudo && echo '#1000 ALL=(ALL:ALL) NOPASSWD: ALL' > /etc/sudoers.d/runtime, "
                "or set user: null to run commands as the sandbox user."
            )
            raise RuntimeError(msg)
        return f"sudo -n -E -H -u {shlex.quote(user)} -- "

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(exclude={"api_key"}), platform.uname()._asdict(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json", exclude={"api_key"}),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the sandbox and return the raw output."""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd
        env = {k: v for k in self.config.forward_env if (v := os.getenv(k)) is not None}
        env.update(self.config.env)
        inner = shlex.join([*self.config.interpreter, f"cd {shlex.quote(cwd)} && {command}"])
        timeout = timeout or self.config.timeout
        try:
            result = self.sandbox.exec(
                ["sh", "-c", self._sudo + inner], env=env or None, timeout_ms=int(timeout * 1000)
            )
            output_text = result.stdout + result.stderr
            if result.timed_out:
                output = {
                    "output": output_text,
                    "returncode": -1,
                    "exception_info": f"An error occurred while executing the command: timed out after {timeout} seconds",
                    "extra": {"exception_type": "TimeoutExpired", "exception": f"timed out after {timeout} seconds"},
                }
            else:
                code = result.exit_code
                # Runtime reports a command a signal ended as the negative signal number; a shell says 128 + n.
                returncode = -1 if code is None else (128 - code if code < 0 else code)
                output = {"output": output_text, "returncode": returncode, "exception_info": ""}
        except Exception as e:
            output = {
                "output": "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
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

    def cleanup(self):
        """Stop the sandbox. Its lease stops it anyway if this never runs."""
        sandbox, self.sandbox = getattr(self, "sandbox", None), None
        if sandbox is None:
            return
        try:
            sandbox.stop(wait=False)
        except Exception as e:
            self.logger.warning(f"Could not stop Runtime sandbox {sandbox.id}: {e}")

    def __del__(self):
        self.cleanup()


@atexit.register
def _stop_all():
    for environment in list(_LIVE):
        environment.cleanup()
