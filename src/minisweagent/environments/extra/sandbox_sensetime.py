"""Sandbox environment using SenseTime Sandbox API."""

import base64
import hashlib
import hmac
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests


@dataclass
class SandboxSensetimeEnvironmentConfig:
    image: str
    """Image name or ID to use."""
    workspace_id: str
    """Workspace ID for Sandbox API."""
    cwd: str = "/testbed"
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
    base_url: str = os.getenv("SANDBOX_API_BASE_URL", "https://sandbox.example.com")
    """Base URL for Sandbox API."""
    access_key_id: str = os.getenv("SANDBOX_ACCESS_KEY_ID", "")
    """Access key ID for HMAC authentication."""
    access_key_secret: str = os.getenv("SANDBOX_ACCESS_KEY_SECRET", "")
    """Access key secret for HMAC authentication."""
    resource_limits: dict[str, int] = field(default_factory=lambda: {"cpu": 2, "memory": 2, "storage": 1})
    """Resource limits for snapshot creation."""
    auto_delete_after: int = 7200
    """Auto delete time in seconds. -1 means no auto delete."""
    pull_timeout: int = 120
    """Timeout in seconds for creating snapshots."""


class SandboxSensetimeEnvironment:
    def __init__(
        self,
        *,
        config_class: type = SandboxSensetimeEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """This class executes bash commands in a Sandbox container using SenseTime Sandbox API.
        See `SandboxSensetimeEnvironmentConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self.sandbox_id: str | None = None
        self._start_sandbox()

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def _generate_auth_headers(self) -> dict[str, str]:
        """Generate HMAC-SHA256 authentication headers."""
        date_now = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
        sign_content = f"x-date: {date_now}"
        signature = hmac.new(
            self.config.access_key_secret.encode("utf-8"),
            sign_content.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        base64_signature = base64.b64encode(signature).decode("utf-8")
        token = (
            f'hmac accesskey="{self.config.access_key_id}", algorithm="hmac-sha256", '
            f'headers="x-date", signature="{base64_signature}"'
        )
        return {
            "Authorization": token,
            "X-Date": date_now,
            "Content-Type": "application/json",
        }

    def _api_request(
        self, method: str, path: str, json: dict[str, Any] | None = None, timeout: int | None = None
    ) -> dict[str, Any]:
        """Make an API request to Sandbox API."""
        url = f"{self.config.base_url}/studio/sandbox/v1{path}"
        headers = self._generate_auth_headers()
        response = requests.request(method, url, json=json, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def _find_image_by_name(self, image_name: str) -> str:
        """Find image ID by name. Returns image ID or raises ValueError if not found."""
        path = f"/workspaces/{self.config.workspace_id}/images"
        params = {"order_by": "name ASC"}
        url = f"{self.config.base_url}/studio/sandbox/v1{path}"
        headers = self._generate_auth_headers()
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        images = data.get("images", [])
        for img in images:
            if img.get("name") == image_name or img.get("id") == image_name:
                return img["id"]
        raise ValueError(f"Image '{image_name}' not found in workspace {self.config.workspace_id}")

    def _find_snapshot_by_image(self, image_id: str) -> str | None:
        """Find existing snapshot by image ID. Returns snapshot ID or None if not found."""
        path = f"/workspaces/{self.config.workspace_id}/snapshots"
        # Filter by image using filter parameter (assuming AIP-160 format)
        filter_expr = f'image.id="{image_id}"'
        params = {"filter": filter_expr, "page_size": 100}
        url = f"{self.config.base_url}/studio/sandbox/v1{path}"
        headers = self._generate_auth_headers()
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        snapshots = data.get("snapshots", [])
        # Find snapshot that matches the image
        for snapshot in snapshots:
            snapshot_image = snapshot.get("image", {})
            if isinstance(snapshot_image, dict):
                if snapshot_image.get("id") == image_id:
                    return snapshot["id"]
            elif snapshot_image == image_id:
                return snapshot["id"]
        return None

    def _create_snapshot(self, image_id: str) -> str:
        """Create a snapshot from image. Returns snapshot ID."""
        snapshot_name = f"minisweagent-{uuid.uuid4().hex[:8]}"
        path = f"/workspaces/{self.config.workspace_id}/snapshots"
        payload = {
            "name": snapshot_name,
            "image": {"id": image_id},
            "resource_limits": self.config.resource_limits,
        }
        data = self._api_request("POST", path, json=payload, timeout=self.config.pull_timeout)
        snapshot = data.get("snapshot", {})
        snapshot_id = snapshot.get("id")
        if not snapshot_id:
            raise ValueError(f"Failed to create snapshot: {data}")
        self.logger.info(f"Created snapshot {snapshot_name} with ID {snapshot_id}")
        return snapshot_id

    def _ensure_snapshot(self, image_name: str) -> str:
        """Ensure snapshot exists for the given image. Returns snapshot ID."""
        # Find image ID
        image_id = self._find_image_by_name(image_name)
        # Try to find existing snapshot
        snapshot_id = self._find_snapshot_by_image(image_id)
        if snapshot_id:
            self.logger.info(f"Found existing snapshot {snapshot_id} for image {image_id}")
            return snapshot_id
        # Create new snapshot
        return self._create_snapshot(image_id)

    def _start_sandbox(self):
        """Start the Sandbox and return the sandbox ID."""
        snapshot_id = self._ensure_snapshot(self.config.image)
        path = f"/workspaces/{self.config.workspace_id}/sandboxes"
        payload = {
            "snapshot": {"id": snapshot_id},
            "auto_delete_after": self.config.auto_delete_after,
        }
        data = self._api_request("POST", path, json=payload, timeout=30)
        sandbox = data.get("sandbox", {})
        sandbox_id = sandbox.get("id")
        if not sandbox_id:
            raise ValueError(f"Failed to create sandbox: {data}")
        self.logger.info(f"Created sandbox with ID {sandbox_id}")
        self.sandbox_id = sandbox_id

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the Sandbox and return the result as a dict."""
        cwd = cwd or self.config.cwd
        assert self.sandbox_id, "Sandbox not started"

        # Build command with environment variables and directory change
        cmd_parts = []
        # Add environment variables
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd_parts.append(f'export {key}="{value}"')
        for key, value in self.config.env.items():
            cmd_parts.append(f'export {key}="{value}"')
        # Add directory change
        if cwd:
            cmd_parts.append(f'cd "{cwd}"')
        # Add actual command
        cmd_parts.append(command)
        full_command = " && ".join(cmd_parts)

        # Execute via API
        path = f"/workspaces/{self.config.workspace_id}/sandboxes/{self.sandbox_id}:execute"
        payload = {
            "code": full_command,
            "language": "bash",
        }
        if timeout is not None:
            payload["execution_timeout"] = timeout
        elif self.config.timeout:
            payload["execution_timeout"] = self.config.timeout

        data = self._api_request("POST", path, json=payload, timeout=(timeout or self.config.timeout) + 10)

        # Map response to match docker format
        # Try to get output and returncode from various possible locations
        output = ""
        returncode = 0

        # Check top-level fields first (preferred format for mini-swe-agent)
        if "output" in data:
            output = data["output"]
        elif "result" in data:
            result = data["result"]
            # Try to get from ExecutionResult format
            if isinstance(result, dict):
                stdout = result.get("stdout", "")
                stderr = result.get("stderr", "")
                # Merge stdout and stderr
                output = stdout + stderr
                returncode = result.get("exit_code", result.get("returncode", 0))
        elif "stdout" in data:
            output = data["stdout"]
            if "stderr" in data:
                output += data["stderr"]

        # Get returncode (try multiple field names for compatibility)
        if "returncode" in data:
            returncode = data["returncode"]
        elif "result" in data and isinstance(data["result"], dict):
            result = data["result"]
            returncode = result.get("returncode", result.get("exit_code", 0))
        elif "exit_code" in data:
            returncode = data["exit_code"]

        return {"output": output, "returncode": returncode}

    def cleanup(self):
        """Stop and remove the Sandbox."""
        if getattr(self, "sandbox_id", None) is not None:
            try:
                path = f"/workspaces/{self.config.workspace_id}/sandboxes/{self.sandbox_id}"
                self._api_request("DELETE", path, timeout=30)
                self.logger.info(f"Deleted sandbox {self.sandbox_id}")
            except Exception as e:
                self.logger.warning(f"Failed to delete sandbox {self.sandbox_id}: {e}")

    def __del__(self):
        """Cleanup sandbox when object is destroyed."""
        self.cleanup()
