"""GitHub Copilot OAuth (device-code) flow.

Ported from pi-mono (packages/ai/src/utils/oauth/github-copilot.ts).
"""

from __future__ import annotations

import base64
import json
import re
import time
from typing import Any
from urllib.parse import urlparse

import requests

from minisweagent.oauth.types import (
    OAuthAuthInfo,
    OAuthCredentials,
    OAuthLoginCallbacks,
    OAuthPrompt,
    OAuthProviderInterface,
    OAuthTransientError,
)

CLIENT_ID = base64.b64decode("SXYxLmI1MDdhMDhjODdlY2ZlOTg=").decode()
COPILOT_HEADERS = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}
INITIAL_POLL_MULTIPLIER = 1.2
SLOW_DOWN_POLL_MULTIPLIER = 1.4
EXPIRY_SAFETY_MS = 5 * 60 * 1000
DEFAULT_BASE_URL = "https://api.individual.githubcopilot.com"


def normalize_domain(value: str) -> str | None:
    trimmed = value.strip()
    if not trimmed:
        return None
    candidate = trimmed if "://" in trimmed else f"https://{trimmed}"
    try:
        host = urlparse(candidate).hostname
    except ValueError:
        return None
    return host or None


def _urls(domain: str) -> dict[str, str]:
    return {
        "device_code": f"https://{domain}/login/device/code",
        "access_token": f"https://{domain}/login/oauth/access_token",
        "copilot_token": f"https://api.{domain}/copilot_internal/v2/token",
    }


def _base_url_from_token(token: str) -> str | None:
    match = re.search(r"proxy-ep=([^;]+)", token)
    if not match:
        return None
    proxy_host = match.group(1)
    api_host = re.sub(r"^proxy\.", "api.", proxy_host)
    return f"https://{api_host}"


def get_github_copilot_base_url(token: str | None = None, enterprise_domain: str | None = None) -> str:
    if token:
        url = _base_url_from_token(token)
        if url:
            return url
    if enterprise_domain:
        return f"https://copilot-api.{enterprise_domain}"
    return DEFAULT_BASE_URL


def _post_form(url: str, data: dict[str, str]) -> dict[str, Any]:
    try:
        response = requests.post(
            url,
            data=data,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": COPILOT_HEADERS["User-Agent"],
            },
            timeout=30,
        )
    except requests.exceptions.RequestException as exc:
        raise OAuthTransientError(
            f"HTTP request failed (network): url={url}; error={type(exc).__name__}: {exc}"
        ) from exc
    if not response.ok:
        # Don't echo response.text — error bodies can echo the access token
        # we sent in the form.
        message = f"HTTP {response.status_code} {response.reason} from {url}"
        if response.status_code >= 500:
            raise OAuthTransientError(message)
        raise RuntimeError(message)
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from {url}") from exc


def _start_device_flow(domain: str) -> dict[str, Any]:
    data = _post_form(
        _urls(domain)["device_code"],
        {"client_id": CLIENT_ID, "scope": "read:user"},
    )
    required = {"device_code", "user_code", "verification_uri", "interval", "expires_in"}
    if not required.issubset(data):
        raise RuntimeError("Invalid device code response fields")
    return data


def _poll_for_access_token(
    domain: str,
    device_code: str,
    interval_seconds: int,
    expires_in: int,
) -> str:
    deadline = time.time() + expires_in
    interval = max(1.0, interval_seconds)
    multiplier = INITIAL_POLL_MULTIPLIER
    slow_downs = 0
    url = _urls(domain)["access_token"]
    while time.time() < deadline:
        wait_for = min(interval * multiplier, max(0.0, deadline - time.time()))
        time.sleep(wait_for)
        try:
            data = _post_form(
                url,
                {
                    "client_id": CLIENT_ID,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Device flow failed: {exc}") from exc
        if isinstance(data.get("access_token"), str):
            return data["access_token"]
        error = data.get("error")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            slow_downs += 1
            new_interval = data.get("interval")
            interval = (
                float(new_interval) if isinstance(new_interval, (int, float)) and new_interval > 0 else interval + 5.0
            )
            multiplier = SLOW_DOWN_POLL_MULTIPLIER
            continue
        if error:
            description = data.get("error_description") or ""
            suffix = f": {description}" if description else ""
            raise RuntimeError(f"Device flow failed: {error}{suffix}")
    if slow_downs > 0:
        raise RuntimeError("Device flow timed out after slow_down responses. Sync your system clock and try again.")
    raise RuntimeError("Device flow timed out")


def refresh_github_copilot_token(refresh_token: str, enterprise_domain: str | None = None) -> OAuthCredentials:
    domain = enterprise_domain or "github.com"
    try:
        response = requests.get(
            _urls(domain)["copilot_token"],
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {refresh_token}",
                **COPILOT_HEADERS,
            },
            timeout=30,
        )
    except requests.exceptions.RequestException as exc:
        raise OAuthTransientError(f"Copilot token request failed (network): {type(exc).__name__}: {exc}") from exc
    if not response.ok:
        # Don't echo response.text — Copilot's bearer token (the
        # ``refresh_token`` arg here) is on the Authorization header and the
        # response body can echo it back in error scenarios.
        message = f"Copilot token request failed: status={response.status_code} reason={response.reason}"
        if response.status_code >= 500:
            raise OAuthTransientError(message)
        raise RuntimeError(message)
    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid JSON from Copilot token endpoint") from exc
    token = data.get("token")
    expires_at = data.get("expires_at")
    if not isinstance(token, str) or not isinstance(expires_at, (int, float)):
        raise RuntimeError("Invalid Copilot token response fields")
    extra: dict[str, Any] = {}
    if enterprise_domain:
        extra["enterprise_url"] = enterprise_domain
    return OAuthCredentials(
        refresh=refresh_token,
        access=token,
        expires=int(expires_at) * 1000 - EXPIRY_SAFETY_MS,
        extra=extra,
    )


def login_github_copilot(callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
    raw = callbacks.on_prompt(
        OAuthPrompt(
            message="GitHub Enterprise URL/domain (blank for github.com)",
            placeholder="company.ghe.com",
            allow_empty=True,
        )
    )
    trimmed = raw.strip() if raw else ""
    enterprise_domain = normalize_domain(trimmed) if trimmed else None
    if trimmed and not enterprise_domain:
        raise RuntimeError(f"Invalid GitHub Enterprise URL/domain: {trimmed!r}")
    domain = enterprise_domain or "github.com"

    device = _start_device_flow(domain)
    callbacks.on_auth(OAuthAuthInfo(url=device["verification_uri"], instructions=f"Enter code: {device['user_code']}"))

    access_token = _poll_for_access_token(
        domain, device["device_code"], int(device["interval"]), int(device["expires_in"])
    )
    return refresh_github_copilot_token(access_token, enterprise_domain)


class GitHubCopilotOAuthProvider(OAuthProviderInterface):
    id = "github-copilot"
    name = "GitHub Copilot"

    def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
        return login_github_copilot(callbacks)

    def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        enterprise = credentials.extra.get("enterprise_url")
        return refresh_github_copilot_token(credentials.refresh, enterprise)

    def get_api_key(self, credentials: OAuthCredentials) -> str:
        return credentials.access


github_copilot_oauth_provider = GitHubCopilotOAuthProvider()
