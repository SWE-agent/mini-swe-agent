"""LiteLLM model with OAuth subscription auth.

Wraps :class:`LitellmModel` and, before each request, fetches a fresh access
token from :mod:`minisweagent.oauth` (refreshing it if needed) and injects the
provider-specific headers, base URL, and (for Anthropic OAuth) the
``"You are Claude Code, Anthropic's official CLI for Claude."`` system prefix.

Supported ``oauth_provider`` values:

- ``"anthropic"``       - Claude Pro/Max
- ``"openai-codex"``    - ChatGPT Plus/Pro (Codex)
- ``"github-copilot"``  - GitHub Copilot

Example config::

    model_class: oauth
    model_name: anthropic/claude-sonnet-4-5-20250929
    oauth_provider: anthropic

"""

from __future__ import annotations

import logging
import os
import platform
from typing import Any

from pydantic import Field

from minisweagent import oauth
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"


def _default_codex_base_url() -> str:
    return os.getenv("MSWEA_CODEX_BASE_URL", DEFAULT_CODEX_BASE_URL)

logger = logging.getLogger("oauth_model")

DEFAULT_CLAUDE_CODE_VERSION = "2.1.75"
CLAUDE_CODE_SYSTEM_PROMPT = "You are Claude Code, Anthropic's official CLI for Claude."


def _claude_code_version() -> str:
    """Resolve the ``user-agent: claude-cli/<version>`` string per request.

    Anthropic ages out client versions periodically; when the current default
    starts being rejected, set ``MSWEA_CLAUDE_CODE_VERSION`` in the
    environment as a hot-fix instead of waiting for a release that bumps the
    constant. Read on every call so test monkeypatches and live env edits
    take effect without a re-import.
    """
    return os.getenv("MSWEA_CLAUDE_CODE_VERSION", DEFAULT_CLAUDE_CODE_VERSION)

# LiteLLM's Anthropic adapter requires a non-empty ``api_key`` and uses it to set
# ``x-api-key``. For OAuth we authenticate via ``Authorization: Bearer <token>``
# instead, so we pass a sentinel here and override the ``x-api-key`` header to
# an empty string to avoid leaking the access token under the wrong header name
# (Anthropic's OAuth path rejects requests that present both x-api-key and
# Authorization with a real OAuth token).
_ANTHROPIC_OAUTH_API_KEY_SENTINEL = "oauth"

_VALID_PROVIDERS = {"anthropic", "openai-codex", "github-copilot"}


def _extract_text(content: Any) -> str:
    """Return the human-readable text from a chat message ``content`` field.

    Anthropic / OpenAI multimodal messages may carry ``content`` as either a
    plain string or a list of ``{"type": "text", "text": ...}`` parts (with
    other part types like ``image`` mixed in). ``str(content)`` would
    accidentally match against Python's repr of the list — fine in practice
    but fragile, so extract real text instead.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


class OAuthLitellmModelConfig(LitellmModelConfig):
    oauth_provider: str
    """OAuth provider id. One of ``anthropic``, ``openai-codex``, ``github-copilot``."""

    inject_claude_code_system: bool = True
    """For Anthropic OAuth, prepend the Claude Code identity system message (required)."""

    codex_base_url: str = Field(default_factory=_default_codex_base_url)
    """Base URL used for OpenAI Codex requests. Reads ``MSWEA_CODEX_BASE_URL`` at construction time."""

    codex_originator: str = "mini-swe-agent"
    """``originator`` header sent to Codex."""


class OAuthLitellmModel(LitellmModel):
    """LiteLLM model that authenticates via stored OAuth credentials."""

    def __init__(self, **kwargs: Any) -> None:
        provider = kwargs.get("oauth_provider")
        if provider not in _VALID_PROVIDERS:
            raise ValueError(
                f"oauth_provider must be one of {sorted(_VALID_PROVIDERS)}, got {provider!r}"
            )
        super().__init__(config_class=OAuthLitellmModelConfig, **kwargs)
        self.config: OAuthLitellmModelConfig

    # -- request building --------------------------------------------------

    def _resolve_oauth_kwargs(self) -> dict[str, Any]:
        """Resolve the per-request kwargs (headers, api_base, api_key) for the active provider."""
        provider_id = self.config.oauth_provider
        creds = oauth.get_credentials(provider_id, refresh_if_expired=True)
        if creds is None:
            raise RuntimeError(
                f"No OAuth credentials for provider {provider_id!r}. "
                f"Run `mini-extra oauth login {provider_id}` first."
            )
        token = creds.access

        if provider_id == "anthropic":
            return {
                "api_key": _ANTHROPIC_OAUTH_API_KEY_SENTINEL,
                "extra_headers": {
                    "Authorization": f"Bearer {token}",
                    "x-api-key": "",
                    "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
                    "user-agent": f"claude-cli/{_claude_code_version()}",
                    "x-app": "cli",
                },
            }

        if provider_id == "github-copilot":
            from minisweagent.oauth.github_copilot import (
                COPILOT_HEADERS,
                get_github_copilot_base_url,
            )

            enterprise = creds.extra.get("enterprise_url")
            base_url = get_github_copilot_base_url(token, enterprise)
            headers: dict[str, str] = {
                "Authorization": f"Bearer {token}",
                **COPILOT_HEADERS,
            }
            return {"api_key": token, "api_base": base_url, "extra_headers": headers}

        if provider_id == "openai-codex":
            account_id = creds.extra.get("account_id")
            if not account_id:
                raise RuntimeError("Codex credentials missing account_id; re-run login.")
            user_agent = f"mini-swe-agent ({platform.system()} {platform.release()}; {platform.machine()})"
            headers = {
                "Authorization": f"Bearer {token}",
                "chatgpt-account-id": account_id,
                "originator": self.config.codex_originator,
                "OpenAI-Beta": "responses=experimental",
                "User-Agent": user_agent,
            }
            return {
                "api_key": token,
                "api_base": self.config.codex_base_url,
                "extra_headers": headers,
            }

        raise RuntimeError(f"Unsupported oauth_provider: {provider_id}")

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = super()._prepare_messages_for_api(messages)
        if self.config.oauth_provider == "anthropic" and self.config.inject_claude_code_system:
            already_present = any(
                msg.get("role") == "system" and CLAUDE_CODE_SYSTEM_PROMPT in _extract_text(msg.get("content"))
                for msg in prepared
            )
            if not already_present:
                prepared = [{"role": "system", "content": CLAUDE_CODE_SYSTEM_PROMPT}, *prepared]
        return prepared

    def _query(self, messages: list[dict[str, str]], **kwargs):
        oauth_kwargs = self._resolve_oauth_kwargs()
        # OAuth identity (api_key, api_base, Authorization, etc.) MUST win over
        # caller-supplied kwargs. Otherwise upstream code that defaults
        # ``api_key=os.getenv("ANTHROPIC_API_KEY")`` would silently replace the
        # OAuth bearer token with a static API key and break the OAuth flow.
        merged = {**kwargs, **oauth_kwargs}
        # Precedence (low -> high) for extra_headers: config.model_kwargs (user
        # defaults), per-call kwargs, OAuth-injected. Parent ``LitellmModel._query``
        # does ``self.config.model_kwargs | kwargs`` shallowly; if we leave the
        # merged ``extra_headers`` dict here, it shadows ``model_kwargs["extra_headers"]``
        # entirely and silently drops user-set audit / tracing headers.
        config_headers = self.config.model_kwargs.get("extra_headers") or {}
        caller_headers = kwargs.get("extra_headers") or {}
        oauth_headers = oauth_kwargs.get("extra_headers") or {}
        if config_headers or caller_headers or oauth_headers:
            merged["extra_headers"] = {**config_headers, **caller_headers, **oauth_headers}
        return super()._query(messages, **merged)


__all__ = ["OAuthLitellmModel", "OAuthLitellmModelConfig"]
