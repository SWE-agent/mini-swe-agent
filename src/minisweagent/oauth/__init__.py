"""OAuth subscription auth for AI providers.

Provides login/refresh/storage for the three subscription flows ported from
pi-mono:

- ``anthropic``      - Claude Pro/Max
- ``openai-codex``   - ChatGPT Plus/Pro (Codex)
- ``github-copilot`` - GitHub Copilot

Public surface:

- :func:`get_oauth_provider`, :func:`register_oauth_provider`,
  :func:`unregister_oauth_provider`, :func:`restore_oauth_provider`,
  :func:`reset_oauth_providers`, :func:`get_oauth_providers`
- :func:`get_oauth_api_key` - returns a fresh access token, refreshing on demand
- :func:`refresh_provider`  - explicit refresh hook (alias)
- :func:`subscribe_refresh` / :func:`unsubscribe_refresh` - listen for refreshes
- :data:`storage` - low level credential persistence
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable

from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.oauth import storage
from minisweagent.oauth.anthropic import anthropic_oauth_provider
from minisweagent.oauth.github_copilot import github_copilot_oauth_provider
from minisweagent.oauth.openai_codex import openai_codex_oauth_provider
from minisweagent.oauth.types import (
    OAuthAuthInfo,
    OAuthCredentials,
    OAuthLoginCallbacks,
    OAuthPrompt,
    OAuthProviderInterface,
    OAuthTransientError,
)

logger = logging.getLogger("oauth")

__all__ = [
    "OAuthAuthInfo",
    "OAuthCredentials",
    "OAuthLoginCallbacks",
    "OAuthPrompt",
    "OAuthProviderInterface",
    "OAuthTransientError",
    "get_credentials",
    "get_oauth_provider",
    "get_oauth_providers",
    "get_oauth_api_key",
    "login_provider",
    "logout_provider",
    "refresh_provider",
    "register_oauth_provider",
    "reset_oauth_providers",
    "restore_oauth_provider",
    "storage",
    "subscribe_refresh",
    "unregister_oauth_provider",
    "unsubscribe_refresh",
]

_BUILT_IN: list[OAuthProviderInterface] = [
    anthropic_oauth_provider,
    github_copilot_oauth_provider,
    openai_codex_oauth_provider,
]

_registry_lock = threading.Lock()
_registry: dict[str, OAuthProviderInterface] = {p.id: p for p in _BUILT_IN}

_RefreshCallback = Callable[[str, OAuthCredentials], None]
_refresh_listeners: list[_RefreshCallback] = []
_refresh_listeners_lock = threading.Lock()

# Per-provider refresh locks. Most IdPs rotate or invalidate the prior refresh
# token on a successful refresh, so two concurrent ``refresh_provider`` calls
# would normally have the loser's credentials become unusable. Serialize the
# load+refresh+save sequence on a per-provider basis. Process-local: cross
# process refreshes still race, but the common case (parallel agents sharing a
# credential file from one process) is covered.
_provider_locks: dict[str, threading.Lock] = {}
_provider_locks_lock = threading.Lock()


def _get_provider_lock(provider_id: str) -> threading.Lock:
    with _provider_locks_lock:
        lock = _provider_locks.get(provider_id)
        if lock is None:
            lock = threading.Lock()
            _provider_locks[provider_id] = lock
        return lock


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


def get_oauth_provider(provider_id: str) -> OAuthProviderInterface | None:
    with _registry_lock:
        return _registry.get(provider_id)


def get_oauth_providers() -> list[OAuthProviderInterface]:
    with _registry_lock:
        return list(_registry.values())


def register_oauth_provider(provider: OAuthProviderInterface) -> None:
    with _registry_lock:
        _registry[provider.id] = provider


def unregister_oauth_provider(provider_id: str) -> bool:
    """Remove ``provider_id`` from the registry. Returns ``True`` if removed.

    Built-ins are removed too — call :func:`restore_oauth_provider` to put the
    built-in implementation back. Tests that register a stub under a built-in
    id (e.g. aliasing a fake to ``"anthropic"``) should call
    :func:`restore_oauth_provider` to clean up, not this function.
    """
    with _registry_lock:
        return _registry.pop(provider_id, None) is not None


def restore_oauth_provider(provider_id: str) -> OAuthProviderInterface:
    """Restore the built-in provider for ``provider_id``, replacing whatever is
    currently registered under that id. Raises ``KeyError`` if ``provider_id``
    is not a built-in.
    """
    builtin = next((p for p in _BUILT_IN if p.id == provider_id), None)
    if builtin is None:
        raise KeyError(f"{provider_id!r} is not a built-in OAuth provider")
    with _registry_lock:
        _registry[provider_id] = builtin
    return builtin


def reset_oauth_providers() -> None:
    with _registry_lock:
        _registry.clear()
        for provider in _BUILT_IN:
            _registry[provider.id] = provider
    with _refresh_listeners_lock:
        _refresh_listeners.clear()


# ---------------------------------------------------------------------------
# Refresh hook
# ---------------------------------------------------------------------------


def subscribe_refresh(callback: _RefreshCallback) -> _RefreshCallback:
    """Register a listener invoked after every successful refresh.

    Returns the callback for convenient stacking with decorators. Pass the
    same callback to :func:`unsubscribe_refresh` to remove it.
    """
    with _refresh_listeners_lock:
        _refresh_listeners.append(callback)
    return callback


def unsubscribe_refresh(callback: _RefreshCallback) -> None:
    with _refresh_listeners_lock:
        try:
            _refresh_listeners.remove(callback)
        except ValueError:
            pass


def _emit_refresh(provider_id: str, creds: OAuthCredentials) -> None:
    with _refresh_listeners_lock:
        snapshot = list(_refresh_listeners)
    for cb in snapshot:
        try:
            cb(provider_id, creds)
        except Exception:  # noqa: BLE001
            # Listener errors must never break a token refresh.
            continue


# ---------------------------------------------------------------------------
# Retry + persistence helpers
# ---------------------------------------------------------------------------


def _refresh_retry_attempts() -> int:
    return max(1, int(os.getenv("MSWEA_OAUTH_REFRESH_RETRY_ATTEMPTS", "3")))


def _refresh_retry_wait_min() -> float:
    return float(os.getenv("MSWEA_OAUTH_REFRESH_RETRY_WAIT_MIN", "1"))


def _refresh_retry_wait_max() -> float:
    return float(os.getenv("MSWEA_OAUTH_REFRESH_RETRY_WAIT_MAX", "10"))


def _refresh_with_retry(
    provider: OAuthProviderInterface, creds: OAuthCredentials
) -> OAuthCredentials:
    """Call ``provider.refresh_token(creds)`` with retries on transient errors.

    Mirrors the LitellmModel retry pattern: tenacity with exponential backoff,
    bounded attempts, and a typed-exception filter. Only :class:`OAuthTransientError`
    (5xx + network failures) is retried — permanent errors (4xx auth failures,
    malformed responses) bubble out immediately so the caller sees an
    actionable failure instead of N back-off cycles.
    """
    refreshed: OAuthCredentials | None = None
    for attempt in Retrying(
        reraise=True,
        stop=stop_after_attempt(_refresh_retry_attempts()),
        wait=wait_exponential(multiplier=1, min=_refresh_retry_wait_min(), max=_refresh_retry_wait_max()),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_exception_type(OAuthTransientError),
    ):
        with attempt:
            refreshed = provider.refresh_token(creds)
    assert refreshed is not None  # tenacity reraise=True guarantees this
    return refreshed


def _persist_or_raise(provider_id: str, refreshed: OAuthCredentials) -> None:
    """Persist refreshed credentials, with an actionable error on failure.

    Most IdPs (notably Anthropic) rotate ``refresh_token`` server-side on a
    successful refresh, invalidating the prior one. If we fail to write the
    new credentials to disk (disk full, permission error, ENOSPC on tmp), the
    rotated refresh token is lost and the next request will fail with an
    auth error. Surface the situation explicitly with a re-login hint
    instead of letting the bare OSError propagate.
    """
    try:
        storage.save(provider_id, refreshed)
    except Exception as exc:
        logger.error(
            "OAuth refresh for %r succeeded but persistence failed (%s: %s). "
            "The IdP may have already invalidated the previous refresh token; "
            "if subsequent requests fail with an authentication error, run "
            "`mini-extra oauth login %s` to recover.",
            provider_id,
            type(exc).__name__,
            exc,
            provider_id,
        )
        raise RuntimeError(
            f"OAuth refresh for {provider_id!r} succeeded but failed to persist "
            f"the rotated credentials ({type(exc).__name__}: {exc}). "
            f"Re-run `mini-extra oauth login {provider_id}` if subsequent "
            f"calls fail with authentication errors."
        ) from exc


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def login_provider(provider_id: str, callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
    """Run the login flow and persist the resulting credentials."""
    provider = _require(provider_id)
    creds = provider.login(callbacks)
    storage.save(provider_id, creds)
    _emit_refresh(provider_id, creds)
    return creds


def logout_provider(provider_id: str) -> bool:
    return storage.delete(provider_id)


def refresh_provider(provider_id: str, credentials: OAuthCredentials | None = None) -> OAuthCredentials:
    """Force-refresh the access token for ``provider_id`` and persist it.

    This is the explicit refresh hook: callers can invoke it manually (e.g.
    from the CLI, on a schedule, or from a long-running process) without
    waiting for the access token to expire.
    """
    provider = _require(provider_id)
    with _get_provider_lock(provider_id):
        creds = credentials or storage.load(provider_id)
        if creds is None:
            raise RuntimeError(f"No stored credentials for provider {provider_id!r}")
        refreshed = _refresh_with_retry(provider, creds)
        _persist_or_raise(provider_id, refreshed)
    _emit_refresh(provider_id, refreshed)
    return refreshed


def _refresh_if_still_expired(provider_id: str) -> OAuthCredentials:
    """Refresh under the provider lock, but skip the network call if another
    thread already refreshed before we acquired the lock."""
    provider = _require(provider_id)
    with _get_provider_lock(provider_id):
        creds = storage.load(provider_id)
        if creds is None:
            raise RuntimeError(f"No stored credentials for provider {provider_id!r}")
        if int(time.time() * 1000) < creds.expires:
            return creds
        refreshed = _refresh_with_retry(provider, creds)
        _persist_or_raise(provider_id, refreshed)
    _emit_refresh(provider_id, refreshed)
    return refreshed


def get_oauth_api_key(provider_id: str, *, force_refresh: bool = False) -> str | None:
    """Return a usable API key for ``provider_id``, refreshing when needed.

    Returns ``None`` when no credentials are stored. Raises ``RuntimeError`` if
    the refresh fails.
    """
    creds = storage.load(provider_id)
    if creds is None:
        return None
    if force_refresh:
        creds = refresh_provider(provider_id, creds)
    elif int(time.time() * 1000) >= creds.expires:
        creds = _refresh_if_still_expired(provider_id)
    provider = _require(provider_id)
    return provider.get_api_key(creds)


def get_credentials(provider_id: str, *, refresh_if_expired: bool = True) -> OAuthCredentials | None:
    creds = storage.load(provider_id)
    if creds is None:
        return None
    if refresh_if_expired and int(time.time() * 1000) >= creds.expires:
        creds = _refresh_if_still_expired(provider_id)
    return creds


def _require(provider_id: str) -> OAuthProviderInterface:
    provider = get_oauth_provider(provider_id)
    if provider is None:
        raise RuntimeError(f"Unknown OAuth provider: {provider_id}")
    return provider
