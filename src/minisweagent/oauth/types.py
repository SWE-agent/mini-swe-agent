"""OAuth credential & provider types.

Ported from pi-mono (packages/ai/src/utils/oauth/types.ts).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OAuthCredentials:
    """Credentials returned by an OAuth login flow.

    ``expires`` is a unix-millis timestamp at which the access token expires.
    Concrete providers may attach extra fields via ``extra`` (e.g. ``accountId``,
    ``enterpriseUrl``).
    """

    refresh: str
    access: str
    expires: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        # Known fields override anything stashed in ``extra`` so a stray
        # ``extra={"refresh": ...}`` cannot shadow the real refresh token.
        return {**self.extra, "refresh": self.refresh, "access": self.access, "expires": self.expires}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OAuthCredentials:
        known = {"refresh", "access", "expires"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(refresh=data["refresh"], access=data["access"], expires=int(data["expires"]), extra=extra)


@dataclass
class OAuthAuthInfo:
    url: str
    instructions: str | None = None


@dataclass
class OAuthPrompt:
    message: str
    placeholder: str | None = None
    allow_empty: bool = False


@dataclass
class OAuthLoginCallbacks:
    """Callbacks given to a provider's ``login()`` to drive an interactive flow."""

    on_auth: Callable[[OAuthAuthInfo], None]
    on_prompt: Callable[[OAuthPrompt], str]
    on_progress: Callable[[str], None] | None = None
    on_manual_code_input: Callable[[], str] | None = None
    cancel_manual_code_input: Callable[[], None] | None = None
    """Optional hook invoked after the callback server resolves the login.

    Lets the caller release a ``on_manual_code_input`` thread that is still
    blocked on stdin (e.g. inside ``prompt_toolkit.prompt``) so it does not
    keep eating keystrokes meant for the next interactive prompt.
    """


class OAuthTransientError(RuntimeError):
    """Refresh/login error that may succeed if the caller retries.

    Raised for HTTP 5xx responses and network/connection failures. Plain
    :class:`RuntimeError` is reserved for permanent failures (4xx auth errors,
    malformed responses) so the retry layer can distinguish them.
    """


class OAuthProviderInterface(ABC):
    """Abstract OAuth provider.

    Concrete subclasses must define ``id`` and ``name`` class attributes and
    implement ``login``/``refresh_token``/``get_api_key``.
    """

    id: str
    name: str
    uses_callback_server: bool = False

    @abstractmethod
    def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials: ...

    @abstractmethod
    def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials: ...

    @abstractmethod
    def get_api_key(self, credentials: OAuthCredentials) -> str: ...
