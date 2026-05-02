from minisweagent import oauth
from minisweagent.oauth.types import (
    OAuthCredentials,
    OAuthLoginCallbacks,
    OAuthProviderInterface,
)


class _Dummy(OAuthProviderInterface):
    id = "dummy"
    name = "Dummy"

    def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
        return OAuthCredentials(refresh="r", access="a", expires=0)

    def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        return OAuthCredentials(refresh="r2", access="a2", expires=99)

    def get_api_key(self, credentials: OAuthCredentials) -> str:
        return credentials.access


def test_built_in_providers_present():
    ids = {p.id for p in oauth.get_oauth_providers()}
    assert {"anthropic", "openai-codex", "github-copilot"}.issubset(ids)


def test_register_and_unregister_custom():
    try:
        oauth.register_oauth_provider(_Dummy())
        assert oauth.get_oauth_provider("dummy") is not None
    finally:
        oauth.unregister_oauth_provider("dummy")
    assert oauth.get_oauth_provider("dummy") is None


def test_unregister_builtin_actually_removes():
    original = oauth.get_oauth_provider("anthropic")
    assert original is not None
    try:
        assert oauth.unregister_oauth_provider("anthropic") is True
        assert oauth.get_oauth_provider("anthropic") is None
    finally:
        oauth.restore_oauth_provider("anthropic")
    restored = oauth.get_oauth_provider("anthropic")
    assert restored is original


def test_unregister_unknown_returns_false():
    assert oauth.unregister_oauth_provider("nope-not-a-real-id") is False


def test_restore_oauth_provider_replaces_stub():
    class _Stub(_Dummy):
        id = "anthropic"
        name = "Stub Anthropic"

    original = oauth.get_oauth_provider("anthropic")
    assert original is not None
    oauth.register_oauth_provider(_Stub())
    assert oauth.get_oauth_provider("anthropic").name == "Stub Anthropic"
    restored = oauth.restore_oauth_provider("anthropic")
    assert restored is original
    assert oauth.get_oauth_provider("anthropic") is original


def test_restore_non_builtin_raises():
    import pytest

    with pytest.raises(KeyError):
        oauth.restore_oauth_provider("not-a-builtin")


def test_reset_oauth_providers():
    oauth.register_oauth_provider(_Dummy())
    oauth.reset_oauth_providers()
    ids = {p.id for p in oauth.get_oauth_providers()}
    assert ids == {"anthropic", "openai-codex", "github-copilot"}


def test_reset_oauth_providers_clears_listeners():
    seen: list[str] = []
    oauth.subscribe_refresh(lambda pid, _creds: seen.append(pid))
    oauth.reset_oauth_providers()
    # Force a refresh path that would emit if any listener survived.
    from minisweagent.oauth.types import OAuthCredentials

    oauth._emit_refresh("anthropic", OAuthCredentials(refresh="r", access="a", expires=0))
    assert seen == []
