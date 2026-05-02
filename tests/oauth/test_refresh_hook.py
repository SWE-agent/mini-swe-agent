"""Refresh hook + automatic refresh-on-expiry coverage."""

import time

import pytest

from minisweagent import oauth
from minisweagent.oauth.types import (
    OAuthCredentials,
    OAuthLoginCallbacks,
    OAuthProviderInterface,
)


class _StubProvider(OAuthProviderInterface):
    id = "stub-refresh"
    name = "Stub"

    def __init__(self) -> None:
        self.refresh_calls = 0

    def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
        return OAuthCredentials(refresh="r0", access="a0", expires=0)

    def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        self.refresh_calls += 1
        return OAuthCredentials(
            refresh=f"r{self.refresh_calls}",
            access=f"a{self.refresh_calls}",
            expires=int(time.time() * 1000) + 60_000,
        )

    def get_api_key(self, credentials: OAuthCredentials) -> str:
        return credentials.access


@pytest.fixture
def stub_provider(monkeypatch, tmp_path):
    path = tmp_path / "oauth.json"
    monkeypatch.setenv("MSWEA_OAUTH_FILE", str(path))
    provider = _StubProvider()
    oauth.register_oauth_provider(provider)
    yield provider
    oauth.unregister_oauth_provider(provider.id)


def test_refresh_provider_persists_and_emits(stub_provider):
    oauth.storage.save(
        stub_provider.id,
        OAuthCredentials(refresh="initial-refresh", access="expired-access", expires=0),
    )
    events: list[tuple[str, str]] = []
    listener = oauth.subscribe_refresh(lambda pid, creds: events.append((pid, creds.access)))
    try:
        new_creds = oauth.refresh_provider(stub_provider.id)
    finally:
        oauth.unsubscribe_refresh(listener)

    assert new_creds.access == "a1"
    assert new_creds.refresh == "r1"
    assert oauth.storage.load(stub_provider.id) is not None
    assert events == [(stub_provider.id, "a1")]


def test_get_oauth_api_key_refreshes_when_expired(stub_provider):
    oauth.storage.save(
        stub_provider.id,
        OAuthCredentials(refresh="r0", access="a-stale", expires=0),
    )
    api_key = oauth.get_oauth_api_key(stub_provider.id)
    assert api_key == "a1"
    assert stub_provider.refresh_calls == 1


def test_get_oauth_api_key_skips_refresh_when_valid(stub_provider):
    future_ms = int(time.time() * 1000) + 60_000
    oauth.storage.save(
        stub_provider.id,
        OAuthCredentials(refresh="r0", access="still-good", expires=future_ms),
    )
    api_key = oauth.get_oauth_api_key(stub_provider.id)
    assert api_key == "still-good"
    assert stub_provider.refresh_calls == 0


def test_get_oauth_api_key_force_refresh(stub_provider):
    future_ms = int(time.time() * 1000) + 60_000
    oauth.storage.save(
        stub_provider.id,
        OAuthCredentials(refresh="r0", access="still-good", expires=future_ms),
    )
    api_key = oauth.get_oauth_api_key(stub_provider.id, force_refresh=True)
    assert api_key == "a1"
    assert stub_provider.refresh_calls == 1


def test_refresh_provider_raises_for_missing(stub_provider):
    with pytest.raises(RuntimeError, match="No stored credentials"):
        oauth.refresh_provider(stub_provider.id)


def test_listener_errors_do_not_break_refresh(stub_provider):
    oauth.storage.save(
        stub_provider.id,
        OAuthCredentials(refresh="r0", access="expired", expires=0),
    )

    def boom(_pid, _creds):
        raise RuntimeError("boom")

    listener = oauth.subscribe_refresh(boom)
    try:
        creds = oauth.refresh_provider(stub_provider.id)
    finally:
        oauth.unsubscribe_refresh(listener)
    assert creds.access == "a1"


# ---------------------------------------------------------------------------
# login_provider / logout_provider
# ---------------------------------------------------------------------------


def test_login_provider_persists_and_emits(stub_provider):
    events: list[tuple[str, str]] = []
    listener = oauth.subscribe_refresh(lambda pid, creds: events.append((pid, creds.access)))
    try:
        from minisweagent.oauth.types import OAuthLoginCallbacks

        creds = oauth.login_provider(
            stub_provider.id,
            OAuthLoginCallbacks(on_auth=lambda _: None, on_prompt=lambda _: ""),
        )
    finally:
        oauth.unsubscribe_refresh(listener)

    assert creds.access == "a0"
    assert oauth.storage.load(stub_provider.id) is not None
    assert events == [(stub_provider.id, "a0")]


def test_logout_provider(stub_provider):
    oauth.storage.save(stub_provider.id, OAuthCredentials(refresh="r", access="a", expires=0))
    assert oauth.logout_provider(stub_provider.id) is True
    assert oauth.storage.load(stub_provider.id) is None
    assert oauth.logout_provider(stub_provider.id) is False


# ---------------------------------------------------------------------------
# get_credentials
# ---------------------------------------------------------------------------


def test_get_credentials_returns_none_when_no_creds(stub_provider):
    assert oauth.get_credentials(stub_provider.id) is None


def test_get_credentials_refreshes_when_expired(stub_provider):
    oauth.storage.save(stub_provider.id, OAuthCredentials(refresh="r0", access="stale", expires=0))
    creds = oauth.get_credentials(stub_provider.id)
    assert creds is not None
    assert creds.access == "a1"
    assert stub_provider.refresh_calls == 1


def test_get_credentials_skips_refresh_when_not_expired(stub_provider):
    future = int(time.time() * 1000) + 60_000
    oauth.storage.save(stub_provider.id, OAuthCredentials(refresh="r0", access="valid", expires=future))
    creds = oauth.get_credentials(stub_provider.id)
    assert creds is not None
    assert creds.access == "valid"
    assert stub_provider.refresh_calls == 0


def test_get_credentials_no_refresh_flag(stub_provider):
    oauth.storage.save(stub_provider.id, OAuthCredentials(refresh="r0", access="expired-but-kept", expires=0))
    creds = oauth.get_credentials(stub_provider.id, refresh_if_expired=False)
    assert creds is not None
    assert creds.access == "expired-but-kept"
    assert stub_provider.refresh_calls == 0


def test_get_oauth_api_key_returns_none_when_no_creds(stub_provider):
    assert oauth.get_oauth_api_key(stub_provider.id) is None


def test_require_unknown_provider_raises():
    with pytest.raises(RuntimeError, match="Unknown OAuth provider"):
        oauth.refresh_provider("totally-nonexistent-provider-xyz")


# ---------------------------------------------------------------------------
# Concurrent refresh: per-provider lock prevents racing refresh_token calls
# ---------------------------------------------------------------------------


class _SerializingProvider(OAuthProviderInterface):
    """Provider that asserts ``refresh_token`` is never re-entered concurrently."""

    id = "stub-serial"
    name = "Stub"

    def __init__(self) -> None:
        import threading as _t
        self._inflight = _t.Lock()
        self._concurrent_breaches = 0
        self.refresh_calls = 0

    def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
        return OAuthCredentials(refresh="r0", access="a0", expires=0)

    def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        if not self._inflight.acquire(blocking=False):
            self._concurrent_breaches += 1
        try:
            time.sleep(0.05)
            self.refresh_calls += 1
            return OAuthCredentials(
                refresh=f"r{self.refresh_calls}",
                access=f"a{self.refresh_calls}",
                expires=int(time.time() * 1000) + 60_000,
            )
        finally:
            try:
                self._inflight.release()
            except RuntimeError:
                pass

    def get_api_key(self, credentials: OAuthCredentials) -> str:
        return credentials.access


def test_refresh_provider_serializes_concurrent_callers(monkeypatch, tmp_path):
    """Two concurrent ``refresh_provider`` calls must not both hit the IdP at
    once — most providers rotate the refresh token, which would invalidate one
    side's credentials."""
    import threading

    monkeypatch.setenv("MSWEA_OAUTH_FILE", str(tmp_path / "oauth.json"))
    provider = _SerializingProvider()
    oauth.register_oauth_provider(provider)
    try:
        oauth.storage.save(provider.id, OAuthCredentials(refresh="r0", access="stale", expires=0))
        results: list[OAuthCredentials] = []
        errors: list[Exception] = []

        def _runner() -> None:
            try:
                results.append(oauth.refresh_provider(provider.id))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_runner) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert provider._concurrent_breaches == 0
        assert provider.refresh_calls == 4  # forced refresh always runs
    finally:
        oauth.unregister_oauth_provider(provider.id)


def test_get_credentials_double_check_skips_redundant_refresh(monkeypatch, tmp_path):
    """When two threads find an expired token, only the first should refresh;
    the second should observe the freshly-saved token under the lock and skip
    the IdP round trip."""
    import threading

    monkeypatch.setenv("MSWEA_OAUTH_FILE", str(tmp_path / "oauth.json"))
    provider = _SerializingProvider()
    oauth.register_oauth_provider(provider)
    try:
        oauth.storage.save(provider.id, OAuthCredentials(refresh="r0", access="stale", expires=0))

        results: list[OAuthCredentials] = []
        errors: list[Exception] = []
        gate = threading.Barrier(4)

        def _runner() -> None:
            try:
                gate.wait(timeout=5)
                creds = oauth.get_credentials(provider.id)
                assert creds is not None
                results.append(creds)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_runner) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # Only the first thread should have hit refresh_token; the rest see the
        # already-fresh credentials under the lock and skip.
        assert provider.refresh_calls == 1
        # All four callers end up with identical fresh creds.
        assert {c.access for c in results} == {"a1"}
    finally:
        oauth.unregister_oauth_provider(provider.id)
