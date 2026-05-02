"""Regression tests for the gap fixes:

- Cancellation of orphan manual-input thread when the callback server wins.
- Tenacity retry on transient (5xx / network) refresh failures.
- Re-login hint when storage.save() fails after a successful refresh.
- Sanitized HTTP error messages (no response.text leak of refresh tokens).
- ``MSWEA_CLAUDE_CODE_VERSION`` env override for the Anthropic user-agent.
- Cross-process file lock on the credential file.
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from minisweagent import oauth
from minisweagent.oauth.types import (
    OAuthAuthInfo,
    OAuthCredentials,
    OAuthLoginCallbacks,
    OAuthProviderInterface,
    OAuthTransientError,
)

# ---------------------------------------------------------------------------
# 1. Manual-input cancellation
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_anthropic_callback(monkeypatch):
    from minisweagent.oauth import anthropic as anthropic_oauth

    state_holder: dict = {}

    class _DummyServer:
        def shutdown(self) -> None:
            return

        def server_close(self) -> None:
            return

    def fake_start(expected_state: str):
        result = anthropic_oauth._CallbackResult()
        state_holder["result"] = result
        state_holder["expected_state"] = expected_state
        return _DummyServer(), result, threading.Thread()

    monkeypatch.setattr(anthropic_oauth, "_start_callback_server", fake_start)
    return state_holder


@pytest.fixture
def patched_codex_callback(monkeypatch):
    from minisweagent.oauth import openai_codex as codex_oauth

    state_holder: dict = {}

    class _DummyServer:
        def shutdown(self) -> None:
            return

        def server_close(self) -> None:
            return

    def fake_start(expected_state: str):
        result = codex_oauth._CallbackResult()
        state_holder["result"] = result
        state_holder["expected_state"] = expected_state
        return _DummyServer(), result

    monkeypatch.setattr(codex_oauth, "_start_callback_server", fake_start)
    return state_holder


def test_anthropic_cancel_hook_called_when_server_wins(patched_anthropic_callback, monkeypatch):
    """Server callback wins → cancel hook fires so the manual-input thread can
    release stdin instead of orphaning a prompt that eats the next keystroke."""
    from minisweagent.oauth import anthropic as anthropic_oauth

    cancel_calls: list[None] = []
    manual_unblock = threading.Event()

    def on_auth(info: OAuthAuthInfo) -> None:
        # Server completes first.
        result = patched_anthropic_callback["result"]
        result.code = "auth-code-123"
        result.state = patched_anthropic_callback["expected_state"]
        result.event.set()

    def on_manual() -> str:
        # Block until the test allows the thread to exit.
        manual_unblock.wait(timeout=5)
        return ""

    def cancel_manual() -> None:
        cancel_calls.append(None)
        manual_unblock.set()

    callbacks = OAuthLoginCallbacks(
        on_auth=on_auth,
        on_prompt=lambda _p: pytest.fail("on_prompt should not run when server wins"),
        on_manual_code_input=on_manual,
        cancel_manual_code_input=cancel_manual,
    )

    # Skip the actual token exchange; we only care about the cancellation.
    monkeypatch.setattr(
        anthropic_oauth,
        "_exchange_code",
        lambda *a, **kw: OAuthCredentials(refresh="r", access="a", expires=0),
    )

    anthropic_oauth.login_anthropic(callbacks)
    assert cancel_calls == [None], "cancel_manual_code_input must be called when server wins"


def test_anthropic_cancel_hook_not_called_when_manual_input_won(patched_anthropic_callback, monkeypatch):
    """If the user finishes pasting before the server fires, no cancellation."""
    from minisweagent.oauth import anthropic as anthropic_oauth

    cancel_calls: list[None] = []

    def on_auth(_info: OAuthAuthInfo) -> None:
        return

    def on_manual() -> str:
        return "code-from-paste#state-from-paste"

    callbacks = OAuthLoginCallbacks(
        on_auth=on_auth,
        on_prompt=lambda _p: "",
        on_manual_code_input=on_manual,
        cancel_manual_code_input=lambda: cancel_calls.append(None),
    )

    monkeypatch.setattr(
        anthropic_oauth,
        "_exchange_code",
        lambda *a, **kw: OAuthCredentials(refresh="r", access="a", expires=0),
    )

    # state mismatch is fine — exchange is mocked, but the state nonce check
    # runs first. Suppress by patching state to match.
    with pytest.raises(RuntimeError, match="state mismatch"):
        anthropic_oauth.login_anthropic(callbacks)

    assert cancel_calls == [], "cancel hook should not fire when manual input completes first"


def test_codex_cancel_hook_called_when_server_wins(patched_codex_callback, monkeypatch):
    from minisweagent.oauth import openai_codex as codex_oauth

    cancel_calls: list[None] = []

    def on_auth(_info: OAuthAuthInfo) -> None:
        result = patched_codex_callback["result"]
        result.code = "auth-code"
        result.event.set()

    def on_manual() -> str:
        threading.Event().wait()
        raise AssertionError("unreachable")

    callbacks = OAuthLoginCallbacks(
        on_auth=on_auth,
        on_prompt=lambda _p: pytest.fail("on_prompt should not run"),
        on_manual_code_input=on_manual,
        cancel_manual_code_input=lambda: cancel_calls.append(None),
    )

    monkeypatch.setattr(
        codex_oauth,
        "_exchange_code",
        lambda *a, **kw: {
            "access_token": "header.eyJobHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOiB7ImNoYXRncHRfYWNjb3VudF9pZCI6ICJhYy0xIn19.sig",
            "refresh_token": "rt",
            "expires_in": 3600,
        },
    )

    codex_oauth.login_openai_codex(callbacks)
    assert cancel_calls == [None]


# ---------------------------------------------------------------------------
# 2. Token-rotation persistence guard
# ---------------------------------------------------------------------------


class _RotatingProvider(OAuthProviderInterface):
    id = "stub-rotate"
    name = "Stub"

    def __init__(self) -> None:
        self.refresh_calls = 0

    def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
        return OAuthCredentials(refresh="r0", access="a0", expires=0)

    def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        self.refresh_calls += 1
        return OAuthCredentials(
            refresh=f"rotated-{self.refresh_calls}",
            access=f"fresh-{self.refresh_calls}",
            expires=int(time.time() * 1000) + 60_000,
        )

    def get_api_key(self, credentials: OAuthCredentials) -> str:
        return credentials.access


def test_refresh_save_failure_raises_actionable_error(monkeypatch, tmp_path):
    """When ``storage.save`` fails after a successful refresh, the error must
    name the provider and tell the user to re-run ``mini-extra oauth login``
    so they understand the rotated refresh token may be lost."""
    monkeypatch.setenv("MSWEA_OAUTH_FILE", str(tmp_path / "oauth.json"))
    provider = _RotatingProvider()
    oauth.register_oauth_provider(provider)
    try:
        oauth.storage.save(provider.id, OAuthCredentials(refresh="r0", access="stale", expires=0))

        def boom(*_a, **_kw) -> None:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(oauth.storage, "save", boom)
        with pytest.raises(RuntimeError) as excinfo:
            oauth.refresh_provider(provider.id)
        msg = str(excinfo.value)
        assert "succeeded but failed to persist" in msg
        assert "mini-extra oauth login stub-rotate" in msg
        # Original cause preserved for debugging.
        assert isinstance(excinfo.value.__cause__, OSError)
    finally:
        oauth.unregister_oauth_provider(provider.id)


# ---------------------------------------------------------------------------
# 3. Tenacity retry on transient errors
# ---------------------------------------------------------------------------


class _FlakyProvider(OAuthProviderInterface):
    id = "stub-flaky"
    name = "Stub"

    def __init__(self, transient_failures: int = 2) -> None:
        self.calls = 0
        self.transient_failures = transient_failures

    def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
        return OAuthCredentials(refresh="r", access="a", expires=0)

    def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        self.calls += 1
        if self.calls <= self.transient_failures:
            raise OAuthTransientError(f"simulated 502 attempt {self.calls}")
        return OAuthCredentials(refresh="rotated", access="fresh", expires=int(time.time() * 1000) + 60_000)

    def get_api_key(self, credentials: OAuthCredentials) -> str:
        return credentials.access


def test_refresh_retries_transient_errors(monkeypatch, tmp_path):
    """Two transient 502s then a success must yield a successful refresh."""
    monkeypatch.setenv("MSWEA_OAUTH_FILE", str(tmp_path / "oauth.json"))
    monkeypatch.setenv("MSWEA_OAUTH_REFRESH_RETRY_ATTEMPTS", "5")
    monkeypatch.setenv("MSWEA_OAUTH_REFRESH_RETRY_WAIT_MIN", "0")
    monkeypatch.setenv("MSWEA_OAUTH_REFRESH_RETRY_WAIT_MAX", "0")

    provider = _FlakyProvider(transient_failures=2)
    oauth.register_oauth_provider(provider)
    try:
        oauth.storage.save(provider.id, OAuthCredentials(refresh="r0", access="stale", expires=0))
        creds = oauth.refresh_provider(provider.id)
        assert creds.access == "fresh"
        assert provider.calls == 3  # 2 transient failures + 1 success
    finally:
        oauth.unregister_oauth_provider(provider.id)


def test_refresh_does_not_retry_permanent_errors(monkeypatch, tmp_path):
    """Plain ``RuntimeError`` (e.g. 401 invalid_grant) must not retry."""
    monkeypatch.setenv("MSWEA_OAUTH_FILE", str(tmp_path / "oauth.json"))
    monkeypatch.setenv("MSWEA_OAUTH_REFRESH_RETRY_ATTEMPTS", "5")
    monkeypatch.setenv("MSWEA_OAUTH_REFRESH_RETRY_WAIT_MIN", "0")
    monkeypatch.setenv("MSWEA_OAUTH_REFRESH_RETRY_WAIT_MAX", "0")

    class _PermProvider(OAuthProviderInterface):
        id = "stub-perm"
        name = "Stub"
        calls = 0

        def login(self, _cb):  # noqa: ANN001
            return OAuthCredentials(refresh="r", access="a", expires=0)

        def refresh_token(self, _c):  # noqa: ANN001
            type(self).calls += 1
            raise RuntimeError("HTTP request failed. status=401; reason=Unauthorized")

        def get_api_key(self, c):  # noqa: ANN001
            return c.access

    provider = _PermProvider()
    oauth.register_oauth_provider(provider)
    try:
        oauth.storage.save(provider.id, OAuthCredentials(refresh="r0", access="stale", expires=0))
        with pytest.raises(RuntimeError, match="status=401"):
            oauth.refresh_provider(provider.id)
        assert _PermProvider.calls == 1, "permanent errors must not be retried"
    finally:
        oauth.unregister_oauth_provider(provider.id)


# ---------------------------------------------------------------------------
# 4. Sanitized HTTP error messages
# ---------------------------------------------------------------------------


def _err_response(status: int, reason: str, text: str) -> MagicMock:
    r = MagicMock()
    r.ok = False
    r.status_code = status
    r.reason = reason
    r.text = text
    return r


def test_anthropic_post_json_error_does_not_leak_body():
    """``response.text`` must not appear in the exception message — IdPs can
    echo the rejected refresh token in error responses."""
    from minisweagent.oauth.anthropic import _post_json

    leaked_token = "sk-ant-refresh-DO-NOT-LEAK-abc123"
    body = json.dumps({"error": "invalid_grant", "refresh_token": leaked_token})
    resp = _err_response(400, "Bad Request", body)
    with patch("minisweagent.oauth.anthropic.requests.post", return_value=resp):
        with pytest.raises(RuntimeError) as excinfo:
            _post_json("https://example.test/oauth/token", {})
    assert leaked_token not in str(excinfo.value)
    assert "status=400" in str(excinfo.value)
    assert "reason=Bad Request" in str(excinfo.value)


def test_anthropic_post_json_5xx_raises_transient():
    from minisweagent.oauth.anthropic import _post_json

    resp = _err_response(503, "Service Unavailable", "leaky body")
    with patch("minisweagent.oauth.anthropic.requests.post", return_value=resp):
        with pytest.raises(OAuthTransientError):
            _post_json("https://example.test/oauth/token", {})


def test_anthropic_invalid_json_does_not_leak_body():
    from minisweagent.oauth.anthropic import _post_json

    leaked = "<html>refresh=sk-ant-LEAK</html>"
    resp = MagicMock()
    resp.ok = True
    resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
    resp.text = leaked
    with patch("minisweagent.oauth.anthropic.requests.post", return_value=resp):
        with pytest.raises(RuntimeError) as excinfo:
            _post_json("https://example.test/oauth/token", {})
    assert leaked not in str(excinfo.value)


def test_codex_refresh_error_does_not_leak_body():
    from minisweagent.oauth.openai_codex import _refresh

    leaked = "refresh=rt-LEAK"
    resp = _err_response(401, "Unauthorized", leaked)
    with patch("minisweagent.oauth.openai_codex.requests.post", return_value=resp):
        with pytest.raises(RuntimeError) as excinfo:
            _refresh("rt-LEAK")
    assert leaked not in str(excinfo.value)
    assert "status=401" in str(excinfo.value)


def test_codex_refresh_5xx_raises_transient():
    from minisweagent.oauth.openai_codex import _refresh

    resp = _err_response(502, "Bad Gateway", "leaky")
    with patch("minisweagent.oauth.openai_codex.requests.post", return_value=resp):
        with pytest.raises(OAuthTransientError):
            _refresh("rt")


def test_copilot_refresh_error_does_not_leak_body():
    from minisweagent.oauth.github_copilot import refresh_github_copilot_token

    leaked = "Bearer gh_pat_LEAK"
    resp = _err_response(401, "Unauthorized", leaked)
    with patch("minisweagent.oauth.github_copilot.requests.get", return_value=resp):
        with pytest.raises(RuntimeError) as excinfo:
            refresh_github_copilot_token("gh_pat_LEAK")
    assert leaked not in str(excinfo.value)


def test_copilot_post_form_error_does_not_leak_body():
    from minisweagent.oauth.github_copilot import _post_form

    leaked = "device_code=secret-LEAK"
    resp = _err_response(400, "Bad Request", leaked)
    with patch("minisweagent.oauth.github_copilot.requests.post", return_value=resp):
        with pytest.raises(RuntimeError) as excinfo:
            _post_form("https://github.com/login/device/code", {})
    assert leaked not in str(excinfo.value)


def test_copilot_post_form_5xx_raises_transient():
    from minisweagent.oauth.github_copilot import _post_form

    resp = _err_response(503, "Service Unavailable", "leaky")
    with patch("minisweagent.oauth.github_copilot.requests.post", return_value=resp):
        with pytest.raises(OAuthTransientError):
            _post_form("https://github.com/x", {})


# ---------------------------------------------------------------------------
# 5. CLAUDE_CODE_VERSION env override
# ---------------------------------------------------------------------------


def test_claude_code_version_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MSWEA_OAUTH_FILE", str(tmp_path / "oauth.json"))
    monkeypatch.setenv("MSWEA_CLAUDE_CODE_VERSION", "9.9.9-hotfix")

    from minisweagent.models.oauth_model import OAuthLitellmModel

    class _Stub(OAuthProviderInterface):
        id = "anthropic"
        name = "stub"

        def login(self, _cb):  # noqa: ANN001
            return OAuthCredentials(refresh="r", access="a", expires=0)

        def refresh_token(self, c):  # noqa: ANN001
            return c

        def get_api_key(self, c):  # noqa: ANN001
            return c.access

    oauth.register_oauth_provider(_Stub())
    try:
        oauth.storage.save(
            "anthropic",
            OAuthCredentials(refresh="r", access="t", expires=int(time.time() * 1000) + 60_000),
        )
        model = OAuthLitellmModel(model_name="anthropic/claude-sonnet-4-5", oauth_provider="anthropic")
        kwargs = model._resolve_oauth_kwargs()
        assert kwargs["extra_headers"]["user-agent"] == "claude-cli/9.9.9-hotfix"
    finally:
        oauth.restore_oauth_provider("anthropic")


def test_claude_code_version_default(monkeypatch, tmp_path):
    monkeypatch.setenv("MSWEA_OAUTH_FILE", str(tmp_path / "oauth.json"))
    monkeypatch.delenv("MSWEA_CLAUDE_CODE_VERSION", raising=False)

    from minisweagent.models.oauth_model import DEFAULT_CLAUDE_CODE_VERSION, OAuthLitellmModel

    class _Stub(OAuthProviderInterface):
        id = "anthropic"
        name = "stub"

        def login(self, _cb):  # noqa: ANN001
            return OAuthCredentials(refresh="r", access="a", expires=0)

        def refresh_token(self, c):  # noqa: ANN001
            return c

        def get_api_key(self, c):  # noqa: ANN001
            return c.access

    oauth.register_oauth_provider(_Stub())
    try:
        oauth.storage.save(
            "anthropic",
            OAuthCredentials(refresh="r", access="t", expires=int(time.time() * 1000) + 60_000),
        )
        model = OAuthLitellmModel(model_name="anthropic/claude-sonnet-4-5", oauth_provider="anthropic")
        kwargs = model._resolve_oauth_kwargs()
        assert kwargs["extra_headers"]["user-agent"] == f"claude-cli/{DEFAULT_CLAUDE_CODE_VERSION}"
    finally:
        oauth.restore_oauth_provider("anthropic")


# ---------------------------------------------------------------------------
# 6. Cross-process file lock (sidecar exists & is honored within process)
# ---------------------------------------------------------------------------


def test_storage_lock_file_created(monkeypatch, tmp_path):
    """The sidecar lock file should be created next to the credentials file."""
    target = tmp_path / "oauth.json"
    monkeypatch.setenv("MSWEA_OAUTH_FILE", str(target))
    oauth.storage.save("p", OAuthCredentials(refresh="r", access="a", expires=1))
    assert (tmp_path / "oauth.json.lock").exists()


def test_storage_concurrent_save_no_data_loss(monkeypatch, tmp_path):
    """Threads in the same process already serialize via _LOCK; assert the
    flock-protected path also produces a consistent file (no torn writes,
    every entry survives)."""
    monkeypatch.setenv("MSWEA_OAUTH_FILE", str(tmp_path / "oauth.json"))

    def _worker(i: int) -> None:
        oauth.storage.save(
            f"p{i}",
            OAuthCredentials(refresh=f"r{i}", access=f"a{i}", expires=i),
        )

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    raw = json.loads((tmp_path / "oauth.json").read_text())
    assert set(raw.keys()) == {f"p{i}" for i in range(8)}


@pytest.mark.skipif(__import__("sys").platform == "win32", reason="fcntl only available on POSIX")
def test_storage_flock_serializes_writers(monkeypatch, tmp_path):
    """Two threads bypassing the in-process ``_LOCK`` must still serialize via
    fcntl.flock — that is the only barrier between separate processes."""
    import fcntl

    target = tmp_path / "oauth.json"
    monkeypatch.setenv("MSWEA_OAUTH_FILE", str(target))
    target.parent.mkdir(parents=True, exist_ok=True)

    # Hold the flock from one "process" (a thread that opens the lock file
    # directly) and confirm a save() from another thread blocks until release.
    lock_path = target.parent / (target.name + ".lock")
    lock_path.touch()
    held_fd = __import__("os").open(str(lock_path), __import__("os").O_RDWR)
    fcntl.flock(held_fd, fcntl.LOCK_EX)

    finished = threading.Event()

    def _saver() -> None:
        oauth.storage.save("p", OAuthCredentials(refresh="r", access="a", expires=1))
        finished.set()

    t = threading.Thread(target=_saver)
    t.start()
    try:
        # save() must block on flock for at least a moment; assert it did not
        # complete before we release the lock.
        assert not finished.wait(timeout=0.3), "save() must block on the file lock"
    finally:
        fcntl.flock(held_fd, fcntl.LOCK_UN)
        __import__("os").close(held_fd)
    t.join(timeout=5)
    assert finished.is_set(), "save() must complete after the lock is released"
