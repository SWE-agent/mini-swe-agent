"""Tests for provider network helpers using mocked requests."""

from __future__ import annotations

import base64
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.oauth.types import OAuthCredentials

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ok_response(data: dict) -> MagicMock:
    r = MagicMock()
    r.ok = True
    r.json.return_value = data
    return r


def _make_error_response(status: int, text: str = "error") -> MagicMock:
    r = MagicMock()
    r.ok = False
    r.status_code = status
    r.text = text
    r.reason = text
    r.url = "https://example.com"
    return r


def _make_jwt(payload: dict) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"header.{encoded}.signature"


# ---------------------------------------------------------------------------
# Anthropic refresh
# ---------------------------------------------------------------------------


def test_refresh_anthropic_token_success():
    from minisweagent.oauth.anthropic import refresh_anthropic_token

    resp = _make_ok_response({"access_token": "at", "refresh_token": "rt", "expires_in": 3600})
    with patch("minisweagent.oauth.anthropic.requests.post", return_value=resp):
        creds = refresh_anthropic_token("old-rt")

    assert creds.access == "at"
    assert creds.refresh == "rt"
    assert creds.expires > int(time.time() * 1000)


def test_refresh_anthropic_token_http_error():
    from minisweagent.oauth.anthropic import refresh_anthropic_token

    resp = _make_error_response(401, "Unauthorized")
    with patch("minisweagent.oauth.anthropic.requests.post", return_value=resp):
        with pytest.raises(RuntimeError, match="HTTP request failed"):
            refresh_anthropic_token("bad-token")


def test_refresh_anthropic_token_invalid_json():
    from minisweagent.oauth.anthropic import refresh_anthropic_token

    resp = MagicMock()
    resp.ok = True
    resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
    resp.text = "not json"
    resp.url = "https://platform.claude.com/v1/oauth/token"
    with patch("minisweagent.oauth.anthropic.requests.post", return_value=resp):
        with pytest.raises(RuntimeError, match="Invalid JSON"):
            refresh_anthropic_token("rt")


# ---------------------------------------------------------------------------
# Anthropic provider class
# ---------------------------------------------------------------------------


def test_anthropic_provider_get_api_key():
    from minisweagent.oauth.anthropic import anthropic_oauth_provider

    creds = OAuthCredentials(refresh="r", access="tok", expires=0)
    assert anthropic_oauth_provider.get_api_key(creds) == "tok"


def test_anthropic_provider_attributes():
    from minisweagent.oauth.anthropic import anthropic_oauth_provider

    assert anthropic_oauth_provider.id == "anthropic"
    assert anthropic_oauth_provider.uses_callback_server is True


def test_anthropic_provider_refresh_delegates():
    from minisweagent.oauth.anthropic import anthropic_oauth_provider

    resp = _make_ok_response({"access_token": "new-at", "refresh_token": "new-rt", "expires_in": 3600})
    with patch("minisweagent.oauth.anthropic.requests.post", return_value=resp):
        creds = anthropic_oauth_provider.refresh_token(OAuthCredentials(refresh="old-rt", access="x", expires=0))
    assert creds.access == "new-at"


# ---------------------------------------------------------------------------
# GitHub Copilot refresh
# ---------------------------------------------------------------------------


def test_refresh_github_copilot_token_success():
    from minisweagent.oauth.github_copilot import refresh_github_copilot_token

    expires_at = int(time.time()) + 1800
    resp = _make_ok_response({"token": "copilot-tok", "expires_at": expires_at})
    with patch("minisweagent.oauth.github_copilot.requests.get", return_value=resp):
        creds = refresh_github_copilot_token("gh-access-token")

    assert creds.access == "copilot-tok"
    assert creds.refresh == "gh-access-token"
    assert creds.extra == {}


def test_refresh_github_copilot_token_enterprise_stored_in_extra():
    from minisweagent.oauth.github_copilot import refresh_github_copilot_token

    expires_at = int(time.time()) + 1800
    resp = _make_ok_response({"token": "ent-tok", "expires_at": expires_at})
    with patch("minisweagent.oauth.github_copilot.requests.get", return_value=resp):
        creds = refresh_github_copilot_token("gh-tok", enterprise_domain="myco.ghe.com")

    assert creds.extra.get("enterprise_url") == "myco.ghe.com"


def test_refresh_github_copilot_token_http_error():
    from minisweagent.oauth.github_copilot import refresh_github_copilot_token

    resp = _make_error_response(403)
    with patch("minisweagent.oauth.github_copilot.requests.get", return_value=resp):
        with pytest.raises(RuntimeError, match="Copilot token request failed"):
            refresh_github_copilot_token("bad-token")


def test_refresh_github_copilot_token_invalid_response_fields():
    from minisweagent.oauth.github_copilot import refresh_github_copilot_token

    resp = _make_ok_response({"no_token": "here"})
    with patch("minisweagent.oauth.github_copilot.requests.get", return_value=resp):
        with pytest.raises(RuntimeError, match="Invalid Copilot token response"):
            refresh_github_copilot_token("tok")


def test_copilot_provider_get_api_key():
    from minisweagent.oauth.github_copilot import github_copilot_oauth_provider

    creds = OAuthCredentials(refresh="r", access="tok", expires=0)
    assert github_copilot_oauth_provider.get_api_key(creds) == "tok"


def test_copilot_provider_attributes():
    from minisweagent.oauth.github_copilot import github_copilot_oauth_provider

    assert github_copilot_oauth_provider.id == "github-copilot"
    assert github_copilot_oauth_provider.uses_callback_server is False


def test_copilot_provider_refresh_reads_enterprise_from_extra():
    from minisweagent.oauth.github_copilot import github_copilot_oauth_provider

    expires_at = int(time.time()) + 1800
    resp = _make_ok_response({"token": "ent-tok", "expires_at": expires_at})
    creds = OAuthCredentials(refresh="gh-tok", access="x", expires=0, extra={"enterprise_url": "corp.ghe.com"})
    with patch("minisweagent.oauth.github_copilot.requests.get", return_value=resp) as mock_get:
        result = github_copilot_oauth_provider.refresh_token(creds)

    assert result.extra.get("enterprise_url") == "corp.ghe.com"
    call_url = mock_get.call_args[0][0]
    assert "corp.ghe.com" in call_url


# ---------------------------------------------------------------------------
# GitHub Copilot _poll_for_access_token
# ---------------------------------------------------------------------------


def test_poll_for_token_success():
    from minisweagent.oauth.github_copilot import _poll_for_access_token

    with (
        patch("minisweagent.oauth.github_copilot.time.sleep"),
        patch("minisweagent.oauth.github_copilot._post_form") as mock_post,
    ):
        mock_post.side_effect = [
            {"error": "authorization_pending"},
            {"access_token": "gh-tok"},
        ]
        result = _poll_for_access_token("github.com", "device-code", 0, 9999)
    assert result == "gh-tok"


def test_poll_for_token_explicit_error():
    from minisweagent.oauth.github_copilot import _poll_for_access_token

    with (
        patch("minisweagent.oauth.github_copilot.time.sleep"),
        patch("minisweagent.oauth.github_copilot._post_form") as mock_post,
    ):
        mock_post.return_value = {"error": "access_denied", "error_description": "Denied by user"}
        with pytest.raises(RuntimeError, match="access_denied: Denied by user"):
            _poll_for_access_token("github.com", "dc", 0, 9999)


def test_poll_for_token_error_no_description():
    from minisweagent.oauth.github_copilot import _poll_for_access_token

    with (
        patch("minisweagent.oauth.github_copilot.time.sleep"),
        patch("minisweagent.oauth.github_copilot._post_form") as mock_post,
    ):
        mock_post.return_value = {"error": "expired_token"}
        with pytest.raises(RuntimeError, match="expired_token"):
            _poll_for_access_token("github.com", "dc", 0, 9999)


def test_poll_for_token_post_form_raises():
    from minisweagent.oauth.github_copilot import _poll_for_access_token

    with (
        patch("minisweagent.oauth.github_copilot.time.sleep"),
        patch("minisweagent.oauth.github_copilot._post_form", side_effect=RuntimeError("network error")),
    ):
        with pytest.raises(RuntimeError, match="Device flow failed: network error"):
            _poll_for_access_token("github.com", "dc", 0, 9999)


def test_poll_for_token_expires_without_slow_downs():
    from minisweagent.oauth.github_copilot import _poll_for_access_token

    with patch("minisweagent.oauth.github_copilot.time.sleep"):
        with pytest.raises(RuntimeError, match="Device flow timed out$"):
            _poll_for_access_token("github.com", "dc", 0, -1)


def test_poll_for_token_slow_down_updates_interval():
    from minisweagent.oauth.github_copilot import _poll_for_access_token

    # time.time() call sequence:
    #   1. deadline = time.time() + expires_in  → 0, deadline=300
    #   2. while time.time() < deadline          → 1, enter loop
    #   3. deadline - time.time() in wait_for    → 1
    #   4. (slow_down → continue) while check    → 99999, exit loop → raises
    with (
        patch("minisweagent.oauth.github_copilot.time.sleep"),
        patch("minisweagent.oauth.github_copilot._post_form") as mock_post,
        patch("minisweagent.oauth.github_copilot.time.time", side_effect=[0, 1, 1, 99999]),
    ):
        mock_post.return_value = {"error": "slow_down", "interval": 15}
        with pytest.raises(RuntimeError, match="slow_down responses"):
            _poll_for_access_token("github.com", "dc", 5, 300)


# ---------------------------------------------------------------------------
# OpenAI Codex
# ---------------------------------------------------------------------------


def test_b64url_decode_no_padding():
    from minisweagent.oauth.openai_codex import _b64url_decode

    assert _b64url_decode("aGVsbG8") == b"hello"


def test_b64url_decode_various_lengths():
    from minisweagent.oauth.openai_codex import _b64url_decode

    for original in (b"a", b"ab", b"abc", b"abcd"):
        encoded = base64.urlsafe_b64encode(original).rstrip(b"=").decode()
        assert _b64url_decode(encoded) == original


def test_refresh_openai_codex_token_success():
    from minisweagent.oauth.openai_codex import refresh_openai_codex_token

    account_id = "acc-42"
    access_token = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": account_id}})
    resp = _make_ok_response({"access_token": access_token, "refresh_token": "new-rt", "expires_in": 3600})
    with patch("minisweagent.oauth.openai_codex.requests.post", return_value=resp):
        creds = refresh_openai_codex_token("old-rt")

    assert creds.access == access_token
    assert creds.refresh == "new-rt"
    assert creds.extra.get("account_id") == account_id


def test_refresh_openai_codex_token_missing_account_id_raises():
    from minisweagent.oauth.openai_codex import refresh_openai_codex_token

    access_token = _make_jwt({})  # no account_id
    resp = _make_ok_response({"access_token": access_token, "refresh_token": "rt", "expires_in": 3600})
    with patch("minisweagent.oauth.openai_codex.requests.post", return_value=resp):
        with pytest.raises(RuntimeError, match="account_id"):
            refresh_openai_codex_token("old-rt")


def test_refresh_openai_codex_token_http_error():
    from minisweagent.oauth.openai_codex import refresh_openai_codex_token

    resp = _make_error_response(400, "Bad request")
    with patch("minisweagent.oauth.openai_codex.requests.post", return_value=resp):
        with pytest.raises(RuntimeError, match="Codex token refresh failed"):
            refresh_openai_codex_token("bad-refresh")


def test_codex_provider_get_api_key():
    from minisweagent.oauth.openai_codex import openai_codex_oauth_provider

    creds = OAuthCredentials(refresh="r", access="tok", expires=0)
    assert openai_codex_oauth_provider.get_api_key(creds) == "tok"


def test_codex_provider_attributes():
    from minisweagent.oauth.openai_codex import openai_codex_oauth_provider

    assert openai_codex_oauth_provider.id == "openai-codex"
    assert openai_codex_oauth_provider.uses_callback_server is True


def test_codex_provider_refresh_delegates():
    from minisweagent.oauth.openai_codex import openai_codex_oauth_provider

    account_id = "acc-1"
    access_token = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": account_id}})
    resp = _make_ok_response({"access_token": access_token, "refresh_token": "rt", "expires_in": 3600})
    with patch("minisweagent.oauth.openai_codex.requests.post", return_value=resp):
        creds = openai_codex_oauth_provider.refresh_token(OAuthCredentials(refresh="old-rt", access="x", expires=0))
    assert creds.extra.get("account_id") == account_id


def test_codex_refresh_applies_expiry_safety_margin():
    """Codex expiry must subtract EXPIRY_SAFETY_MS for parity with anthropic /
    copilot, so a borderline-expired token does not slip past the
    AuthenticationError abort policy without a chance to refresh."""
    from minisweagent.oauth.openai_codex import EXPIRY_SAFETY_MS, refresh_openai_codex_token

    account_id = "acc-7"
    access_token = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": account_id}})
    expires_in = 3600
    resp = _make_ok_response({"access_token": access_token, "refresh_token": "rt", "expires_in": expires_in})
    before_ms = int(time.time() * 1000)
    with patch("minisweagent.oauth.openai_codex.requests.post", return_value=resp):
        creds = refresh_openai_codex_token("old-rt")
    after_ms = int(time.time() * 1000)

    # creds.expires == now_ms_inside + expires_in*1000 - EXPIRY_SAFETY_MS, so
    # it must fall in [before_ms, after_ms] + (expires_in*1000 - EXPIRY_SAFETY_MS).
    expected_lo = before_ms + expires_in * 1000 - EXPIRY_SAFETY_MS
    expected_hi = after_ms + expires_in * 1000 - EXPIRY_SAFETY_MS
    assert expected_lo <= creds.expires <= expected_hi
    # Sanity: the safety margin actually subtracts time vs the naive expiry.
    assert creds.expires < before_ms + expires_in * 1000


def test_copilot_post_form_invalid_json_raises_runtime_error():
    """Non-JSON 200 responses must surface as RuntimeError, not JSONDecodeError."""
    from minisweagent.oauth.github_copilot import _post_form

    resp = MagicMock()
    resp.ok = True
    resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
    resp.text = "not json"
    with patch("minisweagent.oauth.github_copilot.requests.post", return_value=resp):
        with pytest.raises(RuntimeError, match="Invalid JSON"):
            _post_form("https://github.com/x", {})


def test_copilot_refresh_invalid_json_raises_runtime_error():
    from minisweagent.oauth.github_copilot import refresh_github_copilot_token

    resp = MagicMock()
    resp.ok = True
    resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
    resp.text = "not json"
    with patch("minisweagent.oauth.github_copilot.requests.get", return_value=resp):
        with pytest.raises(RuntimeError, match="Invalid JSON from Copilot"):
            refresh_github_copilot_token("gh-tok")


def test_anthropic_login_state_is_not_pkce_verifier(monkeypatch):
    """The PKCE verifier must stay client-side; ``state`` should be a separate
    nonce so the front-channel redirect URL does not leak the verifier."""
    import minisweagent.oauth.anthropic as anth

    captured: dict = {}

    class _FakeServer:
        def shutdown(self) -> None:  # noqa: D401 - protocol stub
            pass

        def server_close(self) -> None:
            pass

    def _fake_start(expected_state: str):
        captured["expected_state"] = expected_state
        result = anth._CallbackResult()
        result.event.set()  # immediately unblock
        return _FakeServer(), result, None

    monkeypatch.setattr(anth, "_start_callback_server", _fake_start)

    auth_url_holder: dict = {}

    def _on_auth(info) -> None:
        auth_url_holder["url"] = info.url

    def _on_prompt(_):
        # Force the flow to bail out before exchanging anything.
        return ""

    from minisweagent.oauth.types import OAuthLoginCallbacks

    callbacks = OAuthLoginCallbacks(on_auth=_on_auth, on_prompt=_on_prompt)
    with pytest.raises(RuntimeError, match="Missing authorization code"):
        anth.login_anthropic(callbacks)

    state_in_url = auth_url_holder["url"].split("state=", 1)[1].split("&", 1)[0]
    # State must be the random nonce passed to the callback server, not the
    # PKCE verifier (which never appears outside ``code_verifier``).
    assert state_in_url == captured["expected_state"]
    assert "code_verifier" not in auth_url_holder["url"]
