"""Regression tests for the OAuth callback HTTP handlers.

Bug: invalid callback requests (missing code/state, state mismatch) used to
respond 400 but never set the wakeup event, hanging the login flow forever.
"""

import threading
from io import BytesIO

import pytest

from minisweagent.oauth import anthropic as anthropic_oauth
from minisweagent.oauth import openai_codex as codex_oauth


class _FakeRequest:
    """Minimal stand-in for an HTTP request object accepted by BaseHTTPRequestHandler."""

    def __init__(self, raw: bytes) -> None:
        self.rfile = BytesIO(raw)
        self.wfile = BytesIO()

    def makefile(self, *_a, **_kw):
        return self.rfile


def _invoke_handler(handler_cls, path: str):
    raw = f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode()
    request = _FakeRequest(raw)

    class _ServerStub:
        server_address = ("127.0.0.1", 0)

    handler = handler_cls.__new__(handler_cls)
    handler.rfile = request.rfile
    handler.wfile = request.wfile
    handler.client_address = ("127.0.0.1", 0)
    handler.server = _ServerStub()
    handler.request = request
    handler.command = "GET"
    handler.path = path
    handler.headers = {}
    handler.request_version = "HTTP/1.1"
    handler.raw_requestline = raw.split(b"\r\n", 1)[0] + b"\r\n"
    handler.protocol_version = "HTTP/1.0"
    # parse_request fills in command/path/headers from raw_requestline.
    handler.parse_request()
    handler.do_GET()


@pytest.mark.parametrize(
    ("path", "expected_error"),
    [
        ("/callback?code=abc", "Missing code or state parameter"),
        ("/callback?state=xyz", "Missing code or state parameter"),
        ("/callback?code=abc&state=wrong", "State mismatch"),
    ],
)
def test_anthropic_handler_400_paths_set_event(path: str, expected_error: str):
    result = anthropic_oauth._CallbackResult()
    handler_cls = anthropic_oauth._make_handler(expected_state="expected-state", result=result)

    _invoke_handler(handler_cls, path)

    assert result.event.is_set(), "event must be set so the login flow does not hang"
    assert result.error == expected_error
    assert result.code is None


@pytest.mark.parametrize(
    ("path", "expected_error"),
    [
        ("/auth/callback?state=wrong", "State mismatch"),
        ("/auth/callback?state=expected-state", "Missing authorization code"),
    ],
)
def test_codex_handler_400_paths_set_event(path: str, expected_error: str):
    result = codex_oauth._CallbackResult()
    handler_cls = codex_oauth._make_handler(expected_state="expected-state", result=result)

    _invoke_handler(handler_cls, path)

    assert result.event.is_set()
    assert result.error == expected_error
    assert result.code is None


def test_anthropic_wait_has_default_timeout():
    """Sanity-check: the configured callback wait must have a finite timeout."""
    assert anthropic_oauth.CALLBACK_WAIT_TIMEOUT_S > 0
    assert anthropic_oauth.CALLBACK_WAIT_TIMEOUT_S < 24 * 60 * 60


def test_codex_wait_has_default_timeout():
    assert codex_oauth.CALLBACK_WAIT_TIMEOUT_S > 0
    assert codex_oauth.CALLBACK_WAIT_TIMEOUT_S < 24 * 60 * 60


def test_anthropic_wait_unblocks_after_400(monkeypatch):
    """Full integration: a 400-triggering hit on the callback unblocks the waiter."""
    from urllib.request import urlopen

    monkeypatch.setattr(anthropic_oauth, "CALLBACK_PORT", 0)  # let OS pick

    result = anthropic_oauth._CallbackResult()
    handler_cls = anthropic_oauth._make_handler(expected_state="expected-state", result=result)

    from http.server import HTTPServer

    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        try:
            urlopen(f"http://127.0.0.1:{port}/callback?code=x&state=wrong", timeout=2)
        except Exception:  # noqa: BLE001
            pass
        assert result.event.wait(timeout=2.0), "event must fire after invalid callback"
        assert result.error == "State mismatch"
    finally:
        server.shutdown()
        server.server_close()
