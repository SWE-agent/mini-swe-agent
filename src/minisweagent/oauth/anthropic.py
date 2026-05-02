"""Anthropic (Claude Pro/Max) OAuth flow.

Ported from pi-mono (packages/ai/src/utils/oauth/anthropic.ts).

Uses authorization-code + PKCE with a localhost redirect listener. If the user
cannot reach the browser (e.g. running on a remote box) they may paste the
final redirect URL or ``code#state`` instead.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import requests

from minisweagent.oauth.oauth_page import oauth_error_html, oauth_success_html
from minisweagent.oauth.pkce import generate_pkce
from minisweagent.oauth.types import (
    OAuthAuthInfo,
    OAuthCredentials,
    OAuthLoginCallbacks,
    OAuthPrompt,
    OAuthProviderInterface,
    OAuthTransientError,
)

CLIENT_ID = base64.b64decode("OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl").decode()
AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
# REDIRECT_URI is hard-coded to ``localhost`` because Anthropic whitelists that
# exact string. The server must bind to the same hostname to avoid a v4/v6
# split on dual-stack Linux where ``localhost`` → ``::1``.
CALLBACK_HOST = "localhost"
CALLBACK_PORT = int(os.getenv("MSWEA_ANTHROPIC_CALLBACK_PORT", "53692"))
CALLBACK_PATH = "/callback"
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}{CALLBACK_PATH}"
SCOPES = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"
EXPIRY_SAFETY_MS = 5 * 60 * 1000
CALLBACK_WAIT_TIMEOUT_S = float(os.getenv("MSWEA_OAUTH_CALLBACK_TIMEOUT", "300"))


class _CallbackResult:
    def __init__(self) -> None:
        self.code: str | None = None
        self.state: str | None = None
        self.error: str | None = None
        self.event = threading.Event()


def _make_handler(expected_state: str, result: _CallbackResult):
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != CALLBACK_PATH:
                self._respond(404, oauth_error_html("Callback route not found."))
                return
            params = parse_qs(parsed.query)
            err = params.get("error", [None])[0]
            if err:
                result.error = err
                self._respond(400, oauth_error_html("Anthropic authentication did not complete.", f"Error: {err}"))
                result.event.set()
                return
            code = params.get("code", [None])[0]
            state = params.get("state", [None])[0]
            if not code or not state:
                result.error = "Missing code or state parameter"
                self._respond(400, oauth_error_html("Missing code or state parameter."))
                result.event.set()
                return
            if state != expected_state:
                result.error = "State mismatch"
                self._respond(400, oauth_error_html("State mismatch."))
                result.event.set()
                return
            result.code = code
            result.state = state
            self._respond(200, oauth_success_html("Anthropic authentication completed. You can close this window."))
            result.event.set()

        def _respond(self, status: int, body: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            data = body.encode("utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return _Handler


def _start_callback_server(
    expected_state: str,
) -> tuple[HTTPServer | None, _CallbackResult, threading.Thread | None]:
    result = _CallbackResult()
    handler = _make_handler(expected_state, result)
    try:
        server = HTTPServer((CALLBACK_HOST, CALLBACK_PORT), handler)
    except OSError:
        # Port busy (another login in flight, or stale binding). Fall back to
        # manual paste flow rather than crashing the login.
        return None, result, None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, result, thread


def _parse_authorization_input(text: str) -> tuple[str | None, str | None]:
    value = text.strip()
    if not value:
        return None, None
    try:
        url = urlparse(value)
        if url.scheme and url.netloc:
            params = parse_qs(url.query)
            return (params.get("code", [None])[0], params.get("state", [None])[0])
    except ValueError:
        pass
    if "#" in value:
        code, state = value.split("#", 1)
        return code or None, state or None
    if "code=" in value:
        query = value[1:] if value.startswith("?") else value
        params = parse_qs(query)
        return (params.get("code", [None])[0], params.get("state", [None])[0])
    return value, None


def _post_json(url: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        response = requests.post(
            url,
            json=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=30,
        )
    except requests.exceptions.RequestException as exc:
        # Network-level failure (connection refused, DNS, timeout). Mark as
        # transient so the retry layer reattempts.
        raise OAuthTransientError(
            f"HTTP request failed (network): url={url}; error={type(exc).__name__}: {exc}"
        ) from exc
    if not response.ok:
        # Don't echo response.text — the IdP can include the refresh token in
        # error responses (e.g. ``invalid_grant`` payloads that quote the
        # offending token). status + reason is enough for the operator log.
        message = f"HTTP request failed. status={response.status_code}; reason={response.reason}; url={url}"
        if response.status_code >= 500:
            raise OAuthTransientError(message)
        raise RuntimeError(message)
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        # Bodies on JSON-decode errors are also untrusted — drop them for the
        # same reason as the HTTP error path above.
        raise RuntimeError(f"Invalid JSON from {url}") from exc


def _credentials_from_token_response(data: dict[str, Any], *, context: str) -> OAuthCredentials:
    """Validate an Anthropic token endpoint response and convert it to credentials.

    Raises ``RuntimeError`` (not ``KeyError``) when required fields are missing
    or malformed, so callers see an actionable error instead of a stack trace
    pointing at a bare dict lookup.
    """
    if not isinstance(data, dict):
        raise RuntimeError(f"Anthropic {context} returned non-object payload: {data!r}")

    missing = [k for k in ("access_token", "refresh_token", "expires_in") if k not in data]
    if missing:
        keys = ", ".join(sorted(data.keys())) or "<empty>"
        raise RuntimeError(
            f"Anthropic {context} response missing required fields {missing}. Received keys: {keys}. Body: {data!r}"
        )

    access = data["access_token"]
    refresh = data["refresh_token"]
    expires_in = data["expires_in"]

    if not isinstance(access, str) or not access:
        raise RuntimeError(f"Anthropic {context} returned non-string access_token: {access!r}")
    if not isinstance(refresh, str) or not refresh:
        raise RuntimeError(f"Anthropic {context} returned non-string refresh_token: {refresh!r}")
    try:
        expires_seconds = int(expires_in)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Anthropic {context} returned non-integer expires_in: {expires_in!r}") from exc

    return OAuthCredentials(
        refresh=refresh,
        access=access,
        expires=int(time.time() * 1000) + expires_seconds * 1000 - EXPIRY_SAFETY_MS,
    )


def _exchange_code(code: str, state: str, verifier: str, redirect_uri: str) -> OAuthCredentials:
    data = _post_json(
        TOKEN_URL,
        {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "state": state,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        },
    )
    return _credentials_from_token_response(data, context="authorization code exchange")


def login_anthropic(callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
    pkce = generate_pkce()
    # The OAuth ``state`` parameter travels through the front channel (browser
    # address bar, history, IdP logs, Referer headers). The PKCE ``code_verifier``
    # MUST stay on this client only, so it cannot double as ``state`` — anyone
    # who sees the redirect URL would otherwise hold both halves of the PKCE
    # exchange. Use a fresh random nonce for state instead.
    state_nonce = secrets.token_hex(16)
    server, result, _ = _start_callback_server(state_nonce)
    try:
        auth_params = {
            "code": "true",
            "client_id": CLIENT_ID,
            "response_type": "code",
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPES,
            "code_challenge": pkce.challenge,
            "code_challenge_method": "S256",
            "state": state_nonce,
        }
        callbacks.on_auth(
            OAuthAuthInfo(
                url=f"{AUTHORIZE_URL}?{urlencode(auth_params)}",
                instructions=(
                    "Complete login in your browser. If the browser is on another machine, "
                    "paste the final redirect URL here."
                ),
            )
        )

        code: str | None = None
        state: str | None = None

        if server is not None:
            if callbacks.on_manual_code_input is not None:
                manual_text: list[str | None] = [None]
                manual_err: list[Exception | None] = [None]

                def _runner() -> None:
                    try:
                        manual_text[0] = callbacks.on_manual_code_input()  # type: ignore[misc]
                    except Exception as e:  # noqa: BLE001
                        manual_err[0] = e
                    finally:
                        result.event.set()

                t = threading.Thread(target=_runner, daemon=True)
                t.start()
                result.event.wait(timeout=CALLBACK_WAIT_TIMEOUT_S)
                # If the server callback won the race, the manual-input runner
                # is still blocked on stdin (e.g. inside ``prompt_toolkit.prompt``).
                # Tell the caller to cancel it so it does not keep eating
                # keystrokes meant for the next interactive prompt.
                if manual_text[0] is None and manual_err[0] is None and callbacks.cancel_manual_code_input is not None:
                    try:
                        callbacks.cancel_manual_code_input()
                    except Exception:  # noqa: BLE001
                        pass
                if manual_err[0]:
                    raise manual_err[0]
                if result.code:
                    code = result.code
                    state = result.state
                elif manual_text[0]:
                    code, state = _parse_authorization_input(manual_text[0])
                elif result.error:
                    raise RuntimeError(f"Anthropic OAuth callback error: {result.error}")
            else:
                result.event.wait(timeout=CALLBACK_WAIT_TIMEOUT_S)
                if result.error and not result.code:
                    raise RuntimeError(f"Anthropic OAuth callback error: {result.error}")
                code = result.code
                state = result.state

        if not code:
            text = callbacks.on_prompt(
                OAuthPrompt(
                    message="Paste the authorization code or full redirect URL:",
                    placeholder=REDIRECT_URI,
                )
            )
            code, state = _parse_authorization_input(text)

        if state and state != state_nonce:
            raise RuntimeError("OAuth state mismatch")
        if not code:
            raise RuntimeError("Missing authorization code")

        if callbacks.on_progress:
            callbacks.on_progress("Exchanging authorization code for tokens...")
        return _exchange_code(code, state or state_nonce, pkce.verifier, REDIRECT_URI)
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()


def refresh_anthropic_token(refresh_token: str) -> OAuthCredentials:
    data = _post_json(
        TOKEN_URL,
        {"grant_type": "refresh_token", "client_id": CLIENT_ID, "refresh_token": refresh_token},
    )
    return _credentials_from_token_response(data, context="token refresh")


class AnthropicOAuthProvider(OAuthProviderInterface):
    id = "anthropic"
    name = "Anthropic (Claude Pro/Max)"
    uses_callback_server = True

    def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
        return login_anthropic(callbacks)

    def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        return refresh_anthropic_token(credentials.refresh)

    def get_api_key(self, credentials: OAuthCredentials) -> str:
        return credentials.access


anthropic_oauth_provider = AnthropicOAuthProvider()
