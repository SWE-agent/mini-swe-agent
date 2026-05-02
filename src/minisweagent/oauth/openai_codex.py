"""OpenAI Codex (ChatGPT Plus/Pro) OAuth flow.

Ported from pi-mono (packages/ai/src/utils/oauth/openai-codex.ts).
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

# OpenAI's Codex OAuth client only allows ``http://localhost:1455/auth/callback``
# as a redirect URI, so the host name in the redirect MUST stay literal
# ``localhost``. Bind the listener to the same hostname so getaddrinfo resolves
# the same address for the browser and for our HTTPServer (avoids a v4/v6
# split on dual-stack Linux where ``localhost`` resolves to ``::1`` while a
# ``127.0.0.1`` listener never receives the callback).
#
# REDIRECT_URI is hard-coded to ``localhost`` because OpenAI whitelists that
# exact string. The server must bind to the same hostname to avoid a v4/v6
# split on dual-stack Linux where ``localhost`` → ``::1``.
CALLBACK_HOST = "localhost"
CALLBACK_PORT = int(os.getenv("MSWEA_CODEX_CALLBACK_PORT", "1455"))
CALLBACK_PATH = "/auth/callback"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}{CALLBACK_PATH}"
SCOPE = "openid profile email offline_access"
JWT_CLAIM_PATH = "https://api.openai.com/auth"
DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"
CALLBACK_WAIT_TIMEOUT_S = float(os.getenv("MSWEA_OAUTH_CALLBACK_TIMEOUT", "300"))
EXPIRY_SAFETY_MS = 5 * 60 * 1000


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
            state = params.get("state", [None])[0]
            if state != expected_state:
                result.error = "State mismatch"
                self._respond(400, oauth_error_html("State mismatch."))
                result.event.set()
                return
            code = params.get("code", [None])[0]
            if not code:
                result.error = "Missing authorization code"
                self._respond(400, oauth_error_html("Missing authorization code."))
                result.event.set()
                return
            result.code = code
            result.state = state
            self._respond(200, oauth_success_html("OpenAI authentication completed. You can close this window."))
            result.event.set()

        def _respond(self, status: int, body: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            data = body.encode("utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return _Handler


def _start_callback_server(expected_state: str) -> tuple[HTTPServer | None, _CallbackResult]:
    result = _CallbackResult()
    handler = _make_handler(expected_state, result)
    try:
        server = HTTPServer((CALLBACK_HOST, CALLBACK_PORT), handler)
    except OSError:
        return None, result
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, result


def _parse_authorization_input(text: str) -> tuple[str | None, str | None]:
    value = text.strip()
    if not value:
        return None, None
    try:
        url = urlparse(value)
        if url.scheme and url.netloc:
            params = parse_qs(url.query)
            return params.get("code", [None])[0], params.get("state", [None])[0]
    except ValueError:
        pass
    if "#" in value:
        code, state = value.split("#", 1)
        return code or None, state or None
    if "code=" in value:
        query = value[1:] if value.startswith("?") else value
        params = parse_qs(query)
        return params.get("code", [None])[0], params.get("state", [None])[0]
    return value, None


def _b64url_decode(payload: str) -> bytes:
    padding = "=" * (-len(payload) % 4)
    return base64.urlsafe_b64decode(payload + padding)


def decode_jwt(token: str) -> dict[str, Any] | None:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        return json.loads(_b64url_decode(parts[1]).decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None


def get_account_id(access_token: str) -> str | None:
    payload = decode_jwt(access_token) or {}
    auth = payload.get(JWT_CLAIM_PATH) or {}
    account_id = auth.get("chatgpt_account_id")
    return account_id if isinstance(account_id, str) and account_id else None


def _exchange_code(code: str, verifier: str) -> dict[str, Any]:
    try:
        response = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": REDIRECT_URI,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
    except requests.exceptions.RequestException as exc:
        raise OAuthTransientError(
            f"Codex token exchange failed (network): {type(exc).__name__}: {exc}"
        ) from exc
    if not response.ok:
        # Don't echo response.text — IdP error bodies can mirror the rejected
        # code or refresh token back to us.
        message = (
            f"Codex token exchange failed: status={response.status_code} "
            f"reason={response.reason}"
        )
        if response.status_code >= 500:
            raise OAuthTransientError(message)
        raise RuntimeError(message)
    return response.json()


def _refresh(refresh_token: str) -> dict[str, Any]:
    try:
        response = requests.post(
            TOKEN_URL,
            data={"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": CLIENT_ID},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
    except requests.exceptions.RequestException as exc:
        raise OAuthTransientError(
            f"Codex token refresh failed (network): {type(exc).__name__}: {exc}"
        ) from exc
    if not response.ok:
        message = (
            f"Codex token refresh failed: status={response.status_code} "
            f"reason={response.reason}"
        )
        if response.status_code >= 500:
            raise OAuthTransientError(message)
        raise RuntimeError(message)
    return response.json()


def _credentials_from_token_response(data: dict[str, Any], refresh_token: str | None, *, context: str) -> OAuthCredentials:
    """Validate a Codex token endpoint response and convert it to credentials.

    ``refresh_token`` is passed explicitly because the refresh endpoint may not
    rotate it — callers should supply the existing value as a fallback.
    """
    if not isinstance(data, dict):
        raise RuntimeError(f"Codex {context} returned non-object payload: {data!r}")

    missing = [k for k in ("access_token", "expires_in") if k not in data]
    if missing:
        keys = ", ".join(sorted(data.keys())) or "<empty>"
        raise RuntimeError(
            f"Codex {context} response missing required fields {missing}. "
            f"Received keys: {keys}. Body: {data!r}"
        )

    access = data["access_token"]
    new_refresh = data.get("refresh_token") or refresh_token
    expires_in = data["expires_in"]

    if not isinstance(access, str) or not access:
        raise RuntimeError(f"Codex {context} returned non-string access_token: {access!r}")
    if not isinstance(new_refresh, str) or not new_refresh:
        raise RuntimeError(f"Codex {context}: no usable refresh_token in response or fallback")
    try:
        expires_seconds = int(expires_in)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Codex {context} returned non-integer expires_in: {expires_in!r}"
        ) from exc

    account_id = get_account_id(access)
    if not account_id:
        raise RuntimeError(f"Codex {context}: failed to extract account_id from access token JWT")

    return OAuthCredentials(
        refresh=new_refresh,
        access=access,
        # Apply the same EXPIRY_SAFETY_MS margin used by anthropic / github-copilot
        # so callers that hit the abort-on-AuthenticationError retry policy don't
        # see hard failures from borderline-expired tokens.
        expires=int(time.time() * 1000) + expires_seconds * 1000 - EXPIRY_SAFETY_MS,
        extra={"account_id": account_id},
    )


def login_openai_codex(callbacks: OAuthLoginCallbacks, originator: str = "mini-swe-agent") -> OAuthCredentials:
    pkce = generate_pkce()
    state = secrets.token_hex(16)

    auth_url = (
        AUTHORIZE_URL
        + "?"
        + urlencode(
            {
                "response_type": "code",
                "client_id": CLIENT_ID,
                "redirect_uri": REDIRECT_URI,
                "scope": SCOPE,
                "code_challenge": pkce.challenge,
                "code_challenge_method": "S256",
                "state": state,
                "id_token_add_organizations": "true",
                "codex_cli_simplified_flow": "true",
                "originator": originator,
            }
        )
    )

    server, result = _start_callback_server(state)
    callbacks.on_auth(
        OAuthAuthInfo(url=auth_url, instructions="A browser window should open. Complete login to finish.")
    )

    try:
        code: str | None = None
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

                threading.Thread(target=_runner, daemon=True).start()
                result.event.wait(timeout=CALLBACK_WAIT_TIMEOUT_S)
                # Server callback won the race: release the manual-input
                # thread that's still blocked on stdin so it does not eat the
                # next interactive prompt's keystrokes.
                if (
                    manual_text[0] is None
                    and manual_err[0] is None
                    and callbacks.cancel_manual_code_input is not None
                ):
                    try:
                        callbacks.cancel_manual_code_input()
                    except Exception:  # noqa: BLE001
                        pass
                if manual_err[0]:
                    raise manual_err[0]
                if result.code:
                    code = result.code
                elif manual_text[0]:
                    parsed_code, parsed_state = _parse_authorization_input(manual_text[0])
                    if parsed_state and parsed_state != state:
                        raise RuntimeError("State mismatch")
                    code = parsed_code
                elif result.error:
                    raise RuntimeError(f"OpenAI Codex OAuth callback error: {result.error}")
            else:
                result.event.wait(timeout=CALLBACK_WAIT_TIMEOUT_S)
                if result.error and not result.code:
                    raise RuntimeError(f"OpenAI Codex OAuth callback error: {result.error}")
                code = result.code

        if not code:
            text = callbacks.on_prompt(
                OAuthPrompt(message="Paste the authorization code (or full redirect URL):")
            )
            parsed_code, parsed_state = _parse_authorization_input(text)
            if parsed_state and parsed_state != state:
                raise RuntimeError("State mismatch")
            code = parsed_code

        if not code:
            raise RuntimeError("Missing authorization code")

        token = _exchange_code(code, pkce.verifier)
        return _credentials_from_token_response(token, None, context="authorization code exchange")
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()


def refresh_openai_codex_token(refresh_token: str) -> OAuthCredentials:
    token = _refresh(refresh_token)
    return _credentials_from_token_response(token, refresh_token, context="token refresh")


class OpenAICodexOAuthProvider(OAuthProviderInterface):
    id = "openai-codex"
    name = "ChatGPT Plus/Pro (Codex Subscription)"
    uses_callback_server = True

    def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
        return login_openai_codex(callbacks)

    def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        return refresh_openai_codex_token(credentials.refresh)

    def get_api_key(self, credentials: OAuthCredentials) -> str:
        return credentials.access


openai_codex_oauth_provider = OpenAICodexOAuthProvider()
