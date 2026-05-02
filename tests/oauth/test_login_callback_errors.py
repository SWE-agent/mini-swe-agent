"""Regression: login flows must surface ?error=... callbacks instead of falling
through to the manual-paste prompt.
"""

from __future__ import annotations

import threading

import pytest

from minisweagent.oauth import anthropic as anthropic_oauth
from minisweagent.oauth import openai_codex as codex_oauth
from minisweagent.oauth.types import OAuthAuthInfo, OAuthLoginCallbacks, OAuthPrompt


def _fail_prompt(_p: OAuthPrompt) -> str:
    raise AssertionError("on_prompt must not be called when the IdP returned an explicit error")


def _noop_auth(_info: OAuthAuthInfo) -> None:
    return


@pytest.fixture
def patched_anthropic_callback(monkeypatch):
    """Replace the callback server so we can inject an error directly."""
    state_holder: dict[str, anthropic_oauth._CallbackResult] = {}

    class _DummyServer:
        def shutdown(self) -> None:
            return

        def server_close(self) -> None:
            return

    def fake_start(expected_state: str):
        result = anthropic_oauth._CallbackResult()
        state_holder["result"] = result
        state_holder["expected_state"] = expected_state  # type: ignore[assignment]
        return _DummyServer(), result, threading.Thread()

    monkeypatch.setattr(anthropic_oauth, "_start_callback_server", fake_start)
    return state_holder


@pytest.fixture
def patched_codex_callback(monkeypatch):
    state_holder: dict[str, codex_oauth._CallbackResult] = {}

    class _DummyServer:
        def shutdown(self) -> None:
            return

        def server_close(self) -> None:
            return

    def fake_start(expected_state: str):
        result = codex_oauth._CallbackResult()
        state_holder["result"] = result
        state_holder["expected_state"] = expected_state  # type: ignore[assignment]
        return _DummyServer(), result

    monkeypatch.setattr(codex_oauth, "_start_callback_server", fake_start)
    return state_holder


def test_anthropic_callback_error_raises_without_prompt(patched_anthropic_callback):
    def on_auth(_info: OAuthAuthInfo) -> None:
        result = patched_anthropic_callback["result"]
        result.error = "access_denied"
        result.event.set()

    callbacks = OAuthLoginCallbacks(on_auth=on_auth, on_prompt=_fail_prompt)
    with pytest.raises(RuntimeError, match="access_denied"):
        anthropic_oauth.login_anthropic(callbacks)


def test_anthropic_callback_error_with_manual_input_branch(patched_anthropic_callback):
    def on_auth(_info: OAuthAuthInfo) -> None:
        result = patched_anthropic_callback["result"]
        result.error = "access_denied"
        result.event.set()

    def on_manual(_event=None) -> str:
        # Block forever; the error path should win.
        threading.Event().wait()
        raise AssertionError("unreachable")

    callbacks = OAuthLoginCallbacks(
        on_auth=on_auth,
        on_prompt=_fail_prompt,
        on_manual_code_input=on_manual,
    )
    with pytest.raises(RuntimeError, match="access_denied"):
        anthropic_oauth.login_anthropic(callbacks)


def test_codex_callback_error_raises_without_prompt(patched_codex_callback):
    def on_auth(_info: OAuthAuthInfo) -> None:
        result = patched_codex_callback["result"]
        result.error = "State mismatch"
        result.event.set()

    callbacks = OAuthLoginCallbacks(on_auth=on_auth, on_prompt=_fail_prompt)
    with pytest.raises(RuntimeError, match="State mismatch"):
        codex_oauth.login_openai_codex(callbacks)


def test_codex_callback_error_with_manual_input_branch(patched_codex_callback):
    def on_auth(_info: OAuthAuthInfo) -> None:
        result = patched_codex_callback["result"]
        result.error = "Missing authorization code"
        result.event.set()

    def on_manual() -> str:
        threading.Event().wait()
        raise AssertionError("unreachable")

    callbacks = OAuthLoginCallbacks(
        on_auth=on_auth,
        on_prompt=_fail_prompt,
        on_manual_code_input=on_manual,
    )
    with pytest.raises(RuntimeError, match="Missing authorization code"):
        codex_oauth.login_openai_codex(callbacks)
