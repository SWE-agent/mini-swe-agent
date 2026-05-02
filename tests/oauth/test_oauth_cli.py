"""Tests for oauth_cli helpers."""

import time

import pytest

from minisweagent.run.utilities.oauth_cli import _format_expiry


def test_format_expiry_future():
    future_ms = int(time.time() * 1000) + 180_000  # 3 minutes from now
    result = _format_expiry(future_ms)
    assert "in 3m" in result


def test_format_expiry_past():
    past_ms = int(time.time() * 1000) - 180_000  # 3 minutes ago
    result = _format_expiry(past_ms)
    assert "expired 3m ago" in result


def test_format_expiry_contains_iso_date():
    ms = 1_700_000_000_000  # 2023-11-14
    result = _format_expiry(ms)
    assert "2023-" in result


def test_format_expiry_just_expired():
    ms = int(time.time() * 1000) - 60_000  # 1 minute ago
    result = _format_expiry(ms)
    assert "expired" in result
    assert "ago" in result


def test_format_expiry_returns_string():
    assert isinstance(_format_expiry(int(time.time() * 1000)), str)


def test_token_command_writes_error_to_stderr(monkeypatch, capsys):
    """``token`` is meant for ``$(mini-extra oauth token X)`` style scripts.
    The error message when no creds exist must go to stderr, not stdout, so
    captured stdout still contains only the access token (or empty)."""
    from typer.testing import CliRunner

    from minisweagent import oauth
    from minisweagent.run.utilities import oauth_cli

    monkeypatch.setattr(oauth, "get_oauth_api_key", lambda *a, **kw: None)
    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(oauth_cli.app, ["token", "anthropic"])

    assert result.exit_code == 2
    assert "No credentials" in (result.stderr or "")
    assert "No credentials" not in result.stdout
