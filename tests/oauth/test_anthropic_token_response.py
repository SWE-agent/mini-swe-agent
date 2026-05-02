"""Validation around malformed responses from Anthropic's token endpoint."""

import pytest

from minisweagent.oauth.anthropic import _credentials_from_token_response


def test_well_formed_response_returns_credentials():
    creds = _credentials_from_token_response(
        {"access_token": "a", "refresh_token": "r", "expires_in": 3600},
        context="test",
    )
    assert creds.access == "a"
    assert creds.refresh == "r"
    assert creds.expires > 0


def test_missing_access_token_raises_runtime_error():
    with pytest.raises(RuntimeError, match="missing required fields"):
        _credentials_from_token_response(
            {"refresh_token": "r", "expires_in": 3600},
            context="test",
        )


def test_missing_refresh_token_raises_runtime_error():
    with pytest.raises(RuntimeError, match="missing required fields"):
        _credentials_from_token_response(
            {"access_token": "a", "expires_in": 3600},
            context="test",
        )


def test_missing_expires_in_raises_runtime_error():
    with pytest.raises(RuntimeError, match="missing required fields"):
        _credentials_from_token_response(
            {"access_token": "a", "refresh_token": "r"},
            context="test",
        )


def test_empty_string_access_token_raises_runtime_error():
    with pytest.raises(RuntimeError, match="non-string access_token"):
        _credentials_from_token_response(
            {"access_token": "", "refresh_token": "r", "expires_in": 3600},
            context="test",
        )


def test_non_string_refresh_token_raises_runtime_error():
    with pytest.raises(RuntimeError, match="non-string refresh_token"):
        _credentials_from_token_response(
            {"access_token": "a", "refresh_token": 123, "expires_in": 3600},
            context="test",
        )


def test_non_integer_expires_in_raises_runtime_error():
    with pytest.raises(RuntimeError, match="non-integer expires_in"):
        _credentials_from_token_response(
            {"access_token": "a", "refresh_token": "r", "expires_in": "not-a-number"},
            context="test",
        )


def test_non_dict_payload_raises_runtime_error():
    with pytest.raises(RuntimeError, match="non-object payload"):
        _credentials_from_token_response(["unexpected"], context="test")  # type: ignore[arg-type]


def test_error_message_mentions_context():
    with pytest.raises(RuntimeError, match="token refresh"):
        _credentials_from_token_response({}, context="token refresh")
