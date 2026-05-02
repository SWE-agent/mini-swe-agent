"""Unit tests for oauth/types.py."""

import pytest

from minisweagent.oauth.types import (
    OAuthCredentials,
    OAuthLoginCallbacks,
    OAuthProviderInterface,
)


class _ConcreteProvider(OAuthProviderInterface):
    id = "concrete"
    name = "Concrete"

    def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
        return OAuthCredentials(refresh="r", access="a", expires=0)

    def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        return credentials

    def get_api_key(self, credentials: OAuthCredentials) -> str:
        return credentials.access


def test_credentials_to_dict_no_extra():
    creds = OAuthCredentials(refresh="r", access="a", expires=42)
    assert creds.to_dict() == {"refresh": "r", "access": "a", "expires": 42}


def test_credentials_to_dict_with_extra():
    creds = OAuthCredentials(refresh="r", access="a", expires=1, extra={"account_id": "u1", "foo": "bar"})
    d = creds.to_dict()
    assert d["account_id"] == "u1"
    assert d["foo"] == "bar"
    assert d["refresh"] == "r"


def test_credentials_from_dict_no_extra():
    creds = OAuthCredentials.from_dict({"refresh": "rt", "access": "at", "expires": 99})
    assert creds.refresh == "rt"
    assert creds.access == "at"
    assert creds.expires == 99
    assert creds.extra == {}


def test_credentials_from_dict_with_extra():
    creds = OAuthCredentials.from_dict(
        {"refresh": "r", "access": "a", "expires": 0, "account_id": "u1", "enterprise_url": "e"}
    )
    assert creds.extra == {"account_id": "u1", "enterprise_url": "e"}


def test_credentials_from_dict_roundtrip():
    original = OAuthCredentials(
        refresh="r", access="a", expires=123456, extra={"k": "v"}
    )
    restored = OAuthCredentials.from_dict(original.to_dict())
    assert restored.refresh == original.refresh
    assert restored.access == original.access
    assert restored.expires == original.expires
    assert restored.extra == original.extra


def test_credentials_from_dict_expires_coerced_to_int():
    creds = OAuthCredentials.from_dict({"refresh": "r", "access": "a", "expires": "77"})
    assert creds.expires == 77
    assert isinstance(creds.expires, int)


def test_credentials_from_dict_missing_key_raises():
    with pytest.raises(KeyError):
        OAuthCredentials.from_dict({"refresh": "r", "expires": 0})  # missing 'access'
