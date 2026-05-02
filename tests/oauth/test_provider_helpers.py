"""Smoke tests for non-network helpers in the OAuth providers."""

import base64
import json

import pytest

from minisweagent.oauth.anthropic import (
    AUTHORIZE_URL,
    REDIRECT_URI,
    SCOPES,
    _parse_authorization_input,
)
from minisweagent.oauth.anthropic import (
    _parse_authorization_input as anthropic_parse,
)
from minisweagent.oauth.github_copilot import (
    DEFAULT_BASE_URL,
    _urls,
    get_github_copilot_base_url,
    login_github_copilot,
    normalize_domain,
)
from minisweagent.oauth.openai_codex import (
    REDIRECT_URI as CODEX_REDIRECT_URI,
)
from minisweagent.oauth.openai_codex import (
    _parse_authorization_input as codex_parse,
)
from minisweagent.oauth.openai_codex import (
    decode_jwt,
    get_account_id,
)
from minisweagent.oauth.types import OAuthLoginCallbacks


def test_anthropic_parse_input_url():
    code, state = _parse_authorization_input(f"{REDIRECT_URI}?code=abc&state=xyz")
    assert code == "abc"
    assert state == "xyz"


def test_anthropic_parse_input_hash_separator():
    code, state = _parse_authorization_input("code-value#state-value")
    assert code == "code-value"
    assert state == "state-value"


def test_anthropic_parse_input_plain_code():
    code, state = _parse_authorization_input("just-a-code")
    assert code == "just-a-code"
    assert state is None


def test_anthropic_authorize_url_constants():
    assert AUTHORIZE_URL.startswith("https://")
    assert "user:inference" in SCOPES


def test_copilot_normalize_domain_strips_scheme():
    assert normalize_domain("https://example.ghe.com/foo") == "example.ghe.com"
    assert normalize_domain("example.ghe.com") == "example.ghe.com"
    assert normalize_domain("   ") is None


def test_copilot_default_base_url():
    assert get_github_copilot_base_url() == DEFAULT_BASE_URL


def test_copilot_base_url_from_token_proxy_ep():
    fake_token = "tid=1;exp=1;proxy-ep=proxy.individual.githubcopilot.com;rest=more"
    assert get_github_copilot_base_url(fake_token) == "https://api.individual.githubcopilot.com"


def test_copilot_base_url_enterprise_fallback():
    assert (
        get_github_copilot_base_url(token=None, enterprise_domain="example.ghe.com")
        == "https://copilot-api.example.ghe.com"
    )


def test_codex_decode_jwt_returns_payload():
    import base64
    import json

    payload = {"https://api.openai.com/auth": {"chatgpt_account_id": "acc-7"}}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    fake_jwt = f"header.{encoded}.signature"
    assert get_account_id(fake_jwt) == "acc-7"
    assert decode_jwt(fake_jwt) == payload


def test_codex_decode_jwt_invalid_returns_none():
    assert decode_jwt("not.a-valid.token??") is None


@pytest.mark.parametrize(
    ("fake_token", "expected_host"),
    [
        ("proxy-ep=proxy.business.githubcopilot.com;exp=99", "https://api.business.githubcopilot.com"),
        ("foo=bar", None),
    ],
)
def test_copilot_base_url_extraction_parametrized(fake_token: str, expected_host: str | None):
    base = get_github_copilot_base_url(fake_token)
    if expected_host is None:
        assert base == DEFAULT_BASE_URL
    else:
        assert base == expected_host


# ---------------------------------------------------------------------------
# Additional anthropic parse edge cases
# ---------------------------------------------------------------------------


def test_anthropic_parse_input_empty():
    assert anthropic_parse("") == (None, None)
    assert anthropic_parse("   ") == (None, None)


def test_anthropic_parse_input_query_string_format():
    code, state = anthropic_parse("code=mycode&state=mystate")
    assert code == "mycode"
    assert state == "mystate"


def test_anthropic_parse_input_query_string_with_leading_question_mark():
    code, state = anthropic_parse("?code=mycode&state=mystate")
    assert code == "mycode"
    assert state == "mystate"


# ---------------------------------------------------------------------------
# OpenAI Codex _parse_authorization_input
# ---------------------------------------------------------------------------


def test_codex_parse_input_full_url():
    code, state = codex_parse(f"{CODEX_REDIRECT_URI}?code=abc&state=xyz")
    assert code == "abc"
    assert state == "xyz"


def test_codex_parse_input_hash_separator():
    code, state = codex_parse("mycode#mystate")
    assert code == "mycode"
    assert state == "mystate"


def test_codex_parse_input_plain_code():
    code, state = codex_parse("just-a-code")
    assert code == "just-a-code"
    assert state is None


def test_codex_parse_input_empty():
    assert codex_parse("") == (None, None)
    assert codex_parse("  ") == (None, None)


def test_codex_parse_input_query_string_format():
    code, state = codex_parse("code=x&state=y")
    assert code == "x"
    assert state == "y"


def test_codex_parse_input_query_string_with_leading_question_mark():
    code, state = codex_parse("?code=x&state=y")
    assert code == "x"
    assert state == "y"


# ---------------------------------------------------------------------------
# GitHub Copilot _urls helper
# ---------------------------------------------------------------------------


def test_copilot_urls_github_com():
    urls = _urls("github.com")
    assert urls["device_code"] == "https://github.com/login/device/code"
    assert urls["access_token"] == "https://github.com/login/oauth/access_token"
    assert "copilot_token" in urls
    assert "api.github.com" in urls["copilot_token"]


def test_copilot_normalize_domain_http_scheme():
    assert normalize_domain("http://company.ghe.com") == "company.ghe.com"


def test_copilot_normalize_domain_with_path():
    assert normalize_domain("https://example.ghe.com/some/path") == "example.ghe.com"


# ---------------------------------------------------------------------------
# Additional decode_jwt / get_account_id edge cases
# ---------------------------------------------------------------------------


def test_codex_decode_jwt_wrong_part_count():
    assert decode_jwt("only.two") is None
    assert decode_jwt("single") is None
    assert decode_jwt("a.b.c.d") is None


def test_codex_get_account_id_missing_claim_path():
    payload = {"other_claim": {"chatgpt_account_id": "acc-9"}}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    assert get_account_id(f"h.{encoded}.s") is None


def test_codex_get_account_id_non_string_value():
    payload = {"https://api.openai.com/auth": {"chatgpt_account_id": 12345}}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    assert get_account_id(f"h.{encoded}.s") is None


def test_copilot_login_invalid_domain_echoes_user_input():
    bad_input = "://not a host"
    callbacks = OAuthLoginCallbacks(
        on_auth=lambda info: None,
        on_prompt=lambda prompt: bad_input,
    )
    with pytest.raises(RuntimeError, match=r"Invalid GitHub Enterprise URL/domain: '://not a host'"):
        login_github_copilot(callbacks)
