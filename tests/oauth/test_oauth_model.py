import time

import pytest

from minisweagent import oauth
from minisweagent.oauth.types import (
    OAuthCredentials,
    OAuthLoginCallbacks,
    OAuthProviderInterface,
)


class _StubProvider(OAuthProviderInterface):
    id = "stub-model"
    name = "Stub"

    def __init__(self, account_id: str | None = None, enterprise: str | None = None) -> None:
        self.refreshes = 0
        self._extra: dict = {}
        if account_id:
            self._extra["account_id"] = account_id
        if enterprise:
            self._extra["enterprise_url"] = enterprise

    def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
        return OAuthCredentials(refresh="r", access="a", expires=0, extra=self._extra)

    def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        self.refreshes += 1
        return OAuthCredentials(
            refresh=credentials.refresh,
            access="fresh-token",
            expires=int(time.time() * 1000) + 60_000,
            extra=self._extra,
        )

    def get_api_key(self, credentials: OAuthCredentials) -> str:
        return credentials.access


@pytest.fixture
def isolated_oauth(monkeypatch, tmp_path):
    monkeypatch.setenv("MSWEA_OAUTH_FILE", str(tmp_path / "oauth.json"))
    return


def _stash_provider_under(provider_id: str, stub: OAuthProviderInterface):
    """Register stub under an alias matching one of the supported provider ids."""
    stub.id = provider_id  # type: ignore[misc]
    oauth.register_oauth_provider(stub)


def test_anthropic_oauth_kwargs(isolated_oauth):
    from minisweagent.models.oauth_model import OAuthLitellmModel

    stub = _StubProvider()
    _stash_provider_under("anthropic", stub)
    try:
        oauth.storage.save(
            "anthropic",
            OAuthCredentials(refresh="r", access="stale", expires=0),
        )
        model = OAuthLitellmModel(model_name="anthropic/claude-sonnet-4-5", oauth_provider="anthropic")
        kwargs = model._resolve_oauth_kwargs()
        # api_key must be a sentinel - never the real OAuth token - so LiteLLM
        # cannot leak it into the x-api-key header. Authorization carries the
        # bearer token instead.
        assert kwargs["api_key"] == "oauth"
        assert kwargs["api_key"] != "fresh-token"
        headers = kwargs["extra_headers"]
        assert headers["Authorization"] == "Bearer fresh-token"
        assert headers["x-api-key"] == ""
        assert "claude-code-20250219" in headers["anthropic-beta"]
        assert "oauth-2025-04-20" in headers["anthropic-beta"]
        assert headers["x-app"] == "cli"
    finally:
        oauth.restore_oauth_provider("anthropic")


def test_anthropic_inject_claude_code_system(isolated_oauth):
    from minisweagent.models.oauth_model import (
        CLAUDE_CODE_SYSTEM_PROMPT,
        OAuthLitellmModel,
    )

    stub = _StubProvider()
    _stash_provider_under("anthropic", stub)
    try:
        oauth.storage.save("anthropic", OAuthCredentials(refresh="r", access="x", expires=0))
        model = OAuthLitellmModel(model_name="anthropic/claude-sonnet-4-5", oauth_provider="anthropic")
        prepared = model._prepare_messages_for_api([{"role": "user", "content": "hi"}])
        assert prepared[0] == {"role": "system", "content": CLAUDE_CODE_SYSTEM_PROMPT}

        # Idempotent
        prepared_again = model._prepare_messages_for_api(prepared)
        assert prepared_again.count({"role": "system", "content": CLAUDE_CODE_SYSTEM_PROMPT}) == 1
    finally:
        oauth.restore_oauth_provider("anthropic")


def test_codex_oauth_kwargs(isolated_oauth):
    from minisweagent.models.oauth_model import OAuthLitellmModel

    stub = _StubProvider(account_id="acc-99")
    _stash_provider_under("openai-codex", stub)
    try:
        oauth.storage.save(
            "openai-codex",
            OAuthCredentials(refresh="r", access="old", expires=0, extra={"account_id": "acc-99"}),
        )
        model = OAuthLitellmModel(model_name="openai/codex-mini", oauth_provider="openai-codex")
        kwargs = model._resolve_oauth_kwargs()
        assert kwargs["api_base"] == "https://chatgpt.com/backend-api/codex"
        headers = kwargs["extra_headers"]
        assert headers["chatgpt-account-id"] == "acc-99"
        assert headers["originator"] == "mini-swe-agent"
        assert headers["OpenAI-Beta"] == "responses=experimental"
    finally:
        oauth.restore_oauth_provider("openai-codex")


def test_copilot_oauth_kwargs(isolated_oauth):
    from minisweagent.models.oauth_model import OAuthLitellmModel

    stub = _StubProvider()
    _stash_provider_under("github-copilot", stub)
    try:
        oauth.storage.save(
            "github-copilot",
            OAuthCredentials(refresh="r", access="proxy-ep=proxy.individual.githubcopilot.com;exp=99", expires=0),
        )
        model = OAuthLitellmModel(model_name="anthropic/claude-3-5-sonnet", oauth_provider="github-copilot")
        kwargs = model._resolve_oauth_kwargs()
        # First refresh returns "fresh-token" which has no proxy-ep so we expect default
        assert kwargs["api_base"].startswith("https://api.")
        headers = kwargs["extra_headers"]
        assert headers["Authorization"] == "Bearer fresh-token"
        assert "Copilot-Integration-Id" in headers
    finally:
        oauth.restore_oauth_provider("github-copilot")


def test_invalid_provider_rejected(isolated_oauth):
    from minisweagent.models.oauth_model import OAuthLitellmModel

    with pytest.raises(ValueError, match="oauth_provider"):
        OAuthLitellmModel(model_name="anthropic/claude", oauth_provider="bogus")


def test_resolve_kwargs_raises_when_logged_out(isolated_oauth):
    from minisweagent.models.oauth_model import OAuthLitellmModel

    model = OAuthLitellmModel(model_name="anthropic/claude", oauth_provider="anthropic")
    with pytest.raises(RuntimeError, match="No OAuth credentials"):
        model._resolve_oauth_kwargs()


def test_oauth_identity_wins_over_caller_kwargs(isolated_oauth):
    """Caller-supplied api_key / api_base must not clobber OAuth identity."""
    from minisweagent.models.oauth_model import OAuthLitellmModel

    stub = _StubProvider()
    _stash_provider_under("anthropic", stub)
    captured: dict = {}

    def fake_super_query(self, messages, **kwargs):  # noqa: ARG001
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    try:
        oauth.storage.save("anthropic", OAuthCredentials(refresh="r", access="x", expires=0))
        model = OAuthLitellmModel(model_name="anthropic/claude-sonnet-4-5", oauth_provider="anthropic")

        from minisweagent.models.litellm_model import LitellmModel

        original_query = LitellmModel._query
        LitellmModel._query = fake_super_query  # type: ignore[assignment]
        try:
            model._query(
                [{"role": "user", "content": "hi"}],
                api_key="sk-ant-static-leak",  # would normally clobber OAuth
                api_base="https://attacker.example.com",
                extra_headers={"Authorization": "Bearer attacker", "X-Custom": "ok"},
            )
        finally:
            LitellmModel._query = original_query  # type: ignore[assignment]
    finally:
        oauth.restore_oauth_provider("anthropic")

    # OAuth wins for api_key and Authorization (the identity-bearing fields)
    assert captured["api_key"] == "oauth"
    assert captured["extra_headers"]["Authorization"] == "Bearer fresh-token"
    assert captured["extra_headers"]["x-api-key"] == ""
    # Caller may still contribute non-conflicting headers
    assert captured["extra_headers"]["X-Custom"] == "ok"


def test_caller_api_base_does_not_override_codex(isolated_oauth):
    from minisweagent.models.oauth_model import OAuthLitellmModel

    stub = _StubProvider(account_id="acc-1")
    _stash_provider_under("openai-codex", stub)
    captured: dict = {}

    def fake_super_query(self, messages, **kwargs):  # noqa: ARG001
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    try:
        oauth.storage.save(
            "openai-codex",
            OAuthCredentials(refresh="r", access="a", expires=0, extra={"account_id": "acc-1"}),
        )
        model = OAuthLitellmModel(model_name="openai/codex", oauth_provider="openai-codex")

        from minisweagent.models.litellm_model import LitellmModel

        original_query = LitellmModel._query
        LitellmModel._query = fake_super_query  # type: ignore[assignment]
        try:
            model._query(
                [{"role": "user", "content": "hi"}],
                api_base="https://attacker.example.com",
                extra_headers={"chatgpt-account-id": "spoofed"},
            )
        finally:
            LitellmModel._query = original_query  # type: ignore[assignment]
    finally:
        oauth.restore_oauth_provider("openai-codex")

    assert captured["api_base"] == "https://chatgpt.com/backend-api/codex"
    assert captured["extra_headers"]["chatgpt-account-id"] == "acc-1"


def test_anthropic_no_inject_claude_code_system(isolated_oauth):
    from minisweagent.models.oauth_model import CLAUDE_CODE_SYSTEM_PROMPT, OAuthLitellmModel

    stub = _StubProvider()
    _stash_provider_under("anthropic", stub)
    try:
        oauth.storage.save("anthropic", OAuthCredentials(refresh="r", access="x", expires=0))
        model = OAuthLitellmModel(
            model_name="anthropic/claude-sonnet-4-5",
            oauth_provider="anthropic",
            inject_claude_code_system=False,
        )
        prepared = model._prepare_messages_for_api([{"role": "user", "content": "hi"}])
        assert not any(
            msg.get("role") == "system" and CLAUDE_CODE_SYSTEM_PROMPT in str(msg.get("content", "")) for msg in prepared
        )
    finally:
        oauth.restore_oauth_provider("anthropic")


def test_non_anthropic_provider_no_system_inject(isolated_oauth):
    from minisweagent.models.oauth_model import CLAUDE_CODE_SYSTEM_PROMPT, OAuthLitellmModel

    model = OAuthLitellmModel(model_name="openai/codex-mini", oauth_provider="openai-codex")
    prepared = model._prepare_messages_for_api([{"role": "user", "content": "hi"}])
    assert not any(
        msg.get("role") == "system" and CLAUDE_CODE_SYSTEM_PROMPT in str(msg.get("content", "")) for msg in prepared
    )


def test_anthropic_inject_idempotent_with_multimodal_system(isolated_oauth):
    """A pre-existing system message whose content is a list of text parts
    (multimodal shape) must be detected so we do not double-inject."""
    from minisweagent.models.oauth_model import CLAUDE_CODE_SYSTEM_PROMPT, OAuthLitellmModel

    stub = _StubProvider()
    _stash_provider_under("anthropic", stub)
    try:
        oauth.storage.save("anthropic", OAuthCredentials(refresh="r", access="x", expires=0))
        model = OAuthLitellmModel(model_name="anthropic/claude-sonnet-4-5", oauth_provider="anthropic")
        prepared = model._prepare_messages_for_api(
            [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": f"{CLAUDE_CODE_SYSTEM_PROMPT}\nExtra system context."},
                    ],
                },
                {"role": "user", "content": "hi"},
            ]
        )
        system_msgs = [m for m in prepared if m.get("role") == "system"]
        assert len(system_msgs) == 1
    finally:
        oauth.restore_oauth_provider("anthropic")


def test_anthropic_inject_when_multimodal_system_lacks_prompt(isolated_oauth):
    """A system message that is multimodal but does NOT contain the Claude Code
    prompt must not satisfy the idempotency check (the old ``str(content)``
    coincidentally worked, but it should not be load-bearing)."""
    from minisweagent.models.oauth_model import CLAUDE_CODE_SYSTEM_PROMPT, OAuthLitellmModel

    stub = _StubProvider()
    _stash_provider_under("anthropic", stub)
    try:
        oauth.storage.save("anthropic", OAuthCredentials(refresh="r", access="x", expires=0))
        model = OAuthLitellmModel(model_name="anthropic/claude-sonnet-4-5", oauth_provider="anthropic")
        prepared = model._prepare_messages_for_api(
            [
                {"role": "system", "content": [{"type": "text", "text": "Other system instructions."}]},
                {"role": "user", "content": "hi"},
            ]
        )
        assert prepared[0] == {"role": "system", "content": CLAUDE_CODE_SYSTEM_PROMPT}
    finally:
        oauth.restore_oauth_provider("anthropic")


def test_codex_missing_account_id_raises(isolated_oauth):
    from minisweagent.models.oauth_model import OAuthLitellmModel

    stub = _StubProvider()  # no account_id in extra
    _stash_provider_under("openai-codex", stub)
    try:
        oauth.storage.save(
            "openai-codex",
            OAuthCredentials(refresh="r", access="old", expires=0),
        )
        model = OAuthLitellmModel(model_name="openai/codex-mini", oauth_provider="openai-codex")
        with pytest.raises(RuntimeError, match="account_id"):
            model._resolve_oauth_kwargs()
    finally:
        oauth.restore_oauth_provider("openai-codex")


def test_query_merges_extra_headers(isolated_oauth):
    from unittest.mock import MagicMock, patch

    from minisweagent.models.litellm_model import LitellmModel
    from minisweagent.models.oauth_model import OAuthLitellmModel

    stub = _StubProvider()
    _stash_provider_under("anthropic", stub)
    try:
        future = int(time.time() * 1000) + 60_000
        oauth.storage.save("anthropic", OAuthCredentials(refresh="r", access="tok", expires=future))
        model = OAuthLitellmModel(model_name="anthropic/claude-sonnet-4-5", oauth_provider="anthropic")

        captured: dict = {}

        def fake_parent_query(self, messages, **kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch.object(LitellmModel, "_query", fake_parent_query):
            model._query([{"role": "user", "content": "hi"}], extra_headers={"X-Custom": "yes"})

        assert captured["extra_headers"]["X-Custom"] == "yes"
        assert captured["extra_headers"]["x-app"] == "cli"
    finally:
        oauth.restore_oauth_provider("anthropic")


def test_codex_base_url_reads_env_at_construction(isolated_oauth, monkeypatch):
    from minisweagent.models.oauth_model import OAuthLitellmModel

    monkeypatch.setenv("MSWEA_CODEX_BASE_URL", "https://example.test/codex")
    model = OAuthLitellmModel(model_name="openai/codex-mini", oauth_provider="openai-codex")
    assert model.config.codex_base_url == "https://example.test/codex"

    monkeypatch.delenv("MSWEA_CODEX_BASE_URL", raising=False)
    default_model = OAuthLitellmModel(model_name="openai/codex-mini", oauth_provider="openai-codex")
    assert default_model.config.codex_base_url == "https://chatgpt.com/backend-api/codex"


def test_query_preserves_model_kwargs_extra_headers(isolated_oauth):
    """``model_kwargs.extra_headers`` (e.g. audit/tracing headers) must survive
    the OAuth header merge. Otherwise the parent ``LitellmModel._query``'s
    shallow ``self.config.model_kwargs | kwargs`` union drops them silently."""
    from unittest.mock import MagicMock, patch

    from minisweagent.models.litellm_model import LitellmModel
    from minisweagent.models.oauth_model import OAuthLitellmModel

    stub = _StubProvider()
    _stash_provider_under("anthropic", stub)
    try:
        future = int(time.time() * 1000) + 60_000
        oauth.storage.save("anthropic", OAuthCredentials(refresh="r", access="tok", expires=future))
        model = OAuthLitellmModel(
            model_name="anthropic/claude-sonnet-4-5",
            oauth_provider="anthropic",
            model_kwargs={"extra_headers": {"X-Audit": "yes", "Authorization": "should-be-overridden"}},
        )

        captured: dict = {}

        def fake_parent_query(self, messages, **kwargs):
            # Mimic parent: merge config.model_kwargs with kwargs shallowly.
            captured.update(self.config.model_kwargs | kwargs)
            return MagicMock()

        with patch.object(LitellmModel, "_query", fake_parent_query):
            model._query(
                [{"role": "user", "content": "hi"}],
                extra_headers={"X-Caller": "yes"},
            )

        # All three precedence levels survive after the parent's shallow union.
        assert captured["extra_headers"]["X-Audit"] == "yes"  # from model_kwargs
        assert captured["extra_headers"]["X-Caller"] == "yes"  # from per-call kwargs
        assert captured["extra_headers"]["Authorization"] == "Bearer tok"  # OAuth wins
        assert captured["extra_headers"]["x-app"] == "cli"  # OAuth-injected
    finally:
        oauth.restore_oauth_provider("anthropic")
