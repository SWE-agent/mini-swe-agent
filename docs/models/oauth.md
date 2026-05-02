# OAuth subscription auth

!!! abstract "Overview"

    * Use Claude Pro/Max, ChatGPT Plus/Pro (Codex), and GitHub Copilot subscriptions instead of pay-per-token API keys.
    * Tokens are stored locally with mode `0600` and refreshed automatically when they expire.
    * Login / logout / refresh / list / status / token are exposed as `mini-extra oauth` subcommands.

!!! warning "Use according to each provider's terms of service"

    OAuth subscription tokens are issued for use with the provider's official clients. Check the relevant provider terms before pointing the agent at one.

## Supported providers

| Provider id        | Name                                | Auth flow                                    |
| ------------------ | ----------------------------------- | -------------------------------------------- |
| `anthropic`        | Claude Pro / Max                    | Authorization code + PKCE, localhost callback |
| `openai-codex`     | ChatGPT Plus / Pro (Codex)          | Authorization code + PKCE, localhost callback |
| `github-copilot`   | GitHub Copilot (incl. Enterprise)   | Device-code flow                             |

## Quick start

```bash
# 1. Log in once (opens a browser; falls back to a paste prompt if needed)
mini-extra oauth login anthropic

# 2. Run mini against the OAuth-backed model class
mini -m anthropic/claude-sonnet-4-5-20250929 --model-class oauth
```

For a config-file workflow, set the `oauth` model class and `oauth_provider` key in the agent YAML:

```yaml
model:
  model_class: oauth
  model_name: anthropic/claude-sonnet-4-5-20250929
  oauth_provider: anthropic
```

The `oauth_provider` key is required and must be one of `anthropic`, `openai-codex`, `github-copilot`.

## `mini-extra oauth` subcommands

```bash
mini-extra oauth login PROVIDER     # run the login flow + persist credentials
mini-extra oauth logout PROVIDER    # delete stored credentials
mini-extra oauth refresh PROVIDER   # force-refresh the access token now
mini-extra oauth list               # show all stored providers + expiries
mini-extra oauth status PROVIDER    # show current credential status
mini-extra oauth token PROVIDER     # print a fresh access token to stdout (scripts)
```

`mini-extra oauth token --refresh PROVIDER` forces a refresh before printing.
`mini-extra oauth login --no-browser PROVIDER` skips opening the browser (useful over SSH; paste the redirect URL manually).

## Per-provider notes

=== "Anthropic (Claude Pro/Max)"

    * Login binds a localhost listener on port `53692` (override with `MSWEA_ANTHROPIC_CALLBACK_PORT`).
    * The `oauth` model class injects the required `Authorization: Bearer <token>`, `anthropic-beta: claude-code-20250219,oauth-2025-04-20`, `user-agent: claude-cli/...`, and `x-app: cli` headers.
    * It also prepends the `"You are Claude Code, Anthropic's official CLI for Claude."` system message (required by Anthropic's OAuth path). Disable with `inject_claude_code_system: false` if you know what you are doing.

=== "OpenAI Codex (ChatGPT Plus/Pro)"

    * Login binds a localhost listener on port `1455` (override with `MSWEA_CODEX_CALLBACK_PORT`).
    * Requests are sent to `https://chatgpt.com/backend-api/codex` by default — override with `MSWEA_CODEX_BASE_URL` or the `codex_base_url` config key.
    * The `originator` header defaults to `mini-swe-agent`; override with the `codex_originator` config key.

=== "GitHub Copilot"

    * Uses the GitHub device-code flow — no localhost listener.
    * The login prompt asks for your GitHub Enterprise URL / domain; leave blank for `github.com`.
    * Requests go to `https://api.individual.githubcopilot.com` by default; the base URL is derived from the token's `proxy-ep=` claim or, for enterprise, from `https://copilot-api.<domain>`.

## Environment variables

```bash
# Path to the OAuth credentials file (default: $MSWEA_GLOBAL_CONFIG_DIR/oauth.json, mode 0600)
MSWEA_OAUTH_FILE="/path/to/oauth.json"

# Seconds to wait for the localhost callback before falling back to a paste prompt
# Used by the anthropic and openai-codex flows (default: 300)
MSWEA_OAUTH_CALLBACK_TIMEOUT="300"

# Localhost port used by the Anthropic OAuth callback (default: 53692)
MSWEA_ANTHROPIC_CALLBACK_PORT="53692"

# Localhost port used by the OpenAI Codex OAuth callback (default: 1455)
MSWEA_CODEX_CALLBACK_PORT="1455"

# Base URL for OpenAI Codex requests (default: https://chatgpt.com/backend-api/codex)
MSWEA_CODEX_BASE_URL="https://chatgpt.com/backend-api/codex"
```

The localhost hostname is hard-coded to `localhost` for both the Anthropic and Codex flows because the providers whitelist that exact string in their redirect URI allowlists.

## Storage layout

Credentials are persisted as JSON keyed by provider id:

```json
{
  "anthropic":      {"refresh": "...", "access": "...", "expires": 1735689600000},
  "openai-codex":   {"refresh": "...", "access": "...", "expires": 1735689600000, "account_id": "..."},
  "github-copilot": {"refresh": "...", "access": "...", "expires": 1735689600000, "enterprise_url": "..."}
}
```

`expires` is unix-millis. The file is created with mode `0600` and parents with `0700`. Override the path with `MSWEA_OAUTH_FILE`.

## Programmatic API

```python
from minisweagent import oauth

# List provider implementations
providers = oauth.get_oauth_providers()

# Run a login flow yourself with custom callbacks
creds = oauth.login_provider("anthropic", callbacks)

# Force a refresh from your own code (e.g. before a long batch run)
oauth.refresh_provider("anthropic")

# Get a fresh access token, refreshing on demand
token = oauth.get_oauth_api_key("anthropic")

# Subscribe to refresh events (e.g. to mirror the token to another store)
oauth.subscribe_refresh(lambda provider_id, creds: ...)
```

Public surface lives in `minisweagent.oauth` — see [the API reference](../reference/models/oauth.md) for full details.

{% include-markdown "../_footer.md" %}
