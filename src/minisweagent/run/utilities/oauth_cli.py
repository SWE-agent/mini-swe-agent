#!/usr/bin/env python3

"""Manage OAuth subscriptions: login / logout / refresh / list / status.

[bold]Subcommands[/bold]

  [bold green]login PROVIDER[/bold green]    Run the OAuth login flow for ``PROVIDER`` and persist the credentials.
  [bold green]logout PROVIDER[/bold green]   Forget the stored credentials for ``PROVIDER``.
  [bold green]refresh PROVIDER[/bold green]  Force-refresh the access token for ``PROVIDER`` (explicit refresh hook).
  [bold green]list[/bold green]              Show all stored providers and their expiry timestamps.
  [bold green]status PROVIDER[/bold green]   Show the current credential status for ``PROVIDER``.
  [bold green]token PROVIDER[/bold green]    Print a fresh access token to stdout (use in scripts).

Available providers: anthropic, openai-codex, github-copilot.
"""

from __future__ import annotations

import time
import webbrowser
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table
from typer import Argument, Option, Typer

from minisweagent import oauth
from minisweagent.oauth.types import OAuthAuthInfo, OAuthLoginCallbacks, OAuthPrompt

app = Typer(
    help=__doc__,
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
)
console = Console(highlight=False)
# Errors go to stderr so subcommands like ``token`` keep stdout clean for
# scripts that capture the access token.
err_console = Console(highlight=False, stderr=True)


def _prompt(*args, **kwargs) -> str:
    from prompt_toolkit.shortcuts.prompt import prompt as _p

    return _p(*args, **kwargs)


def _make_cli_callbacks(open_browser: bool) -> OAuthLoginCallbacks:
    def _on_auth(info: OAuthAuthInfo) -> None:
        console.print(f"\n[bold green]Open this URL to continue:[/bold green]\n{info.url}\n")
        if info.instructions:
            console.print(f"[bold yellow]{info.instructions}[/bold yellow]")
        if open_browser:
            try:
                webbrowser.open(info.url)
            except Exception:  # noqa: BLE001
                pass

    def _on_prompt(prompt: OAuthPrompt) -> str:
        message = prompt.message
        if prompt.placeholder:
            message = f"{message} [{prompt.placeholder}] "
        else:
            message = f"{message} "
        while True:
            value = _prompt(message)
            if value or prompt.allow_empty:
                return value
            console.print("[bold red]Value required.[/bold red]")

    def _on_progress(message: str) -> None:
        console.print(f"[dim]{message}[/dim]")

    return OAuthLoginCallbacks(on_auth=_on_auth, on_prompt=_on_prompt, on_progress=_on_progress)


def _format_expiry(expires_ms: int) -> str:
    seconds = expires_ms / 1000
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    delta = expires_ms - int(time.time() * 1000)
    if delta < 0:
        suffix = f"expired {-delta // 60000}m ago"
    else:
        suffix = f"in {delta // 60000}m"
    return f"{dt.isoformat(timespec='seconds')} ({suffix})"


@app.command("login")
def login(
    provider_id: str = Argument(..., help="Provider id: anthropic / openai-codex / github-copilot"),
    open_browser: bool = Option(True, "--browser/--no-browser", help="Open the auth URL in the default browser."),
) -> None:
    """Run the OAuth login flow and persist credentials."""
    if oauth.get_oauth_provider(provider_id) is None:
        ids = ", ".join(p.id for p in oauth.get_oauth_providers())
        console.print(f"[bold red]Unknown provider: {provider_id}[/bold red]. Available: {ids}")
        raise SystemExit(1)
    callbacks = _make_cli_callbacks(open_browser)
    creds = oauth.login_provider(provider_id, callbacks)
    console.print(
        f"[bold green]Logged in to {provider_id}[/bold green]. Token expires {_format_expiry(creds.expires)}."
    )


@app.command("logout")
def logout(provider_id: str = Argument(..., help="Provider id to log out of")) -> None:
    """Delete stored credentials for ``provider_id``."""
    if oauth.logout_provider(provider_id):
        console.print(f"[bold green]Logged out of {provider_id}[/bold green].")
    else:
        console.print(f"[yellow]No credentials stored for {provider_id}.[/yellow]")


@app.command("refresh")
def refresh(provider_id: str = Argument(..., help="Provider id to refresh")) -> None:
    """Force-refresh the access token (explicit refresh hook)."""
    creds = oauth.refresh_provider(provider_id)
    console.print(
        f"[bold green]Refreshed {provider_id}[/bold green]. New expiry {_format_expiry(creds.expires)}."
    )


@app.command("list")
def list_cmd() -> None:
    """List all OAuth credentials currently stored on disk."""
    providers = oauth.storage.list_providers()
    if not providers:
        console.print("[dim]No OAuth credentials stored. Try `mini-extra oauth login <provider>`.[/dim]")
        return
    table = Table(title="OAuth credentials")
    table.add_column("Provider")
    table.add_column("Expires")
    for pid in providers:
        creds = oauth.storage.load(pid)
        if creds is None:
            continue
        table.add_row(pid, _format_expiry(creds.expires))
    console.print(table)


@app.command("status")
def status(provider_id: str = Argument(..., help="Provider id")) -> None:
    """Show stored credential status."""
    creds = oauth.storage.load(provider_id)
    if creds is None:
        console.print(f"[yellow]No credentials stored for {provider_id}.[/yellow]")
        raise SystemExit(2)
    console.print(f"[bold]{provider_id}[/bold]")
    console.print(f"  expires: {_format_expiry(creds.expires)}")
    if creds.extra:
        for key, value in creds.extra.items():
            console.print(f"  {key}: {value}")


@app.command("token")
def token(
    provider_id: str = Argument(..., help="Provider id"),
    force_refresh: bool = Option(False, "--refresh", help="Force a refresh before printing the token."),
) -> None:
    """Print a fresh access token to stdout. Useful for scripts."""
    api_key = oauth.get_oauth_api_key(provider_id, force_refresh=force_refresh)
    if api_key is None:
        err_console.print(f"[bold red]No credentials for {provider_id}.[/bold red]")
        raise SystemExit(2)
    print(api_key)


if __name__ == "__main__":
    app()
