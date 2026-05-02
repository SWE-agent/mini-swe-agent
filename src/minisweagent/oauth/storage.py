"""Persistent OAuth credential storage.

Stores credentials as JSON, keyed by provider id, in
``$MSWEA_GLOBAL_CONFIG_DIR/oauth.json`` (defaults to the platform user config
directory). The file is created with mode ``0600`` to keep refresh tokens
private to the user.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from minisweagent import global_config_dir
from minisweagent.oauth.types import OAuthCredentials

try:
    import fcntl  # POSIX-only
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]

_LOCK = threading.Lock()


def get_storage_path() -> Path:
    """Return the path of the oauth credentials file."""
    override = os.getenv("MSWEA_OAUTH_FILE")
    if override:
        return Path(override)
    return Path(global_config_dir) / "oauth.json"


@contextlib.contextmanager
def _interprocess_lock(target: Path) -> Iterator[None]:
    """Cross-process exclusive lock for the credential file.

    The in-process :data:`_LOCK` only serializes threads in this Python
    process. Two ``mini-extra oauth refresh`` invocations running side-by-side
    would otherwise read-modify-write the same JSON file and clobber each
    other's rotated refresh tokens, forcing a re-login. ``fcntl.flock`` on a
    sidecar file gives us a real cross-process barrier on POSIX. On Windows
    we fall back to the in-process lock only — multi-process refresh is not a
    primary supported configuration there.
    """
    if fcntl is None:  # pragma: no cover - Windows fallback
        yield
        return
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = target.parent / (target.name + ".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        os.close(fd)


def _read_raw(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(data, indent=2, sort_keys=True)
    # Open with O_CREAT|O_WRONLY|O_TRUNC and explicit 0o600 so the file is
    # never world-readable on disk — Path.write_text would create it with the
    # default umask (typically 0o644) before any chmod could tighten it,
    # leaking the refresh token through that race window.
    fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        # Pre-existing tmp (e.g. crash leftover) keeps its old mode after
        # O_TRUNC; fchmod the fd so the mode is set before any write lands.
        if hasattr(os, "fchmod"):
            try:
                os.fchmod(fd, 0o600)
            except OSError:
                pass
        with os.fdopen(fd, "w") as f:
            f.write(payload)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def load_all() -> dict[str, OAuthCredentials]:
    with _LOCK:
        raw = _read_raw(get_storage_path())
        out: dict[str, OAuthCredentials] = {}
        for provider_id, creds in raw.items():
            if not isinstance(creds, dict):
                continue
            try:
                out[provider_id] = OAuthCredentials.from_dict(creds)
            except (KeyError, TypeError, ValueError):
                continue
        return out


def load(provider_id: str) -> OAuthCredentials | None:
    return load_all().get(provider_id)


def save(provider_id: str, credentials: OAuthCredentials) -> None:
    with _LOCK:
        path = get_storage_path()
        with _interprocess_lock(path):
            raw = _read_raw(path)
            raw[provider_id] = credentials.to_dict()
            _atomic_write(path, raw)


def delete(provider_id: str) -> bool:
    with _LOCK:
        path = get_storage_path()
        with _interprocess_lock(path):
            raw = _read_raw(path)
            if provider_id not in raw:
                return False
            del raw[provider_id]
            if raw:
                _atomic_write(path, raw)
            else:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            return True


def list_providers() -> list[str]:
    return sorted(load_all().keys())
