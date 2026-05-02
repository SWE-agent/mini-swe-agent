import json

import pytest

from minisweagent.oauth import storage
from minisweagent.oauth.types import OAuthCredentials


@pytest.fixture
def tmp_oauth_file(monkeypatch, tmp_path):
    path = tmp_path / "oauth.json"
    monkeypatch.setenv("MSWEA_OAUTH_FILE", str(path))
    return path


def test_save_load_round_trip(tmp_oauth_file):
    creds = OAuthCredentials(
        refresh="refresh-token",
        access="access-token",
        expires=1700000000000,
        extra={"account_id": "acc-1"},
    )
    storage.save("test-provider", creds)

    loaded = storage.load("test-provider")
    assert loaded is not None
    assert loaded.refresh == "refresh-token"
    assert loaded.access == "access-token"
    assert loaded.expires == 1700000000000
    assert loaded.extra == {"account_id": "acc-1"}


def test_load_missing_returns_none(tmp_oauth_file):
    assert storage.load("nope") is None


def test_delete(tmp_oauth_file):
    storage.save("a", OAuthCredentials(refresh="r", access="a", expires=1))
    storage.save("b", OAuthCredentials(refresh="r2", access="a2", expires=2))
    assert storage.delete("a") is True
    assert storage.load("a") is None
    assert storage.load("b") is not None
    assert storage.delete("a") is False


def test_delete_last_entry_unlinks_file(tmp_oauth_file):
    storage.save("only", OAuthCredentials(refresh="r", access="a", expires=1))
    assert tmp_oauth_file.exists()
    assert storage.delete("only") is True
    assert not tmp_oauth_file.exists()


def test_list_providers(tmp_oauth_file):
    storage.save("a", OAuthCredentials(refresh="r", access="a", expires=1))
    storage.save("b", OAuthCredentials(refresh="r2", access="a2", expires=2))
    assert storage.list_providers() == ["a", "b"]


def test_storage_file_is_json(tmp_oauth_file):
    storage.save("a", OAuthCredentials(refresh="r", access="t", expires=42))
    raw = json.loads(tmp_oauth_file.read_text())
    assert raw == {"a": {"refresh": "r", "access": "t", "expires": 42}}


def test_corrupt_file_yields_empty(tmp_oauth_file):
    tmp_oauth_file.write_text("not-json")
    assert storage.load_all() == {}


def test_get_storage_path_uses_env_override(monkeypatch, tmp_path):
    target = tmp_path / "custom.json"
    monkeypatch.setenv("MSWEA_OAUTH_FILE", str(target))
    assert storage.get_storage_path() == target


def test_get_storage_path_default(monkeypatch):
    monkeypatch.delenv("MSWEA_OAUTH_FILE", raising=False)
    path = storage.get_storage_path()
    assert path.name == "oauth.json"


def test_load_all_skips_non_dict_entries(tmp_oauth_file):
    tmp_oauth_file.write_text(
        json.dumps(
            {
                "valid": {"refresh": "r", "access": "a", "expires": 1},
                "bad-string": "not-a-dict",
                "bad-int": 42,
            }
        )
    )
    result = storage.load_all()
    assert "valid" in result
    assert "bad-string" not in result
    assert "bad-int" not in result


def test_load_all_skips_entries_missing_required_fields(tmp_oauth_file):
    tmp_oauth_file.write_text(
        json.dumps(
            {
                "ok": {"refresh": "r", "access": "a", "expires": 1},
                "no-access": {"refresh": "r", "expires": 1},
            }
        )
    )
    result = storage.load_all()
    assert "ok" in result
    assert "no-access" not in result


def test_atomic_write_creates_parent_dirs(monkeypatch, tmp_path):
    nested = tmp_path / "a" / "b" / "c" / "oauth.json"
    monkeypatch.setenv("MSWEA_OAUTH_FILE", str(nested))
    storage.save("p", OAuthCredentials(refresh="r", access="a", expires=1))
    assert nested.exists()


def test_atomic_write_parent_dir_is_private(monkeypatch, tmp_path):
    import os
    import stat

    parent = tmp_path / "private-oauth"
    target = parent / "oauth.json"
    monkeypatch.setenv("MSWEA_OAUTH_FILE", str(target))
    storage.save("p", OAuthCredentials(refresh="r", access="a", expires=1))
    mode = stat.S_IMODE(os.stat(parent).st_mode)
    assert mode == 0o700


def test_atomic_write_tightens_existing_parent_dir(monkeypatch, tmp_path):
    import os
    import stat

    parent = tmp_path / "loose-oauth"
    parent.mkdir(mode=0o755)
    os.chmod(parent, 0o755)
    target = parent / "oauth.json"
    monkeypatch.setenv("MSWEA_OAUTH_FILE", str(target))
    storage.save("p", OAuthCredentials(refresh="r", access="a", expires=1))
    mode = stat.S_IMODE(os.stat(parent).st_mode)
    assert mode == 0o700
