import shutil

import pytest

from minisweagent.models.utils.warpgrep import WarpGrepClient, generate_repo_tree


class TestGenerateRepoTree:
    def test_excludes_hidden_and_venv_dirs(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").touch()
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "node_modules").mkdir()
        (tmp_path / ".venv").mkdir()
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").touch()
        tree = generate_repo_tree(tmp_path)
        assert ".git" not in tree
        assert "__pycache__" not in tree
        assert "node_modules" not in tree
        assert ".venv" not in tree
        assert "src" in tree
        assert "main.py" in tree

    def test_respects_max_depth(self, tmp_path):
        d = tmp_path
        for name in ["alpha", "bravo", "charlie", "delta", "echo"]:
            d = d / name
            d.mkdir()
            (d / "file.txt").touch()
        tree = generate_repo_tree(tmp_path, max_depth=2)
        assert "alpha" in tree
        assert "bravo" in tree
        assert "delta" not in tree
        assert "echo" not in tree

    def test_empty_directory(self, tmp_path):
        tree = generate_repo_tree(tmp_path)
        assert tree == tmp_path.name


class TestParseToolCalls:
    def test_parses_single_ripgrep_call(self):
        client = WarpGrepClient(api_key="test")
        content = '<tool_call><function=ripgrep>{"pattern": "auth", "include": "*.py"}</function></tool_call>'
        calls = client._parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "ripgrep"
        assert calls[0]["pattern"] == "auth"
        assert calls[0]["include"] == "*.py"

    def test_parses_multiple_calls(self):
        client = WarpGrepClient(api_key="test")
        content = (
            '<tool_call><function=ripgrep>{"pattern": "foo"}</function></tool_call>'
            '<tool_call><function=read>{"path": "x.py", "start": 1, "end": 10}</function></tool_call>'
        )
        calls = client._parse_tool_calls(content)
        assert len(calls) == 2
        assert calls[0]["name"] == "ripgrep"
        assert calls[1]["name"] == "read"
        assert calls[1]["path"] == "x.py"

    def test_parses_finish_call(self):
        client = WarpGrepClient(api_key="test")
        content = '<tool_call><function=finish>{"files": ["src/a.py:1-5"]}</function></tool_call>'
        calls = client._parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "finish"
        assert calls[0]["specs"] == ["src/a.py:1-5"]

    def test_no_tool_calls(self):
        client = WarpGrepClient(api_key="test")
        assert client._parse_tool_calls("Just some text with no calls") == []

    def test_invalid_json_propagates(self):
        client = WarpGrepClient(api_key="test")
        content = "<tool_call><function=ripgrep>not valid json</function></tool_call>"
        with pytest.raises(ValueError):
            client._parse_tool_calls(content)


class TestExecuteToolCall:
    def test_read_file(self, tmp_path):
        client = WarpGrepClient(api_key="test")
        f = tmp_path / "hello.py"
        f.write_text("line1\nline2\nline3\nline4\n")
        result = client._execute_tool_call({"name": "read", "path": "hello.py", "start": 2, "end": 3}, tmp_path)
        assert "2: line2" in result
        assert "3: line3" in result
        assert "line1" not in result

    def test_list_directory(self, tmp_path):
        client = WarpGrepClient(api_key="test")
        (tmp_path / "a.py").touch()
        (tmp_path / "b.py").touch()
        result = client._execute_tool_call({"name": "list_directory", "path": "."}, tmp_path)
        assert "a.py" in result
        assert "b.py" in result

    def test_path_traversal_blocked_read(self, tmp_path):
        client = WarpGrepClient(api_key="test")
        result = client._execute_tool_call({"name": "read", "path": "../../etc/passwd"}, tmp_path)
        assert "path traversal blocked" in result

    def test_path_traversal_blocked_list(self, tmp_path):
        client = WarpGrepClient(api_key="test")
        result = client._execute_tool_call({"name": "list_directory", "path": "../.."}, tmp_path)
        assert "path traversal blocked" in result

    @pytest.mark.skipif(
        shutil.which("rg") is None,
        reason="rg not installed",
    )
    def test_ripgrep_execution(self, tmp_path):
        client = WarpGrepClient(api_key="test")
        (tmp_path / "test.py").write_text("def hello():\n    return 42\n")
        result = client._execute_tool_call({"name": "ripgrep", "pattern": "hello", "include": "*.py"}, tmp_path)
        assert "hello" in result


class TestFormatFinish:
    def test_format_line_ranges(self, tmp_path):
        client = WarpGrepClient(api_key="test")
        f = tmp_path / "code.py"
        f.write_text("\n".join(f"line{i}" for i in range(1, 11)))
        result = client._format_finish(["code.py:2-4"], tmp_path)
        assert "2: line2" in result
        assert "4: line4" in result
        assert "line1" not in result

    def test_format_no_specs(self, tmp_path):
        client = WarpGrepClient(api_key="test")
        assert client._format_finish([], tmp_path) == "WarpGrep: no results."


class TestMissingApiKey:
    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("MORPH_API_KEY", raising=False)
        monkeypatch.delenv("WARPGREP_API_KEY", raising=False)
        from minisweagent.models.litellm_warpgrep import LitellmWarpGrepModel

        with pytest.raises(ValueError, match="MORPH_API_KEY"):
            LitellmWarpGrepModel(model_name="test-model", cost_tracking="ignore_errors")
