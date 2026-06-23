import pytest

from minisweagent.models.utils.content_string import get_content_string


class TestGetContentString:
    def test_plain_string_content(self):
        assert get_content_string({"content": "hello world"}) == "hello world"

    def test_openai_multimodal_text_blocks(self):
        msg = {"content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]}
        assert "first" in get_content_string(msg)
        assert "second" in get_content_string(msg)

    def test_openai_tool_calls(self):
        msg = {
            "content": "thinking...",
            "tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "ls -la"}'}}],
        }
        result = get_content_string(msg)
        assert "ls -la" in result
        assert "thinking..." in result

    @pytest.mark.parametrize(
        ("message", "expected_fragments"),
        [
            (
                {
                    "content": [
                        {"type": "text", "text": "Let me check"},
                        {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "ls -la"}},
                    ]
                },
                ["Let me check", "ls -la"],
            ),
            (
                {
                    "content": [
                        {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "echo hello"}},
                        {"type": "tool_use", "id": "toolu_2", "name": "bash", "input": {"command": "pwd"}},
                    ]
                },
                ["echo hello", "pwd"],
            ),
            (
                {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file1.txt\nfile2.txt"},
                    ]
                },
                ["file1.txt", "file2.txt"],
            ),
            (
                {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "<returncode>0</returncode>\n<output>ok</output>",
                        },
                    ]
                },
                ["<returncode>0</returncode>", "ok"],
            ),
        ],
    )
    def test_anthropic_content_blocks(self, message: dict, expected_fragments: list[str]):
        result = get_content_string(message)
        for fragment in expected_fragments:
            assert fragment in result

    def test_anthropic_tool_use_without_text_block(self):
        """tool_use-only messages must still produce visible output."""
        msg = {"content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "cat /etc/hosts"}}]}
        assert get_content_string(msg).strip() != ""

    def test_empty_message(self):
        assert get_content_string({}) == ""

    def test_content_list_skips_non_dict_items(self):
        assert get_content_string({"content": ["bare string", 42]}) == ""
