import pytest

from minisweagent.agents.multimodal import (
    MultimodalAgent,
    MultimodalAgentConfig,
    _expand_content_string,
)
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            "Just plain text",
            [{"type": "text", "content": "Just plain text"}],
        ),
        (
            "Text before <MSWEA_IMG_CONTENT>https://example.com/image.png</MSWEA_IMG_CONTENT> text after",
            [
                {"type": "text", "content": "Text before "},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                {"type": "text", "content": " text after"},
            ],
        ),
        (
            "<MSWEA_IMG_CONTENT>data:image/png;base64,iVBORw0KGgoAAAANS</MSWEA_IMG_CONTENT>",
            [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANS"}}],
        ),
        (
            "First <MSWEA_IMG_CONTENT>image1.png_</MSWEA_IMG_CONTENT> middle <MSWEA_IMG_CONTENT>image2.jpg_</MSWEA_IMG_CONTENT> end",
            [
                {"type": "text", "content": "First "},
                {"type": "image_url", "image_url": {"url": "image1.png_"}},
                {"type": "text", "content": " middle "},
                {"type": "image_url", "image_url": {"url": "image2.jpg_"}},
                {"type": "text", "content": " end"},
            ],
        ),
    ],
)
def test_expand_content_string(content, expected):
    """Test _expand_content_string with various content patterns."""
    assert (
        _expand_content_string(content=content, pattern=r"(?s)<MSWEA_IMG_CONTENT>(.{10,}?)</MSWEA_IMG_CONTENT>")
        == expected
    )


def test_expand_content_string_multiline():
    """Test _expand_content_string handles multiline image content."""
    content = """Here is an image:
<MSWEA_IMG_CONTENT>data:image/png;base64,
iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==</MSWEA_IMG_CONTENT>
After image"""
    result = _expand_content_string(content=content, pattern=r"(?s)<MSWEA_IMG_CONTENT>(.{10,}?)</MSWEA_IMG_CONTENT>")
    assert len(result) == 3
    assert result[0] == {"type": "text", "content": "Here is an image:\n"}
    assert result[1]["type"] == "image_url"
    assert "data:image/png;base64" in result[1]["image_url"]["url"]
    assert result[2] == {"type": "text", "content": "\nAfter image"}


def test_expand_content_string_whitespace_handling():
    """Test that whitespace in image URLs is stripped but preserved in text."""
    content = "Text  \n<MSWEA_IMG_CONTENT>  image_url_10  </MSWEA_IMG_CONTENT>  \nMore text"
    result = _expand_content_string(content=content, pattern=r"(?s)<MSWEA_IMG_CONTENT>(.{10,}?)</MSWEA_IMG_CONTENT>")
    assert result[0]["content"] == "Text  \n"
    assert result[1]["image_url"]["url"] == "image_url_10"
    assert result[2]["content"] == "  \nMore text"


def test_expand_content_string_too_short():
    """Test that image blocks shorter than 10 characters are not matched."""
    content = "Text <MSWEA_IMG_CONTENT>short</MSWEA_IMG_CONTENT> more"
    assert _expand_content_string(content=content, pattern=r"(?s)<MSWEA_IMG_CONTENT>(.{10,}?)</MSWEA_IMG_CONTENT>") == [
        {"type": "text", "content": content}
    ]


def test_expand_content_string_only_text_before():
    """Test content with only text before image."""
    content = "Text before <MSWEA_IMG_CONTENT>image12345</MSWEA_IMG_CONTENT>"
    result = _expand_content_string(content=content, pattern=r"(?s)<MSWEA_IMG_CONTENT>(.{10,}?)</MSWEA_IMG_CONTENT>")
    assert len(result) == 2
    assert result[0] == {"type": "text", "content": "Text before "}
    assert result[1] == {"type": "image_url", "image_url": {"url": "image12345"}}


def test_expand_content_string_only_text_after():
    """Test content with only text after image."""
    content = "<MSWEA_IMG_CONTENT>image12345</MSWEA_IMG_CONTENT> text after"
    result = _expand_content_string(content=content, pattern=r"(?s)<MSWEA_IMG_CONTENT>(.{10,}?)</MSWEA_IMG_CONTENT>")
    assert len(result) == 2
    assert result[0] == {"type": "image_url", "image_url": {"url": "image12345"}}
    assert result[1] == {"type": "text", "content": " text after"}


def test_expand_content_string_adjacent_images():
    """Test multiple images with no text between them."""
    content = "<MSWEA_IMG_CONTENT>image1_url</MSWEA_IMG_CONTENT><MSWEA_IMG_CONTENT>image2_url</MSWEA_IMG_CONTENT>"
    result = _expand_content_string(content=content, pattern=r"(?s)<MSWEA_IMG_CONTENT>(.{10,}?)</MSWEA_IMG_CONTENT>")
    assert len(result) == 2
    assert result[0] == {"type": "image_url", "image_url": {"url": "image1_url"}}
    assert result[1] == {"type": "image_url", "image_url": {"url": "image2_url"}}


def test_multimodal_agent_expand_content_string():
    """Test MultimodalAgent._expand_content with string input."""
    agent = MultimodalAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        system_template="test",
        instance_template="test",
    )
    result = agent._expand_content("Text <MSWEA_IMG_CONTENT>image12345</MSWEA_IMG_CONTENT> more")
    assert len(result) == 3
    assert result[0]["type"] == "text"
    assert result[1]["type"] == "image_url"
    assert result[2]["type"] == "text"


def test_multimodal_agent_expand_content_list():
    """Test MultimodalAgent._expand_content with list input."""
    agent = MultimodalAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        system_template="test",
        instance_template="test",
    )
    result = agent._expand_content(["plain text", "text <MSWEA_IMG_CONTENT>image12345</MSWEA_IMG_CONTENT> more"])
    assert len(result) == 2
    assert result[0] == [{"type": "text", "content": "plain text"}]
    assert len(result[1]) == 3


def test_multimodal_agent_expand_content_dict():
    """Test MultimodalAgent._expand_content with dict input."""
    agent = MultimodalAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        system_template="test",
        instance_template="test",
    )
    result = agent._expand_content(
        {"role": "user", "content": "text <MSWEA_IMG_CONTENT>image12345</MSWEA_IMG_CONTENT>"}
    )
    assert result["role"] == "user"
    assert len(result["content"]) == 2


def test_multimodal_agent_expand_content_dict_no_content_key():
    """Test MultimodalAgent._expand_content with dict without 'content' key."""
    agent = MultimodalAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        system_template="test",
        instance_template="test",
    )
    input_dict = {"role": "user", "other": "data"}
    assert agent._expand_content(input_dict) == input_dict


def test_multimodal_agent_expand_content_nested():
    """Test MultimodalAgent._expand_content with nested structures."""
    agent = MultimodalAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        system_template="test",
        instance_template="test",
    )
    result = agent._expand_content(
        {
            "role": "user",
            "content": [
                "text <MSWEA_IMG_CONTENT>image12345</MSWEA_IMG_CONTENT>",
                {"nested": "value"},
            ],
        }
    )
    assert result["role"] == "user"
    assert len(result["content"]) == 2
    assert len(result["content"][0]) == 2


def test_multimodal_agent_add_messages():
    """Test MultimodalAgent.add_messages expands image content."""
    agent = MultimodalAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        system_template="test",
        instance_template="test",
    )
    agent.add_messages(
        {"role": "user", "content": "Hello <MSWEA_IMG_CONTENT>image12345</MSWEA_IMG_CONTENT>"},
        {"role": "assistant", "content": "Response"},
    )
    assert len(agent.messages) == 2
    assert agent.messages[0]["content"][0]["type"] == "text"
    assert agent.messages[0]["content"][1]["type"] == "image_url"
    assert agent.messages[1]["content"] == [{"type": "text", "content": "Response"}]


def test_multimodal_agent_add_messages_no_images():
    """Test MultimodalAgent.add_messages with plain text."""
    agent = MultimodalAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        system_template="test",
        instance_template="test",
    )
    agent.add_messages({"role": "user", "content": "Just text"})
    assert len(agent.messages) == 1
    assert agent.messages[0]["content"] == [{"type": "text", "content": "Just text"}]


def test_multimodal_agent_custom_regex():
    """Test MultimodalAgent with custom image regex pattern."""

    class CustomConfig(MultimodalAgentConfig):
        image_regex: str = r"(?s)<IMG>(.{5,}?)</IMG>"

    agent = MultimodalAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        config_class=CustomConfig,
        system_template="test",
        instance_template="test",
    )
    agent.add_messages({"role": "user", "content": "Text <IMG>image</IMG> more"})
    assert len(agent.messages) == 1
    assert agent.messages[0]["content"][1]["type"] == "image_url"
    assert agent.messages[0]["content"][1]["image_url"]["url"] == "image"


def test_multimodal_agent_preserves_original_content():
    """Test that _expand_content deep copies and doesn't modify original."""
    agent = MultimodalAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        system_template="test",
        instance_template="test",
    )
    original = {"role": "user", "content": "text <MSWEA_IMG_CONTENT>image12345</MSWEA_IMG_CONTENT>"}
    original_content = original["content"]
    agent._expand_content(original)
    assert original["content"] == original_content
