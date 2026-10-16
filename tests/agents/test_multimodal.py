import copy

import pytest

from minisweagent.models import get_model
from minisweagent.models.utils.openai_multimodal import (
    DEFAULT_MULTIMODAL_REGEX,
    _expand_content_string,
    expand_multimodal_content,
)


@pytest.mark.parametrize(
    ("model_class", "cache_enabled"),
    [
        (model_class, cache_enabled)
        for model_class in ["litellm", "litellm_textbased", "openrouter", "openrouter_textbased"]
        for cache_enabled in [True, False]
    ],
)
def test_prepare_multimodal_messages_with_cache_control(model_class, cache_enabled):
    config = {"model_class": model_class, "multimodal_regex": DEFAULT_MULTIMODAL_REGEX}
    if not cache_enabled:
        config["set_cache_control"] = None
    model = get_model("anthropic/claude-sonnet-4-5", config=config)
    messages = [
        model.format_message(
            role="user",
            content="Before <MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>https://example.com/image.png</MSWEA_MULTIMODAL_CONTENT> after",
            extra={"timestamp": 1},
        )
    ]
    original = copy.deepcopy(messages)
    expected = [{"role": "user", "content": copy.deepcopy(messages[0]["content"])}]
    assert [block["type"] for block in expected[0]["content"]] == ["text", "image_url", "text"]
    if cache_enabled:
        expected[0]["content"][-1]["cache_control"] = {"type": "ephemeral"}

    assert model._prepare_messages_for_api(messages) == expected
    assert messages == original
    expected.append({"role": "user", "content": "Next turn"})
    if cache_enabled:
        expected[0]["content"][-1].pop("cache_control")
        expected[-1]["content"] = [{"type": "text", "text": "Next turn", "cache_control": {"type": "ephemeral"}}]
    assert model._prepare_messages_for_api([*messages, {"role": "user", "content": "Next turn"}]) == expected


@pytest.mark.parametrize(
    ("model_class", "cache_mode"),
    [("litellm", "default_end"), ("litellm", None), ("openrouter", "default_end"), ("openrouter", None)],
)
def test_prepare_thinking_messages_with_cache_control(model_class, cache_mode):
    model = get_model(
        "anthropic/claude-sonnet-4-5", config={"model_class": model_class, "set_cache_control": cache_mode}
    )
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Answer"},
                {"type": "thinking", "thinking": "Reasoning", "signature": "signature"},
                {"type": "redacted_thinking", "data": "redacted"},
            ],
            "extra": {"timestamp": 1},
        }
    ]
    original = copy.deepcopy(messages)
    expected_text = {"type": "text", "text": "Answer"}
    if cache_mode:
        expected_text["cache_control"] = {"type": "ephemeral"}
    assert model._prepare_messages_for_api(messages) == [
        {
            "role": "assistant",
            "content": [*messages[0]["content"][1:], expected_text],
        }
    ]
    assert messages == original


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            "Just plain text",
            [{"type": "text", "text": "Just plain text"}],
        ),
        (
            "Text before <MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>https://example.com/image.png</MSWEA_MULTIMODAL_CONTENT> text after",
            [
                {"type": "text", "text": "Text before "},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                {"type": "text", "text": " text after"},
            ],
        ),
        (
            "<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>data:image/png;base64,iVBORw0KGgoAAAANS</MSWEA_MULTIMODAL_CONTENT>",
            [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANS"}}],
        ),
    ],
)
def test_expand_content_string(content, expected):
    """Test _expand_content_string with various content patterns."""
    assert _expand_content_string(content=content, pattern=DEFAULT_MULTIMODAL_REGEX) == expected


def test_expand_content_string_multiple_images():
    """Test _expand_content_string with multiple images."""
    content = (
        "First <MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>image1.png</MSWEA_MULTIMODAL_CONTENT> "
        "middle <MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>image2.jpg</MSWEA_MULTIMODAL_CONTENT> end"
    )
    result = _expand_content_string(content=content, pattern=DEFAULT_MULTIMODAL_REGEX)
    assert len(result) == 5
    assert result[0] == {"type": "text", "text": "First "}
    assert result[1] == {"type": "image_url", "image_url": {"url": "image1.png"}}
    assert result[2] == {"type": "text", "text": " middle "}
    assert result[3] == {"type": "image_url", "image_url": {"url": "image2.jpg"}}
    assert result[4] == {"type": "text", "text": " end"}


def test_expand_content_string_multiline():
    """Test _expand_content_string handles multiline image content."""
    content = """Here is an image:
<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>data:image/png;base64,
iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk</MSWEA_MULTIMODAL_CONTENT>
After image"""
    result = _expand_content_string(content=content, pattern=DEFAULT_MULTIMODAL_REGEX)
    assert len(result) == 3
    assert result[0] == {"type": "text", "text": "Here is an image:\n"}
    assert result[1]["type"] == "image_url"
    assert "data:image/png;base64" in result[1]["image_url"]["url"]
    assert result[2] == {"type": "text", "text": "\nAfter image"}


def test_expand_content_string_whitespace_handling():
    """Test that whitespace in image URLs is stripped but preserved in text."""
    content = "Text  \n<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>  image_url  </MSWEA_MULTIMODAL_CONTENT>  \nMore text"
    result = _expand_content_string(content=content, pattern=DEFAULT_MULTIMODAL_REGEX)
    assert result[0]["text"] == "Text  \n"
    assert result[1]["image_url"]["url"] == "image_url"
    assert result[2]["text"] == "  \nMore text"


def test_expand_content_string_adjacent_images():
    """Test multiple images with no text between them."""
    content = (
        "<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>img1</MSWEA_MULTIMODAL_CONTENT>"
        "<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>img2</MSWEA_MULTIMODAL_CONTENT>"
    )
    result = _expand_content_string(content=content, pattern=DEFAULT_MULTIMODAL_REGEX)
    assert len(result) == 2
    assert result[0] == {"type": "image_url", "image_url": {"url": "img1"}}
    assert result[1] == {"type": "image_url", "image_url": {"url": "img2"}}


def test_expand_multimodal_content_string():
    """Test expand_multimodal_content with string input."""
    content = (
        "Text <MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>image.png</MSWEA_MULTIMODAL_CONTENT> more"
    )
    result = expand_multimodal_content(content, pattern=DEFAULT_MULTIMODAL_REGEX)
    assert len(result) == 3
    assert result[0]["type"] == "text"
    assert result[1]["type"] == "image_url"
    assert result[2]["type"] == "text"


def test_expand_multimodal_content_list():
    """Test expand_multimodal_content with list input."""
    content = [
        "plain text",
        "text <MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>image.png</MSWEA_MULTIMODAL_CONTENT> more",
    ]
    result = expand_multimodal_content(content, pattern=DEFAULT_MULTIMODAL_REGEX)
    assert len(result) == 2
    assert result[0] == [{"type": "text", "text": "plain text"}]
    assert len(result[1]) == 3


def test_expand_multimodal_content_dict():
    """Test expand_multimodal_content with dict input."""
    content = {
        "role": "user",
        "content": "text <MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>image.png</MSWEA_MULTIMODAL_CONTENT>",
    }
    result = expand_multimodal_content(content, pattern=DEFAULT_MULTIMODAL_REGEX)
    assert result["role"] == "user"
    assert len(result["content"]) == 2


def test_expand_multimodal_content_dict_no_content_key():
    """Test expand_multimodal_content with dict without 'content' key."""
    input_dict = {"role": "user", "other": "data"}
    assert expand_multimodal_content(input_dict, pattern=DEFAULT_MULTIMODAL_REGEX) == input_dict


def test_expand_multimodal_content_nested():
    """Test expand_multimodal_content with nested structures."""
    content = {
        "role": "user",
        "content": [
            "text <MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>image.png</MSWEA_MULTIMODAL_CONTENT>",
            {"nested": "value"},
        ],
    }
    result = expand_multimodal_content(content, pattern=DEFAULT_MULTIMODAL_REGEX)
    assert result["role"] == "user"
    assert len(result["content"]) == 2
    assert len(result["content"][0]) == 2


def test_expand_multimodal_content_preserves_original():
    """Test that expand_multimodal_content deep copies and doesn't modify original."""
    original = {
        "role": "user",
        "content": "text <MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>image.png</MSWEA_MULTIMODAL_CONTENT>",
    }
    original_content = original["content"]
    expand_multimodal_content(original, pattern=DEFAULT_MULTIMODAL_REGEX)
    assert original["content"] == original_content


def test_model_format_message_with_multimodal():
    """Test that model.format_message applies multimodal transformation when configured."""
    from minisweagent.models.test_models import DeterministicModel

    model = DeterministicModel(outputs=[], multimodal_regex=DEFAULT_MULTIMODAL_REGEX)
    result = model.format_message(
        role="user",
        content="Hello <MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>image.png</MSWEA_MULTIMODAL_CONTENT>",
    )
    assert result["role"] == "user"
    assert len(result["content"]) == 2
    assert result["content"][0]["type"] == "text"
    assert result["content"][1]["type"] == "image_url"


def test_model_format_message_without_multimodal():
    """Test that model.format_message returns plain dict when multimodal is disabled."""
    from minisweagent.models.test_models import DeterministicModel

    model = DeterministicModel(outputs=[])
    result = model.format_message(role="user", content="Hello world")
    assert result == {"role": "user", "content": "Hello world"}


def test_unknown_content_type_ignored():
    """Test that unknown content types are ignored."""
    content = (
        "Text <MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>unknown_type</CONTENT_TYPE>data</MSWEA_MULTIMODAL_CONTENT> more"
    )
    result = _expand_content_string(content=content, pattern=DEFAULT_MULTIMODAL_REGEX)
    # Unknown type is not added, so we get text before, nothing for unknown, text after
    assert len(result) == 2
    assert result[0] == {"type": "text", "text": "Text "}
    assert result[1] == {"type": "text", "text": " more"}
