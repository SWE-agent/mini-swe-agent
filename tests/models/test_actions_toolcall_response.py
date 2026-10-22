import copy
import time

import pytest

from minisweagent.models.utils.actions_toolcall_response import format_toolcall_observation_messages
from minisweagent.models.utils.content_string import get_content_string
from minisweagent.models.utils.openai_multimodal import DEFAULT_MULTIMODAL_REGEX


@pytest.mark.parametrize(
    ("action", "message_fields", "content_key"),
    [
        ({"tool_call_id": "call_1"}, {"type": "function_call_output", "call_id": "call_1"}, "output"),
        ({}, {"type": "message", "role": "user"}, "content"),
    ],
)
@pytest.mark.parametrize(
    ("pattern", "tag"),
    [
        (
            DEFAULT_MULTIMODAL_REGEX,
            "<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>{}</MSWEA_MULTIMODAL_CONTENT>",
        ),
        (r"<media type='(.*?)'>(.*?)</media>", "<media type='image_url'>{}</media>"),
    ],
)
def test_multimodal_observations(action, message_fields, content_key, pattern, tag):
    urls = ["https://example.com/image.png", "data:image/png;base64,iVBORw0KGgoAAAANS"]
    raw_output = "before " + " middle ".join(tag.format(f"  {url}  ") for url in urls) + " after"
    outputs = [{"output": raw_output, "returncode": 0, "exception_info": None, "extra": {"detail": "kept"}}]
    original_outputs = copy.deepcopy(outputs)
    started = time.time()
    messages = format_toolcall_observation_messages(
        actions=[action],
        outputs=outputs,
        observation_template="{{ label }}: {{ output.output }}",
        template_vars={"label": "Screenshot"},
        multimodal_regex=pattern,
    )
    assert started <= messages[0]["extra"]["timestamp"] <= time.time()
    assert messages == [
        {
            **message_fields,
            content_key: [
                {"type": "input_text", "text": "Screenshot: before "},
                {"type": "input_image", "image_url": urls[0]},
                {"type": "input_text", "text": " middle "},
                {"type": "input_image", "image_url": urls[1]},
                {"type": "input_text", "text": " after"},
            ],
            "extra": {
                "raw_output": raw_output,
                "returncode": 0,
                "exception_info": None,
                "detail": "kept",
                "timestamp": messages[0]["extra"]["timestamp"],
            },
        }
    ]
    assert get_content_string(messages[0]) == "Screenshot: before \n\n middle \n\n after"
    assert outputs == original_outputs


@pytest.mark.parametrize(("action", "content_key"), [({"tool_call_id": "call_1"}, "output"), ({}, "content")])
@pytest.mark.parametrize(
    ("pattern", "text"),
    [("", "plain text"), ("", ""), (DEFAULT_MULTIMODAL_REGEX, "plain text"), (DEFAULT_MULTIMODAL_REGEX, "")],
)
def test_text_observations(action, content_key, pattern, text):
    messages = format_toolcall_observation_messages(
        actions=[action],
        outputs=[{"output": text, "returncode": 0}],
        observation_template="{{ output.output }}",
        multimodal_regex=pattern,
    )
    assert messages[0][content_key] == (
        [{"type": "input_text", "text": text}] if pattern or content_key == "content" else text
    )
    assert messages[0]["extra"]["raw_output"] == text
    assert get_content_string(messages[0]) == text


def test_disabled_multimodal_preserves_tags_and_pads_outputs():
    text = "<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>image.png</MSWEA_MULTIMODAL_CONTENT>"
    messages = format_toolcall_observation_messages(
        actions=[{"tool_call_id": "call_1"}, {}, {"tool_call_id": "call_2"}],
        outputs=[{"output": text, "returncode": 0}, {"output": text, "returncode": 0}],
        observation_template="{{ output.output }}",
    )
    assert len(messages) == 3
    assert messages[0]["output"] == text
    assert messages[1]["content"] == [{"type": "input_text", "text": text}]
    assert messages[2]["call_id"] == "call_2"
    assert messages[2]["output"] == ""
    assert messages[2]["extra"]["returncode"] == -1
    assert messages[2]["extra"]["exception_info"] == "action was not executed"
