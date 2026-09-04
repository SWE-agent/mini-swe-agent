import json
from pathlib import Path

import yaml

from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.actions_text import format_observation_messages
from minisweagent.models.utils.actions_toolcall import format_toolcall_observation_messages
from minisweagent.models.utils.actions_toolcall_response import (
    format_toolcall_observation_messages as format_response_observation_messages,
)
from minisweagent.models.utils.observations import MAX_OBSERVATION_OUTPUT_BYTES


def _large_output() -> str:
    return "A" * 10000 + "B" * 10000


def test_toolcall_observation_bounds_provider_facing_output():
    output = _large_output()
    messages = format_toolcall_observation_messages(
        actions=[{"command": "generate output", "tool_call_id": "call_1"}],
        outputs=[{"output": output, "returncode": 0}],
        observation_template="{{ output.output }}",
    )

    content = messages[0]["content"]
    assert len(content.encode("utf-8")) <= MAX_OBSERVATION_OUTPUT_BYTES
    assert "output truncated" in content
    assert content.startswith("A")
    assert content.endswith("B")
    assert messages[0]["extra"]["raw_output"] == output


def test_toolcall_observation_stays_bounded_across_format_error_retries():
    output = _large_output()
    messages = format_toolcall_observation_messages(
        actions=[{"command": "generate output", "tool_call_id": "call_1"}],
        outputs=[{"output": output, "returncode": 0}],
        observation_template="{{ output.output }}",
    )

    assert len(messages[0]["content"].encode("utf-8")) * 3 < 8192


def test_mini_template_observation_stays_bounded_for_provider_retries():
    output = _large_output()
    config_path = Path(__file__).parents[2] / "src" / "minisweagent" / "config" / "mini.yaml"
    template = yaml.safe_load(config_path.read_text())["model"]["observation_template"]

    model = LitellmModel(model_name="openai/gpt-4o", observation_template=template)
    messages = model.format_observation_messages(
        {"extra": {"actions": [{"command": "generate output", "tool_call_id": "call_1"}]}},
        [{"output": output, "returncode": 0, "exception_info": ""}],
    )
    provider_messages = model._prepare_messages_for_api(messages)

    content = provider_messages[0]["content"]
    assert len(content.encode("utf-8")) * 3 < 8192
    assert "output truncated" in content
    assert "raw_output" not in json.dumps(provider_messages)
    assert output not in json.dumps(provider_messages)


def test_response_api_observation_bounds_provider_facing_output():
    output = _large_output()
    messages = format_response_observation_messages(
        actions=[{"command": "generate output", "tool_call_id": "call_1"}],
        outputs=[{"output": output, "returncode": 0}],
        observation_template="{{ output.output }}",
    )

    content = messages[0]["output"]
    assert len(content.encode("utf-8")) <= MAX_OBSERVATION_OUTPUT_BYTES
    assert "output truncated" in content
    assert messages[0]["extra"]["raw_output"] == output


def test_text_observation_bounds_provider_facing_output():
    output = _large_output()
    messages = format_observation_messages(
        [{"output": output, "returncode": 0}],
        observation_template="{{ output.output }}",
    )

    content = messages[0]["content"]
    assert len(content.encode("utf-8")) <= MAX_OBSERVATION_OUTPUT_BYTES
    assert "output truncated" in content
    assert messages[0]["extra"]["raw_output"] == output
