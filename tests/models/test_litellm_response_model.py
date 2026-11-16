import json

import pytest
from litellm.types.llms.openai import ResponsesAPIResponse

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_response_model import LitellmResponseModel


@pytest.fixture(params=[False, True], ids=["object", "dict"])
def response_factory(request):
    def build(**kwargs):
        response = ResponsesAPIResponse(id="resp_test", created_at=0, **kwargs)
        return response.model_dump() if request.param else response

    return build


def test_parse_actions_preserves_commands_and_call_ids(response_factory):
    commands = ["printf 'hello\\n'", 'echo "world"']
    response = response_factory(
        status="completed",
        output=[
            {"type": "reasoning", "id": "reasoning_1", "summary": []},
            *[
                {
                    "type": "function_call",
                    "call_id": f"call_{i}",
                    "name": "bash",
                    "arguments": json.dumps({"command": command}),
                }
                for i, command in enumerate(commands)
            ],
        ],
    )
    model = LitellmResponseModel(model_name="test/model")

    assert model._parse_actions(response) == [
        {"command": command, "tool_call_id": f"call_{i}"} for i, command in enumerate(commands)
    ]


def test_parse_actions_reports_unknown_tool(response_factory):
    response = response_factory(
        status="completed",
        output=[{"type": "function_call", "call_id": "call_1", "name": "unknown_tool", "arguments": "{}"}],
    )
    model = LitellmResponseModel(model_name="test/model")

    with pytest.raises(FormatError) as exc:
        model._parse_actions(response)

    assert "Unknown tool 'unknown_tool'" in exc.value.messages[0]["content"][0]["text"]
    assert exc.value.messages[0]["extra"]["interrupt_type"] == "FormatError"
