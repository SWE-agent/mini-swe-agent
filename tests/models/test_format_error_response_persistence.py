"""Regression tests for issue #784 — FormatError must carry the unparsed
LLM response in `e.messages[0]["extra"]["response"]` for trajectory inspection.

Each test triggers FormatError via an unrecognised tool name on the parser,
then asserts that the response payload was persisted into the FormatError's
extra dict before re-raise. On the pre-fix code these tests fail because
`e.messages[0]["extra"]` only contains `{"interrupt_type": "FormatError"}`
and no `response` key.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.exceptions import FormatError

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _bad_tool_call_mock() -> MagicMock:
    """Build a tool-call mock whose function name is not 'bash'."""
    tc = MagicMock()
    tc.id = "call_xyz"
    tc.function.name = "unknown_tool"
    tc.function.arguments = "{}"
    return tc


def _bad_chat_completion_dict() -> dict[str, Any]:
    """Plain-dict chat completion payload (OpenRouter/Requesty shape) with bad tool."""
    return {
        "id": "resp_1",
        "model": "test-model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_xyz",
                            "type": "function",
                            "function": {"name": "unknown_tool", "arguments": "{}"},
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.01},
    }


def _bad_response_api_dict() -> dict[str, Any]:
    """Plain-dict Responses API payload with a bad function_call item."""
    return {
        "id": "resp_2",
        "object": "response",
        "model": "test-model",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_xyz",
                "name": "unknown_tool",
                "arguments": "{}",
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


# --------------------------------------------------------------------------- #
# litellm_model.LitellmModel
# --------------------------------------------------------------------------- #

def test_litellm_model_format_error_persists_response() -> None:
    from minisweagent.models.litellm_model import LitellmModel

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.tool_calls = [_bad_tool_call_mock()]
    serialized = {"id": "resp_1", "choices": [{"message": {"tool_calls": [{"function": {"name": "unknown_tool"}}]}}]}
    response.model_dump.return_value = serialized

    model = LitellmModel(model_name="test/model")

    with patch.object(LitellmModel, "_query", return_value=response), \
         patch.object(LitellmModel, "_calculate_cost", return_value={"cost": 0.0}):
        with pytest.raises(FormatError) as excinfo:
            model.query([{"role": "user", "content": "hi"}])

    extra = excinfo.value.messages[0]["extra"]
    assert "response" in extra, "response key missing — fix not applied"
    assert extra["response"], "response payload empty — fix not applied"
    # model_dump(mode='json') was used: result must be JSON-serialisable
    json.dumps(extra["response"])
    response.model_dump.assert_any_call(mode="json")


# --------------------------------------------------------------------------- #
# litellm_response_model.LitellmResponseModel
# --------------------------------------------------------------------------- #

def test_litellm_response_model_format_error_persists_response_with_model_dump() -> None:
    from minisweagent.models.litellm_response_model import LitellmResponseModel

    response = MagicMock()
    response.output = [
        {"type": "function_call", "call_id": "call_xyz", "name": "unknown_tool", "arguments": "{}"}
    ]
    serialized = {"id": "resp_2", "output": response.output}
    response.model_dump.return_value = serialized

    model = LitellmResponseModel(model_name="test/model")

    with patch.object(LitellmResponseModel, "_query", return_value=response), \
         patch.object(LitellmResponseModel, "_calculate_cost", return_value={"cost": 0.0}):
        with pytest.raises(FormatError) as excinfo:
            model.query([{"role": "user", "content": "hi"}])

    extra = excinfo.value.messages[0]["extra"]
    assert "response" in extra
    assert extra["response"] == serialized
    json.dumps(extra["response"])  # JSON-serialisable


def test_litellm_response_model_format_error_persists_response_plain_dict_fallback() -> None:
    """When the response object lacks model_dump, dict(response) is used."""
    from minisweagent.models.litellm_response_model import LitellmResponseModel

    response = _bad_response_api_dict()  # plain dict, no model_dump
    model = LitellmResponseModel(model_name="test/model")

    with patch.object(LitellmResponseModel, "_query", return_value=response), \
         patch.object(LitellmResponseModel, "_calculate_cost", return_value={"cost": 0.0}):
        with pytest.raises(FormatError) as excinfo:
            model.query([{"role": "user", "content": "hi"}])

    extra = excinfo.value.messages[0]["extra"]
    assert "response" in extra
    assert extra["response"] == response


# --------------------------------------------------------------------------- #
# openrouter_model.OpenRouterModel
# --------------------------------------------------------------------------- #

def test_openrouter_model_format_error_persists_response() -> None:
    from minisweagent.models.openrouter_model import OpenRouterModel

    response = _bad_chat_completion_dict()
    model = OpenRouterModel(model_name="test/model")

    with patch.object(OpenRouterModel, "_query", return_value=response), \
         patch.object(OpenRouterModel, "_calculate_cost", return_value={"cost": 0.0}):
        with pytest.raises(FormatError) as excinfo:
            model.query([{"role": "user", "content": "hi"}])

    extra = excinfo.value.messages[0]["extra"]
    assert "response" in extra
    assert extra["response"] == response  # plain dict round-trip


# --------------------------------------------------------------------------- #
# openrouter_response_model.OpenRouterResponseModel
# --------------------------------------------------------------------------- #

def test_openrouter_response_model_format_error_persists_response() -> None:
    from minisweagent.models.openrouter_response_model import OpenRouterResponseModel

    response = _bad_response_api_dict()
    model = OpenRouterResponseModel(model_name="test/model")

    with patch.object(OpenRouterResponseModel, "_query", return_value=response), \
         patch.object(OpenRouterResponseModel, "_calculate_cost", return_value={"cost": 0.0}):
        with pytest.raises(FormatError) as excinfo:
            model.query([{"role": "user", "content": "hi"}])

    extra = excinfo.value.messages[0]["extra"]
    assert "response" in extra
    assert extra["response"] == response


# --------------------------------------------------------------------------- #
# portkey_model.PortkeyModel
# --------------------------------------------------------------------------- #

def test_portkey_model_format_error_persists_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORTKEY_API_KEY", "test-key")
    from minisweagent.models.portkey_model import PortkeyModel

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.tool_calls = [_bad_tool_call_mock()]
    serialized = {"id": "resp_1", "choices": [{"message": {"tool_calls": [{"function": {"name": "unknown_tool"}}]}}]}
    response.model_dump.return_value = serialized

    with patch("minisweagent.models.portkey_model.Portkey"):
        model = PortkeyModel(model_name="gpt-4o")

    with patch.object(PortkeyModel, "_query", return_value=response), \
         patch.object(PortkeyModel, "_calculate_cost", return_value={"cost": 0.0}):
        with pytest.raises(FormatError) as excinfo:
            model.query([{"role": "user", "content": "hi"}])

    extra = excinfo.value.messages[0]["extra"]
    assert "response" in extra
    assert extra["response"], "response payload empty — fix not applied"
    json.dumps(extra["response"])  # JSON-serialisable
    response.model_dump.assert_any_call(mode="json")


# --------------------------------------------------------------------------- #
# portkey_response_model.PortkeyResponseAPIModel
# --------------------------------------------------------------------------- #

def test_portkey_response_model_format_error_persists_response_with_model_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PORTKEY_API_KEY", "test-key")
    from minisweagent.models.portkey_response_model import PortkeyResponseAPIModel

    response = MagicMock()
    response.output = [
        {"type": "function_call", "call_id": "call_xyz", "name": "unknown_tool", "arguments": "{}"}
    ]
    serialized = {"id": "resp_2", "output": response.output}
    response.model_dump.return_value = serialized

    with patch("minisweagent.models.portkey_response_model.Portkey"):
        model = PortkeyResponseAPIModel(model_name="gpt-4o")

    with patch.object(PortkeyResponseAPIModel, "_query", return_value=response), \
         patch.object(PortkeyResponseAPIModel, "_calculate_cost", return_value={"cost": 0.0}):
        with pytest.raises(FormatError) as excinfo:
            model.query([{"role": "user", "content": "hi"}])

    extra = excinfo.value.messages[0]["extra"]
    assert "response" in extra
    assert extra["response"] == serialized
    json.dumps(extra["response"])  # JSON-serialisable


# --------------------------------------------------------------------------- #
# requesty_model.RequestyModel
# --------------------------------------------------------------------------- #

def test_requesty_model_format_error_persists_response() -> None:
    from minisweagent.models.requesty_model import RequestyModel

    response = _bad_chat_completion_dict()
    model = RequestyModel(model_name="test/model")

    with patch.object(RequestyModel, "_query", return_value=response), \
         patch.object(RequestyModel, "_calculate_cost", return_value={"cost": 0.01}):
        with pytest.raises(FormatError) as excinfo:
            model.query([{"role": "user", "content": "hi"}])

    extra = excinfo.value.messages[0]["extra"]
    assert "response" in extra
    assert extra["response"] == response


# --------------------------------------------------------------------------- #
# Trajectory log round-trip — the persisted response must JSON-serialise
# --------------------------------------------------------------------------- #

def test_persisted_response_round_trips_through_json() -> None:
    """The whole point of the fix is that the trajectory log can dump the
    payload. If model_dump(mode='json') is missing, datetimes/Decimals leak
    through and json.dumps raises TypeError."""
    from minisweagent.models.litellm_model import LitellmModel

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.tool_calls = [_bad_tool_call_mock()]
    response.model_dump.return_value = {"id": "abc", "nested": {"k": "v"}}

    model = LitellmModel(model_name="test/model")

    with patch.object(LitellmModel, "_query", return_value=response), \
         patch.object(LitellmModel, "_calculate_cost", return_value={"cost": 0.0}):
        with pytest.raises(FormatError) as excinfo:
            model.query([{"role": "user", "content": "hi"}])

    # Simulate trajectory log serialisation: dump the whole exception messages.
    payload = excinfo.value.messages[0]
    serialised = json.dumps(payload)
    assert '"response"' in serialised
    reloaded = json.loads(serialised)
    assert reloaded["extra"]["response"] == {"id": "abc", "nested": {"k": "v"}}


# --------------------------------------------------------------------------- #
# Contract — FormatError must still propagate (not be swallowed)
# --------------------------------------------------------------------------- #

def test_format_error_still_propagates_after_persisting_response() -> None:
    """The except block must re-raise. If a future change accidentally swallows
    the FormatError (e.g. forgets `raise`), this test catches it."""
    from minisweagent.models.openrouter_model import OpenRouterModel

    response = _bad_chat_completion_dict()
    model = OpenRouterModel(model_name="test/model")

    with patch.object(OpenRouterModel, "_query", return_value=response), \
         patch.object(OpenRouterModel, "_calculate_cost", return_value={"cost": 0.0}):
        # If the handler silently swallowed FormatError, pytest.raises would fail.
        with pytest.raises(FormatError):
            model.query([{"role": "user", "content": "hi"}])


# --------------------------------------------------------------------------- #
# Text-based models — fix applied via inheritance from parent query()
# --------------------------------------------------------------------------- #

def test_litellm_textbased_model_format_error_persists_response() -> None:
    """LitellmTextbasedModel overrides _parse_actions but not query(); the fix
    must reach the parse_regex_actions code path via inherited query()."""
    from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

    response = MagicMock()
    response.choices = [MagicMock()]
    # Provide zero actions in content — triggers FormatError in parse_regex_actions.
    response.choices[0].message.content = "no backtick block here"
    serialized = {"id": "resp_txt", "choices": [{"message": {"content": "no backtick block here"}}]}
    response.model_dump.return_value = serialized

    model = LitellmTextbasedModel(model_name="test/model")

    with patch.object(LitellmTextbasedModel, "_query", return_value=response), \
         patch.object(LitellmTextbasedModel, "_calculate_cost", return_value={"cost": 0.0}):
        with pytest.raises(FormatError) as excinfo:
            model.query([{"role": "user", "content": "hi"}])

    extra = excinfo.value.messages[0]["extra"]
    assert "response" in extra, "response key missing from FormatError extra"
    assert extra["response"] == serialized
    json.dumps(extra["response"])  # JSON-serialisable
    response.model_dump.assert_any_call(mode="json")


def test_openrouter_textbased_model_format_error_persists_response() -> None:
    """OpenRouterTextbasedModel overrides _parse_actions but not query(); the fix
    must reach the parse_regex_actions code path via inherited query()."""
    from minisweagent.models.openrouter_textbased_model import OpenRouterTextbasedModel

    # OpenRouterTextbasedModel uses plain dict responses (dict from response.json()).
    response = {
        "id": "resp_txt",
        "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": "no backtick block here", "tool_calls": None}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.01},
    }

    model = OpenRouterTextbasedModel(model_name="test/model")

    with patch.object(OpenRouterTextbasedModel, "_query", return_value=response), \
         patch.object(OpenRouterTextbasedModel, "_calculate_cost", return_value={"cost": 0.01}):
        with pytest.raises(FormatError) as excinfo:
            model.query([{"role": "user", "content": "hi"}])

    extra = excinfo.value.messages[0]["extra"]
    assert "response" in extra, "response key missing from FormatError extra"
    # OpenRouter/textbased stores dict(response) — a shallow copy of the plain dict.
    assert extra["response"] == response


# --------------------------------------------------------------------------- #
# ATK-01: model_dump failure inside except block must not swallow FormatError
# --------------------------------------------------------------------------- #

def test_format_error_not_swallowed_when_model_dump_raises() -> None:
    """If response.model_dump(mode='json') raises (e.g. serialization error),
    the original FormatError must still propagate — not an opaque secondary error."""
    from minisweagent.models.litellm_model import LitellmModel

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.tool_calls = [_bad_tool_call_mock()]
    # Simulate model_dump raising regardless of kwargs.
    response.model_dump.side_effect = TypeError("unserializable object")

    model = LitellmModel(model_name="test/model")

    with patch.object(LitellmModel, "_query", return_value=response), \
         patch.object(LitellmModel, "_calculate_cost", return_value={"cost": 0.0}):
        # Must raise FormatError, not TypeError from the failed model_dump.
        with pytest.raises(FormatError):
            model.query([{"role": "user", "content": "hi"}])
