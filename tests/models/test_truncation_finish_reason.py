"""finish_reason is threaded into format_error_template across the parse utilities, so a config can
report a max_tokens truncation as "cut off" instead of a misleading format error."""

import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.actions_text import parse_regex_actions
from minisweagent.models.utils.actions_toolcall_response import (
    finish_reason_from_responses_api,
    parse_toolcall_actions_response,
)

_TEMPLATE = "{% if finish_reason == 'length' %}cut off{% else %}{{ error }}{% endif %}"


class TestFinishReasonFromResponsesApi:
    @pytest.mark.parametrize("response", [{}, {"status": "completed"}, None])
    def test_non_truncation_returns_status(self, response):
        # completed / unknown -> not "length", so templates keep the normal error
        assert finish_reason_from_responses_api(response) != "length"

    def test_incomplete_max_output_tokens_maps_to_length_dict(self):
        response = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
        assert finish_reason_from_responses_api(response) == "length"

    def test_incomplete_max_output_tokens_maps_to_length_obj(self):
        class _Resp:
            status = "incomplete"
            incomplete_details = {"reason": "max_output_tokens"}

        assert finish_reason_from_responses_api(_Resp()) == "length"

    def test_incomplete_other_reason_is_not_length(self):
        response = {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}}
        assert finish_reason_from_responses_api(response) != "length"


class TestRegexActionsTemplateKwargs:
    def test_finish_reason_reported_on_zero_actions(self):
        # a truncated text response yields zero parsed actions
        with pytest.raises(FormatError) as exc:
            parse_regex_actions(
                "no action here",
                action_regex=r"```bash\n(.*?)\n```",
                format_error_template=_TEMPLATE,
                template_kwargs={"finish_reason": "length"},
            )
        assert exc.value.messages[0]["content"] == "cut off"

    def test_without_template_kwargs_still_works(self):
        with pytest.raises(FormatError) as exc:
            parse_regex_actions("nope", action_regex=r"```bash\n(.*?)\n```", format_error_template="{{ error }}")
        assert "found 0" in exc.value.messages[0]["content"]


class TestResponseActionsTemplateKwargs:
    def test_finish_reason_reported_on_no_tool_calls(self):
        with pytest.raises(FormatError) as exc:
            parse_toolcall_actions_response(
                [], format_error_template=_TEMPLATE, template_kwargs={"finish_reason": "length"}
            )
        assert exc.value.messages[0]["content"][0]["text"] == "cut off"
