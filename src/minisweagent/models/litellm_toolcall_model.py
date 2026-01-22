from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)


class LitellmToolcallModelConfig(LitellmModelConfig):
    format_error_template: str = "{{ error }}"


class LitellmToolcallModel(LitellmModel):
    def __init__(self, **kwargs):
        super().__init__(config_class=LitellmToolcallModelConfig, **kwargs)

    def _query(self, messages: list[dict[str, str]], **kwargs):
        return super()._query(messages, tools=[BASH_TOOL], **kwargs)

    def _parse_actions(self, response) -> list[dict]:
        """Parse tool calls from the response. Raises FormatError if unknown tool."""
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_toolcall_actions(tool_calls, format_error_template=self.config.format_error_template)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions, outputs, self.config.observation_template, template_vars, self.config.multimodal_regex
        )
