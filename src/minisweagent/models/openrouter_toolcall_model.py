from minisweagent.models.openrouter_model import OpenRouterModel, OpenRouterModelConfig
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)


class OpenRouterToolcallModelConfig(OpenRouterModelConfig):
    format_error_template: str = "{{ error }}"


class OpenRouterToolcallModel(OpenRouterModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = OpenRouterToolcallModelConfig(**kwargs)

    def _query(self, messages: list[dict[str, str]], **kwargs):
        return super()._query(messages, tools=[BASH_TOOL], **kwargs)

    def _parse_actions(self, response: dict) -> list[dict]:
        """Parse tool calls from the response. Raises FormatError if unknown tool."""
        tool_calls = response["choices"][0]["message"].get("tool_calls") or []
        # Convert dict tool_calls to objects with attributes for compatibility
        tool_calls = [_DictToObj(tc) for tc in tool_calls]
        return parse_toolcall_actions(tool_calls, format_error_template=self.config.format_error_template)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )


class _DictToObj:
    """Simple wrapper to convert dict to object with attribute access."""

    def __init__(self, d: dict):
        self._d = d
        self.id = d.get("id")
        self.function = _DictToObj(d.get("function", {})) if "function" in d else None
        self.name = d.get("name")
        self.arguments = d.get("arguments")
