import json
import logging
import os

import litellm

from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.exceptions import FormatError, Submitted
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL, parse_toolcall_actions
from minisweagent.models.utils.warpgrep import WARPGREP_TOOL, WarpGrepClient

logger = logging.getLogger("litellm_warpgrep")


class LitellmWarpGrepModel(LitellmModel):
    def __init__(self, *, config_class=LitellmModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)
        api_key = os.getenv("MORPH_API_KEY") or os.getenv("WARPGREP_API_KEY")
        if not api_key:
            msg = "Set MORPH_API_KEY or WARPGREP_API_KEY to use warpgrep."
            raise ValueError(msg)
        self.warpgrep = WarpGrepClient(api_key=api_key)

    def _query(self, messages, **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=[BASH_TOOL, WARPGREP_TOOL],
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def _parse_actions(self, response) -> list[dict]:
        tool_calls = response.choices[0].message.tool_calls or []
        bash_calls = [tc for tc in tool_calls if tc.function.name != "warpgrep"]
        warpgrep_calls = [tc for tc in tool_calls if tc.function.name == "warpgrep"]
        if not tool_calls:
            return parse_toolcall_actions([], format_error_template=self.config.format_error_template)
        actions = (
            parse_toolcall_actions(bash_calls, format_error_template=self.config.format_error_template)
            if bash_calls
            else []
        )
        for tc in warpgrep_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                raise FormatError(
                    {
                        "role": "user",
                        "content": f"Error parsing warpgrep arguments: {e}",
                        "extra": {"interrupt_type": "FormatError"},
                    }
                )
            actions.append({"command": args.get("query", ""), "tool_call_id": tc.id, "tool": "warpgrep"})
        return actions


class WarpGrepAgent(InteractiveAgent):
    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        commands = [a["command"] for a in actions if a.get("tool") != "warpgrep"]
        outputs = []
        try:
            if commands:
                self._ask_confirmation_or_interrupt(commands)
            for action in actions:
                if action.get("tool") == "warpgrep":
                    result = self.model.warpgrep.search(os.getcwd(), action["command"])
                    outputs.append({"output": result, "returncode": 0, "exception_info": ""})
                else:
                    outputs.append(self.env.execute(action))
        except Submitted as e:
            self._check_for_new_task_or_submit(e)
        finally:
            result_msgs = self.add_messages(
                *self.model.format_observation_messages(message, outputs, self.get_template_vars())
            )
        return result_msgs
