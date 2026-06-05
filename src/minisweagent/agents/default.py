"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

import json
import logging
import time
import traceback
from collections.abc import Iterable
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import AgentObserver, AgentObserverEvent, Environment, Model, __version__
from minisweagent.exceptions import InterruptAgentFlow, LimitsExceeded, TimeExceeded
from minisweagent.utils.serialize import recursive_merge


class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    wall_time_limit_seconds: int = 0
    """Stop agent after this many seconds of wall-clock time. 0 means no limit."""
    output_path: Path | None = None
    """Save the trajectory to this path."""


class DefaultAgent:
    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: type = AgentConfig,
        observer: AgentObserver | None = None,
        observers: Iterable[AgentObserver] | None = None,
        **kwargs,
    ):
        """See the `AgentConfig` class for permitted keyword arguments.

        `observer` is a convenience shortcut for passing a single observer.
        Use `observers` when attaching multiple observers. If both are
        supplied, `observer` is notified first.
        """
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        observer_list = []
        if observer is not None:
            observer_list.append(observer)
        if observers is not None:
            observer_list.extend(observers)
        self.observers = tuple(observer_list)
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self._start_time = time.time()

    def _notify(self, event_name: AgentObserverEvent, **payload) -> None:
        """Notify an optional observer without letting telemetry break agent execution."""
        for observer in self.observers:
            handler = getattr(observer, event_name.value, None)
            if handler is None:
                continue
            try:
                handler(agent=self, **payload)
            except Exception:
                self.logger.exception("Agent observer failed while handling %s", event_name.value)

    @staticmethod
    def _get_tool_call_id(action: dict) -> str | None:
        for key in ("id", "tool_call_id", "call_id", "tool_use_id"):
            if action.get(key):
                return str(action[key])
        return None

    @staticmethod
    def _tool_call_id(tool_call: dict) -> str | None:
        return tool_call.get("id") or tool_call.get("call_id")

    @staticmethod
    def _tool_call_name(tool_call: dict) -> str | None:
        function = tool_call.get("function")
        if isinstance(function, dict) and function.get("name"):
            return str(function["name"])
        if tool_call.get("name"):
            return str(tool_call["name"])
        return None

    @staticmethod
    def _tool_call_arguments(tool_call: dict) -> dict | None:
        function = tool_call.get("function")
        arguments = function.get("arguments") if isinstance(function, dict) else tool_call.get("arguments")
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return {"raw": arguments}
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        return None

    @staticmethod
    def _message_tool_calls(message: dict) -> list[dict]:
        tool_calls = []
        for tool_call in message.get("tool_calls", []) or []:
            if isinstance(tool_call, dict):
                tool_calls.append(tool_call)
        for item in message.get("output", []) or []:
            if isinstance(item, dict) and item.get("type") == "function_call":
                tool_calls.append(item)
        return tool_calls

    def _observer_action(self, action: dict, message: dict) -> dict:
        """Return a stable action payload for observers without changing execution input."""
        observed_action = dict(action)
        tool_call_id = self._get_tool_call_id(observed_action)
        tool_call = None
        for candidate in self._message_tool_calls(message):
            if tool_call_id and self._tool_call_id(candidate) == tool_call_id:
                tool_call = candidate
                break

        if tool_call_id:
            observed_action.setdefault("id", tool_call_id)
            observed_action.setdefault("tool_call_id", tool_call_id)
        if tool_call is not None:
            if name := self._tool_call_name(tool_call):
                observed_action.setdefault("name", name)
            if arguments := self._tool_call_arguments(tool_call):
                observed_action.setdefault("arguments", arguments)
            observed_action.setdefault("kind", "tool_call")

        if "arguments" not in observed_action and "command" in observed_action:
            observed_action["arguments"] = {"command": observed_action["command"]}
        observed_action.setdefault("name", "action.exec")
        observed_action.setdefault("kind", "action")
        return observed_action

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "elapsed_seconds": int(time.time() - self._start_time),
            },
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        self._notify(AgentObserverEvent.RUN_START, task=task, kwargs=kwargs, messages=list(self.messages))
        while True:
            try:
                self.step()
            except InterruptAgentFlow as e:
                messages = self.add_messages(*e.messages)
                self._notify(AgentObserverEvent.INTERRUPT, exception=e, messages=messages)
                if messages and messages[-1].get("extra", {}).get("exit_status") == "Submitted":
                    self._notify(
                        AgentObserverEvent.SUBMIT,
                        exception=e,
                        messages=messages,
                        submission=messages[-1].get("extra", {}).get("submission", ""),
                    )
            except Exception as e:
                messages = self.handle_uncaught_exception(e)
                self._notify(AgentObserverEvent.ERROR, exception=e, messages=messages)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        result = self.messages[-1].get("extra", {})
        self._notify(AgentObserverEvent.RUN_END, result=result, messages=list(self.messages))
        return result

    def step(self) -> list[dict]:
        """Query the LM, execute actions."""
        step_index = self.n_calls + 1
        self._notify(AgentObserverEvent.STEP_START, step_index=step_index, messages=list(self.messages))
        try:
            message = self.query()
            observations = self.execute_actions(message)
        except Exception as e:
            self._notify(AgentObserverEvent.STEP_END, step_index=step_index, exception=e, messages=list(self.messages))
            raise
        self._notify(
            AgentObserverEvent.STEP_END,
            step_index=step_index,
            message=message,
            observations=observations,
            messages=list(self.messages),
        )
        return observations

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        self.n_calls += 1
        call_index = self.n_calls
        self._notify(AgentObserverEvent.MODEL_START, call_index=call_index, messages=list(self.messages))
        try:
            message = self.model.query(self.messages)
        except Exception as e:
            self._notify(AgentObserverEvent.MODEL_END, call_index=call_index, exception=e, messages=list(self.messages))
            raise
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        self._notify(
            AgentObserverEvent.MODEL_END,
            call_index=call_index,
            message=message,
            cost=message.get("extra", {}).get("cost", 0.0),
            total_cost=self.cost,
            messages=list(self.messages),
        )
        return message

    def _execute_action(self, message: dict, action_index: int, action: dict) -> dict:
        """Execute one action and notify observers around the environment call."""
        observed_action = self._observer_action(action, message) if isinstance(action, dict) else action
        self._notify(
            AgentObserverEvent.ACTION_START,
            action_index=action_index,
            action=observed_action,
            raw_action=action,
            message=message,
        )
        try:
            output = self.env.execute(action)
        except Exception as e:
            self._notify(
                AgentObserverEvent.ACTION_END,
                action_index=action_index,
                action=observed_action,
                raw_action=action,
                exception=e,
                message=message,
            )
            raise
        self._notify(
            AgentObserverEvent.ACTION_END,
            action_index=action_index,
            action=observed_action,
            raw_action=action,
            output=output,
            message=message,
        )
        return output

    def _add_observation_messages(self, message: dict, outputs: list[dict]) -> list[dict]:
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        outputs = [
            self._execute_action(message, action_index, action)
            for action_index, action in enumerate(message.get("extra", {}).get("actions", []))
        ]
        return self._add_observation_messages(message, outputs)

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data
