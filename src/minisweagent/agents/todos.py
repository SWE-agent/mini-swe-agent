"""Interactive agent that isolates todo state per run via a unique file path."""

import uuid

from minisweagent.agents.interactive import InteractiveAgent, InteractiveAgentConfig


class TodosAgent(InteractiveAgent):
    """Same as InteractiveAgent, but injects a per-session ``todo_path`` for prompts."""

    def __init__(self, *args, config_class=InteractiveAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)

    def run(self, task: str = "", **kwargs) -> dict:
        kwargs.setdefault("todo_session_id", uuid.uuid4().hex[:12])
        kwargs.setdefault("todo_path", f".mini_todos/{kwargs['todo_session_id']}.json")
        return super().run(task, **kwargs)
