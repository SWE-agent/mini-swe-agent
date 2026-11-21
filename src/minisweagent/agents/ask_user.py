"""Interactive agent variant with a structured ask-the-user action.

Same three modes as ``InteractiveAgent`` (human/confirm/yolo), plus the
model can emit an ``ask_user`` action — ``{"question": ..., "options": [...]}``
instead of a shell command — to get mid-run clarification. The question is
rendered through the existing prompt sessions and the answer lands in the
conversation as a normal observation message.

A question that never gets answered (empty input, Ctrl-C/Ctrl-D) produces an
explicit placeholder observation rather than silently attributed text.

To use: select this agent class in the run config, e.g.
``mini --agent-class ask_user`` or ``agent: {agent_class: ask_user}``, and
mention the ``ask_user`` action in the instance/system template so the model
knows it exists (see ``config/mini_ask_user.yaml``).
"""

from minisweagent.agents.interactive import InteractiveAgent, InteractiveAgentConfig
from minisweagent.agents.utils.ask_user import ask_user, is_ask_user
from minisweagent.exceptions import Submitted


class AskUserAgentConfig(InteractiveAgentConfig):
    """Same config shape as ``InteractiveAgentConfig`` — ask-the-user needs no
    additional knobs; ``options`` travel inside the action."""


class AskUserAgent(InteractiveAgent):
    def __init__(self, *args, config_class=AskUserAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)

    def execute_actions(self, message: dict) -> list[dict]:
        # ask_user actions never reach the environment — they prompt the user
        # and become observations directly. Mixed messages (some ask_user,
        # some commands) keep working: each action is answered or executed in
        # order, so question/answer pairs stay aligned.
        actions = message.get("extra", {}).get("actions", [])
        if not any(is_ask_user(a) for a in actions):
            return super().execute_actions(message)

        commands = [a["command"] for a in actions if not is_ask_user(a) and a.get("command")]
        outputs: list[dict] = []
        try:
            self._ask_confirmation_or_interrupt(commands)
            for action in actions:
                if is_ask_user(action):
                    outputs.append(ask_user(action))
                else:
                    outputs.append(self.env.execute(action))
        except Submitted as e:
            self._check_for_new_task_or_submit(e)
        finally:
            result = self.add_messages(
                *self.model.format_observation_messages(message, outputs, self.get_template_vars())
            )
        return result
