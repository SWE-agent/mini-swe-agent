import logging
from pathlib import Path

import pytest
import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import FormatError
from minisweagent.models.test_models import DeterministicModel, make_output

AGENT_CONFIG = yaml.safe_load(Path("src/minisweagent/config/default.yaml").read_text())["agent"]
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho done"


def boundary_logs(caplog) -> list[tuple[str, tuple]]:
    """Tag and the first two args of every boundary log record (durations vary)."""
    return [
        (r.msg.split("]")[0] + "]", r.args[:2])
        for r in caplog.records
        if r.name == "agent" and isinstance(r.msg, str) and r.msg.startswith(("[LLM_", "[TOOL_EXECUTION_"))
    ]


@pytest.mark.parametrize(
    ("agent_class", "extra_config"),
    [(DefaultAgent, {}), (InteractiveAgent, {"mode": "yolo", "confirm_exit": False})],
)
def test_llm_and_tool_boundaries_are_logged(caplog, agent_class, extra_config):
    caplog.set_level(logging.DEBUG, logger="agent")
    model = DeterministicModel(
        outputs=[make_output("look", [{"command": "echo hi"}]), make_output("submit", [{"command": SUBMIT}])]
    )
    agent_class(model, LocalEnvironment(), **AGENT_CONFIG, **extra_config).run("Say hi")
    assert boundary_logs(caplog) == [
        ("[LLM_QUERY]", (1, "deterministic")),
        ("[LLM_RESPONSE]", (1, [{"command": "echo hi"}])),
        ("[TOOL_EXECUTION_START]", ("bash", {"command": "echo hi"})),
        ("[TOOL_EXECUTION_END]", ("bash", 0)),
        ("[LLM_QUERY]", (2, "deterministic")),
        ("[LLM_RESPONSE]", (2, [{"command": SUBMIT}])),
        ("[TOOL_EXECUTION_START]", ("bash", {"command": SUBMIT})),
        ("[TOOL_EXECUTION_ERROR]", ("bash", "Submitted")),
    ]


class _FormatErrorModel(DeterministicModel):
    """Raises FormatError (as the real models do on an unparsable reply) for outputs marked {"_format_error": True}."""

    def query(self, messages, **kwargs):
        if self.config.outputs[self.current_index + 1].get("_format_error"):
            self.current_index += 1
            raise FormatError(
                {"role": "user", "content": "Use the bash tool.", "extra": {"interrupt_type": "FormatError"}}
            )
        return super().query(messages, **kwargs)


def test_format_error_is_logged_as_llm_error(caplog):
    caplog.set_level(logging.DEBUG, logger="agent")
    model = _FormatErrorModel(outputs=[{"_format_error": True}, make_output("submit", [{"command": SUBMIT}])])
    assert DefaultAgent(model, LocalEnvironment(), **AGENT_CONFIG).run("Say hi")["exit_status"] == "Submitted"
    assert boundary_logs(caplog)[:3] == [
        ("[LLM_QUERY]", (1, "deterministic")),
        ("[LLM_ERROR]", (1, "FormatError")),
        ("[LLM_QUERY]", (2, "deterministic")),
    ]
