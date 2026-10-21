# ruff: noqa: PT006
# AGENTS.md requires tuple parameter names, including single-parameter cases.

import json
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from minisweagent.agents.extra.resumable import ResumableAgent
from minisweagent.agents.utils.checkpoint import load_checkpoint
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import (
    DeterministicModel,
    DeterministicToolcallModel,
    make_output,
    make_toolcall_output,
)
from tests.agents.resume_worker import continuation
from tests.agents.test_default import _BilledFormatErrorModel


def agent_for(
    tmp_path: Path, *, toolcalls: bool = False, commands: list[str] | None = None, **kwargs
) -> ResumableAgent:
    workspace = tmp_path / "work"
    workspace.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, command in enumerate(
        commands
        or [
            "printf 'A\\n' >> effects.txt",
            "printf 'B\\n' >> effects.txt",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ]
    ):
        if toolcalls:
            outputs.append(
                make_toolcall_output(
                    command,
                    [
                        {
                            "id": str(index),
                            "type": "function",
                            "function": {"name": "bash", "arguments": json.dumps({"command": command})},
                        }
                    ],
                    [{"command": command, "tool_call_id": str(index)}],
                )
            )
        else:
            outputs.append(make_output(command, [{"command": command}]))
    return ResumableAgent(
        (DeterministicToolcallModel if toolcalls else DeterministicModel)(
            outputs=outputs, observation_template="{{ marker }} calls={{ n_model_calls }} {{ output.output }}"
        ),
        LocalEnvironment(cwd=str(workspace)),
        run_dir=tmp_path / "run",
        **({"system_template": "System {{ marker }}", "instance_template": "{{ task }}", "cost_limit": 0} | kwargs),
    )


def worker(path: Path, mode: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "tests.agents.resume_worker", str(path), mode],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.mark.parametrize(("toolcalls",), [(False,), (True,)])
def test_resume_preserves_history_budgets_and_source(tmp_path, toolcalls):
    agent = agent_for(tmp_path, toolcalls=toolcalls, step_limit=1)
    assert agent.run("Append A and B", marker="kept")["exit_status"] == "LimitsExceeded"
    original = agent.config.output_path.read_bytes()
    paused, state = load_checkpoint(agent.run_dir)
    assert (agent.cost, agent.n_calls, (tmp_path / "work/effects.txt").read_text()) == (1, 1, "A\n")
    resumed = continuation(agent.run_dir, step_limit=3)
    assert resumed.resume()["exit_status"] == "Submitted"
    assert (resumed.cost, resumed.n_calls, (tmp_path / "work/effects.txt").read_text()) == (3, 3, "A\nB\n")
    assert resumed.messages[:4] == paused["messages"][:-1]
    assert resumed.messages[5]["content"] == "kept calls=2 "
    assert resumed.extra_template_vars == {"task": "Append A and B", "marker": "kept"}
    assert agent.config.output_path.read_bytes() == original
    assert resumed.parent_checkpoint_id == state.checkpoint_id != resumed.checkpoint_id
    assert (agent.run_dir / f"{state.checkpoint_id}.claimed").is_dir()
    continuous = agent_for(tmp_path / "continuous", toolcalls=toolcalls)
    assert continuous.run("Append A and B", marker="kept")["exit_status"] == "Submitted"
    assert (continuous.cost, continuous.n_calls) == (resumed.cost, resumed.n_calls)
    assert [(m["role"], m["content"], m.get("tool_call_id")) for m in continuous.messages] == [
        (m["role"], m["content"], m.get("tool_call_id")) for m in resumed.messages
    ]
    assert (tmp_path / "continuous/work/effects.txt").read_text() == "A\nB\n"
    with pytest.raises(ValueError, match="paused checkpoint"):
        continuation(agent.run_dir)


def test_exhausted_time_budget_is_not_claimed(tmp_path):
    agent = agent_for(tmp_path, commands=["sleep 1.1"], wall_time_limit_seconds=1)
    assert agent.run("One step", marker="kept")["exit_status"] == "TimeExceeded"
    _, state = load_checkpoint(agent.run_dir)
    with pytest.raises(ValueError, match="wall_time_limit_seconds is exhausted"):
        continuation(agent.run_dir).resume()
    assert not (agent.run_dir / f"{state.checkpoint_id}.claimed").exists()
    assert len(agent.messages) == 5 and agent.n_calls == 1


def test_all_tool_results_are_saved_before_pausing(tmp_path):
    agent = agent_for(tmp_path, toolcalls=True, pause_after=1)
    first, second, submit = agent.model.config.outputs
    first["extra"]["actions"] += second["extra"]["actions"]
    first["tool_calls"] += second["tool_calls"]
    agent.model.config.outputs = [first, submit]
    assert agent.run("Append A and B", marker="kept")["exit_status"] == "Paused"
    assert [message["tool_call_id"] for message in agent.messages[3:5]] == ["0", "1"]
    assert (tmp_path / "work/effects.txt").read_text() == "A\nB\n"
    resumed = continuation(agent.run_dir)
    assert resumed.resume()["exit_status"] == "Submitted"
    assert (resumed.n_calls, resumed.cost) == (2, 2)
    assert (tmp_path / "work/effects.txt").read_text() == "A\nB\n"


def test_pause_time_is_excluded_and_each_session_has_its_own_pause_count(tmp_path):
    agent = agent_for(tmp_path, pause_after=1)
    assert agent.run("Append A and B", marker="kept")["exit_status"] == "Paused"
    _, first = load_checkpoint(agent.run_dir)
    time.sleep(0.15)
    assert agent.serialize()["info"]["resume"]["elapsed_seconds"] == first.elapsed_seconds
    agent.save(agent.config.output_path)
    assert load_checkpoint(agent.run_dir)[1].elapsed_seconds == first.elapsed_seconds
    started = time.time()
    assert agent.resume()["exit_status"] == "Paused"
    duration = time.time() - started
    _, second = load_checkpoint(agent.run_dir)
    assert first.elapsed_seconds < second.elapsed_seconds <= first.elapsed_seconds + duration + 0.02
    assert (agent.cost, agent.n_calls) == (2, 2)
    assert continuation(agent.run_dir).resume()["exit_status"] == "Submitted"


def test_format_error_cost_and_streak_survive_resume(tmp_path):
    config = {
        "system_template": "System",
        "instance_template": "{{task}}",
        "cost_limit": 1,
        "max_consecutive_format_errors": 2,
    }
    agent = ResumableAgent(_BilledFormatErrorModel(outputs=[]), LocalEnvironment(), run_dir=tmp_path / "run", **config)
    assert agent.run("Bad formatting")["exit_status"] == "LimitsExceeded"
    resumed = ResumableAgent(_BilledFormatErrorModel(outputs=[]), LocalEnvironment(), run_dir=agent.run_dir, **config)
    with pytest.raises(ValueError, match="cost_limit is exhausted"):
        resumed.resume()
    assert not list(agent.run_dir.glob("*.claimed"))
    resumed.config.cost_limit = 3
    assert resumed.resume()["exit_status"] == "RepeatedFormatError"
    assert (resumed.cost, resumed.n_calls, resumed.n_consecutive_format_errors) == (2, 2, 2)
    with pytest.raises(ValueError, match="paused checkpoint"):
        load_checkpoint(agent.run_dir)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("missing_result", "incomplete actions or inconsistent counters"),
        ("wrong_cost", "incomplete actions or inconsistent counters"),
        ("wrong_call_id", "missing or interrupted action results"),
    ],
)
def test_invalid_checkpoint_is_rejected_before_side_effects(tmp_path, change, error):
    agent = agent_for(tmp_path, toolcalls=True, pause_after=1)
    agent.run("Append A and B", marker="kept")
    resumed = continuation(agent.run_dir)
    data = json.loads(agent.config.output_path.read_text())
    if change == "missing_result":
        del data["messages"][-2]
    elif change == "wrong_call_id":
        data["messages"][-2]["tool_call_id"] = "unknown"
    else:
        data["info"]["model_stats"]["instance_cost"] = 0
    agent.config.output_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=error):
        resumed.resume()
    assert not list(agent.run_dir.glob("*.claimed"))
    assert (tmp_path / "work/effects.txt").read_text() == "A\n"


def test_model_change_is_rejected_before_claiming(tmp_path):
    agent = agent_for(tmp_path, pause_after=1)
    agent.run("Append A and B", marker="kept")
    resumed = continuation(agent.run_dir)
    resumed.model.config.model_name = "different"
    with pytest.raises(ValueError, match="configuration must match"):
        resumed.resume()
    assert not list(agent.run_dir.glob("*.claimed"))
    assert (tmp_path / "work/effects.txt").read_text() == "A\n"


@pytest.mark.parametrize(
    ("command",),
    [
        ("printf 'A\\n' >> effects.txt; sleep 2",),
        pytest.param(
            "printf 'A\\n' >> effects.txt; kill -KILL $$",
            marks=pytest.mark.skipif(sys.platform == "win32", reason="Requires SIGKILL"),
        ),
    ],
)
def test_interrupted_commands_are_not_resumable(tmp_path, command):
    agent = agent_for(tmp_path, toolcalls=True, commands=[command, "printf 'B\\n' >> effects.txt"], pause_after=1)
    first, second = agent.model.config.outputs
    first["extra"]["actions"] += second["extra"]["actions"]
    first["tool_calls"] += second["tool_calls"]
    agent.env.config.timeout = 1
    assert agent.run("Interrupted batch", marker="kept")["exit_status"] == "EnvironmentError"
    assert (tmp_path / "work/effects.txt").read_text() == "A\n"
    assert agent.messages[4]["tool_call_id"] == "1"
    assert agent.messages[4]["extra"]["exception_info"] == "action was not executed"
    with pytest.raises(ValueError, match="paused checkpoint"):
        load_checkpoint(agent.run_dir)
    # Also reject interrupted histories mislabeled as pauses by an older producer.
    data = json.loads(agent.config.output_path.read_text())
    data["info"]["resume"]["status"] = "paused"
    data["messages"][-1]["extra"]["exit_status"] = "Paused"
    agent.config.output_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="missing or interrupted action results"):
        load_checkpoint(agent.run_dir)


@pytest.mark.parametrize(
    ("command", "returncode"),
    [
        ("exit 1", 1),
        ("exit 137", 137),
        pytest.param(
            f"{shlex.quote(sys.executable)} -c 'import os, signal; os.kill(os.getpid(), signal.SIGKILL)'; "
            'status=$?; exit "$status"',
            137,
            marks=pytest.mark.skipif(sys.platform == "win32", reason="Requires POSIX shell signal status"),
        ),
    ],
)
def test_positive_shell_status_allows_batch_completion_and_resume(tmp_path, command, returncode):
    agent = agent_for(
        tmp_path,
        toolcalls=True,
        commands=[
            f"printf 'A\\n' >> effects.txt; {command}",
            "printf 'B\\n' >> effects.txt",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ],
        pause_after=1,
    )
    first, second, submit = agent.model.config.outputs
    first["extra"]["actions"] += second["extra"]["actions"]
    first["tool_calls"] += second["tool_calls"]
    agent.model.config.outputs = [first, submit]
    assert agent.run("Observe shell exit status", marker="kept")["exit_status"] == "Paused"
    paused, _ = load_checkpoint(agent.run_dir)
    assert [message["extra"]["returncode"] for message in paused["messages"][3:5]] == [returncode, 0]
    assert not any(message["extra"].get("exception_info") for message in paused["messages"][3:5])
    assert (tmp_path / "work/effects.txt").read_text() == "A\nB\n"
    original = agent.config.output_path.read_bytes()
    resumed = continuation(agent.run_dir)
    assert resumed.resume()["exit_status"] == "Submitted"
    assert (resumed.n_calls, resumed.cost, (tmp_path / "work/effects.txt").read_text()) == (2, 2, "A\nB\n")
    assert agent.config.output_path.read_bytes() == original


@pytest.mark.skipif(sys.platform == "win32", reason="Requires SIGKILL")
@pytest.mark.parametrize(("mode", "effects"), [("kill-claimed", "A\n"), ("kill-observation", "A\nB\n")])
def test_killed_resume_cannot_reuse_its_parent(tmp_path, mode, effects):
    agent = agent_for(tmp_path, step_limit=1)
    agent.run("Append A and B", marker="kept")
    original = agent.config.output_path.read_bytes()
    process = worker(agent.run_dir, mode)
    stdout, stderr = process.communicate(timeout=15)
    assert process.returncode == -signal.SIGKILL, (stdout, stderr)
    assert agent.config.output_path.read_bytes() == original
    assert (tmp_path / "work/effects.txt").read_text() == effects
    if mode == "kill-observation":
        with pytest.raises(ValueError, match="paused checkpoint"):
            load_checkpoint(agent.run_dir)
    # Even selecting the intact parent cannot bypass the durable claim.
    (agent.run_dir / "latest").write_text(agent.checkpoint_id)
    with pytest.raises(FileExistsError):
        continuation(agent.run_dir, step_limit=3).resume()
    assert (tmp_path / "work/effects.txt").read_text() == effects


def test_concurrent_new_processes_continue_only_once(tmp_path):
    agent = agent_for(tmp_path, step_limit=1)
    agent.run("Append A and B", marker="kept")
    processes = [worker(agent.run_dir, "resume") for _ in range(2)]
    outputs = [process.communicate(timeout=15) for process in processes]
    assert sorted(process.returncode for process in processes) == [0, 1], outputs
    assert (tmp_path / "work/effects.txt").read_text() == "A\nB\n"
    data = json.loads((agent.run_dir / f"{(agent.run_dir / 'latest').read_text()}.traj.json").read_text())
    assert data["info"]["model_stats"] == {"instance_cost": 3, "api_calls": 3}
