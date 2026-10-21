import json

import pytest
import typer
import yaml
from typer.testing import CliRunner

from minisweagent.models.litellm_model import LitellmModel
from minisweagent.run.extra.resumable import app, report_result
from tests.agents.resume_worker import continuation
from tests.agents.test_resumable import agent_for


def test_start_rejects_unsupported_backends_and_existing_run_directory(tmp_path):
    config = tmp_path / "config.yaml"
    settings = {
        "agent": {"system_template": "System", "instance_template": "{{task}}"},
        "model": {"model_name": "openai/resume-test", "model_class": "unsupported"},
    }
    config.write_text(yaml.safe_dump(settings))
    args = ["start", "--task", "Test", "--run-dir", str(tmp_path / "run"), "--cwd", str(tmp_path), "-c", str(config)]
    assert "supports litellm" in str(CliRunner().invoke(app, args).exception)
    assert not (tmp_path / "run").exists()
    settings["model"]["model_class"] = "litellm"
    settings["environment"] = {"environment_class": "docker"}
    config.write_text(yaml.safe_dump(settings))
    assert "local environment" in str(CliRunner().invoke(app, args).exception)
    settings["environment"]["environment_class"] = "local"
    settings["agent"].update(agent_class="interactive", mode="confirm")
    config.write_text(yaml.safe_dump(settings))
    (tmp_path / "run").mkdir()
    result = CliRunner().invoke(app, args)
    assert isinstance(result.exception, FileExistsError)
    assert "ignores agent_class/mode; commands run without interactive confirmation" in result.output
    assert not list((tmp_path / "run").iterdir())


def test_resume_checks_cumulative_cli_budget_before_claiming(tmp_path):
    agent = agent_for(tmp_path, step_limit=1)
    agent.run("Append A and B", marker="kept")
    data = json.loads(agent.config.output_path.read_text())
    data["info"]["config"].update(LitellmModel(model_name="openai/resume-test").serialize()["info"]["config"])
    agent.config.output_path.write_text(json.dumps(data))
    original = agent.config.output_path.read_bytes()
    result = CliRunner().invoke(app, ["resume", "--run-dir", str(agent.run_dir), "--step-limit", "1"])
    assert isinstance(result.exception, ValueError)
    assert "step_limit is exhausted" in str(result.exception)
    assert not list(agent.run_dir.glob("*.claimed"))
    assert agent.config.output_path.read_bytes() == original
    assert (tmp_path / "work/effects.txt").read_text() == "A\n"


def test_reports_pause_and_failed_continuation(tmp_path, capsys):
    agent = agent_for(tmp_path, commands=["echo first", "sleep 2"], pause_after=1)
    agent.env.config.timeout = 1
    report_result(agent.run("Pause then time out", marker="kept"), agent.run_dir)
    assert "Exit status: Paused" in capsys.readouterr().out
    with pytest.raises(typer.Exit) as error:
        report_result(continuation(agent.run_dir).resume(), agent.run_dir)
    assert error.value.exit_code == 1
    output = capsys.readouterr().out
    assert "Exit status: EnvironmentError" in output
    assert f"Saved run to {agent.run_dir}" in output
