from pathlib import Path

import yaml

CONFIG_PATH = Path("study_planner/config/study_planner.yaml")


def test_study_planner_config_is_valid_yaml():
    with CONFIG_PATH.open() as file:
        config = yaml.safe_load(file)

    assert "agent" in config
    assert "system_template" in config["agent"]
    assert "instance_template" in config["agent"]


def test_prompt_defines_schedule_format():
    with CONFIG_PATH.open() as file:
        config = yaml.safe_load(file)

    prompt = config["agent"]["instance_template"]

    assert "task_id" in prompt
    assert "start" in prompt
    assert "end" in prompt


def test_prompt_defines_validation_and_termination():
    with CONFIG_PATH.open() as file:
        config = yaml.safe_load(file)

    prompt = config["agent"]["instance_template"].lower()

    assert "deadline validation" in prompt
    assert "availability validation" in prompt
    assert "task completion validation" in prompt
    assert "infeasible" in prompt
