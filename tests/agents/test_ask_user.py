"""Tests for the AskUserAgent ask-the-user action."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from minisweagent.agents.ask_user import AskUserAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output


@contextmanager
def _patch_prompts(answers):
    it = iter(answers)

    def se(*args, **kwargs):
        return next(it)

    with patch("minisweagent.agents.utils.prompt_user.prompt_session.prompt", side_effect=se):
        with patch("minisweagent.agents.utils.prompt_user._multiline_prompt_session.prompt", side_effect=se):
            yield


@pytest.fixture
def config():
    return yaml.safe_load(Path("src/minisweagent/config/mini_ask_user.yaml").read_text(encoding="utf-8"))["agent"]


def _agent(outputs: list[dict], config, **kwargs) -> AskUserAgent:
    return AskUserAgent(model=DeterministicModel(outputs=outputs), env=LocalEnvironment(), **{**config, **kwargs})


def test_ask_user_answer_injected_as_observation(config):
    """A question with a free-form answer lands in the conversation as a
    normal observation with the answer text."""
    with _patch_prompts(["tenant A"]):
        agent = _agent(
            [
                make_output(
                    "I need to know the scope first",
                    [{"question": "Apply the migration to all tenants or just tenant A?"}],
                ),
                make_output("Done", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo scoped"}]),
            ],
            config,
            mode="yolo",
            confirm_exit=False,
        )
        info = agent.run("Scope the migration")

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "scoped\n"
    observations = [
        msg
        for msg in agent.messages
        if msg.get("extra", {}).get("ask_user") and "tenant A" in msg["extra"]["raw_output"]
    ]
    assert len(observations) == 1
    assert observations[0]["extra"]["question"] == "Apply the migration to all tenants or just tenant A?"


def test_ask_user_numeric_option_picked(config):
    """Answering with a number resolves to that option's text."""
    with _patch_prompts(["2"]):
        agent = _agent(
            [
                make_output(
                    "Which refactor?",
                    [{"question": "Pick the refactor", "options": ["rename only", "rename + move"]}],
                ),
                make_output("ok", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo picked"}]),
            ],
            config,
            mode="yolo",
            confirm_exit=False,
        )
        info = agent.run("Refactor")

    assert info["exit_status"] == "Submitted"
    observations = [msg for msg in agent.messages if msg.get("extra", {}).get("ask_user")]
    assert observations[0]["extra"]["raw_output"] == "rename + move"
    assert observations[0]["extra"]["options"] == ["rename only", "rename + move"]


def test_ask_user_empty_answer_becomes_placeholder(config):
    """An empty answer must produce an explicit placeholder, never an
    attributed user utterance."""
    with _patch_prompts([""]):
        agent = _agent(
            [
                make_output("q", [{"question": "Which branch should I target?"}]),
                make_output("ok", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo done"}]),
            ],
            config,
            mode="yolo",
            confirm_exit=False,
        )
        info = agent.run("Pick a branch")

    assert info["exit_status"] == "Submitted"
    observations = [msg for msg in agent.messages if msg.get("extra", {}).get("ask_user")]
    assert observations[0]["extra"]["raw_output"] == "(no answer provided)"
    assert observations[0]["extra"]["cancelled"] is False


def test_ask_user_cancelled_becomes_placeholder(config):
    """Ctrl-C/Ctrl-D during the prompt yields an explicit cancelled
    observation instead of crashing or faking an answer."""

    def raise_eof(*a, **k):
        raise EOFError

    with (
        patch("minisweagent.agents.utils.prompt_user.prompt_session.prompt", side_effect=raise_eof),
        patch("minisweagent.agents.utils.prompt_user._multiline_prompt_session.prompt", side_effect=raise_eof),
    ):
        agent = _agent(
            [
                make_output("q", [{"question": "Overwrite the lockfile?"}]),
                make_output("ok", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo kept"}]),
            ],
            config,
            mode="yolo",
            confirm_exit=False,
        )
        info = agent.run("Handle the lockfile")

    assert info["exit_status"] == "Submitted"
    observations = [msg for msg in agent.messages if msg.get("extra", {}).get("ask_user")]
    assert observations[0]["extra"]["raw_output"] == "(cancelled)"
    assert observations[0]["extra"]["cancelled"] is True


def test_ask_user_multi_question_pairing(config):
    """Multiple ask_user actions in one message are asked in order and
    answers pair 1:1 with their questions."""
    with _patch_prompts(["yes", "staging"]):
        agent = _agent(
            [
                make_output(
                    "two questions",
                    [
                        {"question": "Enable feature flag?"},
                        {"question": "Deploy to which env?", "options": ["staging", "prod"]},
                    ],
                ),
                make_output("ok", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo deployed"}]),
            ],
            config,
            mode="yolo",
            confirm_exit=False,
        )
        info = agent.run("Deploy")

    assert info["exit_status"] == "Submitted"
    observations = [msg for msg in agent.messages if msg.get("extra", {}).get("ask_user")]
    assert len(observations) == 2
    assert observations[0]["extra"]["question"] == "Enable feature flag?"
    assert observations[0]["extra"]["raw_output"] == "yes"
    assert observations[1]["extra"]["question"] == "Deploy to which env?"
    assert observations[1]["extra"]["raw_output"] == "staging"


def test_ask_user_mixed_with_command(config):
    """A message mixing an ask_user action and a real command answers the
    question and executes the command 鈥?outputs stay aligned."""
    with _patch_prompts(["green"]):
        agent = _agent(
            [
                make_output(
                    "ask then act",
                    [
                        {"question": "Which colour?"},
                        {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo green it is"},
                    ],
                ),
            ],
            config,
            mode="yolo",
            confirm_exit=False,
        )
        info = agent.run("Set the colour")

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "green it is\n"
    ask_obs = [msg for msg in agent.messages if msg.get("extra", {}).get("ask_user")]
    assert ask_obs[0]["extra"]["raw_output"] == "green"


def test_ask_user_registered_agent_class():
    from minisweagent.agents import get_agent_class

    assert get_agent_class("ask_user") is AskUserAgent
