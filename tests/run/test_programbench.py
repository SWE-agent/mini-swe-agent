import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from minisweagent import package_dir
from minisweagent.exceptions import Submitted
from minisweagent.run.benchmarks.programbench import (
    ProgramBenchAgent,
    copy_submission,
    main,
)
from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent


class _SubmittingModelConfig(BaseModel):
    model_name: str = "submitting_model"


class _SubmittingModel:
    """Test model whose ``query`` immediately raises ``Submitted`` (clean agent exit)."""

    def __init__(self):
        self.cost = 0.0
        self.n_calls = 0
        self.config = _SubmittingModelConfig()

    def query(self, *args, **kwargs):
        self.n_calls += 1
        raise Submitted(
            {"role": "exit", "content": "Submitted", "extra": {"exit_status": "Submitted", "submission": "done"}}
        )

    def format_message(self, **kwargs) -> dict:
        return dict(**kwargs)

    def format_observation_messages(self, message, outputs, template_vars=None) -> list[dict]:
        return [self.format_message(role="user", content=str(o)) for o in outputs]

    def get_template_vars(self, **kwargs) -> dict:
        return self.config.model_dump() | {"n_model_calls": self.n_calls, "model_cost": self.cost}

    def serialize(self) -> dict:
        return {"info": {"model_stats": {"instance_cost": self.cost, "api_calls": self.n_calls}}}


@pytest.fixture
def fake_programbench(monkeypatch):
    """Inject a fake ``programbench`` package exposing ``load_all_instances`` and ``filter_instances``."""
    pb = types.ModuleType("programbench")
    pb_utils = types.ModuleType("programbench.utils")
    pb_load = types.ModuleType("programbench.utils.load_data")
    pb_filters = types.ModuleType("programbench.utils.instance_filters")

    pb_load.load_all_instances = MagicMock(
        return_value=[{"instance_id": "test_repo.abc123", "image_name": "test/test_repo.abc123"}]
    )
    from minisweagent.run.benchmarks.swebench import filter_instances as _filter

    pb_filters.filter_instances = lambda instances, **kw: _filter(
        instances,
        filter_spec=kw.get("filter_spec", ""),
        slice_spec=kw.get("slice_spec", ""),
        shuffle=kw.get("shuffle", False),
    )

    monkeypatch.setitem(sys.modules, "programbench", pb)
    monkeypatch.setitem(sys.modules, "programbench.utils", pb_utils)
    monkeypatch.setitem(sys.modules, "programbench.utils.load_data", pb_load)
    monkeypatch.setitem(sys.modules, "programbench.utils.instance_filters", pb_filters)
    return pb_load


def _fake_docker_env() -> MagicMock:
    """Mock environment that looks Docker-like and tolerates agent.run()."""
    env = MagicMock()
    env.container_id = "mock-container-id"
    env.config.executable = "docker"
    env.execute.side_effect = lambda *args, **kwargs: {"returncode": 0, "output": "", "exception_info": ""}
    env.get_template_vars.return_value = {"system": "linux", "release": "0", "version": "0", "machine": "x86_64"}
    env.serialize.return_value = {}
    return env


def _fake_cp(cmd: list[str], *args, **kwargs):
    """Simulate ``docker cp`` by writing a dummy tarball to the destination path."""
    Path(cmd[-1]).write_bytes(b"fake tarball")
    return MagicMock(returncode=0)


# ---------------------------------------------------------------------------
# copy_submission
# ---------------------------------------------------------------------------


def test_copy_submission_rejects_non_docker_env(tmp_path):
    env = MagicMock(spec=["execute"])  # no container_id / config
    with pytest.raises(RuntimeError, match="container_id"):
        copy_submission(env, tmp_path / "submission.tar.gz")


def test_copy_submission_tars_and_copies(tmp_path):
    env = _fake_docker_env()
    dest = tmp_path / "sub" / "submission.tar.gz"
    with patch("minisweagent.run.benchmarks.programbench.subprocess.run") as mock_run:
        copy_submission(env, dest, src="/workspace")

    cmds = [call.args[0]["command"] for call in env.execute.call_args_list]
    assert any("tar -czf /tmp/_submission.tar.gz -C /workspace ." in c for c in cmds)
    assert any("rm -f /tmp/_submission.tar.gz" in c for c in cmds)
    mock_run.assert_called_once()
    assert mock_run.call_args.args[0] == [
        "docker",
        "cp",
        "mock-container-id:/tmp/_submission.tar.gz",
        str(dest),
    ]
    assert dest.parent.exists()


def test_copy_submission_cleans_up_after_cp_failure(tmp_path):
    env = _fake_docker_env()
    with patch(
        "minisweagent.run.benchmarks.programbench.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "docker cp"),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            copy_submission(env, tmp_path / "submission.tar.gz")

    cleanup = [c.args[0]["command"] for c in env.execute.call_args_list if "rm -f" in c.args[0]["command"]]
    assert cleanup, "Cleanup rm should still run from the finally block"


# ---------------------------------------------------------------------------
# ProgramBenchAgent.serialize
# ---------------------------------------------------------------------------


def test_programbench_agent_strips_raw_output():
    sample = {
        "messages": [
            {"role": "assistant", "content": "hi"},
            {
                "role": "user",
                "content": "...",
                "extra": {
                    "raw_output": "BIG STDOUT BLOB",
                    "kept_field": 1,
                    "observations": [
                        {"raw_output": "nested blob", "name": "obs1"},
                        {"name": "obs2"},
                    ],
                },
            },
        ]
    }
    with patch.object(ProgressTrackingAgent, "serialize", return_value=sample):
        data = ProgramBenchAgent.__new__(ProgramBenchAgent).serialize()

    extra = data["messages"][1]["extra"]
    assert "raw_output" not in extra
    assert extra["kept_field"] == 1
    assert all("raw_output" not in obs for obs in extra["observations"])


# ---------------------------------------------------------------------------
# End-to-end orchestration (no Docker required - env + cp are mocked)
# ---------------------------------------------------------------------------


def test_programbench_end_to_end(fake_programbench, tmp_path):
    env = _fake_docker_env()

    with (
        patch("minisweagent.run.benchmarks.programbench.get_model") as mock_get_model,
        patch("minisweagent.run.benchmarks.programbench.get_environment", return_value=env) as mock_get_env,
        patch("minisweagent.run.benchmarks.programbench.subprocess.run", side_effect=_fake_cp),
    ):
        mock_get_model.side_effect = lambda **kwargs: _SubmittingModel()
        main(
            slice_spec="0:1",
            filter_spec="test_repo",
            shuffle=False,
            output=str(tmp_path),
            workers=1,
            model=None,
            model_class=None,
            redo_existing=False,
            config_spec=[str(package_dir / "config" / "benchmarks" / "programbench.yaml")],
            environment_class=None,
        )

    iid = "test_repo.abc123"
    submission = tmp_path / iid / "submission.tar.gz"
    traj = tmp_path / iid / f"{iid}.traj.json"
    assert submission.exists() and submission.read_bytes() == b"fake tarball"
    assert traj.exists()

    data = json.loads(traj.read_text())
    assert data["instance_id"] == iid
    assert data["info"]["exit_status"] == "Submitted"
    assert data["trajectory_format"] == "mini-swe-agent-1.1"

    # Image was constructed as <image_name>:task_cleanroom
    env_config_passed = mock_get_env.call_args.args[0]
    assert env_config_passed["image"] == "test/test_repo.abc123:task_cleanroom"


def test_redo_existing_false_skips_existing(fake_programbench, tmp_path):
    iid = "test_repo.abc123"
    (tmp_path / iid).mkdir(parents=True)
    (tmp_path / iid / "submission.tar.gz").write_bytes(b"pre-existing")

    with (
        patch("minisweagent.run.benchmarks.programbench.get_model") as mock_get_model,
        patch("minisweagent.run.benchmarks.programbench.get_environment") as mock_get_env,
    ):
        main(
            slice_spec="",
            filter_spec="",
            shuffle=False,
            output=str(tmp_path),
            workers=1,
            model=None,
            model_class=None,
            redo_existing=False,
            config_spec=[str(package_dir / "config" / "benchmarks" / "programbench.yaml")],
            environment_class=None,
        )

    mock_get_model.assert_not_called()
    mock_get_env.assert_not_called()
    assert (tmp_path / iid / "submission.tar.gz").read_bytes() == b"pre-existing"


class _ExceptionModelConfig(BaseModel):
    model_name: str = "exception_model"


class _ExceptionModel:
    def __init__(self, exc_type: type[Exception] = RuntimeError, exc_msg: str = "boom"):
        self.exc_type = exc_type
        self.exc_msg = exc_msg
        self.cost = 0.0
        self.n_calls = 0
        self.config = _ExceptionModelConfig()

    def query(self, *args, **kwargs):
        self.n_calls += 1
        raise self.exc_type(self.exc_msg)

    def format_message(self, **kwargs) -> dict:
        return dict(**kwargs)

    def format_observation_messages(self, message, outputs, template_vars=None) -> list[dict]:
        return [self.format_message(role="user", content=str(o)) for o in outputs]

    def get_template_vars(self, **kwargs) -> dict:
        return self.config.model_dump() | {"n_model_calls": self.n_calls, "model_cost": self.cost}

    def serialize(self) -> dict:
        return {
            "info": {
                "model_stats": {"instance_cost": self.cost, "api_calls": self.n_calls},
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }


def test_agent_exception_recorded_in_trajectory(fake_programbench, tmp_path):
    env = _fake_docker_env()
    with (
        patch(
            "minisweagent.run.benchmarks.programbench.get_model",
            return_value=_ExceptionModel(ValueError, "bad input"),
        ),
        patch("minisweagent.run.benchmarks.programbench.get_environment", return_value=env),
        patch("minisweagent.run.benchmarks.programbench.subprocess.run", side_effect=_fake_cp),
    ):
        main(
            slice_spec="0:1",
            filter_spec="test_repo",
            shuffle=False,
            output=str(tmp_path),
            workers=1,
            model=None,
            model_class=None,
            redo_existing=False,
            config_spec=[str(package_dir / "config" / "benchmarks" / "programbench.yaml")],
            environment_class=None,
        )

    iid = "test_repo.abc123"
    traj = tmp_path / iid / f"{iid}.traj.json"
    assert traj.exists()
    data = json.loads(traj.read_text())
    assert data["info"]["exit_status"] == "ValueError"
    assert data["info"]["exception_str"] == "bad input"
    assert (tmp_path / iid / "submission.tar.gz").exists()
