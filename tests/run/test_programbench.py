"""Integration tests for the programbench runner.

These tests exercise the real ``tar -czf`` + ``docker cp`` flow inside a live
container. They require a working ``docker``/``podman`` (skipped otherwise).
"""

import json
import sys
import tarfile
import types
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import BaseModel

from minisweagent import package_dir
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.exceptions import Submitted
from minisweagent.run.benchmarks.programbench import copy_submission, main

# Lightweight image used for the real-docker tests. Already cached on machines
# that run mini-swe-agent's docker test suite (see tests/environments/test_docker.py).
_TEST_IMAGE = "python:3.11"


@pytest.fixture
def docker_env(container_executable):
    """Spin up a fresh container and yield its DockerEnvironment, tearing it down after."""
    env = DockerEnvironment(image=_TEST_IMAGE, executable=container_executable)
    try:
        yield env
    finally:
        env.cleanup()


@pytest.fixture
def fake_programbench(monkeypatch):
    """Inject a fake ``programbench`` package exposing the loader + filter mini-swe-agent uses."""
    pb = types.ModuleType("programbench")
    pb_utils = types.ModuleType("programbench.utils")
    pb_load = types.ModuleType("programbench.utils.load_data")
    pb_filters = types.ModuleType("programbench.utils.instance_filters")

    pb_load.load_all_instances = MagicMock(return_value=[{"instance_id": "test_repo.abc123", "image_name": "python"}])
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


# ---------------------------------------------------------------------------
# copy_submission integration
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_copy_submission_real_container(tmp_path, docker_env):
    """End-to-end: create files in /workspace, copy them out, verify tar.gz contents."""
    docker_env.execute(
        {"command": "mkdir -p /workspace && echo hello > /workspace/file.txt && echo world > /workspace/other.txt"}
    )

    dest = tmp_path / "submission.tar.gz"
    copy_submission(docker_env, dest)

    assert dest.exists()
    assert dest.stat().st_size > 0
    with tarfile.open(dest, "r:gz") as tf:
        names = set(tf.getnames())
        contents = {
            n: tf.extractfile(n).read().decode()
            for n in names
            if tf.getmember(n).isfile()  # type: ignore[union-attr]
        }
    assert "./file.txt" in names
    assert "./other.txt" in names
    assert contents["./file.txt"].strip() == "hello"
    assert contents["./other.txt"].strip() == "world"


@pytest.mark.slow
def test_copy_submission_removes_container_tar(tmp_path, docker_env):
    """The in-container ``/tmp/_submission.tar.gz`` should be cleaned up after copy."""
    docker_env.execute({"command": "mkdir -p /workspace && touch /workspace/x"})
    copy_submission(docker_env, tmp_path / "submission.tar.gz")

    # The tarball inside the container should be gone after copy_submission returns.
    result = docker_env.execute({"command": "ls /tmp/_submission.tar.gz || echo MISSING"})
    assert "MISSING" in result["output"]


def test_copy_submission_rejects_non_docker_env(tmp_path):
    """No live container needed for this guardrail check."""
    env = MagicMock(spec=["execute"])
    with pytest.raises(RuntimeError, match="container_id"):
        copy_submission(env, tmp_path / "submission.tar.gz")


# ---------------------------------------------------------------------------
# Network isolation: containers must not have internet access
# ---------------------------------------------------------------------------


def test_default_config_disables_network():
    """The shipped ``programbench.yaml`` must contain ``--network none`` in run_args."""
    cfg = yaml.safe_load((package_dir / "config" / "benchmarks" / "programbench.yaml").read_text())
    run_args = cfg["environment"]["run_args"]
    # Argument and value must appear consecutively in the list.
    assert "--network" in run_args
    assert run_args[run_args.index("--network") + 1] == "none"


@pytest.mark.slow
def test_container_with_network_none_has_no_internet(container_executable):
    """A container started with ``--network none`` (our default) cannot reach the internet."""
    env = DockerEnvironment(
        image=_TEST_IMAGE,
        executable=container_executable,
        run_args=["--rm", "--network", "none"],
    )
    try:
        # Try to resolve + reach a well-known external host. Multiple probes so we don't
        # rely on any single tool being available in the image.
        result = env.execute(
            {
                "command": (
                    "python3 -c 'import socket; socket.create_connection((\"1.1.1.1\", 80), timeout=2)' "
                    "&& echo INTERNET_OK || echo INTERNET_BLOCKED"
                )
            }
        )
        assert "INTERNET_BLOCKED" in result["output"], (
            f"Network should be blocked but probe succeeded; output: {result['output']!r}"
        )
        assert "INTERNET_OK" not in result["output"]
    finally:
        env.cleanup()


# ---------------------------------------------------------------------------
# Full main() integration with real docker
# ---------------------------------------------------------------------------


class _SubmittingModelConfig(BaseModel):
    model_name: str = "submitting_model"


class _SubmittingModel:
    """Test model whose ``query`` raises ``Submitted`` so the agent exits cleanly on step 1."""

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


@pytest.mark.slow
def test_programbench_end_to_end_real_docker(fake_programbench, tmp_path, container_executable):
    """Run the full ``main()`` against a real container, with model+image patched.

    We override ``_IMAGE_TAG`` to ``"3.11"`` so the runner picks up ``python:3.11``
    (which the fake programbench advertises as image_name=``python``), yielding a
    working image without depending on the programbench docker registry.
    """
    # Override the prod ``run_args`` (which assume 20 CPUs / 60g RAM / a non-existent
    # ``agent`` user) with minimal flags compatible with stock ``python:3.11``.
    run_args_override = 'environment.run_args=["--rm"]'
    with (
        patch("minisweagent.run.benchmarks.programbench._IMAGE_TAG", "3.11"),
        patch("minisweagent.run.benchmarks.programbench.get_model", side_effect=lambda **kw: _SubmittingModel()),
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
            config_spec=[
                str(package_dir / "config" / "benchmarks" / "programbench.yaml"),
                run_args_override,
            ],
            environment_class=None,
        )

    iid = "test_repo.abc123"
    submission = tmp_path / iid / "submission.tar.gz"
    traj = tmp_path / iid / f"{iid}.traj.json"

    # Real tarball produced by `tar -czf` inside the container
    assert submission.exists() and submission.stat().st_size > 0
    with tarfile.open(submission, "r:gz") as tf:
        assert tf.getnames(), "submission tarball should not be empty"

    assert traj.exists()
    data = json.loads(traj.read_text())
    assert data["instance_id"] == iid
    assert data["info"]["exit_status"] == "Submitted"


@pytest.mark.slow
def test_programbench_skip_existing_real_docker(fake_programbench, tmp_path, container_executable):
    """An existing ``submission.tar.gz`` should make ``main()`` skip the instance (no container started)."""
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
