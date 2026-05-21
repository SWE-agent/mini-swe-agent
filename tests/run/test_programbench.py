"""Integration tests for the programbench runner.

These tests exercise the real ``tar -czf`` + ``docker cp`` flow inside a live
container. They require a working ``docker``/``podman`` (skipped otherwise).

The ``main()`` orchestration tests additionally require the real ``programbench``
package to be installed so we verify API compatibility with what programbench
actually ships — they're skipped via ``pytest.importorskip`` when not available.
"""

import json
import tarfile
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
def real_programbench_first_instance():
    """Return the real first programbench instance (cmatrix at the time of writing).

    Tests using this fixture exercise the real ``load_all_instances`` /
    ``filter_instances`` surface AND the real programbench docker image, so they
    actually verify end-to-end compatibility — at the cost of pulling that image.
    """
    pytest.importorskip("programbench")
    from programbench.utils.load_data import load_all_instances

    instances = load_all_instances(include_tests=False)
    assert instances, "real programbench should ship at least one instance"
    return instances[0]


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
        result = env.execute(
            {
                "command": (
                    "python3 -c 'import socket; socket.create_connection((\"1.1.1.1\", 80), timeout=2)' "
                    "&& echo INTERNET_OK || echo INTERNET_BLOCKED"
                )
            }
        )
        assert "INTERNET_BLOCKED" in result["output"]
        assert "INTERNET_OK" not in result["output"]
    finally:
        env.cleanup()


# ---------------------------------------------------------------------------
# programbench API compatibility (requires programbench installed)
# ---------------------------------------------------------------------------


def test_real_programbench_api_contract():
    """The runner relies on a specific shape from ``load_all_instances`` / ``filter_instances``."""
    pytest.importorskip("programbench")
    from programbench.utils.instance_filters import filter_instances
    from programbench.utils.load_data import load_all_instances

    instances = load_all_instances(include_tests=False)
    assert isinstance(instances, list) and instances
    for inst in instances:
        assert "instance_id" in inst, f"missing instance_id: {inst}"
        assert "image_name" in inst, f"missing image_name: {inst}"

    # The runner passes filter_spec / slice_spec / shuffle by keyword
    filtered = filter_instances(instances, filter_spec="^abishekvashok", slice_spec="0:1", shuffle=False)
    assert isinstance(filtered, list)


# ---------------------------------------------------------------------------
# Full main() integration against real programbench + real docker
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
def test_programbench_end_to_end_real_docker(real_programbench_first_instance, tmp_path, container_executable):
    """Real programbench API + real programbench docker image. Only the model is mocked."""
    instance = real_programbench_first_instance
    # The default config asks for resources (--user agent / 20 CPUs / 60g RAM) that
    # CI runners may not have. Override run_args with the bare minimum (keeping
    # ``--network none`` since the agent is supposed to run offline).
    run_args_override = 'environment.run_args=["--rm", "--network", "none"]'
    with patch("minisweagent.run.benchmarks.programbench.get_model", side_effect=lambda **kw: _SubmittingModel()):
        main(
            slice_spec="",
            filter_spec=f"^{instance['instance_id']}$",
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

    iid = instance["instance_id"]
    submission = tmp_path / iid / "submission.tar.gz"
    traj = tmp_path / iid / f"{iid}.traj.json"

    assert submission.exists() and submission.stat().st_size > 0
    with tarfile.open(submission, "r:gz") as tf:
        assert tf.getnames(), "submission tarball should not be empty"

    assert traj.exists()
    data = json.loads(traj.read_text())
    assert data["instance_id"] == iid
    assert data["info"]["exit_status"] == "Submitted"


@pytest.mark.slow
def test_programbench_skip_existing_real_docker(real_programbench_first_instance, tmp_path, container_executable):
    """Pre-existing ``submission.tar.gz`` should make ``main()`` skip the instance."""
    instance = real_programbench_first_instance
    iid = instance["instance_id"]
    (tmp_path / iid).mkdir(parents=True)
    (tmp_path / iid / "submission.tar.gz").write_bytes(b"pre-existing")

    with (
        patch("minisweagent.run.benchmarks.programbench.get_model") as mock_get_model,
        patch("minisweagent.run.benchmarks.programbench.get_environment") as mock_get_env,
    ):
        main(
            slice_spec="",
            filter_spec=f"^{iid}$",
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
