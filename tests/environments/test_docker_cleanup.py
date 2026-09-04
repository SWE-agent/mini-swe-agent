"""Tests for DockerEnvironment cleanup idempotency and atexit behavior."""

import subprocess

import pytest

from minisweagent.environments.docker import DockerEnvironment


def _docker_available(executable="docker"):
    try:
        subprocess.run([executable, "version"], capture_output=True, check=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


env_params = [
    pytest.param(
        "docker", marks=pytest.mark.skipif(not _docker_available(), reason="Docker not available"), id="docker"
    ),
    pytest.param(
        "podman",
        marks=pytest.mark.skipif(not _docker_available("podman"), reason="Podman not available"),
        id="podman",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize("executable", env_params)
def test_cleanup_idempotent(executable):
    """Multiple cleanup() calls should not raise."""
    env = DockerEnvironment(image="python:3.11", executable=executable)
    env.cleanup()
    env.cleanup()
    env.cleanup()


@pytest.mark.slow
@pytest.mark.parametrize("executable", env_params)
def test_cleaned_flag(executable):
    """_cleaned flag should be True after cleanup()."""
    env = DockerEnvironment(image="python:3.11", executable=executable)
    assert not env._cleaned
    env.cleanup()
    assert env._cleaned
