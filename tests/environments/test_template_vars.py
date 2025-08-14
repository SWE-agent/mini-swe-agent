import pytest

from minisweagent.environments.local import LocalEnvironment
from minisweagent.environments.utils.template_vars import get_remote_template_vars


def test_local_environment_basic():
    """Test get_remote_template_vars with LocalEnvironment returns expected data structure."""
    env = LocalEnvironment(cwd="/tmp", env={"TEST_VAR": "test_value"})
    result = get_remote_template_vars(env, strict=True)

    # Should contain config and env vars (platform info may fail due to shell escaping)
    assert isinstance(result, dict)
    assert "cwd" in result  # from config
    assert "env" in result  # from config
    assert "timeout" in result  # from config
    assert "TEST_VAR" in result  # from env vars
    assert result["TEST_VAR"] == "test_value"

    # Verify config values
    assert result["cwd"] == "/tmp"
    assert result["timeout"] == 30
    assert isinstance(result["env"], dict)
    assert result["env"]["TEST_VAR"] == "test_value"


def test_local_environment_env_override_priority():
    """Test that environment variables override config values in template vars."""
    import os

    original_timeout = os.environ.get("timeout")
    os.environ["timeout"] = "999"

    try:
        env = LocalEnvironment(timeout=30)
        result = get_remote_template_vars(env, strict=True)
        # Environment variable should override config
        assert result["timeout"] == "999"
    finally:
        if original_timeout is None:
            os.environ.pop("timeout", None)
        else:
            os.environ["timeout"] = original_timeout


def test_local_environment_platform_info_failure():
    """Test handling when platform info command fails with strict=True."""

    class FailingPlatformEnv:
        def __init__(self):
            from minisweagent.environments.local import LocalEnvironmentConfig

            self.config = LocalEnvironmentConfig(cwd="/test")

        def execute(self, command: str, cwd: str = ""):
            if "platform.uname" in command:
                return {"output": "invalid json", "returncode": 0}
            elif command == "env":
                return {"output": "PATH=/usr/bin\nHOME=/home/test", "returncode": 0}
            return {"output": "", "returncode": 1}

    env = FailingPlatformEnv()
    # With strict=True, should raise exception when platform info fails to parse
    with pytest.raises(ValueError):
        get_remote_template_vars(env, strict=True)


def test_local_environment_platform_info_failure_non_strict():
    """Test handling when platform info command fails with strict=False."""

    class FailingPlatformEnv:
        def __init__(self):
            from minisweagent.environments.local import LocalEnvironmentConfig

            self.config = LocalEnvironmentConfig(cwd="/test")

        def execute(self, command: str, cwd: str = ""):
            if "platform.uname" in command:
                return {"output": "invalid json", "returncode": 0}
            elif command == "env":
                return {"output": "PATH=/usr/bin\nHOME=/home/test", "returncode": 0}
            return {"output": "", "returncode": 1}

    env = FailingPlatformEnv()
    result = get_remote_template_vars(env, strict=False)

    # Should still work without platform info
    assert isinstance(result, dict)
    assert "cwd" in result
    assert "PATH" in result
    assert "HOME" in result


def test_docker_environment():
    """Test get_remote_template_vars with DockerEnvironment if Docker is available."""
    try:
        # Try to import docker library
        import docker  # noqa: F401
    except ImportError:
        pytest.skip("Docker not available")

    from minisweagent.environments.docker import DockerEnvironment

    env = DockerEnvironment(image="python:3.10-slim", env={"DOCKER_TEST": "docker_value"}, cwd="/app")
    result = get_remote_template_vars(env, strict=True)

    # Should contain all expected sections
    assert isinstance(result, dict)
    assert "image" in result  # from config
    assert "cwd" in result  # from config
    assert "DOCKER_TEST" in result  # from env vars
    assert result["DOCKER_TEST"] == "docker_value"

    # Should have platform info from container
    assert "system" in result

    env.cleanup()


def test_singularity_environment():
    """Test get_remote_template_vars with SingularityEnvironment if available."""
    try:
        import subprocess

        subprocess.run(["singularity", "--version"], check=True, capture_output=True)

        from minisweagent.environments.singularity import SingularityEnvironment

        env = SingularityEnvironment(image="docker://python:3.10-slim", env={"SINGULARITY_TEST": "singularity_value"})
        result = get_remote_template_vars(env, strict=True)

        assert isinstance(result, dict)
        assert "image" in result
        assert "SINGULARITY_TEST" in result
        assert result["SINGULARITY_TEST"] == "singularity_value"

        env.cleanup()
    except Exception as e:
        pytest.skip(f"Singularity environment not available: {e}")


def test_swerex_docker_environment():
    """Test get_remote_template_vars with SwerexDockerEnvironment if available."""
    try:
        import docker

        docker.from_env().ping()

        from minisweagent.environments.extra.swerex_docker import SwerexDockerEnvironment

        env = SwerexDockerEnvironment(image="python:3.10-slim", cwd="/workspace")
        result = get_remote_template_vars(env, strict=True)

        assert isinstance(result, dict)
        assert "image" in result
        assert "cwd" in result
        assert result["cwd"] == "/workspace"

        # Platform info should be available
        assert "system" in result
    except Exception as e:
        pytest.skip(f"SwerexDockerEnvironment not available: {e}")
