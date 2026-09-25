import shlex
from unittest.mock import patch

import pytest

from minisweagent.environments import get_environment
from minisweagent.environments.extra import runtime as runtime_module
from minisweagent.environments.extra.runtime import RuntimeEnvironment
from minisweagent.exceptions import Submitted


class FakeResult:
    def __init__(self, exit_code=0, stdout="", stderr="", timed_out=False):
        self.exit_code, self.stdout, self.stderr, self.timed_out = exit_code, stdout, stderr, timed_out


class FakeSandbox:
    def __init__(self, world, fields):
        self.world, self.fields, self.id = world, fields, f"sbx_{len(world.sandboxes)}"
        self.stopped = False
        self.kept_alive = None

    def exec(self, command, *, env=None, timeout_ms=None):
        self.world.execs.append({"command": command, "env": env, "timeout_ms": timeout_ms})
        if command[:3] == ["sh", "-c", "id -u; id -un; command -v sudo || true"]:
            return FakeResult(stdout=self.world.probe)
        if command == ["sudo", "-n", "true"]:
            return FakeResult(exit_code=0 if self.world.sudo_ok else 1)
        return self.world.next_result

    def keep_alive(self, every_seconds=60, margin_seconds=600):
        self.kept_alive = margin_seconds

    def stop(self, wait=True):
        self.stopped = True


class FakeImages:
    def __init__(self, world):
        self.world = world

    def resolve(self, ref):
        self.world.resolved.append(ref)
        if ref in self.world.images:
            return {"id": self.world.images[ref], "state": "ready"}
        raise LookupError(ref)

    def build(self, **fields):
        self.world.builds.append(fields)
        ref = f"{fields['name']}:{fields['tags'][0]}"
        self.world.images[ref] = f"img_{len(self.world.builds)}"
        return {"id": self.world.images[ref], "state": "ready"}


class FakeWorld:
    def __init__(self):
        self.sandboxes, self.execs, self.builds, self.resolved, self.images = [], [], [], [], {}
        self.probe = "1000\nruntime\n/usr/bin/sudo\n"
        self.sudo_ok = True
        self.next_result = FakeResult(stdout="ok\n")

    def runtime(self, api_key=None):
        world = self

        class Sandboxes:
            def create(self, **fields):
                sandbox = FakeSandbox(world, fields)
                world.sandboxes.append(sandbox)
                return sandbox

        class Client:
            sandboxes = Sandboxes()
            images = FakeImages(world)

        world.api_key = api_key
        return Client()


@pytest.fixture
def world():
    w = FakeWorld()
    with patch.object(runtime_module, "Runtime", side_effect=w.runtime):
        yield w


def _last_command(world):
    return world.execs[-1]["command"]


def test_registered_under_runtime(world):
    env = get_environment({"environment_class": "runtime"})
    assert isinstance(env, RuntimeEnvironment)


def test_creates_a_sandbox_with_a_bounded_lease_and_keeps_it_alive(world):
    env = RuntimeEnvironment(vcpu=2, memory_mib=4096, create={"disk_mib": 8192})
    fields = world.sandboxes[0].fields
    assert fields["vcpu"] == 2 and fields["memory_mib"] == 4096 and fields["disk_mib"] == 8192
    assert fields["on_lease_end"] == "stop" and fields["timeout_seconds"] == 600
    assert "image" not in fields
    assert world.sandboxes[0].kept_alive == 600
    env.cleanup()
    assert world.sandboxes[0].stopped


def test_runs_as_root_through_sudo_in_cwd_with_env(world, monkeypatch):
    monkeypatch.setenv("HOST_TOKEN", "t")
    env = RuntimeEnvironment(cwd="/testbed", env={"PAGER": "cat"}, forward_env=["HOST_TOKEN", "UNSET_VAR"])
    result = env.execute({"command": "ls -la"})
    command = _last_command(world)
    assert command[:2] == ["sh", "-c"]
    assert command[2] == "sudo -n -E -H -u root -- " + shlex.join(["bash", "-lc", "cd /testbed && ls -la"])
    assert world.execs[-1]["env"] == {"HOST_TOKEN": "t", "PAGER": "cat"}
    assert world.execs[-1]["timeout_ms"] == 60_000
    assert result == {"output": "ok\n", "returncode": 0, "exception_info": ""}


def test_no_sudo_when_the_sandbox_user_is_asked_for_or_is_root(world):
    env = RuntimeEnvironment(user=None)
    env.execute({"command": "true"})
    assert not _last_command(world)[2].startswith("sudo")
    world.probe = "0\nroot\n\n"
    env = RuntimeEnvironment()
    env.execute({"command": "true"})
    assert not _last_command(world)[2].startswith("sudo")


def test_an_image_without_sudo_is_refused_and_its_sandbox_stopped(world):
    world.probe = "1000\nruntime\n\n"
    with pytest.raises(RuntimeError, match="passwordless sudo"):
        RuntimeEnvironment()
    assert world.sandboxes[0].stopped


def test_a_docker_image_is_built_once_and_found_again(world):
    RuntimeEnvironment(image="docker.io/swebench/sweb.eval.x86_64.django_1776_django-11099:latest")
    RuntimeEnvironment(image="docker.io/swebench/sweb.eval.x86_64.django_1776_django-11099:latest")
    assert len(world.builds) == 1
    build = world.builds[0]
    assert build["image"] == "docker.io/swebench/sweb.eval.x86_64.django_1776_django-11099:latest"
    assert build["name"] == "mini-swe-agent-sweb.eval.x86_64.django_1776_django-11099"
    assert build["no_start"] is True
    assert world.sandboxes[0].fields["image"] == world.sandboxes[1].fields["image"] == "img_1"


def test_docker_prefix_is_stripped_and_runtime_image_wins(world):
    RuntimeEnvironment(image="docker://python:3.12")
    assert world.builds[0]["image"] == "python:3.12"
    RuntimeEnvironment(image="python:3.12", runtime_image="my-image:v1")
    assert world.sandboxes[1].fields["image"] == "my-image:v1"
    assert len(world.builds) == 1


def test_timeout_is_an_exception_result_with_partial_output(world):
    env = RuntimeEnvironment(timeout=5)
    world.next_result = FakeResult(exit_code=-9, stdout="partial", timed_out=True)
    result = env.execute({"command": "sleep 60"})
    assert result["returncode"] == -1
    assert result["output"] == "partial"
    assert "timed out after 5 seconds" in result["exception_info"]
    assert result["extra"]["exception_type"] == "TimeoutExpired"
    env.execute({"command": "sleep 1"}, timeout=90)
    assert world.execs[-1]["timeout_ms"] == 90_000


def test_a_signal_is_reported_as_a_shell_does(world):
    env = RuntimeEnvironment()
    world.next_result = FakeResult(exit_code=-15)
    assert env.execute({"command": "kill -TERM $$"})["returncode"] == 143
    world.next_result = FakeResult(exit_code=2, stderr="no such file\n")
    assert env.execute({"command": "ls nope"}) == {"output": "no such file\n", "returncode": 2, "exception_info": ""}


def test_submission_raises_submitted(world):
    env = RuntimeEnvironment()
    world.next_result = FakeResult(stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndiff --git a b\n")
    with pytest.raises(Submitted) as raised:
        env.execute({"command": "git diff"})
    assert raised.value.messages[0]["extra"]["submission"] == "diff --git a b\n"


def test_api_errors_become_results_and_the_key_is_never_serialized(world):
    env = RuntimeEnvironment(api_key="secret-key")
    assert world.api_key == "secret-key"
    world.sandboxes[0].exec = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("reset"))
    result = env.execute({"command": "true"})
    assert result["returncode"] == -1 and "reset" in result["exception_info"]
    assert "secret-key" not in str(env.serialize()) and "secret-key" not in str(env.get_template_vars())
