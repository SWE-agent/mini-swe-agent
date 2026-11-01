import json
import os
import stat
import subprocess
import sys

import pytest

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel


@pytest.fixture
def agent():
    return DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        system_template="system",
        instance_template="{{ task }}",
    )


@pytest.mark.parametrize(("mode", "expected_mode"), [(None, None), (0o640, 0o640)])
def test_save_roundtrip(agent, tmp_path, mode, expected_mode):
    agent.config.output_path = tmp_path / "configured.json"
    agent.add_messages({"role": "user", "content": "轨迹 café 🐟"})
    assert agent.save(None)["messages"] == agent.messages
    path = tmp_path / "nested" / "轨迹.json"
    control = tmp_path / "control"
    control.write_text("")
    if expected_mode is None:
        expected_mode = stat.S_IMODE(control.stat().st_mode)
    control.unlink()
    assert agent.save(path)["messages"] == json.loads(path.read_text())["messages"]
    if mode is not None:
        path.chmod(mode)
    agent.add_messages({"role": "assistant", "content": "next"})
    data = agent.save(path, {"info": {"extra_field": "retained"}})
    assert json.loads(path.read_text()) == data
    assert path.read_text() == json.dumps(data, indent=2)
    assert data["messages"][0]["content"] == "轨迹 café 🐟"
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == expected_mode
    assert not agent.config.output_path.exists()
    assert list(path.parent.iterdir()) == [path]


def test_save_serialization_error_keeps_checkpoint(agent, tmp_path):
    path = tmp_path / "trajectory.json"
    agent.save(path)
    before = path.read_bytes()
    agent.add_messages({"role": "user", "content": {"not JSON serializable"}})
    with pytest.raises(TypeError):
        agent.save(path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX symlinks")
def test_save_preserves_symlink(agent, tmp_path):
    target = tmp_path / "target.json"
    agent.save(target)
    target.chmod(0o640)
    link = tmp_path / "trajectory.json"
    link.symlink_to(target.name)
    agent.add_messages({"role": "user", "content": "updated through link"})
    assert json.loads(target.read_text()) != agent.serialize()
    assert agent.save(link) == json.loads(target.read_text())
    assert link.is_symlink()
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert set(tmp_path.iterdir()) == {target, link}


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="requires non-root POSIX permissions")
@pytest.mark.parametrize(("target_mode", "parent_mode"), [(0o400, 0o700), (0o600, 0o500)])
def test_save_permission_failure_keeps_checkpoint(agent, tmp_path, target_mode, parent_mode):
    path = tmp_path / "trajectory.json"
    agent.save(path)
    before = path.read_bytes()
    agent.add_messages({"role": "user", "content": "next"})
    path.chmod(target_mode)
    tmp_path.chmod(parent_mode)
    try:
        with pytest.raises(PermissionError):
            agent.save(path)
        assert path.read_bytes() == before
        assert stat.S_IMODE(path.stat().st_mode) == target_mode
        assert list(tmp_path.iterdir()) == [path]
    finally:
        tmp_path.chmod(0o700)
        path.chmod(0o600)


_WRITE_FAILURE_SCRIPT = """
import errno, json, resource, signal, sys
from pathlib import Path
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel

path = Path(sys.argv[1])
existing = sys.argv[2] == 'True'
interrupted = sys.argv[3] == 'True'
agent = DefaultAgent(model=DeterministicModel(outputs=[]), env=LocalEnvironment(),
                     system_template='system', instance_template='{{ task }}')
if existing:
    agent.save(path)
before = path.read_bytes() if existing else None
agent.add_messages({'role': 'user', 'content': 'x' * 16384})
def interrupt(*args):
    raise KeyboardInterrupt
signal.signal(signal.SIGXFSZ, interrupt if interrupted else signal.SIG_IGN)
previous_limit = resource.getrlimit(resource.RLIMIT_FSIZE)
try:
    resource.setrlimit(resource.RLIMIT_FSIZE, (4096, previous_limit[1]))
    try:
        agent.save(path)
    except BaseException as error:
        if interrupted:
            assert isinstance(error, KeyboardInterrupt), repr(error)
        else:
            assert isinstance(error, OSError) and error.errno == errno.EFBIG, repr(error)
    else:
        raise AssertionError('write limit was not exercised')
finally:
    resource.setrlimit(resource.RLIMIT_FSIZE, previous_limit)
if existing:
    assert path.read_bytes() == before, 'previous checkpoint was overwritten'
    assert json.loads(path.read_text())['trajectory_format'] == 'mini-swe-agent-1.1'
else:
    assert not path.exists(), 'partial checkpoint was left behind'
assert set(path.parent.iterdir()) == ({path} if existing else set()), 'temporary artifacts leaked'
"""


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX file-size limit")
@pytest.mark.parametrize(("existing", "interrupted"), [(True, False), (False, False), (True, True), (False, True)])
def test_save_os_write_failure_keeps_checkpoint(tmp_path, existing, interrupted):
    subprocess.run(
        [
            sys.executable,
            "-c",
            _WRITE_FAILURE_SCRIPT,
            str(tmp_path / "trajectory.json"),
            str(existing),
            str(interrupted),
        ],
        env=os.environ | {"MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path.parent / "config"), "MSWEA_SILENT_STARTUP": "1"},
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
