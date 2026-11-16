import copy
import json
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from minisweagent.models.test_models import make_output
from minisweagent.run.benchmarks.swebench import process_instance
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager


@pytest.mark.parametrize(("shared_output", "submit"), [(False, False), (False, True), (True, False), (True, True)])
def test_batch_checkpoints_are_saved_per_instance(tmp_path, shared_output, submit, reset_global_stats):
    config = {
        "agent": {"system_template": "system", "instance_template": "{{task}}", "cost_limit": 0},
        "environment": {"environment_class": "local", "cwd": str(tmp_path)},
    }
    if shared_output:
        config["agent"]["output_path"] = str(tmp_path / "shared.traj.json")
    original_config = copy.deepcopy(config)
    progress = RunBatchProgressManager(2)

    def run_instance(index: int) -> None:
        iid = f"test__instance-{index}"
        trajectory_path = tmp_path / iid / f"{iid}.traj.json"
        check_checkpoint = (
            "import json; from pathlib import Path; "
            f"data=json.loads(Path({str(trajectory_path)!r}).read_text()); "
            f"assert data['messages'][1]['content']=={iid!r}; "
            "assert data['info']['model_stats']['api_calls']==1; "
            "print('checkpoint contains first step')"
        )
        outputs = [
            make_output("first step", [{"command": "echo first-step"}]),
            make_output(
                "check checkpoint", [{"command": f"{shlex.quote(sys.executable)} -c {shlex.quote(check_checkpoint)}"}]
            ),
        ]
        if submit:
            outputs.append(
                make_output("submit", [{"command": "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\npatch\\n'"}])
            )
        process_instance(
            {"instance_id": iid, "problem_statement": iid},
            tmp_path,
            config | {"model": {"model_class": "deterministic", "model_name": "deterministic", "outputs": outputs}},
            progress,
        )
        trajectory = json.loads(trajectory_path.read_text())
        assert "checkpoint contains first step" in trajectory["messages"][5]["content"]
        assert trajectory["instance_id"] == iid
        assert trajectory["info"]["config"]["agent"]["output_path"] == str(trajectory_path)
        assert trajectory["info"]["exit_status"] == ("Submitted" if submit else "IndexError")
        assert trajectory["info"]["submission"] == ("patch\n" if submit else "")

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run_instance, range(2)))

    predictions = json.loads((tmp_path / "preds.json").read_text())
    assert predictions == {
        f"test__instance-{i}": {
            "model_name_or_path": "deterministic",
            "instance_id": f"test__instance-{i}",
            "model_patch": "patch\n" if submit else "",
        }
        for i in range(2)
    }
    assert config == original_config
    assert not (tmp_path / "shared.traj.json").exists()
    assert progress.n_completed == 2
