import asyncio
from typing import Dict, List, Optional, Any, Tuple
from omegaconf import DictConfig
import yaml
import traceback
import time
import os
from pathlib import Path, PurePosixPath
from loguru import logger 
import tempfile
import shutil

from loguru import logger

from minisweagent.environments import get_environment
from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent
from minisweagent.run.extra.utils.swebench_utils import get_sb_environment, get_swebench_docker_image_name

import swebench

def process_git_patch(patch):
    if not isinstance(patch, str):
        return ''

    if not patch.strip():
        # skip empty patches
        return ''

    patch = patch.replace('\r\n', '\n')
    # There might be some weird characters at the beginning of the patch
    # due to some OpenHands inference command outputs

    # FOR EXAMPLE:
    # git diff --no-color --cached 895f28f9cbed817c00ab68770433170d83132d90
    # [A[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[K0
    # diff --git a/django/db/models/sql/.backup.query.py b/django/db/models/sql/.backup.query.py
    # new file mode 100644
    # index 0000000000..fc13db5948

    # We "find" the first line that starts with "diff" and then we remove lines before it
    lines = patch.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('diff --git'):
            patch = '\n'.join(lines[i:])
            break

    patch = patch.rstrip() + '\n'  # Make sure the last line ends with a newline
    return patch

def copy_to(src: str | Path, dst_in_container: str, sandbox_dir: Path) -> Path:
    """Copy a file/dir from the host into the sandbox at the given container path."""
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(src)

    # Map container path (e.g. "/app/file.txt") to host path under sandbox_dir
    dst_rel = PurePosixPath(dst_in_container).relative_to("/")  # strips leading "/"
    host_dst = (sandbox_dir / Path(*dst_rel.parts)).resolve()

    # Prevent path traversal outside sandbox
    sandbox_root = sandbox_dir.resolve()
    if not str(host_dst).startswith(str(sandbox_root)):
        raise ValueError("Destination escapes sandbox_dir")

    host_dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, host_dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, host_dst)
    return host_dst

def evaluate_result(instance, model_patch, instance_id, dataset, sweagent_config) -> Tuple[bool, str]:
        """Apply patch and evaluate the solution."""
        from swebench.harness.grading import get_eval_report
        from swebench.harness.constants import (
            APPLY_PATCH_FAIL,
            APPLY_PATCH_PASS,
        )
        from swebench.harness.test_spec.test_spec import make_test_spec

        print("git patch: ", model_patch)
        
        if not model_patch:
            raise Exception(f"No git patch found for instance {instance_id}")
    
        env = None
        extra_info = None
        try:
            env = get_sb_environment(sweagent_config, instance)
            assert hasattr(env, "sandbox_dir"), "expected singularity env with 'sandbox_dir' attribute"
        except Exception as e:
            return False, f"Env creation failed with {e}"

        test_spec = make_test_spec(instance=instance)
        model_patch = process_git_patch(model_patch)
        print("model patch after processing: ", model_patch)
        
        # Get patch and save it to /tmp/patch.diff
        with tempfile.TemporaryDirectory() as temp_dir:
            # Patch file
            patch_file_path = os.path.join(temp_dir, 'patch.diff')
            with open(patch_file_path, 'w') as f:
                f.write(model_patch)
            dst = os.path.join(str(env.sandbox_dir), "tmp")
            shutil.copy(patch_file_path,  os.path.join(str(env.sandbox_dir), "tmp") )
            print(f'destination directory: {os.path.join(str(env.sandbox_dir), "tmp")}, Input file path: {patch_file_path}')
            # Eval script
            eval_script_path = os.path.join(temp_dir, 'eval.sh')
            with open(eval_script_path, 'w') as f:
                f.write(test_spec.eval_script)
            shutil.copy(eval_script_path,  os.path.join(str(env.sandbox_dir), "tmp"))
            print(f'destination directory: {dst}, Eval script path: {eval_script_path}')
            print(f"destination fies: {os.listdir(dst)}")

        # Set +x
        obs = env.execute("ls", cwd="/tmp")
        print("observation for ls: ", obs)
        logger.info("chmod /tmp/eval.sh", extra={'msg_type': 'ACTION'})
        obs = env.execute("chmod +x /tmp/eval.sh",  cwd="/")
        logger.info(f"observation logs: {obs['output']}, {obs['returncode']}", extra={'msg_type': 'OBSERVATION'})
        assert obs["returncode"] == 0, f"Got bad observation: {obs} for environment dir: {env.sandbox_dir}"

        # Apply patch
        if 'swe-smith' in dataset:
            # need to fetch and checkout the branch first
            exec_command = (
                "cd /testbed && "
                "git fetch && "
                f"git checkout {instance['instance_id']} && "
                "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
                "(echo 'Failed to apply patch with git apply, trying with patch command...' && "
                "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
                "echo 'APPLY_PATCH_FAIL')))"
            )
        else:
            exec_command = (
                'cd /testbed && '
                "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
                "(echo 'Failed to apply patch with git apply, trying with patch command...' && "
                "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
                "echo 'APPLY_PATCH_FAIL')))"
            )
        obs = env.execute(exec_command, cwd="/")
        apply_patch_output = obs["output"]
        assert isinstance(apply_patch_output, str)
        print("Output for apply patch: ", apply_patch_output)
        # instance['test_result']['apply_patch_output'] = apply_patch_output

        if 'APPLY_PATCH_FAIL' in apply_patch_output:
            raise Exception(f"Instance {instance_id} {APPLY_PATCH_FAIL}:\n{apply_patch_output}")
        elif 'APPLY_PATCH_PASS' in apply_patch_output:
            logger.info(f'[{instance_id}] {APPLY_PATCH_PASS}:\n{apply_patch_output}')

            # Run eval script in background and save output to log file
            log_file = '/tmp/eval_output.log'
            command=f'/tmp/eval.sh > {log_file} 2>&1 & echo $!'
            obs = env.execute(command, cwd="/")

            if isinstance(obs, dict) and obs["returncode"] == 0:
                pid = obs["output"].split()[-1].strip()
                logger.info(
                    f'[{instance_id}] Evaluation process started with PID: {pid}'
                )

                # Poll for completion
                start_time = time.time()
                timeout = 1200  # 20 minutes
                while True:
                    seconds_elapsed = time.time() - start_time
                    if seconds_elapsed > timeout:
                        raise Exception(
                            f'[{instance_id}] Evaluation timed out after {timeout} seconds'
                        )
                    command=f'ps -p {pid} > /dev/null; echo $?'
                    check_obs = env.execute(command, cwd="/")
                    if (
                        isinstance(check_obs, dict)
                        and check_obs["output"].split()[-1].strip() == '1'
                    ):
                        logger.info(
                            f'[{instance_id}] Evaluation process completed after {seconds_elapsed} seconds'
                        )
                        break
                    logger.info(
                        f'[{instance_id}] [{seconds_elapsed:.0f}s] Evaluation still running, waiting...'
                    )
                    time.sleep(30)  # Wait for 30 seconds before checking again

                # Read the log file
                command=f'cat {log_file}'
                cat_obs = env.execute(command, cwd='/')

                # Grade answer
                if isinstance(cat_obs, dict) and cat_obs["returncode"] == 0:
                    test_output = cat_obs["output"]
                    assert isinstance(test_output, str)
                    # instance['test_result']['test_output'] = test_output

                    # Get report from test output
                    logger.info(f'[{instance_id}] Grading answer...')
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Create a directory structure that matches the expected format
                        # NOTE: this is a hack to make the eval report format consistent
                        # with the original SWE-Bench eval script
                        log_dir = os.path.join(temp_dir, 'logs', instance_id.lower())
                        os.makedirs(log_dir, exist_ok=True)
                        test_output_path = os.path.join(log_dir, 'test_output.txt')
                        with open(test_output_path, 'w') as f:
                            f.write(test_output)
                        try:
                            extra_kwargs = {}
                            extra_kwargs['test_log_path'] = test_output_path
                            
                            if 'swe-smith' in dataset:
                                extra_kwargs['inst'] = instance
                            else:
                                extra_kwargs['test_spec'] = test_spec
                                extra_kwargs['include_tests_status'] = True
                            
                            _report = get_eval_report(
                                prediction={
                                    'model_patch': model_patch,
                                    'instance_id': instance_id,
                                },
                                **extra_kwargs,
                            )
                            # in swe-smith, the report is a single dict
                            # in swe-gym and swe-bench, the report is a dict with instance_id
                            report = _report if 'swe-smith' in dataset else _report[instance_id]
                            logger.info(
                                f"[{instance_id}] report: {report}\nResult for [{instance_id}]: resolved: {report['resolved']}"
                            )
                            return report['resolved'], "NOERROR"
                        except Exception as e:
                            logger.error(
                                f'[{instance_id}] Error when getting eval report: {e}'
                            )
                            return False, f"Error when getting eval report: {e}"
            else:
                raise Exception(f'[{instance_id}] Error when starting eval:\n{obs["output"]}')
        else:
            raise Exception(
                f'[{instance_id}] Unexpected output when applying patch:\n{apply_patch_output}'
            )