import json
from dataclasses import asdict, dataclass
from typing import Any
from unittest.mock import patch

import pytest

from minisweagent import package_dir
from minisweagent.models.test_models import DeterministicModel
from minisweagent.run.extra.swebench import (
    filter_instances,
    get_swebench_docker_image_name,
    main,
    remove_from_preds_file,
    update_preds_file,
)


@pytest.mark.slow
@pytest.mark.parametrize("workers", [1, 2])
def test_swebench_end_to_end(github_test_data, tmp_path, workers):
    """Test the complete SWEBench flow using the _test subset with deterministic model"""

    model_responses = github_test_data["model_responses"]

    with patch("minisweagent.run.extra.swebench.get_model") as mock_get_model:
        mock_get_model.return_value = DeterministicModel(outputs=model_responses, cost_per_call=0.1)

        main(
            subset="_test",
            split="test",
            slice_spec="0:1",
            output=str(tmp_path),
            workers=workers,
            filter_spec="swe-agent__test-repo-1",
            config_spec=package_dir / "config" / "extra" / "swebench.yaml",
            environment_class="docker",
        )

    traj_file_path = package_dir.parent.parent / "tests" / "test_data" / "github_issue.traj.json"
    trajectory = json.loads(traj_file_path.read_text())

    last_message = trajectory[-1]["content"]

    instance_id = "swe-agent__test-repo-1"
    expected_result = {
        instance_id: {
            "model_name_or_path": "deterministic",
            "instance_id": instance_id,
            "model_patch": last_message,
        }
    }

    with open(tmp_path / "preds.json") as f:
        actual_result = json.load(f)

    assert actual_result == expected_result

    traj_output_file = tmp_path / instance_id / f"{instance_id}.traj.json"
    output_trajectory = json.loads(traj_output_file.read_text())
    assert output_trajectory["messages"][-1]["content"] == last_message


def test_get_image_name_with_existing_image_name():
    """Test get_image_name when image_name is already provided"""
    instance = {"image_name": "custom/image:tag", "instance_id": "test__repo__1"}
    assert get_swebench_docker_image_name(instance) == "custom/image:tag"


def test_get_image_name_without_image_name():
    """Test get_image_name when image_name needs to be constructed"""
    instance = {"instance_id": "swe-agent__test-repo__1"}
    expected = "docker.io/swebench/sweb.eval.x86_64.swe-agent_1776_test-repo_1776_1:latest"
    assert get_swebench_docker_image_name(instance) == expected


def test_get_image_name_with_none_image_name():
    """Test get_image_name when image_name is explicitly None"""
    instance = {"image_name": None, "instance_id": "django__django__4.0"}
    expected = "docker.io/swebench/sweb.eval.x86_64.django_1776_django_1776_4.0:latest"
    assert get_swebench_docker_image_name(instance) == expected


def test_get_image_name_with_complex_instance_id():
    """Test get_image_name with complex instance_id containing multiple double underscores"""
    instance = {"instance_id": "project__sub__module__version__1.2.3"}
    expected = "docker.io/swebench/sweb.eval.x86_64.project_1776_sub_1776_module_1776_version_1776_1.2.3:latest"
    assert get_swebench_docker_image_name(instance) == expected


def test_filter_instances_no_filters():
    """Test filter_instances with no filtering applied"""
    instances = [{"instance_id": "repo1__test1"}, {"instance_id": "repo2__test2"}, {"instance_id": "repo3__test3"}]
    result = filter_instances(instances, filter_spec="", slice_spec="")
    assert result == instances


def test_filter_instances_regex_filter():
    """Test filter_instances with regex filtering"""
    instances = [
        {"instance_id": "django__test1"},
        {"instance_id": "flask__test2"},
        {"instance_id": "django__test3"},
        {"instance_id": "requests__test4"},
    ]
    result = filter_instances(instances, filter_spec=r"django__.*", slice_spec="")
    expected = [{"instance_id": "django__test1"}, {"instance_id": "django__test3"}]
    assert result == expected


def test_filter_instances_slice_only():
    """Test filter_instances with slice specification"""
    instances = [{"instance_id": f"repo{i}__test{i}"} for i in range(10)]
    result = filter_instances(instances, filter_spec="", slice_spec="2:5")
    expected = [{"instance_id": "repo2__test2"}, {"instance_id": "repo3__test3"}, {"instance_id": "repo4__test4"}]
    assert result == expected


def test_filter_instances_slice_start_only():
    """Test filter_instances with slice start only"""
    instances = [{"instance_id": f"repo{i}__test{i}"} for i in range(5)]
    result = filter_instances(instances, filter_spec="", slice_spec="3:")
    expected = [{"instance_id": "repo3__test3"}, {"instance_id": "repo4__test4"}]
    assert result == expected


def test_filter_instances_slice_end_only():
    """Test filter_instances with slice end only"""
    instances = [{"instance_id": f"repo{i}__test{i}"} for i in range(5)]
    result = filter_instances(instances, filter_spec="", slice_spec=":2")
    expected = [{"instance_id": "repo0__test0"}, {"instance_id": "repo1__test1"}]
    assert result == expected


def test_filter_instances_filter_and_slice():
    """Test filter_instances with both filtering and slicing"""
    instances = [
        {"instance_id": "django__test1"},
        {"instance_id": "flask__test2"},
        {"instance_id": "django__test3"},
        {"instance_id": "django__test4"},
        {"instance_id": "requests__test5"},
    ]
    result = filter_instances(instances, filter_spec=r"django__.*", slice_spec="1:3")
    expected = [{"instance_id": "django__test3"}, {"instance_id": "django__test4"}]
    assert result == expected


def test_filter_instances_shuffle():
    """Test filter_instances with shuffle enabled produces deterministic results"""
    instances = [{"instance_id": f"repo{i:02d}__test{i}"} for i in range(10)]
    # Test that shuffle produces same result with same seed
    result1 = filter_instances(instances.copy(), filter_spec="", slice_spec="", shuffle=True)
    result2 = filter_instances(instances.copy(), filter_spec="", slice_spec="", shuffle=True)
    assert result1 == result2
    # Test that shuffled result is different from original order
    result_no_shuffle = filter_instances(instances.copy(), filter_spec="", slice_spec="", shuffle=False)
    assert result1 != result_no_shuffle


def test_filter_instances_empty_list():
    """Test filter_instances with empty input list"""
    result = filter_instances([], filter_spec=r".*", slice_spec="0:5", shuffle=True)
    assert result == []


def test_filter_instances_no_matches():
    """Test filter_instances when regex matches nothing"""
    instances = [{"instance_id": "django__test1"}, {"instance_id": "flask__test2"}]
    result = filter_instances(instances, filter_spec=r"nonexistent__.*", slice_spec="")
    assert result == []


def test_update_preds_file_new_file(tmp_path):
    """Test update_preds_file when output file doesn't exist"""
    output_path = tmp_path / "preds.json"
    update_preds_file(output_path, "test__instance__1", "test_model", "test_result")

    assert output_path.exists()
    result = json.loads(output_path.read_text())
    expected = {
        "test__instance__1": {
            "model_name_or_path": "test_model",
            "instance_id": "test__instance__1",
            "model_patch": "test_result",
        }
    }
    assert result == expected


def test_update_preds_file_existing_file(tmp_path):
    """Test update_preds_file when output file already exists"""
    output_path = tmp_path / "preds.json"

    # Create initial file with one instance
    initial_data = {
        "existing__instance": {
            "model_name_or_path": "old_model",
            "instance_id": "existing__instance",
            "model_patch": "old_result",
        }
    }
    output_path.write_text(json.dumps(initial_data))

    # Add new instance
    update_preds_file(output_path, "new__instance", "new_model", "new_result")

    result = json.loads(output_path.read_text())
    expected = {
        "existing__instance": {
            "model_name_or_path": "old_model",
            "instance_id": "existing__instance",
            "model_patch": "old_result",
        },
        "new__instance": {
            "model_name_or_path": "new_model",
            "instance_id": "new__instance",
            "model_patch": "new_result",
        },
    }
    assert result == expected


def test_update_preds_file_overwrite_existing(tmp_path):
    """Test update_preds_file overwrites existing instance"""
    output_path = tmp_path / "preds.json"

    # Create initial file
    initial_data = {
        "test__instance": {
            "model_name_or_path": "old_model",
            "instance_id": "test__instance",
            "model_patch": "old_result",
        }
    }
    output_path.write_text(json.dumps(initial_data))

    # Update existing instance
    update_preds_file(output_path, "test__instance", "new_model", "new_result")

    result = json.loads(output_path.read_text())
    expected = {
        "test__instance": {
            "model_name_or_path": "new_model",
            "instance_id": "test__instance",
            "model_patch": "new_result",
        }
    }
    assert result == expected


def test_remove_from_preds_file_existing(tmp_path):
    """Test remove_from_preds_file removes existing instance"""
    output_path = tmp_path / "preds.json"

    # Create file with multiple instances
    initial_data = {
        "instance1": {"model_name_or_path": "model1", "instance_id": "instance1", "model_patch": "result1"},
        "instance2": {"model_name_or_path": "model2", "instance_id": "instance2", "model_patch": "result2"},
    }
    output_path.write_text(json.dumps(initial_data))

    # Remove one instance
    remove_from_preds_file(output_path, "instance1")

    result = json.loads(output_path.read_text())
    expected = {"instance2": {"model_name_or_path": "model2", "instance_id": "instance2", "model_patch": "result2"}}
    assert result == expected


def test_remove_from_preds_file_nonexistent_instance(tmp_path):
    """Test remove_from_preds_file with nonexistent instance"""
    output_path = tmp_path / "preds.json"

    initial_data = {"instance1": {"model_name_or_path": "model1", "instance_id": "instance1", "model_patch": "result1"}}
    output_path.write_text(json.dumps(initial_data))

    # Try to remove nonexistent instance
    remove_from_preds_file(output_path, "nonexistent")

    # File should be unchanged
    result = json.loads(output_path.read_text())
    assert result == initial_data


def test_remove_from_preds_file_no_file(tmp_path):
    """Test remove_from_preds_file when file doesn't exist"""
    output_path = tmp_path / "preds.json"

    # Should not raise an error
    remove_from_preds_file(output_path, "any_instance")

    # File should still not exist
    assert not output_path.exists()


@pytest.mark.slow
def test_redo_existing_false_skips_existing(github_test_data, tmp_path):
    """Test that redo_existing=False skips instances that already have results"""
    model_responses = github_test_data["model_responses"]

    # Create existing preds.json with one instance
    preds_file = tmp_path / "preds.json"
    existing_data = {
        "swe-agent__test-repo-1": {
            "model_name_or_path": "previous_model",
            "instance_id": "swe-agent__test-repo-1",
            "model_patch": "previous_result",
        }
    }
    preds_file.write_text(json.dumps(existing_data))

    with patch("minisweagent.run.extra.swebench.get_model") as mock_get_model:
        mock_get_model.return_value = DeterministicModel(outputs=model_responses)

        main(
            subset="_test",
            split="test",
            slice_spec="0:1",
            output=str(tmp_path),
            workers=1,
            filter_spec="swe-agent__test-repo-1",
            redo_existing=False,
            config_spec=package_dir / "config" / "extra" / "swebench.yaml",
        )

    # Should still have the original result
    result = json.loads(preds_file.read_text())
    assert result == existing_data


@pytest.mark.slow
def test_redo_existing_true_overwrites_existing(github_test_data, tmp_path):
    """Test that redo_existing=True processes instances even if they already have results"""
    model_responses = github_test_data["model_responses"]

    # Create existing preds.json with one instance
    preds_file = tmp_path / "preds.json"
    existing_data = {
        "swe-agent__test-repo-1": {
            "model_name_or_path": "previous_model",
            "instance_id": "swe-agent__test-repo-1",
            "model_patch": "previous_result",
        }
    }
    preds_file.write_text(json.dumps(existing_data))

    with patch("minisweagent.run.extra.swebench.get_model") as mock_get_model:
        mock_get_model.return_value = DeterministicModel(outputs=model_responses, cost_per_call=0.1)

        main(
            subset="_test",
            split="test",
            slice_spec="0:1",
            output=str(tmp_path),
            workers=1,
            filter_spec="swe-agent__test-repo-1",
            redo_existing=True,
            config_spec=package_dir / "config" / "extra" / "swebench.yaml",
            environment_class="docker",
        )

    # Should have new result from deterministic model
    traj_file_path = package_dir.parent.parent / "tests" / "test_data" / "github_issue.traj.json"
    trajectory = json.loads(traj_file_path.read_text())
    expected_result = trajectory[-1]["content"]

    result = json.loads(preds_file.read_text())
    assert result["swe-agent__test-repo-1"]["model_patch"] == expected_result
    assert result["swe-agent__test-repo-1"]["model_name_or_path"] == "deterministic"


@dataclass
class ExceptionModelConfig:
    model_name: str = "exception_model"


class ExceptionModel:
    """Test model that raises exceptions during processing."""

    def __init__(self, exception_type: type[Exception] = RuntimeError, exception_message: str = "Test exception"):
        self.exception_type = exception_type
        self.exception_message = exception_message
        self.cost = 0.0
        self.n_calls = 0
        self.config = ExceptionModelConfig()

    def query(self, *args, **kwargs):
        self.n_calls += 1
        raise self.exception_type(self.exception_message)

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}


@pytest.mark.slow
@pytest.mark.parametrize("workers", [1, 2])
def test_exception_handling_in_agent_run(tmp_path, workers):
    """Test that exceptions during agent.run() are properly handled and recorded"""
    with patch("minisweagent.run.extra.swebench.get_model") as mock_get_model:
        mock_get_model.return_value = ExceptionModel(RuntimeError, "Agent processing failed")

        with patch("minisweagent.run.extra.swebench.RunBatchProgressManager") as mock_progress_class:
            mock_progress_manager = mock_progress_class.return_value
            mock_progress_manager.render_group = None  # For Live context manager

            main(
                subset="_test",
                split="test",
                slice_spec="0:1",
                output=str(tmp_path),
                workers=workers,
                filter_spec="swe-agent__test-repo-1",
                config_spec=package_dir / "config" / "extra" / "swebench.yaml",
                environment_class="docker",
            )

    # Check that prediction file contains exception information
    preds_file = tmp_path / "preds.json"
    assert preds_file.exists()

    result = json.loads(preds_file.read_text())
    instance_id = "swe-agent__test-repo-1"
    assert instance_id in result
    assert result[instance_id]["model_patch"] == "Agent processing failed"
    assert result[instance_id]["model_name_or_path"] == "exception_model"

    # Check that trajectory file contains exception information
    traj_file = tmp_path / instance_id / f"{instance_id}.traj.json"
    assert traj_file.exists()

    traj_data = json.loads(traj_file.read_text())
    assert traj_data["instance_id"] == instance_id
    assert traj_data["info"]["exit_status"] == "RuntimeError"
    assert traj_data["info"]["submission"] == "Agent processing failed"


@pytest.mark.slow
@pytest.mark.parametrize("workers", [1, 2])
def test_different_exception_types(tmp_path, workers):
    """Test that different exception types are properly recorded"""
    with patch("minisweagent.run.extra.swebench.get_model") as mock_get_model:
        mock_get_model.return_value = ExceptionModel(ValueError, "Invalid input provided")

        with patch("minisweagent.run.extra.swebench.RunBatchProgressManager") as mock_progress_class:
            mock_progress_manager = mock_progress_class.return_value
            mock_progress_manager.render_group = None  # For Live context manager

            main(
                subset="_test",
                split="test",
                slice_spec="0:1",
                output=str(tmp_path),
                workers=workers,
                filter_spec="swe-agent__test-repo-1",
                config_spec=package_dir / "config" / "extra" / "swebench.yaml",
                environment_class="docker",
            )

    # Check trajectory file for correct exception type
    instance_id = "swe-agent__test-repo-1"
    traj_file = tmp_path / instance_id / f"{instance_id}.traj.json"
    traj_data = json.loads(traj_file.read_text())

    assert traj_data["info"]["exit_status"] == "ValueError"
    assert traj_data["info"]["submission"] == "Invalid input provided"


@pytest.mark.slow
def test_exception_handling_with_progress_manager(tmp_path):
    """Test that progress manager receives exception notifications in multithreaded mode"""
    with patch("minisweagent.run.extra.swebench.get_model") as mock_get_model:
        mock_get_model.return_value = ExceptionModel(ConnectionError, "Network timeout")

        with patch("minisweagent.run.extra.swebench.RunBatchProgressManager") as mock_progress_class:
            mock_progress_manager = mock_progress_class.return_value
            mock_progress_manager.render_group = None  # For Live context manager

            main(
                subset="_test",
                split="test",
                slice_spec="0:1",
                output=str(tmp_path),
                workers=2,  # Use multithreaded to test progress manager
                filter_spec="swe-agent__test-repo-1",
                config_spec=package_dir / "config" / "extra" / "swebench.yaml",
                environment_class="docker",
            )

            # Verify progress manager methods were called
            mock_progress_manager.on_instance_start.assert_called_once_with("swe-agent__test-repo-1")
            mock_progress_manager.on_instance_end.assert_called_once_with("swe-agent__test-repo-1", "ConnectionError")

            # on_uncaught_exception should not be called since exceptions are handled properly
            mock_progress_manager.on_uncaught_exception.assert_not_called()
