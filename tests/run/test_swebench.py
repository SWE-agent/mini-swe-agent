import json
from unittest.mock import patch

import pytest

from microswea import package_dir
from microswea.models.test_models import DeterministicModel
from microswea.run.extra.swebench import filter_instances, get_swebench_docker_image_name, main


@pytest.mark.slow
@pytest.mark.parametrize("n_workers", [1, 2])
def test_swebench_end_to_end(github_test_data, tmp_path, n_workers):
    """Test the complete SWEBench flow using the _test subset with deterministic model"""

    model_responses = github_test_data["model_responses"]

    with patch("microswea.run.extra.swebench.get_model") as mock_get_model:
        mock_get_model.return_value = DeterministicModel(outputs=model_responses)

        main(
            subset="_test",
            split="test",
            slice_spec="0:1",
            output=str(tmp_path),
            n_workers=n_workers,
            filter_spec="swe-agent__test-repo-1",
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
    expected = "swebench/sweb.eval.x86_64.swe-agent_1776_test-repo_1776_1:latest"
    assert get_swebench_docker_image_name(instance) == expected


def test_get_image_name_with_none_image_name():
    """Test get_image_name when image_name is explicitly None"""
    instance = {"image_name": None, "instance_id": "django__django__4.0"}
    expected = "swebench/sweb.eval.x86_64.django_1776_django_1776_4.0:latest"
    assert get_swebench_docker_image_name(instance) == expected


def test_get_image_name_with_complex_instance_id():
    """Test get_image_name with complex instance_id containing multiple double underscores"""
    instance = {"instance_id": "project__sub__module__version__1.2.3"}
    expected = "swebench/sweb.eval.x86_64.project_1776_sub_1776_module_1776_version_1776_1.2.3:latest"
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
