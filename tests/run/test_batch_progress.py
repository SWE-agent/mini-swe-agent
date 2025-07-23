import pytest
import yaml

from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager, _shorten_str


@pytest.fixture
def manager():
    """Create a basic RunBatchProgressManager for testing."""
    return RunBatchProgressManager(num_instances=5)


@pytest.fixture
def manager_with_yaml(tmp_path):
    """Create a RunBatchProgressManager with yaml reporting."""
    yaml_path = tmp_path / "report.yaml"
    return RunBatchProgressManager(num_instances=3, yaml_report_path=yaml_path), yaml_path


@pytest.mark.parametrize(
    ("text", "max_len", "shorten_left", "expected"),
    [
        ("hello", 10, False, "hello     "),
        ("hello world", 8, False, "hello..."),
        ("hello world", 8, True, "...world"),
        ("hello", 5, False, "hello"),
        ("hi", 5, False, "hi   "),
    ],
)
def test_shorten_str(text, max_len, shorten_left, expected):
    assert _shorten_str(text, max_len, shorten_left) == expected


def test_manager_initialization(manager):
    assert manager.n_completed == 0
    assert manager._instances_by_exit_status == {}


def test_manager_with_yaml_path(manager_with_yaml):
    manager, yaml_path = manager_with_yaml
    assert manager._yaml_report_path == yaml_path


def test_instance_lifecycle(manager):
    manager.on_instance_start("task_1")
    assert "task_1" in manager._spinner_tasks
    assert manager.n_completed == 0

    manager.on_instance_end("task_1", "success")
    assert manager.n_completed == 1
    assert manager._instances_by_exit_status["success"] == ["task_1"]


@pytest.mark.parametrize(
    "statuses",
    [
        ["success", "failed", "success", "timeout"],
        ["error", "error", "error"],
        ["success"] * 5,
    ],
)
def test_multiple_instances(manager, statuses):
    for i, status in enumerate(statuses, 1):
        instance_id = f"task_{i}"
        manager.on_instance_start(instance_id)
        manager.on_instance_end(instance_id, status)

    assert manager.n_completed == len(statuses)
    for status in set(statuses):
        expected_count = statuses.count(status)
        assert len(manager._instances_by_exit_status[status]) == expected_count


def test_uncaught_exception(manager):
    manager.on_instance_start("task_1")
    manager.on_uncaught_exception("task_1", ValueError("test error"))

    assert manager.n_completed == 1
    assert "Uncaught ValueError" in manager._instances_by_exit_status


def test_update_instance_status(manager):
    manager.on_instance_start("task_1")
    manager.update_instance_status("task_1", "Processing files...")


def test_yaml_report_generation(manager_with_yaml):
    manager, yaml_path = manager_with_yaml

    manager.on_instance_start("task_1")
    manager.on_instance_end("task_1", "success")
    manager.on_instance_start("task_2")
    manager.on_instance_end("task_2", "failed")

    assert yaml_path.exists()
    data = yaml.safe_load(yaml_path.read_text())
    assert data["instances_by_exit_status"]["success"] == ["task_1"]
    assert data["instances_by_exit_status"]["failed"] == ["task_2"]


def test_get_overview_data(manager):
    manager.on_instance_start("task_1")
    manager.on_instance_end("task_1", "success")

    overview_data = manager._get_overview_data()
    assert overview_data == {"instances_by_exit_status": {"success": ["task_1"]}}


def test_print_report(manager, capsys):
    """Test that print_report produces expected output."""
    manager.on_instance_start("task_1")
    manager.on_instance_end("task_1", "success")
    manager.on_instance_start("task_2")
    manager.on_instance_end("task_2", "failed")

    manager.print_report()

    captured = capsys.readouterr()
    assert "success: 1" in captured.out
    assert "failed: 1" in captured.out
    assert "task_1" in captured.out
    assert "task_2" in captured.out


def test_concurrent_operations(manager):
    """Test handling multiple operations without corruption."""
    instance_ids = [f"task_{i}" for i in range(10)]
    statuses = ["success", "failed", "timeout"] * 4

    for i, instance_id in enumerate(instance_ids):
        manager.on_instance_start(instance_id)
        manager.update_instance_status(instance_id, f"step {i}")
        manager.on_instance_end(instance_id, statuses[i % 3])

    assert manager.n_completed == 10
    assert sum(len(instances) for instances in manager._instances_by_exit_status.values()) == 10
