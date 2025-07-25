import json
import threading
from pathlib import Path

import pytest

from minisweagent.models import GLOBAL_MODEL_STATS

# Global lock for tests that modify global state - this works across threads
_global_stats_lock = threading.Lock()


@pytest.fixture
def reset_global_stats():
    """Reset global model stats and ensure exclusive access for tests that need it.

    This fixture should be used by any test that depends on global model stats
    to ensure thread safety and test isolation.
    """
    with _global_stats_lock:
        # Reset at start
        GLOBAL_MODEL_STATS._cost = 0.0  # noqa: protected-access
        GLOBAL_MODEL_STATS._n_calls = 0  # noqa: protected-access
        yield
        # Reset at end to clean up
        GLOBAL_MODEL_STATS._cost = 0.0  # noqa: protected-access
        GLOBAL_MODEL_STATS._n_calls = 0  # noqa: protected-access


def get_test_data(trajectory_name: str) -> dict[str, list[str]]:
    """Load test fixtures from a trajectory JSON file"""
    json_path = Path(__file__).parent / "test_data" / f"{trajectory_name}.traj.json"
    with json_path.open() as f:
        trajectory = json.load(f)

    # Extract model responses (assistant messages, starting from index 2)
    model_responses = []
    # Extract expected observations (user messages, starting from index 3)
    expected_observations = []

    for i, message in enumerate(trajectory):
        if i < 2:  # Skip system message (0) and initial user message (1)
            continue

        if message["role"] == "assistant":
            model_responses.append(message["content"])
        elif message["role"] == "user":
            expected_observations.append(message["content"])

    return {"model_responses": model_responses, "expected_observations": expected_observations}


@pytest.fixture
def github_test_data():
    """Load GitHub issue test fixtures"""
    return get_test_data("github_issue")


@pytest.fixture
def local_test_data():
    """Load local test fixtures"""
    return get_test_data("local")
