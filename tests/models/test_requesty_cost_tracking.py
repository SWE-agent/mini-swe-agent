import json
import os
import subprocess
import sys

import pytest

from minisweagent.models import get_model
from minisweagent.models.requesty_model import RequestyAPIError, RequestyModel


@pytest.mark.parametrize(
    ("cost_tracking", "response", "expected_cost"),
    [
        ("default", {}, None),
        ("default", {"usage": {}}, None),
        ("default", {"usage": {"cost": 0.0}}, None),
        ("ignore_errors", {}, 0.0),
        ("ignore_errors", {"usage": {}}, 0.0),
        ("ignore_errors", {"usage": {"cost": 0.0}}, 0.0),
        ("default", {"usage": {"cost": 0.125}}, 0.125),
        ("ignore_errors", {"usage": {"cost": 0.125}}, 0.125),
    ],
)
def test_cost_tracking(cost_tracking, response, expected_cost):
    model = RequestyModel(model_name="offline-test", cost_tracking=cost_tracking)
    if expected_cost is None:
        with pytest.raises(RequestyAPIError, match="No cost information"):
            model._calculate_cost(response)
    else:
        assert model._calculate_cost(response) == {"cost": expected_cost}


def test_factory_preserves_explicit_cost_tracking():
    model = get_model("offline-test", {"model_class": "requesty", "cost_tracking": "ignore_errors"})
    assert model.serialize()["info"]["config"]["model"]["cost_tracking"] == "ignore_errors"
    assert model.get_template_vars()["cost_tracking"] == "ignore_errors"
    assert model._calculate_cost({}) == {"cost": 0.0}


@pytest.mark.parametrize(
    ("environment_mode", "explicit_mode", "expected_mode"),
    [
        (None, None, "default"),
        ("ignore_errors", None, "ignore_errors"),
        ("default", None, "default"),
        ("ignore_errors", "default", "default"),
        ("default", "ignore_errors", "ignore_errors"),
    ],
)
def test_environment_default_and_explicit_precedence(environment_mode, explicit_mode, expected_mode, tmp_path):
    script = """
import json
import sys

from minisweagent.models.requesty_model import RequestyAPIError, RequestyModel

kwargs = {"cost_tracking": sys.argv[1]} if sys.argv[1] else {}
model = RequestyModel(model_name="offline-test", **kwargs)
results = []
for response in ({}, {"usage": {}}, {"usage": {"cost": 0.0}}, {"usage": {"cost": 0.125}}):
    try:
        results.append(model._calculate_cost(response))
    except RequestyAPIError:
        results.append("RequestyAPIError")
print(json.dumps({"mode": model.config.model_dump().get("cost_tracking"), "results": results}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, explicit_mode or ""],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": os.defpath,
            "PYTHONPATH": os.pathsep.join(sys.path),
            "MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path),
            "MSWEA_SILENT_STARTUP": "1",
            **({"MSWEA_COST_TRACKING": environment_mode} if environment_mode is not None else {}),
        },
    )
    assert json.loads(result.stdout) == {
        "mode": expected_mode,
        "results": ([{"cost": 0.0}] * 3 if expected_mode == "ignore_errors" else ["RequestyAPIError"] * 3)
        + [{"cost": 0.125}],
    }
