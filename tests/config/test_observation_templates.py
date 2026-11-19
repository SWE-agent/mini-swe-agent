from pathlib import Path

import pytest
import yaml
from jinja2 import StrictUndefined, Template

CONFIG_DIR = Path(__file__).parents[2] / "src" / "minisweagent" / "config"
OUTPUT = "def f(x: list[int]) -> bool:\n    return x and x[0] < 1 && echo 'ok' # é ✓\n"


@pytest.mark.parametrize(
    "config_path",
    [p for p in sorted(CONFIG_DIR.rglob("*.yaml")) if "observation_template" in p.read_text()],
    ids=lambda p: p.name,
)
def test_observation_template_shows_output_verbatim(config_path: Path):
    template = yaml.safe_load(config_path.read_text())["model"]["observation_template"]
    limits = dict.fromkeys(("step_limit", "n_model_calls", "wall_time_limit_seconds", "elapsed_seconds"), 0)
    output = {"returncode": 0, "output": OUTPUT, "exception_info": ""}
    assert OUTPUT.rstrip() in Template(template, undefined=StrictUndefined).render(output=output, **limits)
