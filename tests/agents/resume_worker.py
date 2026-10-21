"""Subprocess driver for deterministic resume and real process interruption tests."""

import signal
import sys
from pathlib import Path

from minisweagent.agents.extra.resumable import ResumableAgent
from minisweagent.agents.utils.checkpoint import load_checkpoint
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, DeterministicToolcallModel


def continuation(run_dir: Path, **overrides) -> ResumableAgent:
    data, _ = load_checkpoint(run_dir)
    config = data["info"]["config"]
    model_class = DeterministicToolcallModel if "Toolcall" in config["model_type"] else DeterministicModel
    model = model_class(**config["model"])
    # Fixture playback has a cursor; the supported production adapters use messages instead.
    model.current_index = data["info"]["model_stats"]["api_calls"] - 1
    config["agent"].pop("output_path")
    return ResumableAgent(
        model,
        LocalEnvironment(**config["environment"]),
        run_dir=run_dir,
        **(config["agent"] | overrides),
    )


def main():
    path, mode = Path(sys.argv[1]), sys.argv[2]

    def interrupt(frame, event, arg):
        if event == "call" and (
            (mode == "kill-claimed" and frame.f_code.co_name == "_run_loop")
            or (mode == "kill-observation" and frame.f_code.co_name == "format_observation_messages")
        ):
            signal.raise_signal(signal.SIGKILL)
        return interrupt

    if mode.startswith("kill-"):
        sys.settrace(interrupt)
    continuation(path, step_limit=3).resume()


if __name__ == "__main__":
    main()
