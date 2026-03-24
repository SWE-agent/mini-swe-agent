"""Multi-agent orchestrator: runs a main agent followed by one or more reviewer passes."""

import copy
import json
import logging
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from minisweagent import Environment
from minisweagent.agents import get_agent
from minisweagent.agents.default import DefaultAgent
from minisweagent.models import get_model


class OrchestratorConfig(BaseModel):
    max_review_iterations: int = 1
    output_path: Path | None = None


class MultiAgentOrchestrator:
    """Runs a main agent to completion, then hands off to a reviewer agent.

    The reviewer receives the original task and inspects the shared environment
    (via git) to verify and optionally fix the main agent's work. Agents share
    a single Docker environment and a /testbed/__CLAUDE.md__ file for factual
    codebase context (no solution reasoning).

    Exit logic:
    - Main agent must exit with "Submitted" (clean confident exit) to trigger review.
    - Reviewer exits with "ReviewerApproved" (no changes needed) or
      "ReviewerComplete" (fixes applied). Either terminates or loops up to
      max_review_iterations.
    - Any LimitsExceeded / exception exit skips further iterations.
    """

    SHARED_CONTEXT_FILE = "/testbed/__CLAUDE.md__"

    def __init__(
        self,
        env: Environment,
        main_agent: DefaultAgent,
        reviewer_model_config: dict,
        reviewer_agent_config: dict,
        on_step: Callable[[str, int, float], None] | None = None,
        **kwargs,
    ):
        self.config = OrchestratorConfig(**kwargs)
        self.env = env
        self.main_agent = main_agent
        self.reviewer_model_config = reviewer_model_config
        self.reviewer_agent_config = reviewer_agent_config
        self.on_step = on_step
        self.logger = logging.getLogger("orchestrator")
        self._agent_data: list[dict] = []

    def _build_reviewer(self) -> DefaultAgent:
        model = get_model(config=copy.deepcopy(self.reviewer_model_config))
        return get_agent(model, self.env, copy.deepcopy(self.reviewer_agent_config), default_type="reviewer")

    def _attach_progress(self, agent: DefaultAgent, label: str) -> DefaultAgent:
        """Wrap agent.step to fire the on_step callback before each step."""
        if self.on_step is None:
            return agent
        original_step = agent.step

        def step_with_progress():
            self.on_step(label, agent.n_calls + 1, agent.cost)
            return original_step()

        agent.step = step_with_progress
        return agent

    def _clear_shared_context(self):
        self.env.execute({"command": f"rm -f {self.SHARED_CONTEXT_FILE} && touch {self.SHARED_CONTEXT_FILE}"})

    def run(self, task: str) -> dict:
        self._clear_shared_context()

        # --- Main agent ---
        self._attach_progress(self.main_agent, "Main")
        main_result = self.main_agent.run(task)
        self._agent_data.append({"role": "main", **self.main_agent.serialize()})

        if main_result.get("exit_status") != "Submitted":
            self.logger.info("Main agent did not submit cleanly; skipping review")
            return self._finalize(
                submission=main_result.get("submission", ""),
                exit_status=main_result.get("exit_status", ""),
            )

        current_submission = main_result.get("submission", "")
        final_exit_status = "Submitted"

        # --- Reviewer iterations ---
        for i in range(self.config.max_review_iterations):
            self.logger.info(f"Review iteration {i + 1}/{self.config.max_review_iterations}")
            reviewer = self._attach_progress(self._build_reviewer(), f"Reviewer {i + 1}")
            reviewer_result = reviewer.run(task)
            self._agent_data.append({"role": "reviewer", "iteration": i, **reviewer.serialize()})

            exit_status = reviewer_result.get("exit_status", "")
            final_exit_status = exit_status

            if exit_status in ("ReviewerApproved", "ReviewerComplete"):
                current_submission = reviewer_result.get("submission", current_submission)

            if exit_status != "ReviewerComplete":
                # ReviewerApproved, limits exceeded, or exception: stop
                break

        return self._finalize(submission=current_submission, exit_status=final_exit_status)

    def _finalize(self, submission: str = "", exit_status: str = "") -> dict:
        """Serialize and save, always called at the end of run()."""
        return self.save(submission=submission, exit_status=exit_status)

    def serialize(self, submission: str = "", exit_status: str = "") -> dict:
        total_cost = sum(a.get("info", {}).get("model_stats", {}).get("instance_cost", 0.0) for a in self._agent_data)
        total_calls = sum(a.get("info", {}).get("model_stats", {}).get("api_calls", 0) for a in self._agent_data)
        n_reviewer_iterations = sum(1 for a in self._agent_data if a.get("role") == "reviewer")
        messages = [message for agent in self._agent_data for message in agent.get("messages", [])]
        return {
            "info": {
                "model_stats": {
                    "instance_cost": total_cost,
                    "api_calls": total_calls,
                },
                "exit_status": exit_status,
                "submission": submission,
                "n_review_iterations": n_reviewer_iterations,
            },
            "agents": self._agent_data,
            "messages": messages,
            "trajectory_format": "mini-swe-agent-multiagent-1.0",
        }

    def save(self, submission: str = "", exit_status: str = "") -> dict:
        data = self.serialize(submission=submission, exit_status=exit_status)
        if self.config.output_path:
            path = self.config.output_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data
