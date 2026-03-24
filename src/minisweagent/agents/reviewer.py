"""Reviewer agent for verifying and fixing another agent's work."""

from minisweagent.agents.default import DefaultAgent


class ReviewerAgent(DefaultAgent):
    """Independent reviewer agent.

    Behaviorally identical to DefaultAgent but uses different exit signals
    (REVIEWER_APPROVED / REVIEWER_COMPLETE) configured at the environment layer,
    and is distinguishable by class name in trajectory files.
    """
