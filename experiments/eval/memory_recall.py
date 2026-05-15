"""Memory-quality eval harness: recall@k on synthetic transcripts.

Independent of SWE-bench end-to-end accuracy. Builds short, controllable
conversation histories with *known* gold-relevant node ids, runs the
agent's beam-search retrieval under various phase configurations, and
reports recall@k. This lets us claim "the framework works" with a metric
that isolates retrieval quality from LLM stochasticity.

Run:
    python -m experiments.eval.memory_recall

Outputs a small table to stdout. No GPU / no API keys needed.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from minisweagent import Environment, Model
from minisweagent.agents.memory_search import MemorySearchAgent
from minisweagent.agents.planmem.memory_controller import AdaptiveMemoryController
from minisweagent.agents.planmem.types import PlanningSignal, TaskPhase

# ── Test harness primitives ─────────────────────────────────────────────────


class _StubModel(Model):
    """Just enough Model surface to construct the agent without an API."""

    def __init__(self):
        self.cost = 0.0
        self._n_calls = 0

    @property
    def n_calls(self) -> int:
        return self._n_calls

    def query(self, _messages: list[dict]) -> dict:
        self._n_calls += 1
        return {"role": "assistant", "content": ""}

    def get_template_vars(self) -> dict:
        return {}


class _StubEnv(Environment):
    cwd = "/repo"

    def execute(self, _cmd: str):
        return {"output": "", "returncode": 0}

    def get_template_vars(self) -> dict:
        return {}

    def close(self) -> None:
        pass


@dataclass
class Scenario:
    """A synthetic eval case.

    Builds a graph by appending nodes to ``MemorySearchAgent`` directly,
    bypassing the LLM/env. ``gold_ids`` lists the node ids the retrieval
    is expected to surface; ``query_anchor`` is appended as the final
    user node so beam search treats it as the anchor.
    """

    name: str
    nodes: list[tuple[str, str, dict]] = field(default_factory=list)
    gold_descriptions: list[str] = field(default_factory=list)
    query_anchor: str = ""

    seed_task: str = ""  # default: same as query_anchor

    def build(self, agent: MemorySearchAgent) -> list[int]:
        seed_task = self.seed_task or f"task: {self.query_anchor}"
        agent.add_messages(
            {"role": "system", "content": "system"},
            {"role": "user", "content": seed_task},
        )
        gold_ids: list[int] = []
        # Allow the seeded task description (node 1) to count as gold too.
        if any(g in seed_task for g in self.gold_descriptions):
            gold_ids.append(agent.memory_graph[-1].id)
        for role, content, metadata in self.nodes:
            agent.add_messages({"role": role, "content": content})
            if metadata:
                agent.memory_graph[-1].metadata = metadata
            if any(g in content for g in self.gold_descriptions):
                gold_ids.append(agent.memory_graph[-1].id)
        agent.add_messages({"role": "user", "content": self.query_anchor})
        return gold_ids


# ── Concrete scenarios ──────────────────────────────────────────────────────


def needle_in_haystack() -> Scenario:
    """Critical observation buried under unrelated chatter."""
    nodes = []
    # Distractors first.
    for i in range(20):
        nodes.append(
            (
                "user",
                f"Observation: ran command {i} returncode 0 generic output {i}",
                {"command": f"ls /tmp/{i}"},
            )
        )
    # The needle.
    nodes.append(
        (
            "user",
            "GOLD: connection failure caused by DB_HOST='prod-db-1' in config.py",
            {"command": "cat config.py", "filenames": ["config.py"]},
        )
    )
    for i in range(20, 30):
        nodes.append(
            (
                "user",
                f"Observation: irrelevant step {i}",
                {"command": f"ls /var/{i}"},
            )
        )
    return Scenario(
        name="needle_in_haystack",
        nodes=nodes,
        gold_descriptions=["GOLD: connection failure"],
        query_anchor="Why is the database connection failing in production?",
    )


def file_graph_chain() -> Scenario:
    """Target requires following a Python import edge to reach the cause file."""
    nodes = [
        # main.py imports from config — file-graph signal should help.
        (
            "user",
            "Observation: cat main.py:\nfrom config import DB_HOST\nconnect(DB_HOST)",
            {"command": "cat main.py", "filenames": ["main.py"]},
        ),
    ]
    # Distractors with no overlap to either main.py or config.
    for i in range(15):
        nodes.append(
            (
                "user",
                f"Observation: noise step {i} apple banana cherry",
                {"command": f"date {i}"},
            )
        )
    nodes.append(
        (
            "user",
            "GOLD: cat config.py:\nDB_HOST = 'localhost'  # WRONG — should be prod-db-1",
            {"command": "cat config.py", "filenames": ["config.py"]},
        )
    )
    return Scenario(
        name="file_graph_chain",
        nodes=nodes,
        gold_descriptions=["GOLD: cat config.py"],
        query_anchor="The connection in main.py is failing — investigate.",
    )


def drift_recovery() -> Scenario:
    """Original task is at node 1; recent chatter should not crowd it out."""
    nodes = []
    for i in range(25):
        nodes.append(
            (
                "user",
                f"Observation: tangential exploration {i} of unrelated module zeta",
                {"command": f"ls /unrelated/{i}"},
            )
        )
    return Scenario(
        name="drift_recovery",
        nodes=nodes,
        # Gold = the original task description seeded at node 1.
        gold_descriptions=["GOLD-TASK"],
        seed_task="GOLD-TASK: connection in main.py is failing — fix DB_HOST",
        query_anchor="Need to check zeta module status",
    )


SCENARIOS = [needle_in_haystack, file_graph_chain, drift_recovery]


# ── Recall computation ──────────────────────────────────────────────────────


def recall_at_k(selected_ids: set[int], gold_ids: set[int]) -> float:
    if not gold_ids:
        return 0.0
    return len(selected_ids & gold_ids) / len(gold_ids)


def run_scenario(
    scenario_factory,
    *,
    phase: TaskPhase | None,
    token_budget: int = 4000,
) -> dict:
    agent = MemorySearchAgent(
        model=_StubModel(),
        env=_StubEnv(),
        system_template="",
        instance_template="",
        token_budget=token_budget,
        enable_repo_background_card=False,  # don't pollute system in eval
    )
    scenario = scenario_factory()
    gold_ids = set(scenario.build(agent))

    if phase is None:
        # Default (non-adaptive) retrieval.
        selected = agent.construct_context_via_search()
    else:
        ctrl = AdaptiveMemoryController(base_agent=agent)
        signal = PlanningSignal(current_phase=phase, suggested_token_budget=token_budget)
        selected = ctrl.construct_context(signal, base_agent=agent)

    selected_ids = {n.id for n in selected}
    return {
        "scenario": scenario.name,
        "phase": phase.value if phase is not None else "default",
        "k": len(selected_ids),
        "gold": len(gold_ids),
        "recall@k": recall_at_k(selected_ids, gold_ids),
    }


def main() -> None:
    rows: list[dict] = []
    phases = [None, TaskPhase.EXPLORATION, TaskPhase.IMPLEMENTATION, TaskPhase.BACKTRACK]

    for scenario_fn in SCENARIOS:
        for phase in phases:
            t0 = time.perf_counter()
            row = run_scenario(scenario_fn, phase=phase)
            row["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            rows.append(row)

    # Pretty-print.
    print()
    header = f"{'scenario':<20} {'phase':<15} {'k':>4} {'gold':>5} {'recall@k':>10} {'ms':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['scenario']:<20} {r['phase']:<15} {r['k']:>4} {r['gold']:>5} {r['recall@k']:>10.2f} {r['ms']:>7.1f}",
        )
    overall = statistics.mean(r["recall@k"] for r in rows)
    print()
    print(f"Mean recall@k across {len(rows)} runs: {overall:.3f}")


if __name__ == "__main__":
    main()
