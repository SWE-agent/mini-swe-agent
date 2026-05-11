# Design Spec: PlanMem Integration with mini-swe-agent

**Date:** 2026-03-19
**Status:** Draft
**Author:** Jiahao (with Claude)
**Goal:** Research prototype for NeurIPS 2026 / ICLR 2027 submission

---

## 1. Executive Summary

We integrate the PlanMem architecture (hierarchical planning + adaptive memory co-design) into mini-swe-agent by building a new `PlanMemAgent` class that extends the existing `MemorySearchAgent`. The system adds four modules: (1) a hierarchical planner with goal tracking and phase detection, (2) an adaptive memory controller that replaces fixed beam search parameters, (3) a cross-session persistent memory store, and (4) a joint policy network trainable via behavioral cloning and RL (FoldGRPO on SWE-Gym).

The inheritance chain `DefaultAgent → MemorySearchAgent → PlanMemAgent` directly supports the paper's ablation story.

---

## 2. Scope and Constraints

### In Scope
- New `PlanMemAgent` class with 4 core modules
- SWE-bench Verified and SWE-EVO evaluation
- RL training pipeline (behavioral cloning → FoldGRPO)
- Cross-session persistent memory (JSON/SQLite)
- Full ablation suite (6 experimental conditions)
- New config files for PlanMem variants

### Out of Scope
- Modifying `DefaultAgent` or upstream mini-swe-agent code
- Multi-agent architectures (PlanMem is single-agent)
- Production deployment or UI integration
- Vector embedding-based retrieval (stay with lexical + graph signals for now; can be added later)

### Constraints
- Each new file should stay under 400 lines (per coding-style rules)
- Python 3.10+, type hints required
- Compatible with existing SWE-bench run infrastructure (`mini-extra`, Docker environments)
- All configs via dataclass + YAML (consistent with existing pattern)

---

## 3. Architecture Overview

### 3.1 Inheritance Chain

```
DefaultAgent (agents/default.py, 131 lines)
│   - ReAct loop: query → parse_action → execute → observe
│   - Linear message history
│   - No memory management
│
└── MemorySearchAgent (agents/memory_search.py, 1067 lines)
    │   - memory_graph: List[MemoryNode]
    │   - Beam Search + MMR context selection
    │   - File graph + AST import analysis
    │   - Repo background card
    │
    └── PlanMemAgent (agents/planmem.py, NEW)
        │   - Orchestrates all 4 modules
        │   - Overrides: step(), query(), add_message()
        │
        ├── HierarchicalPlanner (planmem/planner.py, NEW)
        │   - Goal stack management
        │   - Sub-task decomposition
        │   - Phase detection (explore/implement/verify)
        │
        ├── AdaptiveMemoryController (planmem/memory_controller.py, NEW)
        │   - Phase-aware token budget allocation
        │   - Dynamic retrieval weight adjustment
        │   - Retention/eviction policy
        │
        ├── CrossSessionStore (planmem/cross_session.py, NEW)
        │   - Persistent semantic index (SQLite)
        │   - Knowledge distillation after task completion
        │   - Retrieval on task start
        │
        └── PolicyNetwork (planmem/policy.py, NEW)
            - Joint action + memory + planning decisions
            - Behavioral cloning interface
            - FoldGRPO RL training interface
```

### 3.2 File Layout

```
src/minisweagent/
├── agents/
│   ├── default.py              # Unchanged
│   ├── memory_search.py        # Unchanged (or minimal refactor for hook points)
│   ├── planmem.py              # PlanMemAgent orchestrator (~300 lines)
│   └── planmem/
│       ├── __init__.py          # Exports
│       ├── planner.py           # HierarchicalPlanner (~350 lines)
│       ├── memory_controller.py # AdaptiveMemoryController (~300 lines)
│       ├── cross_session.py     # CrossSessionStore (~250 lines)
│       ├── policy.py            # PolicyNetwork interface (~200 lines)
│       └── types.py             # Shared dataclasses (~100 lines)
├── config/
│   ├── planmem.yaml             # Default PlanMem config
│   └── extra/
│       ├── swebench_planmem.yaml         # SWE-bench with full PlanMem
│       ├── swebench_planmem_noplanner.yaml
│       ├── swebench_planmem_nomemory.yaml
│       └── swebench_planmem_static.yaml
├── training/                    # NEW: RL training infrastructure
│   ├── __init__.py
│   ├── trajectory_collector.py  # Collect + annotate trajectories
│   ├── behavioral_cloning.py    # Supervised training
│   ├── fold_grpo.py             # FoldGRPO RL training
│   └── rewards.py               # Process reward definitions
```

---

## 4. Module Designs

### 4.1 Shared Types (`planmem/types.py`)

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskPhase(Enum):
    """Agent's current phase in the task lifecycle."""
    EXPLORATION = "exploration"        # Reading code, understanding structure
    HYPOTHESIS = "hypothesis"          # Forming a theory about the bug/fix
    IMPLEMENTATION = "implementation"  # Writing code, editing files
    VERIFICATION = "verification"      # Running tests, checking fixes
    BACKTRACK = "backtrack"           # Reverting a failed approach


@dataclass(frozen=True)
class SubTask:
    """A decomposed unit of work."""
    id: int
    description: str
    phase: TaskPhase
    parent_id: Optional[int] = None
    estimated_context_tokens: int = 2000  # How much context this subtask needs
    status: str = "pending"  # pending, active, completed, failed


@dataclass(frozen=True)
class PlanningSignal:
    """Output from the planner that guides memory and action."""
    current_phase: TaskPhase
    active_subtask: Optional[SubTask]
    goal_summary: str              # Current high-level goal (for drift detection)
    context_priority_files: list[str] = field(default_factory=list)  # Files the planner considers critical
    suggested_token_budget: int = 16000
    should_backtrack: bool = False


@dataclass
class MemoryAction(Enum):
    """Memory operations the policy can emit."""
    KEEP = "keep"          # Keep node in working memory
    COMPRESS = "compress"  # Compress node content
    EVICT = "evict"        # Remove from working memory (stays in episodic store)
    COMMIT = "commit"      # Save to cross-session store
```

### 4.2 HierarchicalPlanner (`planmem/planner.py`)

**Responsibility:** Maintain an explicit goal stack, decompose tasks into sub-tasks, detect the current phase, and emit `PlanningSignal` to guide the memory controller.

**Key Design Decisions:**

1. **LLM-driven decomposition, not rule-based.** The planner calls the LLM with a specialized prompt to decompose the task and detect phases. This costs extra tokens but is more flexible than heuristics.

2. **Phase detection uses a lightweight classifier.** Rather than a full LLM call per step, we use a regex + heuristic classifier on the agent's recent actions to detect phase transitions (e.g., "agent just ran `sed` → IMPLEMENTATION phase").

3. **Goal drift detection.** At each step, the planner checks whether the agent's recent actions are aligned with the registered goal. If drift is detected, it injects a "goal reminder" into the context.

```python
class HierarchicalPlanner:
    def __init__(self, config: PlannerConfig):
        self.goal_stack: list[SubTask] = []
        self.completed_subtasks: list[SubTask] = []
        self.current_phase: TaskPhase = TaskPhase.EXPLORATION
        self.config = config

    def initialize(self, task_description: str, model: Model) -> PlanningSignal:
        """Called once at task start. Uses LLM to decompose the task."""
        # 1. Call LLM with decomposition prompt
        # 2. Parse sub-tasks from response
        # 3. Push sub-tasks onto goal stack
        # 4. Return initial PlanningSignal
        ...

    def update(self, latest_action: str, observation: str, memory_graph: list) -> PlanningSignal:
        """Called each step. Detects phase, checks goal alignment, manages sub-tasks."""
        # 1. Detect phase from action/observation patterns
        new_phase = self._detect_phase(latest_action, observation)

        # 2. Check if current sub-task is complete
        if self._subtask_completed(observation):
            self._complete_current_subtask()

        # 3. Check for goal drift
        if self._detect_goal_drift(memory_graph):
            # Will trigger goal reminder injection

        # 4. Emit planning signal
        return PlanningSignal(
            current_phase=new_phase,
            active_subtask=self.goal_stack[-1] if self.goal_stack else None,
            goal_summary=self._format_goal_summary(),
            context_priority_files=self._get_priority_files(),
            suggested_token_budget=self._phase_token_budget(new_phase),
            should_backtrack=self._should_backtrack(observation),
        )

    def _detect_phase(self, action: str, observation: str) -> TaskPhase:
        """Heuristic phase detection based on action patterns."""
        # grep/find/cat/ls → EXPLORATION
        # sed/cat<<EOF/apply_patch → IMPLEMENTATION
        # python/pytest/test → VERIFICATION
        # Repeated failures → BACKTRACK
        ...

    def _detect_goal_drift(self, memory_graph: list) -> bool:
        """Check if recent actions diverge from the registered goal."""
        # Compare recent action keywords with goal keywords
        # If overlap drops below threshold → drift detected
        ...

    def _phase_token_budget(self, phase: TaskPhase) -> int:
        """Different phases need different context sizes."""
        budgets = {
            TaskPhase.EXPLORATION: self.config.exploration_budget,     # Large: need broad context
            TaskPhase.HYPOTHESIS: self.config.hypothesis_budget,       # Medium: focused analysis
            TaskPhase.IMPLEMENTATION: self.config.implementation_budget, # Small: focused on target files
            TaskPhase.VERIFICATION: self.config.verification_budget,    # Medium: need test context
            TaskPhase.BACKTRACK: self.config.backtrack_budget,          # Large: need to review what went wrong
        }
        return budgets.get(phase, self.config.default_budget)
```

**Phase Detection Heuristics:**

| Pattern | Detected Phase |
|---------|---------------|
| `find`, `grep`, `ls`, `cat`, `head`, `tree` | EXPLORATION |
| Keywords: "understand", "look at", "check" in THOUGHT | EXPLORATION |
| `sed`, `cat <<EOF`, `apply_patch`, `echo >` | IMPLEMENTATION |
| `python`, `pytest`, `./test`, `make test` | VERIFICATION |
| Same file edited 3+ times, test failure after edit | BACKTRACK |
| Keywords: "submit", "COMPLETE_TASK" | (terminal) |

### 4.3 AdaptiveMemoryController (`planmem/memory_controller.py`)

**Responsibility:** Replace MemorySearchAgent's fixed `construct_context_via_search()` with a phase-aware, adaptive version.

**What it changes from MemorySearchAgent's beam search:**

| Parameter | MemorySearchAgent (Fixed) | AdaptiveMemoryController (Phase-Aware) |
|-----------|--------------------------|---------------------------------------|
| `token_budget` | 16000 always | EXPLORATION: 20000, IMPLEMENTATION: 12000, VERIFICATION: 16000 |
| `diversity_lambda` | 0.7 always | EXPLORATION: 0.5 (more diverse), IMPLEMENTATION: 0.9 (more focused) |
| `w_content` / `w_graph` | 0.5 / 0.5 | EXPLORATION: 0.7/0.3, IMPLEMENTATION: 0.3/0.7 |
| `n_recent_fixed` | 6 always | EXPLORATION: 4, IMPLEMENTATION: 8, VERIFICATION: 6 |
| Priority files | None | Planner's `context_priority_files` get boosted relevance |
| Goal reminder | None | Injected when drift detected |

```python
class AdaptiveMemoryController:
    def __init__(self, config: MemoryControllerConfig, base_agent: MemorySearchAgent):
        self.config = config
        self.base_agent = base_agent  # Access to memory_graph, scoring methods

    def construct_context(self, planning_signal: PlanningSignal) -> list[MemoryNode]:
        """Phase-aware context construction."""
        # 1. Apply phase-specific parameters
        params = self._get_phase_params(planning_signal.current_phase)

        # 2. Boost priority files from planner
        file_boost = self._build_file_boost(planning_signal.context_priority_files)

        # 3. Run modified beam search with adaptive params
        selected = self._adaptive_beam_search(
            token_budget=planning_signal.suggested_token_budget,
            diversity_lambda=params.diversity_lambda,
            w_content=params.w_content,
            w_graph=params.w_graph,
            n_recent=params.n_recent,
            file_boost=file_boost,
        )

        # 4. Inject goal reminder if drift detected
        if planning_signal.should_backtrack or self._goal_needs_reminder(planning_signal):
            selected = self._inject_goal_reminder(selected, planning_signal.goal_summary)

        return selected

    def _adaptive_beam_search(self, *, token_budget, diversity_lambda,
                               w_content, w_graph, n_recent, file_boost) -> list[MemoryNode]:
        """Modified version of MemorySearchAgent.construct_context_via_search()
        that accepts dynamic parameters instead of reading from fixed config."""
        # Core logic reused from MemorySearchAgent, but parameterized
        ...

    def _inject_goal_reminder(self, nodes: list[MemoryNode], goal_summary: str) -> list[MemoryNode]:
        """Insert a synthetic node reminding the agent of its goal."""
        reminder = MemoryNode(
            id=-1,  # Synthetic
            role="system",
            content=f"GOAL REMINDER: {goal_summary}",
            timestamp=time.time(),
        )
        # Insert after system prompt, before other nodes
        return [nodes[0], reminder] + nodes[1:]
```

**Learned Policy (Phase 2):** In the RL phase, `_get_phase_params()` is replaced by the PolicyNetwork, which outputs continuous parameters (token_budget, lambda, weights) conditioned on the current state (phase, recent actions, memory statistics).

### 4.4 CrossSessionStore (`planmem/cross_session.py`)

**Responsibility:** Persist repository-specific knowledge across tasks. Load relevant prior knowledge at task start.

**Storage Schema (SQLite):**

```sql
CREATE TABLE knowledge (
    id INTEGER PRIMARY KEY,
    repo_name TEXT NOT NULL,
    knowledge_type TEXT NOT NULL,  -- 'architecture', 'pattern', 'bug_class', 'fix_strategy'
    content TEXT NOT NULL,
    source_task TEXT,              -- Which task produced this knowledge
    confidence REAL DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    created_at REAL,
    last_accessed REAL,
    decay_factor REAL DEFAULT 1.0  -- Ebbinghaus-style decay
);

CREATE INDEX idx_repo ON knowledge(repo_name);
CREATE INDEX idx_type ON knowledge(knowledge_type, repo_name);
```

**Lifecycle:**

```
Task Start                          Task End
    │                                   │
    ▼                                   ▼
load_for_repo(repo_name)          distill_and_store(trajectory, outcome)
    │                                   │
    ▼                                   ▼
Inject into system prompt          LLM summarizes successful strategies
or repo background card            → INSERT into knowledge table
                                   Failed strategies → decay confidence
```

```python
class CrossSessionStore:
    def __init__(self, db_path: str = "~/.minisweagent/cross_session.db"):
        self.db = sqlite3.connect(Path(db_path).expanduser())
        self._ensure_tables()

    def load_for_repo(self, repo_name: str, max_entries: int = 10) -> str:
        """Load relevant knowledge for a repo, return as text block."""
        # Query by repo_name, order by (confidence * decay_factor * recency)
        # Apply Ebbinghaus decay: decay_factor *= exp(-lambda * time_since_last_access)
        # Return formatted text block for injection into system prompt
        ...

    def distill_and_store(self, trajectory: list[dict], outcome: str,
                          repo_name: str, model: Model):
        """After task completion, use LLM to extract reusable knowledge."""
        # 1. If outcome == "Submitted" (success):
        #    - Ask LLM to summarize: key files, architecture patterns, fix strategy
        #    - Store as knowledge entries
        # 2. If outcome == "LimitsExceeded" (failure):
        #    - Ask LLM to summarize: what went wrong, dead ends to avoid
        #    - Store with lower confidence
        ...

    def update_decay(self):
        """Periodic decay of unused knowledge (Ebbinghaus curve)."""
        # decay_factor = decay_factor * exp(-0.1 * days_since_last_access)
        # Delete entries where decay_factor < 0.05
        ...
```

**Integration with Repo Background Card:** The loaded cross-session knowledge is appended to `MemorySearchAgent`'s repo background card:

```
<!-- REPO_BACKGROUND_CARD:BEGIN -->
Repo background card v3
...existing layout...

Prior session knowledge (3 entries):
- Architecture: Django app with custom middleware chain in myapp/middleware/
- Known pattern: Date parsing bugs usually in utils/dateparse.py, use dateutil
- Avoid: Do not modify __init__.py directly, use the factory pattern
<!-- REPO_BACKGROUND_CARD:END -->
```

### 4.5 PolicyNetwork (`planmem/policy.py`)

**Responsibility:** Unified interface for decision-making that can transition from heuristic → behavioral cloning → RL.

**Three Modes:**

| Mode | Implementation | When |
|------|---------------|------|
| `heuristic` | Rule-based phase detection + fixed parameter tables | Initial prototype |
| `bc` | Neural network trained on annotated expert trajectories | Phase 1 training |
| `rl` | Same network fine-tuned with FoldGRPO | Phase 2 training |

```python
@dataclass
class PolicyOutput:
    """What the policy decides at each step."""
    # Planning
    detected_phase: TaskPhase
    should_decompose: bool       # Break current subtask further?
    should_backtrack: bool       # Revert current approach?

    # Memory
    token_budget: int
    diversity_lambda: float
    w_content: float
    w_graph: float
    n_recent: int
    priority_files: list[str]

    # Action (optional, for full joint policy)
    action_prefix: str = ""      # Optional hint prepended to LLM prompt


class PolicyNetwork:
    def __init__(self, mode: str = "heuristic", model_path: str = ""):
        self.mode = mode
        if mode in ("bc", "rl") and model_path:
            self.network = self._load_network(model_path)

    def decide(self, state: PolicyState) -> PolicyOutput:
        if self.mode == "heuristic":
            return self._heuristic_policy(state)
        else:
            return self._neural_policy(state)

    def _heuristic_policy(self, state: PolicyState) -> PolicyOutput:
        """Rule-based: maps phase → fixed parameter set."""
        phase = self._detect_phase_heuristic(state)
        params = PHASE_PARAM_TABLE[phase]
        return PolicyOutput(detected_phase=phase, **params)

    def _neural_policy(self, state: PolicyState) -> PolicyOutput:
        """Neural: encodes state → continuous outputs."""
        features = self._encode_state(state)
        raw_output = self.network(features)
        return self._decode_output(raw_output)
```

**PolicyState (input features for neural policy):**

```python
@dataclass
class PolicyState:
    step_number: int
    total_cost: float
    n_memory_nodes: int
    recent_actions: list[str]        # Last 5 action strings
    recent_return_codes: list[int]   # Last 5 return codes
    current_file_count: int          # Files touched so far
    goal_keyword_overlap: float      # Overlap between recent actions and goal
    phase_history: list[TaskPhase]   # Last 10 phases
    memory_utilization: float        # current_chars / budget
```

---

## 5. PlanMemAgent Orchestrator (`agents/planmem.py`)

The orchestrator wires the four modules together by overriding key methods from `MemorySearchAgent`:

```python
class PlanMemAgent(MemorySearchAgent):
    def __init__(self, model, env, *, config_class=PlanMemConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.planner = HierarchicalPlanner(self.config.planner)
        self.memory_controller = AdaptiveMemoryController(self.config.memory, self)
        self.cross_session = CrossSessionStore(self.config.cross_session_db)
        self.policy = PolicyNetwork(
            mode=self.config.policy_mode,
            model_path=self.config.policy_model_path,
        )
        self._planning_signal: Optional[PlanningSignal] = None

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Override run to add initialization and teardown."""
        # 1. Load cross-session knowledge
        repo_name = self._infer_repo_name()
        prior_knowledge = self.cross_session.load_for_repo(repo_name)
        if prior_knowledge:
            self.extra_template_vars["prior_knowledge"] = prior_knowledge

        # 2. Initialize planner with task decomposition
        self.extra_template_vars |= {"task": task, **kwargs}
        self._planning_signal = self.planner.initialize(task, self.model)

        # 3. Run the main loop (inherited)
        try:
            result = super().run(task, **kwargs)
        finally:
            # 4. Distill and store knowledge after task completion
            self.cross_session.distill_and_store(
                self.messages, result[0] if result else "unknown",
                repo_name, self.model,
            )
        return result

    def query(self) -> dict:
        """Override to use adaptive memory controller instead of fixed beam search."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()

        self._ensure_repo_background_card()

        # 1. Get policy decision based on current state
        state = self._build_policy_state()
        policy_output = self.policy.decide(state)

        # 2. Update planner
        if len(self.memory_graph) > 1:
            last_action = self.memory_graph[-1].metadata.get("command", "")
            last_obs = self.memory_graph[-1].content
            self._planning_signal = self.planner.update(last_action, last_obs, self.memory_graph)

        # 3. Merge policy output with planning signal
        merged_signal = self._merge_signals(policy_output, self._planning_signal)

        # 4. Use adaptive memory controller to build context
        selected_nodes = self.memory_controller.construct_context(merged_signal)
        selected_nodes.sort(key=lambda n: n.id)

        # 5. Query LLM with selected context (same swap trick as MemorySearchAgent)
        original_messages = self.messages
        max_chars = self._max_node_chars()
        search_context_messages = [
            {"role": n.role, "content": self._compress_content(
                getattr(n, "raw_content", n.content), max_chars
            )}
            for n in selected_nodes
        ]

        self.messages = search_context_messages
        try:
            response = self.model.query(self.messages)
        finally:
            self.messages = original_messages

        self.add_message("assistant", **response)
        return response
```

---

## 6. Training Pipeline

### 6.1 Trajectory Collection (`training/trajectory_collector.py`)

```
Strong baselines (GCC, OpenHands)
        │
        ▼
Run on SWE-bench Verified (500 instances)
        │
        ▼
Collect raw trajectories (messages + actions + observations)
        │
        ▼
Auto-annotate with heuristic labels:
  - Phase per step (via regex classifier)
  - Sub-task boundaries (via action pattern changes)
  - Memory operations (which nodes were useful, measured by outcome)
        │
        ▼
Annotated trajectory dataset (JSON)
```

### 6.2 Behavioral Cloning (`training/behavioral_cloning.py`)

Train `PolicyNetwork` to predict:
- `PolicyOutput.detected_phase` (classification)
- `PolicyOutput.token_budget` (regression)
- `PolicyOutput.diversity_lambda` (regression)
- `PolicyOutput.w_content / w_graph` (regression)
- `PolicyOutput.should_backtrack` (binary classification)

**Loss:** Multi-task loss = CrossEntropy(phase) + MSE(continuous params) + BCE(backtrack)

### 6.3 FoldGRPO (`training/fold_grpo.py`)

**Environment:** SWE-Gym (real GitHub issues with test suites)

**Process Rewards (`training/rewards.py`):**

| Reward Signal | Weight | Description |
|---------------|--------|-------------|
| `r_resolve` | 1.0 | +1 if issue resolved (test passes) |
| `r_cost` | 0.2 | -cost_per_step / max_cost (efficiency) |
| `r_subtask` | 0.3 | +0.1 per sub-task completed |
| `r_context_util` | 0.2 | (relevant_tokens / total_tokens) per step |
| `r_goal_align` | 0.1 | keyword overlap between actions and goal |
| `r_no_drift` | 0.2 | -1 if agent submits without addressing the original issue |

**Training Loop:**
1. Sample batch of SWE-Gym tasks
2. Run PlanMemAgent with current policy
3. Compute per-step process rewards
4. Update policy with GRPO (group relative policy optimization)

---

## 7. Configuration

### 7.1 PlanMemConfig

```python
@dataclass
class PlannerConfig:
    decomposition_model: str = ""   # If empty, use main model
    max_subtasks: int = 8
    drift_threshold: float = 0.3    # Below this keyword overlap → drift
    exploration_budget: int = 20000
    hypothesis_budget: int = 16000
    implementation_budget: int = 12000
    verification_budget: int = 16000
    backtrack_budget: int = 20000
    default_budget: int = 16000


@dataclass
class MemoryControllerConfig:
    # Phase-specific parameter tables
    phase_params: dict = field(default_factory=lambda: {
        "exploration":    {"diversity_lambda": 0.5, "w_content": 0.7, "w_graph": 0.3, "n_recent": 4},
        "hypothesis":     {"diversity_lambda": 0.6, "w_content": 0.5, "w_graph": 0.5, "n_recent": 6},
        "implementation": {"diversity_lambda": 0.9, "w_content": 0.3, "w_graph": 0.7, "n_recent": 8},
        "verification":   {"diversity_lambda": 0.7, "w_content": 0.5, "w_graph": 0.5, "n_recent": 6},
        "backtrack":      {"diversity_lambda": 0.4, "w_content": 0.6, "w_graph": 0.4, "n_recent": 4},
    })
    file_boost_factor: float = 1.5  # Multiplier for planner-prioritized files
    goal_reminder_interval: int = 5  # Steps between goal reminders (if drift)


@dataclass
class PlanMemConfig(MemorySearchConfig):
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    memory: MemoryControllerConfig = field(default_factory=MemoryControllerConfig)
    cross_session_db: str = "~/.minisweagent/cross_session.db"
    policy_mode: str = "heuristic"  # "heuristic", "bc", "rl"
    policy_model_path: str = ""
```

### 7.2 SWE-bench Config Example (`config/extra/swebench_planmem.yaml`)

```yaml
agent:
  agent_class: "minisweagent.agents.planmem.PlanMemAgent"
  # Inherits all MemorySearchConfig defaults, plus:
  planner:
    max_subtasks: 8
    drift_threshold: 0.3
  memory:
    phase_params:
      exploration:
        diversity_lambda: 0.5
        w_content: 0.7
        w_graph: 0.3
        n_recent: 4
      implementation:
        diversity_lambda: 0.9
        w_content: 0.3
        w_graph: 0.7
        n_recent: 8
  policy_mode: "heuristic"
  cross_session_db: "~/.minisweagent/cross_session.db"
  step_limit: 250
  cost_limit: 3.0

environment:
  cwd: "/testbed"
  timeout: 60
  environment_class: docker

model:
  model_name: "anthropic/claude-sonnet-4-5-20250929"
  model_kwargs:
    drop_params: true
    temperature: 0.0
```

---

## 8. Ablation Matrix

| Condition | Planner | Adaptive Memory | Cross-Session | Policy | Config |
|-----------|---------|----------------|---------------|--------|--------|
| `DefaultAgent` | - | - | - | - | `swebench.yaml` |
| `MemorySearchAgent` | - | Fixed beam search | - | - | `swebench_memsearch.yaml` |
| `PlanMem-NoPlanner` | - | Adaptive | Cross-session | Heuristic | `swebench_planmem_noplanner.yaml` |
| `PlanMem-NoMemory` | Hierarchical | Fixed beam search | - | Heuristic | `swebench_planmem_nomemory.yaml` |
| `PlanMem-Static` | Hierarchical | Fixed params | Cross-session | Heuristic | `swebench_planmem_static.yaml` |
| `PlanMem-NoCross` | Hierarchical | Adaptive | - | Heuristic | `swebench_planmem_nocross.yaml` |
| `PlanMem-Heuristic` | Hierarchical | Adaptive | Cross-session | Heuristic | `swebench_planmem.yaml` |
| `PlanMem-BC` | Hierarchical | Adaptive | Cross-session | BC | `swebench_planmem_bc.yaml` |
| `PlanMem-RL` | Hierarchical | Adaptive | Cross-session | RL | `swebench_planmem_rl.yaml` |

---

## 9. Evaluation Plan

### 9.1 Benchmarks

| Benchmark | Instances | Purpose |
|-----------|-----------|---------|
| SWE-bench Verified | 500 | Primary: resolve rate + cost |
| SWE-bench Lite | 300 | Efficiency comparison at fixed budget |
| SWE-EVO | TBD | Long-horizon multi-step tasks |

### 9.2 Metrics

| Metric | How Measured |
|--------|-------------|
| Resolve rate | Test pass rate on SWE-bench |
| Cost per instance | Total API cost (tokens * price) |
| Context utilization | Retrospective: fraction of context tokens that appear in successful patches |
| Planning efficiency | Completed subtasks / attempted subtasks |
| Goal retention | Average keyword overlap between actions and goal over trajectory |
| Recovery rate | Fraction of backtrack events that lead to eventual resolution |
| Phase detection accuracy | Manual annotation of 50 trajectories → compare with detected phases |

---

## 10. Implementation Phases

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| **Phase 1: Core Architecture** | Weeks 1-3 | `PlanMemAgent`, `HierarchicalPlanner`, `AdaptiveMemoryController`, heuristic policy. Run on SWE-bench Lite. |
| **Phase 2: Cross-Session** | Weeks 3-4 | `CrossSessionStore`, integration with repo background card. Run repeated tasks on same repos. |
| **Phase 3: Trajectory Collection** | Weeks 4-5 | Collect GCC/OpenHands trajectories, auto-annotate, build training dataset. |
| **Phase 4: Behavioral Cloning** | Weeks 5-6 | Train `PolicyNetwork` in BC mode. Evaluate on SWE-bench Verified. |
| **Phase 5: RL Training** | Weeks 6-8 | FoldGRPO training on SWE-Gym. Full ablation suite. |
| **Phase 6: Evaluation** | Weeks 8-9 | Full benchmark runs, diagnostic analysis, paper figures. |
| **Phase 7: Writing** | Weeks 9-12 | Paper targeting NeurIPS 2026 or ICLR 2027. |

---

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Planner LLM calls add too much cost | High | Use heuristic phase detection (free) for most steps; LLM decomposition only at task start |
| Adaptive params don't beat fixed beam search (Complexity Trap) | High | Include `PlanMem-Static` ablation to quantify; design graceful degradation |
| Cross-session memory injects stale/wrong knowledge | Medium | Ebbinghaus decay + confidence thresholds + validation query before injection |
| RL training doesn't converge | Medium | Start with BC (guaranteed to match baseline); GRPO is conservative |
| MemorySearchAgent refactoring needed for hook points | Low | Minimal: only need to make `construct_context_via_search()` overridable |
| Phase detection heuristics are too noisy | Medium | Measure accuracy on manually annotated set; fall back to "EXPLORATION" when uncertain |
