# Research Proposal: Adaptive Planning-Memory Co-Design for Long-Horizon Coding Agents

**Date:** 2026-03-19
**Based on:** Literature review of ~45 papers on planning and memory/context engineering for LLM coding agents

---

## 1. Problem Statement

LLM-based coding agents have achieved impressive results on benchmarks like SWE-bench, but they still fail systematically on long-horizon, multi-file tasks that require sustained reasoning across many steps. Our literature review identifies a critical gap: **planning systems and memory/context management systems are designed independently**, leading to suboptimal interactions:

- Planning systems generate sub-goals without awareness of what the memory system can retain
- Memory systems compress or discard context without awareness of which information the planner needs
- Neither system adapts its strategy based on task phase (exploration vs. exploitation) or difficulty

The most successful system to date—GCC (Git Context Controller)—achieves SOTA by implicitly coupling planning (BRANCH/MERGE) with memory (COMMIT/CONTEXT), but this coupling is structural rather than learned. We propose to make this coupling **explicit, adaptive, and learnable**.

---

## 2. Research Questions

**RQ1**: Can a co-designed planning-memory architecture outperform independently designed planning and memory systems on long-horizon coding tasks?

**RQ2**: Can reinforcement learning train an agent to jointly optimize planning decisions and context management policies?

**RQ3**: How does cross-session episodic memory (accumulated repository knowledge) affect planning efficiency on recurring task types?

---

## 3. Proposed Approach: PlanMem

We propose **PlanMem**, an architecture that co-designs hierarchical planning with adaptive context management for coding agents.

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    PlanMem Agent                     │
│                                                      │
│  ┌──────────────┐    ┌──────────────────────────┐   │
│  │   Planner    │◄──►│   Memory Controller      │   │
│  │              │    │                          │   │
│  │ - Goal Stack │    │ - Working Memory (WM)    │   │
│  │ - Sub-task   │    │ - Episodic Store (ES)    │   │
│  │   Decomposer │    │ - Semantic Index (SI)    │   │
│  │ - Phase      │    │ - Context Budget Alloc.  │   │
│  │   Detector   │    │ - Retention Policy       │   │
│  └──────┬───────┘    └──────────┬───────────────┘   │
│         │                       │                    │
│         ▼                       ▼                    │
│  ┌──────────────────────────────────────────────┐   │
│  │          Joint Policy Network (π)             │   │
│  │   Decides: what to do AND what to remember    │   │
│  └──────────────────────────────────────────────┘   │
│                        │                             │
│                        ▼                             │
│  ┌──────────────────────────────────────────────┐   │
│  │           Code Environment (Sandbox)          │   │
│  │   Repository + Shell + Test Suite + Git       │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 3.2 Key Components

#### A. Hierarchical Goal-Aware Planner

The planner maintains an explicit goal stack with sub-task decomposition:

1. **Goal Registration**: Each high-level goal (e.g., "fix issue #123") is registered with required context (issue description, relevant files, test expectations)
2. **Sub-task Decomposition**: Goals are decomposed into sub-tasks (localize fault → understand code → generate patch → verify fix), each with estimated context requirements
3. **Phase Detection**: The planner detects which phase the agent is in (exploration, hypothesis formation, implementation, verification) and signals the memory controller to adjust retention policies accordingly

#### B. Adaptive Memory Controller

The memory controller manages three tiers:

1. **Working Memory (WM)**: The active context window, managed via a learned retention policy that decides what to keep, compress, or evict at each step
2. **Episodic Store (ES)**: A structured log of past actions, observations, and reasoning, indexed by task phase and file path. Uses selective retrieval rather than full replay
3. **Semantic Index (SI)**: A persistent representation of repository structure, code semantics, and accumulated knowledge (similar to RepoMap but dynamically updated)

#### C. Joint Policy Network

Instead of separate policies for action selection and context management, PlanMem uses a **joint policy** that outputs:
- **Action**: What tool to call next (edit file, run test, search code, etc.)
- **Memory operation**: What to commit to episodic store, what to evict from working memory, what to retrieve from semantic index
- **Planning signal**: Whether to decompose current sub-task further, backtrack, or declare sub-task complete

### 3.3 Training Approach

We propose a two-phase training strategy:

**Phase 1: Behavioral Cloning from Expert Trajectories**
- Collect successful trajectories from strong baselines (GCC-equipped agents, OpenHands)
- Annotate trajectories with implicit planning decisions and context management operations
- Train the joint policy via supervised learning on annotated trajectories

**Phase 2: Reinforcement Learning with Process Rewards**
- Use SWE-Gym as the training environment
- Define process rewards for:
  - Planning quality: sub-task completion rate, backtrack frequency
  - Memory efficiency: context utilization ratio (relevant tokens / total tokens)
  - Overall: issue resolution + cost efficiency
- Train with a variant of GRPO (similar to FoldGRPO from Context-Folding)

### 3.4 Cross-Session Memory

PlanMem includes a **persistent memory layer** inspired by Lore and GCC:

- After each task, successful strategies and repository-specific knowledge are distilled into the semantic index
- On new tasks, the planner queries the semantic index for relevant prior experience
- A forgetting mechanism (inspired by MemoryBank's Ebbinghaus curve) prevents stale knowledge from accumulating

---

## 4. Experimental Design

### 4.1 Benchmarks

| Benchmark | Purpose | Metric |
|-----------|---------|--------|
| SWE-bench Verified (500 instances) | Standard coding agent evaluation | Resolve rate, cost/instance |
| SWE-EVO | Long-horizon software evolution | Multi-step resolution rate |
| SWE-bench Lite | Efficiency comparison | Resolve rate at fixed budget |

### 4.2 Baselines

1. **ReAct** (no structured planning or memory)
2. **ReAct + Observation Masking** (simple context management, strong baseline per Complexity Trap)
3. **GCC** (structured context management, current SOTA)
4. **OpenHands CodeAct** (free-form planning)
5. **MoatlessTools** (workflow-constrained planning)
6. **Context-Folding** (learned context management, no explicit planning)

### 4.3 Ablation Studies

- **PlanMem-NoPlanner**: Memory controller only (context management without hierarchical planning)
- **PlanMem-NoMemory**: Planner only (hierarchical planning without adaptive context management)
- **PlanMem-Static**: Joint architecture with fixed (non-learned) policies
- **PlanMem-NoCrossSession**: No persistent memory across tasks

### 4.4 Diagnostic Metrics

Beyond resolve rate, we measure:
- **Context utilization**: Fraction of context tokens that are relevant to the current sub-task (measured by retrospective analysis)
- **Planning efficiency**: Number of sub-tasks completed / total sub-tasks attempted
- **Goal retention**: Whether the agent's actions remain aligned with the original goal over time (measuring context drift)
- **Recovery rate**: How often the agent successfully recovers from failed sub-tasks via backtracking

---

## 5. Expected Contributions

1. **Architectural**: First co-designed planning-memory architecture for coding agents, demonstrating that joint optimization outperforms independent optimization
2. **Empirical**: Comprehensive ablation study quantifying the individual and combined effects of planning and memory on coding agent performance
3. **Training methodology**: RL-based training framework for joint planning and context management policies
4. **Diagnostic toolkit**: New metrics and benchmarks for evaluating planning quality and memory effectiveness in coding agents separately

---

## 6. Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| Phase 1: Foundation | Months 1-2 | Implement PlanMem architecture, set up SWE-Gym training environment |
| Phase 2: Behavioral Cloning | Months 2-3 | Collect expert trajectories, annotate, train initial policy |
| Phase 3: RL Training | Months 3-5 | Design process rewards, train joint policy with GRPO |
| Phase 4: Evaluation | Months 5-6 | Run benchmarks, ablation studies, diagnostic analysis |
| Phase 5: Writing | Months 6-7 | Paper writing, targeting NeurIPS 2026 or ICLR 2027 |

---

## 7. Target Venues

- **Primary**: NeurIPS 2026, ICLR 2027
- **Secondary**: ICML 2027, EMNLP 2026
- **Workshop**: MemAgents @ ICLR 2026 (Workshop on Memory for LLM-Based Agentic Systems)

---

## 8. Related Work Positioning

PlanMem differentiates from:
- **GCC**: GCC provides structural coupling between planning and memory via git semantics; PlanMem provides learned, adaptive coupling via a joint policy network
- **Context-Folding**: Focuses on sub-task context management but lacks explicit hierarchical planning and cross-session memory
- **SagaLLM**: Provides transactional guarantees for multi-agent systems; PlanMem focuses on single-agent planning-memory co-optimization
- **A-MEM**: Self-organizing memory without task-aware planning integration
- **MapCoder**: Multi-agent planning for competitive programming; PlanMem targets repository-level software engineering with persistent memory

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| RL training instability | Start with behavioral cloning; use conservative GRPO updates |
| Complexity Trap applies (simple baselines win) | Include observation masking as a strong baseline; design PlanMem to gracefully degrade to simple strategies |
| Cross-session memory introduces noise | Ebbinghaus-style forgetting + validation before retrieval |
| Compute cost of training | Use SWE-bench Lite for development; scale to Verified for final evaluation |
| Architecture too complex for the LLM backbone | Design modular components that can be individually enabled/disabled |
