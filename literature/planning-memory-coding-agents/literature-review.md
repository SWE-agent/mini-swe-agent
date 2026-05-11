# Literature Review: Planning and Memory (Context Engineering) for LLM-Based Coding Agents

**Date:** 2026-03-19
**Scope:** Focused (2023–2026)
**Papers Surveyed:** ~45 key works
**Topic:** Combining planning and memory/context engineering techniques for LLM agents, with emphasis on coding agents

---

## 1. Introduction

LLM-based coding agents—systems that autonomously navigate repositories, localize bugs, generate patches, and verify fixes—have rapidly emerged as one of the most impactful applications of large language models. Systems like SWE-agent (Yang et al., 2024), OpenHands (Wang et al., 2024), Devin (Cognition AI, 2024), and Claude Code (Anthropic, 2025) demonstrate that LLMs augmented with tools, planning, and memory can tackle real-world software engineering tasks.

However, two fundamental challenges constrain these agents:

1. **Planning complexity**: Real-world coding tasks require multi-step reasoning—fault localization, understanding cross-file dependencies, formulating repair strategies, and iterating on failed attempts. Flat, reactive agent loops (e.g., basic ReAct) struggle with long-horizon, multi-file tasks.

2. **Context limitations**: Even with 1M-token context windows, agents face *context drift* (losing sight of original goals), *cognitive overload* (working memory cluttered with irrelevant observations), and *information loss* (critical reasoning discarded during context management).

This review surveys the state of the art at the intersection of **planning** and **memory/context engineering** for LLM-based agents, with particular focus on coding agents. We organize the literature into four themes: (1) planning architectures, (2) memory systems, (3) context management strategies, and (4) integrated systems for coding agents.

---

## 2. Planning Architectures for LLM Agents

### 2.1 Reactive and Linear Planning

The foundational agent paradigm is **ReAct** (Yao et al., 2023), which interleaves reasoning traces with actions in a linear chain. While effective for simple tasks, ReAct lacks backtracking capability and degrades on long-horizon tasks where early mistakes compound.

**Reflexion** (Shinn et al., 2023) extends ReAct with verbal self-reflection: after task failure, the agent critiques its trajectory and stores linguistic feedback in an episodic memory buffer, which is prepended to future attempts. This enables learning from mistakes without weight updates. Reflexion demonstrated significant improvements on HumanEval (pass@1: 91.0%) and ALFWorld.

### 2.2 Tree-Structured and Search-Based Planning

**Tree of Thoughts (ToT)** (Yao et al., 2024) generalizes chain-of-thought by allowing branching exploration of multiple reasoning paths, with evaluation and backtracking. This structured deliberation is particularly valuable for tasks with multiple viable approaches.

**Language Agent Tree Search (LATS)** (Zhou et al., 2024; ICML 2024) unifies reasoning, acting, and planning by integrating Monte Carlo Tree Search (MCTS) into the agent loop. LLMs serve simultaneously as agents, value functions, and optimizers. LATS achieved SOTA pass@1 of 92.7% on HumanEval with GPT-4, outperforming ReAct, Reflexion, CoT, and ToT, though at higher token cost.

**ToolTree** (2026) applies dual-feedback MCTS with bidirectional pruning specifically for tool-use planning, reducing exploration overhead while maintaining planning quality.

### 2.3 Hierarchical and Decomposition-Based Planning

**MapCoder** (Islam et al., 2024; ACL 2024) replicates the human programming cycle through four specialized agents—retrieval, planning, coding, and debugging. The planning agent decomposes problems into sub-steps with confidence scores that guide a dynamic traversal strategy. MapCoder achieved SOTA on HumanEval (93.9%), MBPP (83.1%), and CodeContests (28.5%).

**CodePlan** (2024; ICLR 2025) unlocks reasoning potential by scaling code-form planning, where plans are expressed as structured pseudocode rather than natural language, improving both plan quality and downstream code generation.

**GoalAct** (2025) introduces global planning with hierarchical execution, decomposing agent tasks into high-level skills (searching, coding, writing) to reduce planning complexity for web and coding agents.

### 2.4 Workflow-Based vs. Free-Form Planning

A key architectural distinction in coding agents is between **free-form planning** (e.g., OpenHands CodeAct, which gives the agent extensive freedom in long-horizon planning) and **workflow-constrained planning** (e.g., MoatlessTools, Agentless), which pre-defines the action space and execution stages.

- **Agentless** (Xia et al., 2024) argues that much perceived complexity in agent scaffolds can be replaced by dedicated prompting and workflow design, achieving competitive results without agent-style autonomy.
- **MoatlessTools** constrains the model's action space with pre-defined workflows, effectively reducing task horizons and achieving 39% solve rate on SWE-bench at $0.14/issue with Claude 3.5 Sonnet.

This debate highlights a fundamental tension: free-form planning enables more creative problem-solving but increases the risk of context drift and planning failures; structured workflows reduce variance but may miss novel solution paths.

---

## 3. Memory Systems for LLM Agents

### 3.1 Taxonomies and Cognitive Foundations

The survey "Memory in the Age of AI Agents" (Liu et al., 2025) provides a comprehensive taxonomy drawing on cognitive science:

- **Working memory**: The agent's active context window, analogous to human short-term memory
- **Episodic memory**: Records of past interactions and experiences (e.g., Reflexion's verbal feedback)
- **Semantic memory**: Structured knowledge about the world (e.g., knowledge graphs, code repositories)
- **Procedural memory**: Learned behavioral patterns and skills

A parallel evolutionary framework (2026) proposes three stages of memory development: **Storage** (trajectory preservation), **Reflection** (trajectory refinement), and **Experience** (trajectory abstraction).

### 3.2 Agentic and Self-Organizing Memory

**A-MEM** (Xu et al., 2025) introduces agentic memory inspired by the Zettelkasten method: memories are dynamically organized as interconnected knowledge networks through indexing and linking. The four-step process—note construction, link generation, memory evolution, and retrieval—enables the memory system to self-organize and evolve.

**MemoryBank** (Zhong et al., 2024) applies the Ebbinghaus forgetting curve to memory management, with exponential decay modeling for memory strength. This enables natural prioritization where frequently accessed and recent memories are privileged.

**Hierarchical Procedural Memory** (2025) learns reusable procedures from agent trajectories through Bayesian selection and contrastive refinement, enabling transfer of learned skills across tasks.

### 3.3 Production-Ready Memory Systems

**Mem0** (2025) provides a scalable memory architecture for production agents, achieving 26% relative improvement over OpenAI's memory solution with 91% lower p95 latency and >90% token cost savings compared to full-context methods. An enhanced variant uses graph-based memory to capture relational structures.

**MemGPT / Letta** (Packer et al., 2023) draws inspiration from operating system virtual memory management, introducing a hierarchical memory system with explicit page-in/page-out operations. The agent manages its own context window through function calls that move information between fast (context) and slow (external storage) memory tiers. As of 2024, MemGPT evolved into the Letta framework with **Context Repositories**—git-based persistent memory for coding agents.

### 3.4 Memory for Coding Agents Specifically

**Lore** (2026) repurposes git commit messages as a structured knowledge protocol, capturing what the paper terms the "Decision Shadow"—constraints, rejected alternatives, agent directives, and verification metadata that are normally lost after a commit. Lore requires no infrastructure beyond git and is queryable via CLI.

**Aider's RepoMap** provides a concise text-based representation of repository structure, highlighting the most relevant parts for the current task context. This functions as a form of compressed semantic memory about code structure.

**SWE-agent's Agent-Computer Interface (ACI)** (Yang et al., 2024; NeurIPS 2024) enhances agent-code interaction through custom interfaces for file navigation, editing, and search, implicitly structuring the agent's working memory around code-centric operations.

---

## 4. Context Management Strategies

### 4.1 The Context Engineering Paradigm

The "Survey of Context Engineering for Large Language Models" (Mei et al., 2025) formalizes context engineering as a discipline encompassing context retrieval, processing, and management. **Agentic Context Engineering (ACE)** (2025) treats contexts as evolving playbooks that accumulate and refine strategies through generation, reflection, and curation, achieving +10.6% improvement on agent tasks.

### 4.2 Observation Masking vs. LLM Summarization

The **JetBrains Research** study (2025) and the "Complexity Trap" paper (2025) provide critical empirical insights for coding agents:

- **Observation masking** (hiding all but the M most recent observations) preserves full action and reasoning history while drastically reducing token count
- **LLM summarization** compresses entire turns into compact summaries but risks information loss
- **Key finding**: Simple observation masking matches or outperforms LLM summarization in solve rate while being ~52% cheaper, because observation tokens dominate context and chain-of-thought reasoning is better preserved

With Qwen3-Coder 480B, observation masking boosted solve rates by 2.6% over unmanaged context on SWE-bench Verified, while both approaches cut costs by >50%.

### 4.3 Structured Context Management

**Git Context Controller (GCC)** (Wang et al., 2025) is a breakthrough framework that manages agent context using version control semantics:

- **COMMIT**: Persist checkpoints summarizing progress at milestones
- **BRANCH**: Create isolated workspaces for alternative reasoning paths
- **MERGE**: Integrate successful branches back into the main trajectory
- **CONTEXT**: Hierarchically retrieve relevant historical context

GCC achieved SOTA on SWE-bench Verified (>80% success rate with strong models), outperforming 26 existing systems and improving task resolution by >13% over long-context baselines. This represents the most compelling evidence that structured context management is a key differentiator for coding agent performance.

### 4.4 Context Folding and Proactive Management

**Context-Folding** (2025) enables agents to procedurally branch into sub-trajectories and fold them upon completion, collapsing intermediate steps while retaining concise outcome summaries. The FoldGRPO reinforcement learning framework trains agents to self-manage context, achieving comparable performance to ReAct baselines with 10x smaller active context.

**CORAL (Cognitive Resource Self-Allocation)** (2025) addresses cognitive overload by empowering agents to proactively optimize their context, preventing the attention dilution that degrades long-horizon planning.

**AgentFold** (2025) maintains highly concise working context (~7k tokens after 100 turns), enabling scaling to 500+ turn interactions for web agents.

### 4.5 Transactional Context Management

**SagaLLM** (Chang et al., 2025; VLDB 2025) integrates the Saga transactional pattern with persistent memory for multi-agent planning:

- Explicit preservation of goals, justifications, and dependencies
- Modular checkpointing and compensable execution for recovery
- Automated compensation and independent validation agents

This database-inspired approach addresses context loss, unreliable self-validation, and insufficient inter-agent coordination—common failure modes in multi-agent coding systems.

---

## 5. Integrated Systems: Coding Agents in Practice

### 5.1 Academic Systems

| System | Architecture | Planning | Memory/Context | SWE-bench Performance |
|--------|-------------|----------|----------------|----------------------|
| SWE-agent (2024) | Single agent + ACI | ReAct-style | Custom file navigation interface | NeurIPS 2024 |
| OpenHands (2024) | CodeAct free-form | Long-horizon free planning | Full context + code sandbox | ICLR 2025 |
| Agentless (2024) | Pipeline workflow | Pre-defined stages | Minimal context, targeted retrieval | Competitive |
| MoatlessTools (2024) | Constrained workflow | Workflow-guided | Focused context windows | 39% at $0.14/issue |
| AutoCodeRover (2024) | Tool-augmented agent | Perceive-think-act | Efficient (6-step avg.) | ISSTA 2024 |
| RepairAgent (2024) | LLM-controlled repair | Iterative (34-step avg.) | Full trajectory retention | — |
| GCC-equipped agents (2025) | Version-controlled context | Branch/merge planning | COMMIT/BRANCH/MERGE/CONTEXT | SOTA: >80% on SWE-bench Verified |

### 5.2 Commercial Systems

- **Devin** (Cognition AI, 2024): First claimed fully autonomous AI software engineer with shell, editor, and browser access in a sandboxed environment
- **Claude Code** (Anthropic, 2025): Terminal-based agent with 1M-token context window enabling whole-repository reasoning; category-defining advantage for multi-file refactoring (40+ files)
- **Cursor** (2025): IDE-integrated agent with AI-native editing workflows; 1M+ users
- **Windsurf/Cascade** (2025): AI-native editor with "Flows" collaboration model; acquired by Cognition AI in July 2025

### 5.3 Self-Evolving and Training-Based Approaches

**SWE-Gym** (2024) provides a training environment for software engineering agents, enabling RL-based optimization of agent behavior on real GitHub issues.

**LIVE-SWE-AGENT** (2025) explores whether coding agents can self-evolve their own scaffolds on-the-fly, addressing the challenge that manually designing optimal agent architectures is extremely costly.

**SE-Agent** (2025) uses self-evolution trajectory optimization, iteratively refining its approach based on execution feedback.

---

## 6. Gap Analysis and Research Opportunities

### 6.1 Gap: Unified Planning-Memory Architectures

**Current state**: Planning and memory/context management are largely treated as orthogonal concerns. GCC is the notable exception, but even GCC's planning is implicit (through BRANCH/MERGE) rather than explicit hierarchical task decomposition.

**Opportunity**: Design architectures where the planning system and memory system are co-designed—where planning decisions inform what to remember (and forget), and memory contents shape planning strategies. For coding agents, this means a planner that can reason about repository structure (semantic memory), past debugging attempts (episodic memory), and known repair patterns (procedural memory) simultaneously.

### 6.2 Gap: Adaptive Context Management Policies

**Current state**: Most context management strategies are static (fixed masking window, fixed summarization triggers). The Complexity Trap paper shows that simple approaches work surprisingly well, but this may not hold as tasks become more complex.

**Opportunity**: Develop RL-trained or meta-learned context management policies that adapt to task difficulty, phase (exploration vs. exploitation), and agent confidence. Context-Folding's FoldGRPO is a step in this direction, but focuses on sub-task structure rather than adaptive granularity.

### 6.3 Gap: Cross-Session and Cross-Task Memory Transfer

**Current state**: Most coding agents treat each issue as independent. Lore and Letta's Context Repositories are early attempts at persistent memory, but lack mechanisms for selective transfer and forgetting across tasks.

**Opportunity**: Build memory systems that accumulate repository-specific knowledge (architecture patterns, common bug classes, team conventions) across sessions and selectively apply this knowledge to new tasks—analogous to how experienced developers build mental models of codebases.

### 6.4 Gap: Evaluation Frameworks for Planning+Memory

**Current state**: SWE-bench evaluates end-to-end task resolution but doesn't decompose performance into planning quality vs. memory effectiveness. There are no benchmarks specifically for measuring how well agents maintain context over long horizons in coding tasks.

**Opportunity**: Develop diagnostic benchmarks that separately measure: (a) planning depth and adaptability, (b) context retention under pressure, (c) memory retrieval precision for code-specific queries, and (d) graceful degradation as task horizon increases.

### 6.5 Gap: Multi-Agent Memory Coordination

**Current state**: SagaLLM addresses transactional guarantees but focuses on general planning. Multi-agent coding systems lack principled memory sharing protocols—how should a bug-localizer agent communicate relevant context to a patch-generator agent?

**Opportunity**: Design memory-sharing protocols for multi-agent coding workflows that minimize redundant exploration while maximizing relevant context transfer between specialized agents.

---

## 7. Key Trends and Synthesis

1. **From prompt engineering to context engineering**: The field has definitively moved beyond crafting prompts to designing entire information management systems. For coding agents, this means managing repository context, execution traces, and reasoning history as first-class concerns.

2. **Simplicity can win**: The Complexity Trap result—that observation masking outperforms LLM summarization—challenges the assumption that more sophisticated memory management always helps. The right baseline matters.

3. **Version control as a memory metaphor**: GCC, Lore, and Letta's Context Repositories all draw on git semantics for agent memory. This is a natural fit for coding agents where version control is already part of the domain.

4. **RL for context management**: Context-Folding's FoldGRPO and SWE-Gym's agent training suggest that learning to manage context (not just learning to code) is a viable and important training objective.

5. **The 1M-token context window is necessary but not sufficient**: Claude Code's success demonstrates the value of large context windows, but GCC's structured management on top of large contexts achieves even better results. Raw context capacity is a foundation, not a solution.

---

## 8. References

See `references.bib` for full bibliography. Key papers by theme:

**Planning**: Yao et al. (2023) ReAct; Shinn et al. (2023) Reflexion; Yao et al. (2024) ToT; Zhou et al. (2024) LATS; Islam et al. (2024) MapCoder
**Memory**: Packer et al. (2023) MemGPT; Xu et al. (2025) A-MEM; Zhong et al. (2024) MemoryBank; Liu et al. (2025) Memory Survey; Mem0 (2025)
**Context Management**: Mei et al. (2025) CE Survey; Wang et al. (2025) GCC; Context-Folding (2025); SagaLLM (2025); Complexity Trap (2025)
**Coding Agents**: Yang et al. (2024) SWE-agent; Wang et al. (2024) OpenHands; Xia et al. (2024) Agentless; Zhang et al. (2024) AutoCodeRover; Lore (2026)
