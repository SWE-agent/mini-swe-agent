# Literature Review: Context Engineering for Large Language Models

**Date:** 2026-02-27
**Scope:** Focused (last 3 years, 2023–2026)
**Papers Surveyed:** ~40 key works

---

## 1. Introduction and Definition

Context engineering for LLMs has emerged as a formal discipline that transcends traditional prompt engineering. While prompt engineering focuses on crafting individual instructions or queries, context engineering encompasses the **systematic design, curation, and management of the entire information payload** provided to a model at inference time.

Gartner (2025) declared: *"Context engineering is in, and prompt engineering is out. AI leaders must prioritize context over prompts."* This shift reflects the recognition that model performance depends less on how questions are phrased and more on what information surrounds those questions.

A recent comprehensive survey (Mei et al., 2025) analyzed over 1,400 research papers and established a formal taxonomy of context engineering, decomposing it into:
- **Foundational Components**: context retrieval and generation, context processing, context management
- **System Implementations**: RAG, memory systems with tool-integrated reasoning, multi-agent systems

---

## 2. Taxonomy of Context Engineering

### 2.1 Context Retrieval and Generation

#### 2.1.1 Retrieval-Augmented Generation (RAG)

RAG has become the dominant paradigm for injecting external knowledge into LLM context windows. Since Lewis et al. (2020) introduced the foundational approach, the field has evolved substantially through 2024–2025.

**Naive RAG** retrieves chunks based on query similarity, but suffers from precision-recall trade-offs and noisy context injection.

**Advanced RAG** introduces pre-retrieval (query rewriting, decomposition), during-retrieval (reranking, filtering), and post-retrieval (context compression, reordering) stages.

**Modular RAG** (Gao et al., 2023) decomposes the pipeline into swappable modules, enabling flexible architectures. A 2024 systematic review documents the evolution through multimodal extensions (SAM-RAG, OmniSearch) and structured retrieval-reflection loops (mR2AG, M3DocRAG).

**GraphRAG** (Edge et al., 2024) introduces knowledge-graph-based retrieval, where an LLM first extracts entities and relationships to build a graph, then uses community detection (Louvain algorithm) and hierarchical summaries to answer both local (entity-specific) and global (holistic) queries. Microsoft's GraphRAG achieves significant improvements on query-focused summarization tasks compared to naive vector-search RAG.

**Self-RAG** (Asia et al., 2023) trains models to adaptively retrieve on demand and critically evaluate generated outputs using special "reflection tokens" for relevance, support, and usefulness, achieving improved factuality and citation accuracy.

#### 2.1.2 Knowledge Graph Integration

Structured external knowledge via knowledge graphs provides factual grounding beyond vector-similarity retrieval. KG-augmented approaches offer interpretable reasoning chains and explicit entity-relationship modeling.

### 2.2 Context Processing

#### 2.2.1 Context Compression and Prompt Compression

Long contexts are expensive (compute, memory, latency) and can degrade model quality. Compression approaches aim to retain maximum information with fewer tokens.

**LLMLingua** (Jiang et al., 2023; Microsoft Research) uses a small LM to score token importance and prune low-information tokens, achieving up to **20x compression with minimal performance loss**.

**LLMLingua-2** (2024) reformulates compression as a token classification problem using a BERT-level encoder, achieving **3–6x speedup** over the original while maintaining task-agnostic applicability.

**LongLLMLingua** (2024, ACL) targets long-context scenarios (≥10k tokens), achieving 1.4–2.6x end-to-end latency reduction at 2–6x compression ratios, with a **+21.4% performance boost** on NaturalQuestions (GPT-3.5-Turbo, ~4x fewer tokens).

**GIST tokens** compress soft prompts into compact gist representations via attention mask modification, achieving up to **26x compression** for instruction-following contexts.

A 2025 survey on prompt compression (ACL 2025) categorizes approaches as:
- **Hard prompt methods**: extractive (token deletion) and abstractive (paraphrase/summarize)
- **Soft prompt methods**: learned continuous representations

Extractive reranker-based compression achieves **+7.89 F1** on 2WikiMultihopQA at 4.5x compression, often outperforming abstractive methods.

#### 2.2.2 Structured Output and Constrained Generation

Providing structured context and receiving structured output forms a two-sided context engineering challenge. OpenAI (2024) and others introduced **Structured Outputs** through JSON-schema-constrained decoding.

Research (Tam et al., 2024) reveals that requiring JSON output causes accuracy drops of **>27 percentage points** on reasoning tasks. Wang et al. (2025) mitigate this by passing unstructured outputs through a secondary formatting model, recovering **>20 percentage points** in accuracy.

The Awesome-LLM-Constrained-Decoding survey documents a growing ecosystem of constrained decoding approaches leveraging grammar-based finite-state machines and logit masking.

### 2.3 Context Management

#### 2.3.1 Long Context Challenges and "Lost in the Middle"

Liu et al. (2023) documented the **"lost in the middle"** phenomenon: LLM performance degrades significantly when relevant information appears in the middle of long contexts. The root cause is a **U-shaped attention bias** introduced by RoPE's long-term decay effect—models attend disproportionately to the beginning and end of sequences.

Subsequent work (2024) addresses this through:
- **Found-in-the-Middle calibration** (He et al., 2024, ACL Findings): corrects positional attention bias, outperforming prior methods by up to **15 percentage points**
- **IN2 Training** (information-intensive training): teaches FILM-7B to retrieve information from anywhere in 32K contexts
- **Strategic document reordering**: placing highest-ranked documents at context boundaries

Recent studies reveal additional challenges (2024–2025):
- Long-context LLMs show **label position bias** in long few-shot sequences
- **Context Rot** (Chroma Research, 2024): performance degrades gracefully but consistently as input token counts increase
- Test-time training (Akyürek et al., 2025) uses targeted gradient updates on context to overcome static attention limitations

#### 2.3.2 KV Cache Optimization

The KV cache stores key/value tensors from prior attention computations. For long sequences, this becomes a major memory bottleneck (grows linearly with sequence length).

**vLLM's PagedAttention** (Kwon et al., 2023) applies virtual memory principles to KV cache allocation, reducing waste from 60–80% to **under 4%**, enabling 2–4x throughput gains.

**MiniCache** (2024): compresses KV caches by merging adjacent deep-layer states (high inter-layer similarity), reducing memory with minimal quality degradation.

**RetrievalAttention** (2024): offloads past KV pairs to CPU and uses ANNS to retrieve only the top 1–3%, achieving near-full accuracy with **massive memory savings**.

**MorphKV** (2025): adaptive fixed-size KV cache using attention-pattern-guided iterative refinement, achieving **>50% memory savings** while improving long-form accuracy.

**KVPR** (ACL 2025 Findings): I/O-aware KV cache pre-population for efficient inference.

#### 2.3.3 Context Window Extension

Modern LLMs have seen dramatic context window expansions, from 2K (early models) to millions of tokens. Key technical approaches:

**Position Interpolation (PI)** (Chen et al., 2023): down-scales position indices to fit expanded context, enabling fine-tuning for longer windows.

**YaRN** (Peng et al., 2024, ICLR): efficient RoPE context extension requiring **10x fewer tokens** and **2.5x fewer training steps** than PI.

**LongRoPE** (Ding et al., 2024): extends context to **2048K tokens** using non-uniform RoPE rescaling via evolutionary search.

**LongRoPE2** (2025): near-lossless scaling via improved analysis of the left-skewed position frequency distribution problem.

---

## 3. System-Level Implementations

### 3.1 Memory Systems

Memory for LLMs operates at multiple granularities: **in-context** (within the active context window), **external** (databases, vector stores), and **parametric** (encoded in model weights).

**Titans** (Behrouz et al., 2024): introduces MLP-based neural memory with surprise-driven KL divergence updates for real-time parameter modification, enabling adaptive context retention beyond fixed windows.

**MemLong** (Liu et al., 2024): extends context to 80K tokens via retrieval from external embedding storage while preserving core model parameters.

**M+** (Wang et al., 2025): splits KV cache into on-GPU working memory and CPU-resident long-term bank with a co-trained retriever/scheduler.

**LightMem** (2025): lightweight memory-augmented generation focused on efficiency.

**Cognitive Workspace** (2025): proposes a paradigm transcending RAG by emulating human cognitive mechanisms of active memory management, providing "functional infinite context."

**A-Mem** (2025): agentic memory specifically designed for LLM agents with dynamic memory organization.

For multi-turn dialogue, memory operates at turn, utterance, segment, session, and topic granularities (survey: ACM Computing Surveys, 2025). Key challenges include: relevance filtering, temporal coherence, conflict resolution, and scalability.

### 3.2 Tool-Integrated Reasoning

Agentic LLMs combine context engineering with tool use to ground outputs in real-world actions and data. The **Model Context Protocol (MCP)** (Anthropic, 2024) standardizes tool/resource access across agent implementations.

LLMs get lost in multi-turn conversation (Shi et al., 2025): even with long context windows, maintaining coherent multi-turn agent context remains challenging due to instruction drift and context accumulation.

### 3.3 Multi-Agent Systems

Multi-agent systems introduce the **"disconnected models problem"**—maintaining coherent context across multiple agent interactions. Research in 2024–2025 addresses this through:
- Shared vector store memory (FAISS, Pinecone)
- Hierarchical context passing protocols
- Difficulty-aware agent orchestration (Zheng et al., 2025)
- The Model Context Protocol (MCP) for standardized agent-tool interaction

---

## 4. Hallucination and Context Grounding

Context quality directly impacts hallucination rates. A 2024 Stanford study found that combining RAG, RLHF, and guardrails reduced hallucinations by **96%** compared to baseline.

Key techniques:
- **Self-RAG**: reflection tokens for relevance and factuality evaluation
- **MEGA-RAG**: multi-evidence guided answer refinement with cross-encoder reranking
- **Hierarchical Semantic Piece (HSP)**: reduces token expansion ratio by 30%
- **Intermediate-layer integration**: outperforms input-level context injection for factual accuracy

---

## 5. Benchmarks and Evaluation

Long-context evaluation is an active area (Comprehensive Survey, arXiv:2503.17407):
- **RULER**: synthetic tasks for long-context comprehension
- **SCROLLS**: summarization and question answering over long documents
- **L-Eval**: multi-task long-context evaluation with diverse domains
- **∞Bench**: evaluation with context lengths beyond 100K
- **HELMET**: holistic evaluation of LLMs for extended tasks

LLMs struggle with challenging long-context tasks: position bias, reasoning degradation, and instruction drift are persistent issues even in the latest models.

---

## 6. Research Gap Analysis

### Gap 1: Comprehension-Generation Asymmetry
Mei et al. (2025) identify the **comprehension-generation asymmetry** as the field's defining challenge: LLMs demonstrate strong understanding of complex contexts yet show pronounced limitations in generating equivalently sophisticated long-form outputs. Current context engineering research focuses heavily on retrieval and compression (input side), with insufficient attention to generation-side context management.

**Research opportunity**: Develop context-aware generation frameworks that use structured intermediate representations to guide long-form output coherence, factuality, and completeness.

### Gap 2: Unified Context Quality Metrics
The field lacks standardized metrics for context quality—relevance, faithfulness, completeness, and efficiency. Different RAG systems, compression methods, and memory architectures are evaluated on disparate benchmarks, making cross-system comparison difficult.

**Research opportunity**: A unified context quality evaluation framework combining automatic metrics (faithfulness, coverage, compression ratio) with human-aligned preference measures.

### Gap 3: Dynamic Context Adaptation
Current approaches use largely static context management policies: fixed retrieval strategies, fixed compression ratios, fixed memory sizes. Real-world use cases require dynamic adaptation based on query complexity, domain, and uncertainty.

**Research opportunity**: Reinforcement learning or adaptive inference frameworks that select context engineering strategies (retrieval depth, compression level, memory scope) based on query characteristics and model confidence.

### Gap 4: Cross-Modal and Multi-Source Context Fusion
While multimodal RAG has emerged (SAM-RAG, M3DocRAG), systematic frameworks for reasoning over heterogeneous context sources (text + structured data + images + video + code) remain limited.

**Research opportunity**: Unified multi-source context encoding and fusion architectures with cross-modal attention and structured knowledge grounding.

### Gap 5: Context Security and Privacy
Injecting external content into context windows introduces security risks: prompt injection attacks, data leakage, and context poisoning. This dimension is largely under-studied in the context engineering literature.

**Research opportunity**: Formal threat models for context injection attacks and corresponding defenses (context sanitization, role-based context access control, differential privacy for RAG retrieval).

---

## 7. Conclusion

Context engineering has rapidly matured from an informal practice into a formal engineering discipline. The field spans a rich ecosystem: retrieval methods (RAG, GraphRAG, Self-RAG), compression techniques (LLMLingua series, GIST tokens), memory architectures (Titans, MemLong, Cognitive Workspace), KV cache management (PagedAttention, MorphKV), and system-level integration (multi-agent, MCP).

The trajectory is clear: as model context windows expand toward millions of tokens, the challenge shifts from "can the model accept this context?" to "how do we engineer context that maximally benefits model performance, efficiency, and reliability?" Future work must address the comprehension-generation asymmetry, unify evaluation frameworks, enable dynamic adaptation, and tackle emerging security concerns.

---

## References

See `references.bib` for full BibTeX entries.

Key papers cited:
- Mei et al. (2025). *A Survey of Context Engineering for Large Language Models*. arXiv:2507.13334
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020
- Edge et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. arXiv:2404.16130
- Asia et al. (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. ICLR 2024
- Jiang et al. (2023). *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models*. EMNLP 2023
- Wu et al. (2024). *LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression*. ACL 2024
- Jiang et al. (2024). *LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression*. ACL 2024
- Liu et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts*. TACL 2024
- He et al. (2024). *Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization*. ACL Findings 2024
- Kwon et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023
- Peng et al. (2024). *YaRN: Efficient Context Window Extension of Large Language Models*. ICLR 2024
- Ding et al. (2024). *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens*. arXiv:2402.13753
- Chen et al. (2023). *Extending Context Window of Large Language Models via Positional Interpolation*. arXiv:2306.15595
- Behrouz et al. (2024). *Titans: Learning to Memorize at Test Time*. arXiv
- Gao et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey*. arXiv:2312.10997
- Wang et al. (2024). *LongRoPE2: Near-Lossless LLM Context Window Scaling*. arXiv:2502.20082
- Akyürek et al. (2025). *Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs*. arXiv:2512.13898
- Shi et al. (2025). *LLMs Get Lost In Multi-Turn Conversation*. arXiv:2505.06120
- Tam et al. (2024). *Let Me Speak Freely? A Study of Language Models on Structured Data*. arXiv
- Zheng et al. (2025). *Difficulty-Aware Agent Orchestration in LLM-Powered Workflows*. arXiv:2509.11079
