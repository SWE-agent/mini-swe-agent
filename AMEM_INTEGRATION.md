# A-mem Integration for Mini-SWE-Agent

This document describes the integration of the A-mem (Agentic Memory) system into mini-swe-agent for intelligent context management and performance evaluation.

## Overview

This integration adds a memory layer to mini-swe-agent using the A-mem system to:

1. **Store experiences**: Captures command executions, outputs, errors, and solutions
2. **Retrieve relevant memories**: Uses semantic search to find past similar experiences
3. **Compress conversation history**: Summarizes long conversations to ~50% of original length
4. **Build hybrid context**: Combines recent messages + summary + retrieved memories

## Architecture

### Components

```
src/minisweagent/memory/
├── __init__.py           # Module initialization
├── prompts.py            # Code-specific prompts for A-mem
├── summarizer.py         # Conversation summarization (50% compression)
├── amem_wrapper.py       # A-mem integration wrapper
└── memory_agent.py       # MemoryAgent class (extends DefaultAgent)

experiments/
├── configs/
│   ├── baseline_config.yaml   # Baseline experiment config
│   └── memory_config.yaml     # Memory experiment config
├── run_baseline.sh           # Run baseline experiment
├── run_memory.sh             # Run memory experiment
└── analyze_results.py        # Compare results

run_full_experiment.sh        # Master script for full workflow
```

### Memory Agent Design

**MemoryAgent** extends `DefaultAgent` with three key enhancements:

1. **Hybrid Context Building** (`_build_hybrid_context`):
   - System message (always included)
   - Conversation summary (when messages > threshold)
   - Retrieved memories (top-k relevant past experiences)
   - Recent messages (last N messages)

2. **Memory Storage** (`_store_experience`):
   - Auto-decides what's worth storing (errors, complex commands, novel patterns)
   - Extracts keywords, tags, and context
   - Stores in A-mem with code-specific metadata

3. **Intelligent Summarization**:
   - Uses A-mem's LLM to compress conversation to 50% length
   - Preserves critical information (file paths, errors, solutions)
   - Removes redundant information (repeated failures, verbose outputs)

### Key Features

- **Code-Specific Prompts**: Tailored for software engineering tasks
- **Scenario-Based Retrieval**: Different retrieval strategies for errors, file search, testing, etc.
- **Auto-Storage Decision**: Intelligently filters what to store
- **Chunked Summarization**: Handles very long conversations (>30 messages)
- **Persistent Memory**: ChromaDB vector database for cross-session memory

## Installation

### Prerequisites

1. Python environment with mini-swe-agent dependencies
2. OpenAI API key (for LLM)
3. A-mem system installed

### Setup

```bash
# Activate environment
source /common/users/wx139/env/minisweagentenv/bin/activate

# Install A-mem
pip install -e /common/users/wx139/code/opensource_all/A-mem-sys

# Verify installation
python -c "from agentic_memory import AgenticMemorySystem; print('A-mem installed!')"
```

## Usage

### Quick Start

Run the full experiment (baseline + memory + analysis):

```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-key-here'

# Run full experiment
./run_full_experiment.sh
```

### Individual Components

**Run baseline only:**
```bash
./run_full_experiment.sh --baseline-only
```

**Run memory experiment only:**
```bash
./run_full_experiment.sh --memory-only
```

**Run analysis only:**
```bash
./run_full_experiment.sh --analysis-only
```

### Manual Execution

**Baseline experiment:**
```bash
bash experiments/run_baseline.sh
```

**Memory experiment:**
```bash
bash experiments/run_memory.sh
```

**Analysis:**
```bash
python experiments/analyze_results.py \
    --baseline-dir experiments/results/baseline \
    --memory-dir experiments/results/memory \
    --output experiments/results/comparison.json
```

## Configuration

### Baseline Config (`experiments/configs/baseline_config.yaml`)

```yaml
agent:
  type: "default"
  config:
    step_limit: 50
    cost_limit: 5.0

model:
  name: "gpt-4o-mini"
  temperature: 0.0

dataset:
  name: "swe_bench_lite"
  subset:
    enabled: true
    size: 10  # Start small for testing
    seed: 42
```

### Memory Config (`experiments/configs/memory_config.yaml`)

```yaml
agent:
  type: "memory"
  memory_config:
    embedding_model: "all-MiniLM-L6-v2"
    llm_backend: "openai"
    llm_model: "gpt-4o-mini"
    persist_directory: "./experiments/memory_db"

  summarization_threshold: 20  # Trigger summary at 20 messages
  recent_messages_keep: 5      # Keep last 5 messages raw
  memory_retrieval_k: 3        # Retrieve top 3 memories
```

## Experiments

### Dataset

- **SWE-bench Lite**: Smaller subset of SWE-bench for faster experiments
- **Default size**: 10 instances (configurable in YAML)
- **Seed**: 42 (for reproducibility)

### Metrics Tracked

- **Resolution rate**: % of instances resolved
- **Average cost**: USD per instance
- **Average tokens**: Tokens used per instance
- **Average steps**: Actions taken per instance
- **Average time**: Seconds per instance

### Results Analysis

The `analyze_results.py` script compares baseline vs memory experiments:

```
📊 RESOLUTION RATE:
  Baseline:    40.0%
  Memory:      60.0%
  Improvement: +20.0% (+50.0%)

💰 COST (USD):
  Baseline:   $0.0234
  Memory:     $0.0189
  Difference: -$0.0045 (-19.2%)

🔢 TOKENS:
  Baseline:   1234
  Memory:     998
  Difference: -236 (-19.1%)
```

## Implementation Details

### Memory Storage Strategy

**Always store:**
- Errors and exceptions
- Complex commands (grep, find, pytest, git)
- Commands with error indicators in output

**Skip storage:**
- Simple commands (ls, pwd, cd, echo)
- Routine successful operations

### Retrieval Strategy

**Query construction:**
- Uses last 3 messages as context
- Limits query to 500 characters
- Determines scenario (error, testing, file_search, etc.)

**Scenario-based retrieval:**
- `error_encountered`: Finds similar errors and solutions
- `file_search`: Finds successful file location patterns
- `testing`: Finds test commands and outcomes
- `general`: Finds relevant past experiences

### Summarization Strategy

**Compression ratio:** 50% of original length

**Preserves:**
- File paths and directory structures
- Commands executed and outcomes
- Errors with types and solutions
- Code changes made
- Test and build results

**Removes:**
- Repeated failed attempts (keeps pattern + final outcome)
- Verbose command outputs (keeps essentials)
- Duplicate file listings
- Intermediate debugging steps that didn't lead anywhere

## Troubleshooting

### Common Issues

**1. ImportError: No module named 'agentic_memory'**
```bash
# Install A-mem
pip install -e /common/users/wx139/code/opensource_all/A-mem-sys
```

**2. OpenAI API key not set**
```bash
export OPENAI_API_KEY='your-key-here'
```

**3. Permission denied when running scripts**
```bash
chmod +x run_full_experiment.sh
chmod +x experiments/run_*.sh
```

**4. ChromaDB errors**
```bash
# Clear memory database and restart
rm -rf experiments/memory_db
```

## Performance Expectations

Based on the design:

**Expected improvements with memory:**
- **Higher resolution rate**: Learns from past failures
- **Lower token usage**: Compression reduces context size
- **Lower cost**: Fewer tokens = lower cost
- **Faster convergence**: Relevant memories guide agent

**Trade-offs:**
- **Slightly more overhead**: Memory storage/retrieval adds latency
- **Storage space**: ChromaDB grows with usage
- **LLM calls for summarization**: Additional cost for summaries

## Future Enhancements

Potential improvements:

1. **Memory pruning**: Automatically remove outdated/irrelevant memories
2. **Multi-agent sharing**: Share memories across agent instances
3. **Advanced retrieval**: Use more sophisticated retrieval strategies
4. **Memory evolution**: Let A-mem evolve memories over time
5. **Fine-tuned embeddings**: Train embeddings on code-specific data

## References

- **A-mem Paper**: "A-mem: Agentic Memory for LLM Agents"
- **A-mem GitHub**: https://github.com/WujiangXu/A-mem-sys
- **Mini-SWE-Agent**: https://github.com/SWE-agent/mini-swe-agent
- **SWE-bench**: https://github.com/princeton-nlp/SWE-bench

## Contact

For questions or issues with this integration, please refer to:
- Mini-SWE-Agent documentation
- A-mem documentation and issues
