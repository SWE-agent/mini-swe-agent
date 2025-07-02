<div align="center">

<img src="docs/assets/nano-swe-agent-banner.svg" alt="nano-swe-agent banner" style="height: 12em"/>
<h1>The 100 line AI agent that solves GitHub issues</h1>

</div>

`nano-SWE-agent` is an AI agent implemented in [100 lines of python](https://github.com/SWE-agent/nano-swe-agent/blob/main/nanoswea/agent.py)!
Okay, maybe add another 100 lines for a [minimal sandboxed environment](https://github.com/SWE-agent/nano-swe-agent/blob/main/nanoswea/environment.py) 
and [model config](https://github.com/SWE-agent/nano-swe-agent/blob/main/nanoswea/model.py) -- but that's it, and no fancy packages used!

- 🔥 Run instantly without installation: `pip install pipx && pipx run nano-swe-agent`
- ⚙️ Take full control & quickly prototype your own agent ideas
- ✅ Solves XX % of GitHub issues in SWE-Bench verified
- 🏋 Lean assumptions-free baseline system made for reinforcement learning and finetuning

The project builds on our experience building [SWE-agent](https://swe-agent.com), one of the earliest and most successful software engineering agents for research.

## 🔥 Try it without permanent installation <a target="fire"/>

```bash
pip install pipx && pipx run nano-swe-agent
```

## Install & experiment

```bash
git clone https://github.com/SWE-agent/nano-swe-agent.git
cd nano-swe-agent
pip install -e .
```

This will expose `nswea` (local agent) and `nswea-gh` (run on github issues with docker as sandbox).
But you can also just run the executables as

```bash
python nanoswea/run_local.py
```

etc.

You can put your LM API keys in a `.env` at the repository root or make sure they're set in your shell.
You can also change default behaviors by setting the following variables in `.env` or in the environment:

```bash
NSWEA_MODEL_NAME="claude-sonnet-4-20250514"  # your favorite model
NSWEA_LOCAL_CONFIG_PATH="/path/to/your/own/config"  # override the default config for nswea 
```
