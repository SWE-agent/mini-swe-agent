<div align="center">

<img src="docs/assets/nano-swe-agent-banner.svg" alt="nano-swe-agent banner" style="height: 12em"/>
<h1>The 100 line AI agent that solves GitHub issues</h1>

</div>

nano-SWE-agent is an AI agent implemented in [100 lines of python](https://github.com/SWE-agent/nano-swe-agent/blob/main/nanoswea/agent.py)!
Okay, maybe add another 100 lines for a [minimal sandboxed environment](https://github.com/SWE-agent/nano-swe-agent/blob/main/nanoswea/environment.py) 
and [model config](https://github.com/SWE-agent/nano-swe-agent/blob/main/nanoswea/model.py) -- but that's it, and no fancy packages used!

This is enough to solve XX% of the issues in SWE-bench verified, one of the most popular benchmarks to test agentic capabilities.

It's simple, readable, and an excellent starting point for your AI agent
journey!

It's also modular & composable, and we expect to have lots of compatible goodies in the `extra/` directories soon.

## 🔥 Let's fire it up!

```bash
git clone https://github.com/SWE-agent/nano-swe-agent.git
cd nano-swe-agent
pip install -e .
python -m nanoswea/run_github_issue.py https://github.com/SWE-agent/test-repo/issues/1
```
