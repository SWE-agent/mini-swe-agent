# Running the kit and the agents with SLMs

The SLMs answer through a local [litellm](https://docs.litellm.ai/) proxy that speaks the Anthropic API, so Claude Code (which runs the kit's sessions) and the toy agent (which uses the Anthropic SDK) can use them unchanged. Models are served by DeepInfra (`DEEPINFRA_API_KEY`).

```bash
# 1. the proxy
DEEPINFRA_API_KEY=... uvx --from 'litellm[proxy]' litellm --config litellm-deepinfra.yaml --host 127.0.0.1 --port 4004

# 2. wire an agent (a clean copy with the observatory/ kit in it) with a model
bash wire.sh ../path/to/agent qwen
```

## Toy agent run shown on the dashboard (round 15, `toy-qwen-demo`)

Qwen3.8-27B wired the toy agent, and the agent itself ran on Qwen3.8-27B. The toy agent asks for `claude-haiku-5-5` in its code, so `run_toy_qwen.py` sets the model to `Qwen/Qwen3.8-27B` before running it, without changing the agent: the logs record the model that answered.

```powershell
cd ..\15_qwen_nemotron_current_kit\toy-qwen-demo   # the wired toy agent with the kit, as it was run
$env:UV_PROJECT_ENVIRONMENT = "$HOME\.venvs\toy-qwen"   # a short path: some anthropic file names exceed Windows' path limit
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:4004"
$env:ANTHROPIC_API_KEY = "sk-local-slm-test"
$env:OBSERVATORY_RUNS = "$PWD\observatory\runs"
uv run python ..\..\setup\run_toy_qwen.py "python 도구로 이 폴더의 .py 파일 개수를 세어서 알려줘"
uv run observatory/dashboard.py   # the loop on the left, the runs on the right
```

`toy-qwen-demo` is the whole wired agent (the `toy-qwen` folder next to it keeps only the diff). Its run is `observatory/runs/20261023-053143-47108.jsonl`: the task, two LLM calls, one `python` tool call and the user's answer to the confirmation.
