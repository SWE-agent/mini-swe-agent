# Wire an agent with an SLM: Claude Code runs the kit, the proxy answers with the model.
# usage: bash wire.sh <agent repo with observatory/ in it> <model name in litellm-deepinfra.yaml> [proxy port]
cd "$1" && env ANTHROPIC_BASE_URL=http://127.0.0.1:${3:-4004} ANTHROPIC_AUTH_TOKEN=sk-local-slm-test ANTHROPIC_API_KEY= \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=$2 ANTHROPIC_SMALL_FAST_MODEL=$2 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
  uv run observatory/run.py --model $2 --budget 0
