"""Standalone test to check if Anthropic prompt caching works via litellm + Llama API.

Tests three scenarios:
1. No caching (no cache_control markers)
2. Explicit cache_control markers on the system message
3. Explicit cache_control markers on the last user message (like mini-swe-agent does)

For each scenario, we make repeated calls with the same long prompt and compare:
- Whether cache-related token counts appear in the response
- Average cost per call
"""

import copy
import json
import os
import statistics
import time
from pathlib import Path

import litellm

# ---------------------------------------------------------------------------
# Configuration – override via env vars if needed
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("TEST_MODEL_NAME", "claude-4-5-sonnet-genai")
API_BASE = os.getenv("TEST_API_BASE", "https://api.llama.com/experimental/passthrough/openai/v1/")
API_KEY = os.getenv("TEST_API_KEY", os.getenv("LLAMA_API_KEY", ""))
CUSTOM_LLM_PROVIDER = os.getenv("TEST_CUSTOM_LLM_PROVIDER", "openai")
NUM_CALLS = int(os.getenv("TEST_NUM_CALLS", "10"))
MODEL_PRICES_PATH = Path(
    os.getenv(
        "TEST_MODEL_PRICES_PATH",
        "/Users/klieret/Documents/repos/RevEngBench/reveng/configs/mini/model_prices.json",
    )
)

# Register custom model pricing so litellm can compute costs
if MODEL_PRICES_PATH.is_file():
    print(f"Loading model prices from {MODEL_PRICES_PATH}")
    litellm.utils.register_model(json.loads(MODEL_PRICES_PATH.read_text()))
else:
    print(f"WARNING: Model prices file not found at {MODEL_PRICES_PATH}")

LONG_SYSTEM_PROMPT = "You are a helpful coding assistant. " * 200 + "\nAlways answer concisely in one sentence."

USER_MESSAGE = "What is 2+2? Answer in one word."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_messages_no_cache() -> list[dict]:
    return [
        {"role": "system", "content": LONG_SYSTEM_PROMPT},
        {"role": "user", "content": USER_MESSAGE},
    ]


def _make_messages_cache_system() -> list[dict]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": LONG_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "user", "content": USER_MESSAGE},
    ]


def _make_messages_cache_last_user() -> list[dict]:
    return [
        {"role": "system", "content": LONG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": USER_MESSAGE,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]


def _make_messages_cache_both() -> list[dict]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": LONG_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": USER_MESSAGE,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]


def call_model(messages: list[dict]) -> dict:
    kwargs: dict = {
        "model": MODEL_NAME,
        "messages": copy.deepcopy(messages),
        "max_tokens": 50,
        "drop_params": True,
    }
    if API_BASE:
        kwargs["api_base"] = API_BASE
    if CUSTOM_LLM_PROVIDER:
        kwargs["custom_llm_provider"] = CUSTOM_LLM_PROVIDER
    if API_KEY:
        kwargs["api_key"] = API_KEY

    return litellm.completion(**kwargs)


def extract_cache_info(response) -> dict:
    """Pull out any cache-related fields from the response."""
    info: dict = {}
    usage = response.usage
    if usage is None:
        return {"raw_usage": None}

    info["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
    info["completion_tokens"] = getattr(usage, "completion_tokens", None)
    info["total_tokens"] = getattr(usage, "total_tokens", None)
    info["cache_creation_input_tokens"] = getattr(usage, "cache_creation_input_tokens", None)
    info["cache_read_input_tokens"] = getattr(usage, "cache_read_input_tokens", None)

    # Some providers put it in prompt_tokens_details
    pdf = getattr(usage, "prompt_tokens_details", None)
    if pdf is not None:
        info["prompt_tokens_details"] = pdf.model_dump() if hasattr(pdf, "model_dump") else pdf

    # Dump full usage for inspection
    info["raw_usage"] = usage.model_dump() if hasattr(usage, "model_dump") else str(usage)
    return info


def compute_cost(response) -> float:
    try:
        return litellm.cost_calculator.completion_cost(response, model=MODEL_NAME)
    except Exception as e:
        print(f"  [cost calc failed: {e}]")
        return 0.0


def run_scenario(name: str, messages_fn, num_calls: int = NUM_CALLS):
    print(f"\n{'=' * 70}")
    print(f"SCENARIO: {name}")
    print(f"{'=' * 70}")

    costs: list[float] = []
    for i in range(num_calls):
        messages = messages_fn()
        t0 = time.time()
        response = call_model(messages)
        elapsed = time.time() - t0

        cache_info = extract_cache_info(response)
        cost = compute_cost(response)
        costs.append(cost)

        reply_text = response.choices[0].message.content[:80] if response.choices[0].message.content else "(empty)"
        print(f"\n  Call {i + 1}/{num_calls} ({elapsed:.1f}s, cost=${cost:.6f}):")
        print(f"    Reply: {reply_text}")
        print(f"    prompt_tokens:               {cache_info.get('prompt_tokens')}")
        print(f"    completion_tokens:            {cache_info.get('completion_tokens')}")
        print(f"    cache_creation_input_tokens:  {cache_info.get('cache_creation_input_tokens')}")
        print(f"    cache_read_input_tokens:      {cache_info.get('cache_read_input_tokens')}")
        if cache_info.get("prompt_tokens_details"):
            print(f"    prompt_tokens_details:        {cache_info['prompt_tokens_details']}")
        if i == 0:
            print(f"    raw_usage (first call only):  {json.dumps(cache_info.get('raw_usage'), indent=6)}")

    print(f"\n  --- Summary for '{name}' ---")
    print(f"  Calls:        {num_calls}")
    print(f"  Total cost:   ${sum(costs):.6f}")
    print(f"  Avg cost:     ${statistics.mean(costs):.6f}")
    if len(costs) > 1:
        print(f"  Stddev cost:  ${statistics.stdev(costs):.6f}")
    print(f"  Min cost:     ${min(costs):.6f}")
    print(f"  Max cost:     ${max(costs):.6f}")
    return costs


def main():
    print(f"Model:    {MODEL_NAME}")
    print(f"API base: {API_BASE}")
    print(f"Provider: {CUSTOM_LLM_PROVIDER}")
    print(f"Calls per scenario: {NUM_CALLS}")
    print(f"System prompt length: {len(LONG_SYSTEM_PROMPT)} chars")

    all_results: dict[str, list[float]] = {}

    all_results["no_cache"] = run_scenario(
        "No cache control markers",
        _make_messages_no_cache,
    )

    all_results["cache_system"] = run_scenario(
        "cache_control on system message",
        _make_messages_cache_system,
    )

    all_results["cache_last_user"] = run_scenario(
        "cache_control on last user message (mini-swe-agent style)",
        _make_messages_cache_last_user,
    )

    all_results["cache_both"] = run_scenario(
        "cache_control on both system + user message",
        _make_messages_cache_both,
    )

    # Final comparison
    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print(f"{'=' * 70}")
    for key, costs in all_results.items():
        avg = statistics.mean(costs)
        print(f"  {key:30s}  avg=${avg:.6f}  total=${sum(costs):.6f}")


if __name__ == "__main__":
    main()
