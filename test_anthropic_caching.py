"""Test Anthropic prompt caching by simulating a growing multi-turn conversation.

Each scenario builds up a conversation: call 1 sends system + user1, call 2
sends the same prefix + assistant1 + user2, call 3 adds assistant2 + user3, etc.
If caching works, calls 2+ should show cache_read_input_tokens > 0 because the
prefix was already seen (and cached) in the previous call.

We compare several scenarios:
1. cache_control on the system message
2. Top-level cache_control via extra_body
3. cache_control on the last user message
4. cache_control on both system + last user message
"""

import os
import time

import anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# MODEL = os.getenv("TEST_MODEL", "claude-sonnet-4-5")
# MODEL = os.getenv("TEST_MODEL", "claude-opus-4-6")
MODEL = os.getenv("TEST_MODEL", "claude-4-6-opus-genai")
API_BASE = os.getenv(
    "TEST_API_BASE",
    "https://api.llama.com/experimental/passthrough/anthropic",
)
API_KEY = os.getenv("TEST_API_KEY", os.getenv("LLAMA_API_KEY", ""))
# API_BASE = None
# API_KEY = os.getenv("TEST_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))
NUM_TURNS = int(os.getenv("TEST_NUM_TURNS", "3"))

LONG_SYSTEM_PROMPT_BASE = "you are a helpful coding assistant. " * 2000

_scenario_counter = 0


def make_system_prompt() -> str:
    """Return a unique system prompt per scenario to avoid cross-scenario cache hits."""
    global _scenario_counter
    _scenario_counter += 1
    return LONG_SYSTEM_PROMPT_BASE + f"\nScenario {_scenario_counter}. Always answer concisely in one sentence."


FOLLOW_UP_QUESTIONS = [
    "What is 2+2? Answer in one word." * 100,
    "Now multiply that by 3. Answer in one word." * 100,
    "Subtract 5 from that. Answer in one word." * 100,
    "Square it. Answer in one word." * 100,
    "Is the result prime? Answer yes or no." * 100,
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_client() -> anthropic.Anthropic:
    kwargs = {"api_key": API_KEY}
    if API_BASE:
        kwargs["base_url"] = API_BASE
    return anthropic.Anthropic(**kwargs)


def call_model(
    client: anthropic.Anthropic,
    system,
    messages: list[dict],
    extra_body: dict | None = None,
) -> anthropic.types.Message:
    kwargs: dict = {
        "model": MODEL,
        "max_tokens": 1024,
        "system": system,
        "messages": messages,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    return client.messages.create(**kwargs)


def print_usage(turn: int, elapsed: float, response: anthropic.types.Message, num_messages: int):
    usage = response.usage
    reply = response.content[0].text[:80] if response.content else "(empty)"
    cache_create = getattr(usage, "cache_creation_input_tokens", None)
    cache_read = getattr(usage, "cache_read_input_tokens", None)

    print(f"\n  Turn {turn} ({elapsed:.1f}s, {num_messages} messages):")
    print(f"    Reply: {reply}")
    print(f"    input_tokens:                {usage.input_tokens}")
    print(f"    output_tokens:               {usage.output_tokens}")
    print(f"    cache_creation_input_tokens:  {cache_create}")
    print(f"    cache_read_input_tokens:      {cache_read}")
    if turn == 1:
        print(f"    raw_usage:                   {usage}")
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": cache_create,
        "cache_read_input_tokens": cache_read,
    }


def run_conversation(*args, **kwargs) -> list[dict]:
    try:
        return _run_conversation(*args, **kwargs)
    except Exception as e:
        print(f"Error running conversation: {e}")
        return []


def _run_conversation(
    client: anthropic.Anthropic,
    name: str,
    system,
    num_turns: int = NUM_TURNS,
    extra_body: dict | None = None,
    cache_last_user: bool = False,
) -> list[dict]:
    print(f"\n{'=' * 70}")
    print(f"SCENARIO: {name}")
    print(f"{'=' * 70}")

    messages: list[dict] = []
    results: list[dict] = []

    for i in range(num_turns):
        if cache_last_user:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": FOLLOW_UP_QUESTIONS[i],
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": FOLLOW_UP_QUESTIONS[i]})

        t0 = time.time()
        response = call_model(client, system, messages, extra_body=extra_body)
        elapsed = time.time() - t0

        info = print_usage(i + 1, elapsed, response, len(messages))
        results.append(info)

        assistant_text = response.content[0].text if response.content else ""
        messages.append({"role": "assistant", "content": assistant_text})

    return results


def main():
    print(f"Model:     {MODEL}")
    print(f"API base:  {API_BASE or '(default)'}")
    print(f"Turns per scenario: {NUM_TURNS}")
    print(f"System prompt base length: {len(LONG_SYSTEM_PROMPT_BASE)} chars")

    client = make_client()
    all_results: dict[str, list[dict]] = {}

    # all_results["no_cache"] = run_conversation(
    #     client,
    #     "No cache_control markers (baseline)",
    #     make_system_prompt(),
    # )

    sp = make_system_prompt()
    all_results["cache_system"] = run_conversation(
        client,
        "cache_control on system message",
        [
            {
                "type": "text",
                "text": sp,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    )

    all_results["cache_toplevel"] = run_conversation(
        client,
        "top-level cache_control (extra_body)",
        make_system_prompt(),
        extra_body={"cache_control": {"type": "ephemeral"}},
    )

    all_results["cache_last_user"] = run_conversation(
        client,
        "cache_control on last user message",
        make_system_prompt(),
        cache_last_user=True,
    )

    sp = make_system_prompt()
    all_results["cache_system_and_user"] = run_conversation(
        client,
        "cache_control on system + last user message",
        [
            {
                "type": "text",
                "text": sp,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        cache_last_user=True,
    )

    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'':30s}  {'input':>6s}  {'cache_create':>12s}  {'cache_read':>10s}")
    for key, results in all_results.items():
        for i, r in enumerate(results):
            label = f"{key} turn {i + 1}"
            print(
                f"  {label:30s}  {r['input_tokens']:6d}"
                f"  {r['cache_creation_input_tokens'] or 0:12d}"
                f"  {r['cache_read_input_tokens'] or 0:10d}"
            )


if __name__ == "__main__":
    main()
