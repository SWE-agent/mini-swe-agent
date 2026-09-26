"""Superfast Decision Gate (shadow mode, off by default).

Concept and reference implementation by Andrea Bruno, licensed CC BY 4.0
(https://github.com/Andrea-Bruno/harness-superfast). The decision models
themselves (Von, OpenJev, Laya) are third-party open models; only this
integration architecture and routing method are covered by that credit.

A small, fast "System One" decision model (Von, or any Jev-compatible server)
classifies the pending turn in a single forward pass and returns typed,
calibrated answers without generating text. This module turns those answers into
a conservative routing recommendation.

Safety contract for this first increment:

    Off by default. Nothing runs unless ``SUPERFAST_ENABLED`` is truthy.
    Shadow mode. The caller only logs the recommendation; routing and the model
    call are never changed.
    Fail open. Any error, timeout, non-2xx response, or malformed body returns
    ``None`` ("no opinion"), so the agent behaves exactly as if the gate were off.
    No new dependencies. The local endpoint is reached with ``requests``, which
    the project already uses.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import requests

DEFAULT_ENDPOINT = "http://localhost:8000/v1/systemone"
DEFAULT_MODEL = "von-1.2.0"
DEFAULT_TIMEOUT_MS = 150

# The three typed questions asked in a single forward pass.
_QUESTIONS = {
    "needs_tool": {
        "type": "noul",
        "instructions": (
            "Does answering this request require taking an action with a tool "
            "(reading, writing, running, searching), rather than replying from "
            "what is already known?"
        ),
    },
    "answerable_from_context": {
        "type": "noul",
        "instructions": (
            "Can this request be answered from information already present in "
            "the conversation, without any new investigation?"
        ),
    },
    "intent": {
        "type": "choice",
        "instructions": "Classify the primary intent of the user request.",
        "criteria": {
            "code_change": "Create, edit, or delete code or files.",
            "code_question": "Explain or reason about code without changing it.",
            "command": "Run a command or operation.",
            "chat": "Casual conversation or a question needing no tools.",
            "other": "None of the above.",
        },
    },
}


def _enabled() -> bool:
    """Master switch: true only when SUPERFAST_ENABLED is a truthy string."""
    return os.getenv("SUPERFAST_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


def _endpoint() -> str:
    return os.getenv("SUPERFAST_ENDPOINT") or DEFAULT_ENDPOINT


def _model() -> str:
    return os.getenv("SUPERFAST_MODEL") or DEFAULT_MODEL


def _timeout_s() -> float:
    """Timeout in seconds, falling back to the default on a missing or bad value."""
    try:
        value = float(os.getenv("SUPERFAST_TIMEOUT_MS", ""))
    except ValueError:
        return DEFAULT_TIMEOUT_MS / 1000.0
    return value / 1000.0 if 0 < value <= 10_000 else DEFAULT_TIMEOUT_MS / 1000.0


def _noul(answers: dict, key: str) -> float | None:
    """Return a real, finite probability in [0, 1] for ``key``, else None (no evidence)."""
    value = answers.get(key, {}).get("noul")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if 0.0 <= value <= 1.0 else None


def _is_chat(answers: dict, floor: float) -> bool:
    """True only when intent.choice == 'chat' with a finite confidence >= floor."""
    intent = answers.get("intent", {})
    confidence = intent.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return False
    confidence = float(confidence)
    return intent.get("choice") == "chat" and 0.0 <= confidence <= 1.0 and confidence >= floor


def _derive_route(answers: dict) -> str:
    """Derive a conservative route; only recommend a fast route when decisive."""
    needs_tool = _noul(answers, "needs_tool")
    from_context = _noul(answers, "answerable_from_context")

    # A decisive tool need wins first: the harness must not skip required work.
    if needs_tool is not None and needs_tool >= 0.85:
        return "needs_tool"
    # Strongly answerable from context, with a present and low tool-need signal.
    if from_context is not None and from_context >= 0.85 and needs_tool is not None and needs_tool <= 0.3:
        return "answer_from_context"
    # Clearly chat, with a calibrated intent and a present, low tool-need signal.
    if _is_chat(answers, 0.5) and needs_tool is not None and needs_tool <= 0.2:
        return "plain_chat"
    return "unknown"


def classify_turn(state: str) -> dict | None:
    """Ask the System One backend about a turn. Returns ``{route, latency_ms}`` or None.

    Returns None ("no opinion") on any failure so the caller can fail open.
    Never raises.
    """
    if not state:
        return None
    started = time.perf_counter()
    try:
        resp = requests.post(
            _endpoint(),
            json={"model": _model(), "state": state, "questions": _QUESTIONS},
            timeout=_timeout_s(),
        )
        if not resp.ok:
            return None
        answers = resp.json().get("answers")
        if not isinstance(answers, dict):
            return None
    except Exception:  # noqa: BLE001 - fail open on any transport/parse error by design
        return None
    return {"route": _derive_route(answers), "latency_ms": round((time.perf_counter() - started) * 1000, 1)}


def _last_user_content(messages: list[dict]) -> str:
    """Return the content of the most recent user message, or '' if none."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def shadow_gate(messages: list[dict], logger: logging.Logger) -> None:
    """Shadow-mode hook: classify the pending turn off-thread and log the route.

    Call right before the model call. Returns immediately (adds no latency to the
    real turn). Does nothing unless ``SUPERFAST_ENABLED`` is set. The result is
    only logged; it never changes routing or skips the model call.
    """
    if not _enabled():
        return
    state = _last_user_content(messages)
    if not state:
        return

    def _run() -> None:
        result = classify_turn(state)
        if result:
            logger.info("superfast shadow: route=%s latency_ms=%s", result["route"], result["latency_ms"])

    threading.Thread(target=_run, daemon=True).start()
