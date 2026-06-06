"""Patch-repair loop: evaluate a patch, ask the Reviewer LM for corrections, retry."""

import base64
import logging
import time

import litellm

from minisweagent import Environment, Model
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.retry import retry
from minisweagent.patch_repair.prompts import (
    REVIEWER_SYSTEM_PROMPT,
    build_reviewer_prompt,
    extract_patch,
)

logger = logging.getLogger("patch_repair")

# ---------------------------------------------------------------------------
# Patch evaluation in the container
# ---------------------------------------------------------------------------


def evaluate_patch(patch: str, env: Environment, instance: dict) -> "tuple[bool, bool, str]":
    """Evaluate a patch inside *env* (the same container the agent used).

    1. ``git reset --hard HEAD``
    2. Write *patch* to ``/tmp/repair.diff`` (base64-encoded, avoids heredoc collision)
    3. ``git apply --check``
    4. ``git apply``
    5. Run regression tests (from *instance* ``FAIL_TO_PASS``)

    Returns ``(apply_ok, test_ok, trace)``.  *trace* collects all command
    outputs so the Reviewer can see what went wrong.
    """
    lines: list[str] = []
    _run = _runner(env, lines)

    # 1. Clean working tree ------------------------------------------------
    out = _run("git reset --hard HEAD")
    if out["returncode"] != 0:
        return False, False, "\n".join(lines)

    # 2. Write patch file (base64 to avoid heredoc delimiter issues) --------
    b64patch = base64.b64encode(patch.encode()).decode()
    write_cmd = f"echo '{b64patch}' | base64 -d > /tmp/repair.diff"
    out = _run(write_cmd)
    if out["returncode"] != 0:
        return False, False, "\n".join(lines)

    # 3. git apply --check -------------------------------------------------
    out = _run("git apply --check /tmp/repair.diff")
    if out["returncode"] != 0:
        return False, False, "\n".join(lines)

    # 4. git apply ---------------------------------------------------------
    out = _run("git apply /tmp/repair.diff")
    if out["returncode"] != 0:
        return True, False, "\n".join(lines)  # patch syntax ok, but doesn't apply

    # 5. Run tests ---------------------------------------------------------
    fail_to_pass: list[str] = instance.get("FAIL_TO_PASS", []) or []
    if not fail_to_pass:
        return True, True, "\n".join(lines)  # no tests to run

    test_cmd = "python -m pytest " + " ".join(fail_to_pass)
    out = _run(test_cmd)
    test_ok = out["returncode"] == 0
    return True, test_ok, "\n".join(lines)


def _runner(env: Environment, lines: list[str]):
    """Return a closure that runs a command in *env* and appends output to *lines*."""

    def _run(command: str) -> dict:
        result = env.execute({"command": command})
        lines.append(f"$ {command}")
        lines.append(result.get("output", ""))
        if result.get("returncode", -1) != 0 and result.get("exception_info"):
            lines.append(f"[exception] {result['exception_info']}")
        return result

    return _run


# ---------------------------------------------------------------------------
# Reviewer LM call (single-turn, plain-text, no tools)
# ---------------------------------------------------------------------------


def _call_reviewer_lm(model: Model, task: str, patch: str, trace: str) -> str | None:
    """Ask the Reviewer to fix *patch*.  Returns corrected patch text or ``None``.

    Uses litellm directly (without tools) so the model outputs a plain-text
    unified diff instead of a bash tool call.
    """
    messages = [
        model.format_message(role="system", content=REVIEWER_SYSTEM_PROMPT),
        model.format_message(role="user", content=build_reviewer_prompt(task, patch, trace)),
    ]
    prepared = model._prepare_messages_for_api(messages)

    # Drop tool-specific kwargs not applicable to a plain-text call
    safe_kwargs = {
        k: v for k, v in model.config.model_kwargs.items() if k not in ("parallel_tool_calls", "tool_choice")
    }

    try:
        for attempt in retry(logger=logger, abort_exceptions=model.abort_exceptions):
            with attempt:
                response = litellm.completion(
                    model=model.config.model_name,
                    messages=prepared,
                    **safe_kwargs,
                )
    except litellm.exceptions.AuthenticationError as e:
        e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
        logger.error("Reviewer LM authentication failed: %s", e.message)
        return None
    except Exception:
        logger.exception("Reviewer LM call failed")
        return None

    # Track cost (mirrors LitellmModel._calculate_cost pattern)
    try:
        cost = litellm.cost_calculator.completion_cost(response, model=model.config.model_name)
        if cost > 0:
            GLOBAL_MODEL_STATS.add(cost)
            logger.debug("Reviewer LM cost: $%.4f", cost)
    except Exception:
        logger.debug("Could not calculate reviewer LM cost")

    content: str = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    if usage:
        logger.info("Reviewer tokens: prompt=%s, completion=%s", usage.prompt_tokens, usage.completion_tokens)
    return content


# ---------------------------------------------------------------------------
# Main repair loop
# ---------------------------------------------------------------------------


def attempt_patch_repair(
    task: str,
    patch: str,
    env: Environment,
    model: Model,
    instance: dict,
    max_rounds: int = 2,
) -> dict:
    """Run the Coder–Reviewer repair loop (up to *max_rounds* retries).

    Parameters
    ----------
    task:
        Original task / PR description.
    patch:
        The patch produced by the agent's baseline run.
    env:
        Container environment (still alive after the agent finished).
    model:
        The same Model instance the agent used.
    instance:
        SWE-bench instance dict (used for ``FAIL_TO_PASS`` test list).
    max_rounds:
        Maximum number of repair attempts (1 = evaluate original patch only,
        no reviewer call).

    Returns
    -------
    dict
        ``patch`` (str), ``success`` (bool), ``rounds_used`` (int),
        ``trace`` (list[dict] — one entry per round with apply/test results
        and reviewer response).
    """
    _rounds: list[dict] = []
    current_patch = patch

    for rnd in range(1, max_rounds + 1):
        t0 = time.time()
        apply_ok, test_ok, eval_trace = evaluate_patch(current_patch, env, instance)
        elapsed = time.time() - t0

        entry: dict = {
            "round": rnd,
            "apply_ok": apply_ok,
            "test_ok": test_ok,
            "elapsed_s": round(elapsed, 1),
            "eval_trace": eval_trace[-4000:],  # keep trajectory compact
        }
        logger.info("Round %d: apply=%s test=%s (%.1fs)", rnd, apply_ok, test_ok, elapsed)

        if apply_ok and test_ok:
            entry["reviewer_called"] = False
            _rounds.append(entry)
            return {"patch": current_patch, "success": True, "rounds_used": rnd, "trace": _rounds}

        if rnd >= max_rounds:
            entry["reviewer_called"] = False
            _rounds.append(entry)
            break

        # Ask reviewer for a corrected patch ---------------------------------
        reviewer_output = _call_reviewer_lm(model, task, current_patch, eval_trace)
        entry["reviewer_called"] = True
        entry["reviewer_output_truncated"] = (reviewer_output or "")[-2000:]

        if reviewer_output is None:
            logger.warning("Reviewer call failed; stopping repair loop")
            _rounds.append(entry)
            break

        corrected = extract_patch(reviewer_output)
        if not corrected:
            logger.warning("Could not extract patch from reviewer response")
            entry["extraction_failed"] = True
            _rounds.append(entry)
            break

        current_patch = corrected
        _rounds.append(entry)

    return {"patch": current_patch, "success": False, "rounds_used": rnd, "trace": _rounds}
