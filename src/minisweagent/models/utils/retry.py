"""Retry utility for model queries."""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable, Mapping
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import TypeVar

from pydantic import BaseModel
from tenacity import RetryCallState, Retrying, before_sleep_log, stop_after_attempt

CREDITS_LIMIT_SOURCE = re.compile(r'"limit_source"\s*:\s*"openrouter_credits"')

T = TypeVar("T")


class RetrySettings(BaseModel):
    """Retry policy for one model call."""

    max_attempts: int = 10
    """Total attempts before giving up, including the initial call."""
    backoff_min_seconds: float = 4
    backoff_max_seconds: float = 60
    """Bounds of the fallback exponential delay ``2 ** (attempt - 1)`` when no Retry-After header is present."""
    retry_after_max_seconds: float = 300
    """Maximum Retry-After delay honoured; larger values are clamped to this cap."""
    max_total_seconds: float = 900
    """Wall-clock time budget for one model call including waits."""


def _headers(error: BaseException) -> Mapping[str, str]:
    """Extract response headers from an SDK/LiteLLM exception if present."""
    for source in (
        getattr(error, "litellm_response_headers", None),
        getattr(getattr(error, "response", None), "headers", None),
        getattr(error, "headers", None),
    ):
        if source:
            return {str(key).lower(): str(value) for key, value in dict(source).items()}
    return {}


def retry_after_seconds(error: BaseException, wall_clock: Callable[[], float] = time.time) -> float | None:
    """Parse ``Retry-After`` (delta-seconds or HTTP-date) from ``error`` response headers."""
    value = _headers(error).get("retry-after", "").strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, when.timestamp() - wall_clock())


def backoff_seconds(attempt: int, settings: RetrySettings) -> float:
    """Fallback exponential schedule ``2 ** (attempt - 1)`` clamped to configured bounds."""
    return float(min(max(2 ** (attempt - 1), settings.backoff_min_seconds), settings.backoff_max_seconds))


def provider_budget_exhausted(error: BaseException) -> bool:
    """Return True when the provider rejected the call due to exhausted account credits (HTTP 402)."""
    return getattr(error, "status_code", None) == 402 and CREDITS_LIMIT_SOURCE.search(str(error)) is not None


def _scheduled_delay(
    error: BaseException,
    attempt: int,
    settings: RetrySettings,
    wall_clock: Callable[[], float],
) -> float:
    requested = retry_after_seconds(error, wall_clock)
    if requested is None:
        return backoff_seconds(attempt, settings)
    return min(requested, settings.retry_after_max_seconds)


def retry(
    *,
    logger: logging.Logger,
    abort_exceptions: list[type[BaseException]] | tuple[type[BaseException], ...],
    settings: RetrySettings | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    wall_clock: Callable[[], float] = time.time,
) -> Retrying:
    """Build a ``tenacity.Retrying`` loop that honours ``Retry-After`` and ``RetrySettings``."""
    cfg = settings or RetrySettings()
    env_attempts = os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT")
    max_attempts = int(env_attempts) if env_attempts else cfg.max_attempts
    abort_tuple = tuple(abort_exceptions)
    started = clock()

    def should_retry(retry_state: RetryCallState) -> bool:
        if retry_state.outcome is None or not retry_state.outcome.failed:
            return False
        error = retry_state.outcome.exception()
        if error is None or not isinstance(error, Exception) or isinstance(error, abort_tuple):
            return False
        if provider_budget_exhausted(error):
            return False
        if retry_state.attempt_number >= max_attempts:
            return False
        delay = _scheduled_delay(error, retry_state.attempt_number, cfg, wall_clock)
        return clock() - started + delay <= cfg.max_total_seconds

    def wait(retry_state: RetryCallState) -> float:
        error = retry_state.outcome.exception()  # type: ignore[union-attr]
        return _scheduled_delay(error, retry_state.attempt_number, cfg, wall_clock)  # type: ignore[arg-type]

    return Retrying(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait,
        sleep=sleep,
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=should_retry,
    )


def run_with_retries(
    call: Callable[[], T],
    settings: RetrySettings | None = None,
    abort_exceptions: tuple[type[BaseException], ...] = (),
    *,
    logger: logging.Logger | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    wall_clock: Callable[[], float] = time.time,
) -> T:
    """Run ``call`` until it returns, retrying recoverable failures under ``settings``."""
    log = logger or logging.getLogger("retry_model")
    for attempt in retry(
        logger=log,
        abort_exceptions=abort_exceptions,
        settings=settings,
        clock=clock,
        sleep=sleep,
        wall_clock=wall_clock,
    ):
        with attempt:
            return call()
    raise RuntimeError("unreachable")  # pragma: no cover
