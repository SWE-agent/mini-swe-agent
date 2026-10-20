from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.retry import (
    RetrySettings,
    backoff_seconds,
    provider_budget_exhausted,
    retry_after_seconds,
    run_with_retries,
)

WALL = 2000000000.0  # 2033-05-18 03:33:20 UTC


class _Err(Exception):
    def __init__(
        self,
        message: str,
        status: int | None = None,
        headers: dict[str, str] | None = None,
        source: str = "litellm",
    ):
        super().__init__(message)
        self.status_code = status
        if source == "response":
            self.response = SimpleNamespace(headers=headers or {})
        elif source == "direct":
            self.headers = headers or {}
        else:
            self.litellm_response_headers = headers or {}


class _Fatal(_Err):
    pass


def _run(
    failures: list[BaseException],
    settings: RetrySettings | None = None,
    abort: tuple[type[BaseException], ...] = (),
    sleep_error: BaseException | None = None,
) -> dict:
    now = {"t": 1000.0}
    sleeps: list[float] = []
    calls = {"n": 0}
    pending = list(failures)

    def call() -> str:
        calls["n"] += 1
        if pending:
            raise pending.pop(0)
        return "ok"

    def sleep(seconds: float) -> None:
        if sleep_error is not None:
            raise sleep_error
        sleeps.append(seconds)
        now["t"] += seconds + 0.25

    try:
        outcome = run_with_retries(
            call,
            settings or RetrySettings(),
            abort,
            clock=lambda: now["t"],
            sleep=sleep,
            wall_clock=lambda: WALL,
        )
    except BaseException as error:
        outcome = f"raised:{type(error).__name__}"
    return {"outcome": outcome, "calls": calls["n"], "sleeps": sleeps}


def _throttled(value: str, **kwargs) -> _Err:
    return _Err("slow down", 429, {"retry-after": value}, **kwargs)


def _credits(source: str, **kwargs) -> _Err:
    msg = (
        'litellm.BadRequestError: OpenrouterException - {"error":{"message":"This request requires more credits",'
        f'"code":402,"metadata":{{"limit_source":"{source}"}}}}'
    )
    return _Err(msg, 402, **kwargs)


def test_retry_after_delta_seconds_and_http_date():
    assert _run([_throttled("7")]) == {"outcome": "ok", "calls": 2, "sleeps": [7.0]}
    assert _run([_throttled("Wed, 18 May 2033 03:33:50 GMT")]) == {
        "outcome": "ok",
        "calls": 2,
        "sleeps": [30.0],
    }
    # Naive HTTP-date without timezone defaults to UTC
    assert retry_after_seconds(_throttled("Wed, 18 May 2033 03:33:45"), wall_clock=lambda: WALL) == 25.0


def test_retry_after_reads_response_or_direct_headers_and_caps_delay():
    assert _run([_Err("slow down", 429, {"Retry-After": "3"}, source="response")])["sleeps"] == [3.0]
    assert _run([_Err("slow down", 429, {"Retry-After": "5"}, source="direct")])["sleeps"] == [5.0]
    assert _run([_throttled("3600")])["sleeps"] == [300.0]


def test_fallback_exponential_backoff_and_exhaustion():
    assert _run([_throttled("soon")])["sleeps"] == [4.0]
    ok_run = _run([_Err("boom", 500) for _ in range(9)])
    assert ok_run == {
        "outcome": "ok",
        "calls": 10,
        "sleeps": [4.0, 4.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0, 60.0],
    }
    exhausted = _run([_Err("boom", 500) for _ in range(10)])
    assert exhausted["outcome"] == "raised:_Err"
    assert exhausted["calls"] == 10
    assert len(exhausted["sleeps"]) == 9


def test_abort_exceptions_and_provider_credit_exhaustion():
    assert _run([_Fatal("denied", 401)], abort=(_Fatal, KeyboardInterrupt)) == {
        "outcome": "raised:_Fatal",
        "calls": 1,
        "sleeps": [],
    }
    assert _run([_credits("openrouter_credits") for _ in range(5)]) == {
        "outcome": "raised:_Err",
        "calls": 1,
        "sleeps": [],
    }
    assert provider_budget_exhausted(_credits("openrouter_credits")) is True
    assert provider_budget_exhausted(_credits("openrouter_in_flight_budget")) is False
    # Temporary in-flight budget 402 and generic 402 still retry
    assert _run([_credits("openrouter_in_flight_budget", headers={"retry-after": "5"})]) == {
        "outcome": "ok",
        "calls": 2,
        "sleeps": [5.0],
    }
    assert _run([_Err("payment required", 402)]) == {
        "outcome": "ok",
        "calls": 2,
        "sleeps": [4.0],
    }


def test_time_budget_and_interrupt_passthrough():
    assert _run([_throttled("20")], settings=RetrySettings(max_total_seconds=10)) == {
        "outcome": "raised:_Err",
        "calls": 1,
        "sleeps": [],
    }
    assert _run([KeyboardInterrupt()], abort=(_Fatal, KeyboardInterrupt)) == {
        "outcome": "raised:KeyboardInterrupt",
        "calls": 1,
        "sleeps": [],
    }
    assert _run([_throttled("7")], sleep_error=KeyboardInterrupt()) == {
        "outcome": "raised:KeyboardInterrupt",
        "calls": 1,
        "sleeps": [],
    }


def test_env_override_and_litellm_model_retry_config():
    with patch.dict(os.environ, {"MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": "2"}):
        res = _run([_Err("boom", 500) for _ in range(5)])
        assert res["outcome"] == "raised:_Err"
        assert res["calls"] == 2
        assert res["sleeps"] == [4.0]

    assert backoff_seconds(1, RetrySettings()) == 4.0
    model = LitellmModel(model_name="gpt-4", retry={"max_attempts": 3, "backoff_min_seconds": 0.0})
    assert model.config.retry.max_attempts == 3
    with patch("minisweagent.models.litellm_model.litellm.completion", side_effect=_Err("boom", 500)) as mock_comp:
        with pytest.raises(_Err):
            model.query([{"role": "user", "content": "hi"}])
        assert mock_comp.call_count == 3
