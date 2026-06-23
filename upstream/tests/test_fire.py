#!/usr/bin/env python3
"""Fire tests: Real API integration tests that cost money.

################################################################################
#                                                                              #
#                         ⚠️  CRITICAL WARNING ⚠️                              #
#                                                                              #
#   THIS TEST FILE SHOULD NEVER BE RUN BY AN AI AGENT.                         #
#   IT REQUIRES EXPLICIT HUMAN REQUEST AND SUPERVISION.                        #
#                                                                              #
#   These tests make REAL API calls that:                                      #
#   - Cost real money (API usage fees)                                         #
#   - Require valid API keys for multiple providers                            #
#   - May have rate limits and quotas                                          #
#                                                                              #
#   To run: pytest tests/test_fire.py -v --run-fire                            #
#   Only run when explicitly requested by a human operator.                    #
#                                                                              #
################################################################################
"""

import os
import subprocess
import sys

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "fire: mark test as a fire test (real API calls)")


@pytest.fixture(autouse=True)
def skip_without_fire_flag(request):
    """Skip fire tests unless --run-fire is provided."""
    if not request.config.getoption("--run-fire", default=False):
        pytest.skip("Fire tests require --run-fire flag and cost real money")


SIMPLE_TASK = "Your job is to run `ls`, verify that you see files, then quit."

requires_openai = pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
requires_openrouter = pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
requires_portkey = pytest.mark.skipif(not os.environ.get("PORTKEY_API_KEY"), reason="PORTKEY_API_KEY not set")
requires_requesty = pytest.mark.skipif(not os.environ.get("REQUESTY_API_KEY"), reason="REQUESTY_API_KEY not set")


def run_mini_command(extra_options: list[str]) -> subprocess.CompletedProcess:
    """Run the mini command with the given extra options."""
    cmd = [
        sys.executable,
        "-m",
        "minisweagent",
        "--exit-immediately",
        "-y",
        "--cost-limit",
        "0.03",
        "-t",
        SIMPLE_TASK,
        *extra_options,
    ]
    env = os.environ.copy()
    env["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = "8"
    return subprocess.run(cmd, timeout=120, env=env)


# =============================================================================
# LiteLLM Models (default, toolcall, response_toolcall)
# =============================================================================


@requires_openai
def test_litellm_textbased():
    """Test with litellm_textbased model class."""
    result = run_mini_command(
        ["--model", "openai/gpt-5-mini", "--model-class", "litellm_textbased", "-c", "mini_textbased"]
    )
    assert result.returncode == 0


@requires_openai
def test_litellm_toolcall():
    """Test with litellm_toolcall model class."""
    result = run_mini_command(["--model", "openai/gpt-5.2"])
    assert result.returncode == 0


@requires_openai
def test_litellm_toolcall_explicit():
    """Test with litellm_toolcall model class."""
    result = run_mini_command(["--model", "openai/gpt-5.2", "--model-class", "litellm", "-c", "mini"])
    assert result.returncode == 0


@requires_openai
def test_litellm_response_toolcall():
    """Test with litellm_response_toolcall model class (OpenAI Responses API)."""
    result = run_mini_command(["--model", "openai/gpt-5.2", "--model-class", "litellm_response"])
    assert result.returncode == 0


# =============================================================================
# OpenRouter Models
# =============================================================================


@requires_openrouter
def test_openrouter_textbased():
    """Test with openrouter_textbased model class."""
    result = run_mini_command(
        ["--model", "anthropic/claude-sonnet-4", "--model-class", "openrouter_textbased", "-c", "mini_textbased"]
    )
    assert result.returncode == 0


@requires_openrouter
def test_openrouter_toolcall():
    """Test with openrouter_toolcall model class."""
    result = run_mini_command(["--model", "anthropic/claude-sonnet-4", "--model-class", "openrouter"])
    assert result.returncode == 0


@requires_openrouter
def test_openrouter_response_toolcall():
    """Test with openrouter_response_toolcall model class (OpenAI Responses API via OpenRouter)."""
    result = run_mini_command(["--model", "openai/gpt-5.2", "--model-class", "openrouter_response"])
    assert result.returncode == 0


# =============================================================================
# Portkey Models
# =============================================================================


@requires_portkey
def test_portkey_default():
    """Test with default portkey model class."""
    result = run_mini_command(["--model", "@openai/gpt-5-mini", "--model-class", "portkey"])
    assert result.returncode == 0


@requires_portkey
def test_portkey_response():
    """Test with portkey_response model class (OpenAI Responses API via Portkey)."""
    result = run_mini_command(["--model", "@openai/gpt-5.2", "--model-class", "portkey_response"])
    assert result.returncode == 0


# =============================================================================
# Requesty Models
# =============================================================================


@requires_requesty
def test_requesty():
    """Test with requesty model class."""
    result = run_mini_command(["--model", "openai/gpt-5-mini", "--model-class", "requesty"])
    assert result.returncode == 0
