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


def run_mini_command(extra_options: list[str]) -> subprocess.CompletedProcess:
    """Run the mini command with the given extra options."""
    cmd = [
        sys.executable,
        "-m",
        "minisweagent",
        "--exit-immediately",
        "-y",
        "--cost-limit",
        "0.1",
        "-t",
        SIMPLE_TASK,
        *extra_options,
    ]
    env = os.environ.copy()
    env["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = "1"
    return subprocess.run(cmd, timeout=120, env=env)


# =============================================================================
# LiteLLM Models (default, toolcall, response_toolcall)
# =============================================================================


def test_litellm_default():
    """Test with default litellm model class."""
    result = run_mini_command(["--model", "openai/gpt-5-mini"])
    assert result.returncode == 0


def test_litellm_toolcall():
    """Test with litellm_toolcall model class."""
    result = run_mini_command(["--model", "openai/gpt-5.2", "--model-class", "litellm_toolcall", "-c", "mini_toolcall"])
    assert result.returncode == 0


def test_litellm_response_toolcall():
    """Test with litellm_response_toolcall model class (OpenAI Responses API)."""
    result = run_mini_command(
        ["--model", "openai/gpt-5.2", "--model-class", "litellm_response_toolcall", "-c", "mini_toolcall"]
    )
    assert result.returncode == 0


# =============================================================================
# OpenRouter Models
# =============================================================================


def test_openrouter_default():
    """Test with default openrouter model class."""
    result = run_mini_command(["--model", "anthropic/claude-sonnet-4", "--model-class", "openrouter"])
    assert result.returncode == 0


def test_openrouter_toolcall():
    """Test with openrouter_toolcall model class."""
    result = run_mini_command(
        ["--model", "anthropic/claude-sonnet-4", "--model-class", "openrouter_toolcall", "-c", "mini_toolcall"]
    )
    assert result.returncode == 0


def test_openrouter_response_toolcall():
    """Test with openrouter_response_toolcall model class (OpenAI Responses API via OpenRouter)."""
    result = run_mini_command(
        ["--model", "openai/gpt-5.2", "--model-class", "openrouter_response_toolcall", "-c", "mini_toolcall"]
    )
    assert result.returncode == 0


# =============================================================================
# Portkey Models
# =============================================================================


def test_portkey_default():
    """Test with default portkey model class."""
    result = run_mini_command(["--model", "openai/gpt-5-mini", "--model-class", "portkey"])
    assert result.returncode == 0


def test_portkey_response():
    """Test with portkey_response model class (OpenAI Responses API via Portkey)."""
    result = run_mini_command(["--model", "openai/gpt-5.2", "--model-class", "portkey_response", "-c", "mini_toolcall"])
    assert result.returncode == 0


# =============================================================================
# Requesty Models
# =============================================================================


def test_requesty():
    """Test with requesty model class."""
    result = run_mini_command(["--model", "openai/gpt-5-mini", "--model-class", "requesty"])
    assert result.returncode == 0
