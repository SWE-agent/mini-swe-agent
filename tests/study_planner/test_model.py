import pytest

from study_planner.model import build_model


def test_build_model_requires_api_key(monkeypatch):
    monkeypatch.delenv("NRP_API_KEY", raising=False)

    with pytest.raises(
        ValueError,
        match="NRP_API_KEY",
    ):
        build_model()


def test_build_model_uses_environment_configuration(
    monkeypatch,
):
    monkeypatch.setenv("NRP_API_KEY", "test-key")
    monkeypatch.setenv("NRP_MODEL", "gpt-oss")
    monkeypatch.setenv(
        "NRP_API_BASE",
        "https://example.com/v1",
    )

    model = build_model()

    assert model.config.model_name == "openai/gpt-oss"

    assert model.config.model_kwargs["api_key"] == "test-key"

    assert model.config.model_kwargs["api_base"] == "https://example.com/v1"
