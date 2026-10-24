"""Tests for the roulette and interleaving meta-models."""

from minisweagent.models.extra.roulette import InterleavingModel, RouletteModel
from minisweagent.models.test_models import make_output


def _det(model_name: str, n: int = 10) -> dict:
    """Build a deterministic sub-model config with n available responses."""
    return {
        "model_class": "deterministic",
        "model_name": model_name,
        "outputs": [make_output("ok", [{"command": "echo hi"}]) for _ in range(n)],
    }


def test_roulette_selects_only_from_configured_models():
    m = RouletteModel(model_kwargs=[_det("a"), _det("b")])
    chosen = {m.select_model().config.model_name for _ in range(200)}
    assert chosen <= {"a", "b"}


def test_roulette_uses_all_models():
    m = RouletteModel(model_kwargs=[_det("a"), _det("b"), _det("c")])
    chosen = {m.select_model().config.model_name for _ in range(500)}
    assert chosen == {"a", "b", "c"}


def test_interleaving_round_robin_without_sequence():
    """Round-robin ordering is observable through the public query() path."""
    m = InterleavingModel(model_kwargs=[_det("a"), _det("b"), _det("c")])
    msgs = [{"role": "user", "content": "hi"}]
    got = [m.query(msgs)["model_name"] for _ in range(6)]
    assert got == ["a", "b", "c", "a", "b", "c"]


def test_interleaving_explicit_sequence():
    """An explicit sequence repeats as documented, observable via query()."""
    m = InterleavingModel(model_kwargs=[_det("a"), _det("b")], sequence=[0, 0, 1])
    msgs = [{"role": "user", "content": "hi"}]
    got = [m.query(msgs)["model_name"] for _ in range(6)]
    assert got == ["a", "a", "b", "a", "a", "b"]


def test_query_records_model_name_and_counts_calls():
    m = InterleavingModel(model_kwargs=[_det("a"), _det("b")])
    msgs = [{"role": "user", "content": "hi"}]
    r1 = m.query(msgs)
    r2 = m.query(msgs)
    assert r1["model_name"] == "a"
    assert r2["model_name"] == "b"
    assert m._n_calls == 2  # noqa: SLF001


def test_serialize_contains_config_and_type():
    m = RouletteModel(model_kwargs=[_det("a")])
    data = m.serialize()
    assert data["info"]["config"]["model_type"].endswith("RouletteModel")
    assert "model_kwargs" in data["info"]["config"]["model"]


def test_get_template_vars_returns_dumped_config():
    m = RouletteModel(model_kwargs=[_det("a")])
    tv = m.get_template_vars()
    assert tv["model_name"] == "roulette"
    assert isinstance(tv["model_kwargs"], list)
