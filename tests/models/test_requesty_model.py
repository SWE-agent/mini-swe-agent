"""Regression tests for RequestyModel cost accounting.

The Requesty API reports cost through ``usage.cost``. Every other model class in
this package rejects non-positive costs (``cost <= 0.0`` / ``assert cost > 0.0``)
because ``GLOBAL_MODEL_STATS.add()`` accumulates whatever the model hands back.
These tests pin that contract for the Requesty model.
"""

import pytest

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.requesty_model import RequestyAPIError, RequestyModel


def _model() -> RequestyModel:
    return RequestyModel(model_name="test/model")


def test_requesty_negative_cost_is_rejected():
    """A negative usage cost must not slip past the cost guard.

    ``cost == 0.0`` only rejects exactly zero, so a negative cost (bad provider
    accounting) used to be returned verbatim and silently lowered the global
    cost accumulator, which also lets a configured cost limit be evaded.
    """
    model = _model()
    before = GLOBAL_MODEL_STATS.cost

    with pytest.raises(RequestyAPIError):
        model._calculate_cost({"usage": {"cost": -0.01}})

    assert GLOBAL_MODEL_STATS.cost == before


def test_requesty_zero_cost_is_rejected():
    """Zero cost stays rejected (behavior preserved)."""
    with pytest.raises(RequestyAPIError):
        _model()._calculate_cost({"usage": {"cost": 0.0}})


def test_requesty_missing_cost_is_rejected():
    """A response without a cost field stays rejected (behavior preserved)."""
    with pytest.raises(RequestyAPIError):
        _model()._calculate_cost({"usage": {}})


def test_requesty_positive_cost_is_returned():
    """A valid positive cost is still returned unchanged."""
    assert _model()._calculate_cost({"usage": {"cost": 0.01}}) == {"cost": 0.01}
