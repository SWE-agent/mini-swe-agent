import pytest

from minisweagent.run.eval_module import EvalError, eval_expr


def test_eval_basic_arithmetic():
    assert eval_expr("1+2") == 3
    assert eval_expr("10-3") == 7
    assert eval_expr("4*5") == 20
    assert eval_expr("10/2") == 5


def test_eval_left_to_right_no_precedence():
    # Evaluator applies operators left-to-right, not PEMDAS.
    assert eval_expr("1+2*3") == 9
    assert eval_expr("10-2/2") == 4


def test_eval_whitespace_and_decimals():
    assert eval_expr(" 1 + 2 ") == 3
    assert eval_expr("1.5+2.5") == 4.0


def test_eval_division_by_zero():
    with pytest.raises(EvalError, match="division by zero"):
        eval_expr("1/0")


def test_eval_invalid_input():
    with pytest.raises(EvalError, match="empty expression"):
        eval_expr("")
    with pytest.raises(EvalError, match="invalid character"):
        eval_expr("1+a")
