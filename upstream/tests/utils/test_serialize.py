from minisweagent.utils.serialize import UNSET, recursive_merge


def test_empty_input():
    """Test recursive_merge with no arguments returns empty dict."""
    assert recursive_merge() == {}


def test_single_dictionary():
    """Test recursive_merge with single dictionary returns copy of that dict."""
    assert recursive_merge({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_simple_override():
    """Test that later dictionaries override earlier ones for simple values."""
    assert recursive_merge({"a": 1, "b": 2}, {"b": 3, "c": 4}) == {"a": 1, "b": 3, "c": 4}


def test_nested_dict_merge():
    """Test that nested dictionaries are merged recursively."""
    dict1 = {"a": {"x": 1, "y": 2}, "b": 3}
    dict2 = {"a": {"y": 3, "z": 4}, "c": 5}
    assert recursive_merge(dict1, dict2) == {"a": {"x": 1, "y": 3, "z": 4}, "b": 3, "c": 5}


def test_deeply_nested_merge():
    """Test merging deeply nested dictionaries."""
    dict1 = {"a": {"b": {"c": 1, "d": 2}}}
    dict2 = {"a": {"b": {"d": 3, "e": 4}}}
    assert recursive_merge(dict1, dict2) == {"a": {"b": {"c": 1, "d": 3, "e": 4}}}


def test_override_dict_with_non_dict():
    """Test that non-dict values override dict values."""
    dict1 = {"a": {"x": 1, "y": 2}}
    dict2 = {"a": "string"}
    assert recursive_merge(dict1, dict2) == {"a": "string"}


def test_override_non_dict_with_dict():
    """Test that dict values override non-dict values."""
    dict1 = {"a": "string"}
    dict2 = {"a": {"x": 1, "y": 2}}
    assert recursive_merge(dict1, dict2) == {"a": {"x": 1, "y": 2}}


def test_multiple_dictionaries():
    """Test merging more than two dictionaries."""
    dict1 = {"a": 1, "b": 2}
    dict2 = {"b": 3, "c": 4}
    dict3 = {"c": 5, "d": 6}
    assert recursive_merge(dict1, dict2, dict3) == {"a": 1, "b": 3, "c": 5, "d": 6}


def test_multiple_nested_dictionaries():
    """Test merging multiple nested dictionaries where later ones take precedence."""
    dict1 = {"a": {"x": 1, "y": 2}}
    dict2 = {"a": {"y": 3, "z": 4}}
    dict3 = {"a": {"z": 5, "w": 6}}
    assert recursive_merge(dict1, dict2, dict3) == {"a": {"x": 1, "y": 3, "z": 5, "w": 6}}


def test_mixed_value_types():
    """Test merging dictionaries with various value types."""
    dict1 = {"int": 1, "str": "hello", "list": [1, 2], "dict": {"a": 1}}
    dict2 = {"int": 2, "bool": True, "none": None, "dict": {"b": 2}}
    assert recursive_merge(dict1, dict2) == {
        "int": 2,
        "str": "hello",
        "list": [1, 2],
        "bool": True,
        "none": None,
        "dict": {"a": 1, "b": 2},
    }


def test_list_override_not_merge():
    """Test that lists are overridden, not merged."""
    dict1 = {"list": [1, 2, 3]}
    dict2 = {"list": [4, 5]}
    assert recursive_merge(dict1, dict2) == {"list": [4, 5]}


def test_empty_nested_dicts():
    """Test merging with empty nested dictionaries."""
    dict1 = {"a": {}}
    dict2 = {"a": {"x": 1}}
    assert recursive_merge(dict1, dict2) == {"a": {"x": 1}}
    assert recursive_merge(dict2, dict1) == {"a": {"x": 1}}


def test_complex_nested_structure():
    """Test complex nested structure with multiple levels and mixed types."""
    dict1 = {
        "config": {"server": {"host": "localhost", "port": 8080}, "debug": True},
        "data": [1, 2, 3],
    }
    dict2 = {
        "config": {"server": {"port": 9090, "ssl": True}, "log_level": "info"},
        "extra": "value",
    }
    assert recursive_merge(dict1, dict2) == {
        "config": {
            "server": {"host": "localhost", "port": 9090, "ssl": True},
            "debug": True,
            "log_level": "info",
        },
        "data": [1, 2, 3],
        "extra": "value",
    }


def test_original_dicts_unchanged():
    """Test that original dictionaries are not modified."""
    dict1 = {"a": {"x": 1}}
    dict2 = {"a": {"y": 2}}
    dict1_copy = dict1.copy()
    dict2_copy = dict2.copy()
    recursive_merge(dict1, dict2)
    assert dict1 == dict1_copy
    assert dict2 == dict2_copy


def test_none_dictionaries_skipped():
    """Test that None dictionaries are skipped during merge."""
    assert recursive_merge(None) == {}
    assert recursive_merge({"a": 1}, None, {"b": 2}) == {"a": 1, "b": 2}
    assert recursive_merge(None, {"a": 1}, None, {"b": 2}, None) == {"a": 1, "b": 2}


def test_unset_values_skipped():
    """Test that UNSET values are skipped during merge."""
    assert recursive_merge({"a": 1, "b": UNSET}) == {"a": 1}
    assert recursive_merge({"a": 1}, {"a": UNSET, "b": 2}) == {"a": 1, "b": 2}
    assert recursive_merge(
        {"a": {"x": 1, "y": 2}},
        {"a": {"y": UNSET, "z": 3}, "b": UNSET, "c": 4},
    ) == {"a": {"x": 1, "y": 2, "z": 3}, "c": 4}


def test_nested_dict_with_only_unset():
    """Test that nested dicts containing only UNSET values are filtered out."""
    # This is the bug case from mini.py where task=None resulted in {"run": {"task": UNSET}}
    assert recursive_merge({"run": {"task": UNSET}}) == {"run": {}}
    assert recursive_merge({"a": 1}, {"run": {"task": UNSET}}) == {"a": 1, "run": {}}
    # Deeply nested UNSET values should also be filtered
    assert recursive_merge({"a": {"b": {"c": UNSET}}}) == {"a": {"b": {}}}
    # Mixed: some UNSET, some real values
    assert recursive_merge({"run": {"task": UNSET, "other": "value"}}) == {"run": {"other": "value"}}
