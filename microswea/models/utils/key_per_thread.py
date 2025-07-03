"""Utility for anthropic where we need different keys for different parallel
agents to not mess up prompt caching.
"""

import threading
from typing import Any

_THREADS_THAT_USED_API_KEYS: list[Any] = []


def get_key_per_thread(api_keys: list[Any]) -> Any:
    """Choose key based on thread name. Returns None if no keys are available."""
    thread_name = threading.current_thread().name
    if thread_name not in _THREADS_THAT_USED_API_KEYS:
        _THREADS_THAT_USED_API_KEYS.append(thread_name)
    thread_idx = _THREADS_THAT_USED_API_KEYS.index(thread_name)
    key_idx = thread_idx % len(api_keys)
    return api_keys[key_idx] or None
