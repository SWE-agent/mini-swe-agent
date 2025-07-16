"""Configuration files and utilities for micro-SWE-agent."""

import os
from pathlib import Path

builtin_config_dir = Path(__file__).parent


def get_config_path(config_spec: str | Path) -> Path:
    """Get the path to a config file."""
    candidates = [
        Path(config_spec),
        Path(os.getenv("MSWEA_CONFIG_DIR", ".")) / config_spec,
        builtin_config_dir / config_spec,
        builtin_config_dir / "extra" / config_spec,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find config file for {config_spec} (tried: {candidates})")


__all__ = ["builtin_config_dir", "get_config_path"]
