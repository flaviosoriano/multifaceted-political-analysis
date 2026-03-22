from __future__ import annotations

"""Helpers for loading and validating JSON stage configuration files."""

import json
from pathlib import Path
from typing import Any

from .paths import CONFIGS_DIR, resolve_repo_path


def load_json_config(path_like: str | Path) -> dict[str, Any]:
    """Load a JSON configuration file.

    Relative paths are resolved from the repository root so scripts can be run
    from any working directory.
    """

    path = resolve_repo_path(path_like)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stage_config_path(filename: str) -> Path:
    """Return the path for a config file in the configs directory."""

    return CONFIGS_DIR / filename

