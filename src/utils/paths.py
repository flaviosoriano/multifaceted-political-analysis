from __future__ import annotations

"""Filesystem helpers for locating repository resources."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
DOCS_DIR = REPO_ROOT / "docs"
SCRIPTS_DIR = REPO_ROOT / "scripts"


def resolve_repo_path(path_like: str | Path) -> Path:
    """Resolve a repository-relative path against the project root."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()

