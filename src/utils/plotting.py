from __future__ import annotations

"""Small plotting helpers shared by the research modules."""

from pathlib import Path

import matplotlib.pyplot as plt


def set_default_style() -> None:
    """Apply the seaborn style used throughout the original notebooks."""

    plt.style.use("seaborn-v0_8-whitegrid")


def save_figure(path_like: str | Path, dpi: int = 300, bbox_inches: str | None = None) -> Path:
    """Persist the current matplotlib figure and ensure the parent exists."""

    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    if bbox_inches is None:
        plt.savefig(path, dpi=dpi)
    else:
        plt.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    return path

