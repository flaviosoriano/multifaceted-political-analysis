#!/usr/bin/env python3
from __future__ import annotations

"""CLI wrapper for the yearwise party-distance experiment."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_json_config, stage_config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run yearwise Euclidean distance analysis for party embeddings.")
    parser.add_argument(
        "--config",
        default=str(stage_config_path("distance_analysis.json")),
        help="Path to the distance-analysis JSON config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_json_config(args.config)

    from src.analysis.distance import run_distance_analysis

    matrices = run_distance_analysis(
        config["csv_path"],
        config["embeddings_path"],
        parties=config.get("parties"),
        tables_dir=config["tables_dir"],
        figures_dir=config["figures_dir"],
    )
    print(f"Distance analysis complete for {len(matrices)} years.")


if __name__ == "__main__":
    main()
