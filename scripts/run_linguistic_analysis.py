#!/usr/bin/env python3
from __future__ import annotations

"""CLI wrapper for yearly linguistic analysis."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_json_config, stage_config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run yearly linguistic analysis over cleaned files.")
    parser.add_argument(
        "--config",
        default=str(stage_config_path("linguistic_analysis.json")),
        help="Path to the linguistic-analysis JSON config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_json_config(args.config)

    from src.analysis.linguistic import run_yearly_linguistic_analysis

    yearly_df, successful_years, failed_years = run_yearly_linguistic_analysis(
        config["input_glob"],
        config["output_dir"],
        spacy_model=config.get("spacy_model", "pt_core_news_lg"),
    )
    print(
        "Linguistic analysis complete:",
        f"{len(successful_years)} years succeeded, {len(failed_years)} failed,",
        f"aggregate table rows={len(yearly_df)}.",
    )


if __name__ == "__main__":
    main()
