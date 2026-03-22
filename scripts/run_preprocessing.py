#!/usr/bin/env python3
from __future__ import annotations

"""CLI wrapper for the preprocessing stage."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_json_config, stage_config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run combined and/or yearly speech preprocessing.")
    parser.add_argument(
        "--config",
        default=str(stage_config_path("preprocessing.json")),
        help="Path to the preprocessing JSON config.",
    )
    parser.add_argument(
        "--mode",
        choices=["combined", "yearly", "both"],
        default="both",
        help="Which preprocessing output mode to run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_json_config(args.config)

    from src.preprocessing.cleaning import clean_combined_file, clean_yearly_directory

    keep_first_speaker = config.get("keep_first_speaker", True)
    text_column = config.get("text_column", "transcricao")

    if args.mode in {"combined", "both"} and config.get("combined", {}).get("enabled", True):
        combined = config["combined"]
        summary = clean_combined_file(
            combined["input_path"],
            combined["output_path"],
            text_column=text_column,
            keep_first_speaker=keep_first_speaker,
        )
        print(
            f"Combined corpus cleaned: {summary.original_rows} -> {summary.cleaned_rows} rows "
            f"({summary.retained_pct:.1f}% retained)."
        )

    if args.mode in {"yearly", "both"} and config.get("yearly", {}).get("enabled", True):
        yearly = config["yearly"]
        summaries, summary_df = clean_yearly_directory(
            yearly["input_dir"],
            yearly["output_dir"],
            pattern=yearly.get("pattern", "discursos_*.csv"),
            text_column=text_column,
            keep_first_speaker=keep_first_speaker,
        )
        summary_path = yearly.get("summary_path")
        if summary_path:
            summary_df.to_csv(summary_path, index=False)
        print(f"Yearly preprocessing finished for {len(summaries)} files.")


if __name__ == "__main__":
    main()
