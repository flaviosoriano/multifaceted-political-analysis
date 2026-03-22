#!/usr/bin/env python3
from __future__ import annotations

"""CLI wrapper for yearly BERTopic analysis."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_json_config, stage_config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run yearly BERTopic analysis over cleaned files.")
    parser.add_argument(
        "--config",
        default=str(stage_config_path("topic_modeling.json")),
        help="Path to the topic-modeling JSON config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_json_config(args.config)

    from src.analysis.topic_modeling import run_topic_modeling

    successful_years, failed_years = run_topic_modeling(
        config["input_glob"],
        config["output_dir"],
        spacy_model=config.get("spacy_model", "pt_core_news_sm"),
        sentence_model_name=config.get("sentence_model_name", "distiluse-base-multilingual-cased-v2"),
        lemmatize_results=config.get("lemmatize_results", False),
        min_words=config.get("min_words", 10),
        min_topic_size=config.get("min_topic_size", 10),
    )
    print(f"Topic modeling complete: {len(successful_years)} years succeeded, {len(failed_years)} failed.")


if __name__ == "__main__":
    main()
