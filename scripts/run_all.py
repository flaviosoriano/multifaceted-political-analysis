#!/usr/bin/env python3
from __future__ import annotations

"""Run multiple configured stages in sequence."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_json_config, stage_config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multiple pipeline stages from a single config.")
    parser.add_argument(
        "--config",
        default=str(stage_config_path("pipeline.json")),
        help="Path to the top-level pipeline JSON config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_json_config(args.config)

    from src.analysis.distance import run_distance_analysis
    from src.analysis.linguistic import run_yearly_linguistic_analysis
    from src.analysis.topic_modeling import run_topic_modeling
    from src.clustering.optimization import run_optimization_pipeline
    from src.embeddings.generation import generate_embeddings
    from src.preprocessing.cleaning import clean_combined_file, clean_yearly_directory

    if config.get("preprocessing", {}).get("run", False):
        preprocessing = config["preprocessing"]
        if preprocessing.get("run_combined", True):
            clean_combined_file(
                preprocessing["combined_input_path"],
                preprocessing["combined_output_path"],
                keep_first_speaker=preprocessing.get("keep_first_speaker", True),
            )
        if preprocessing.get("run_yearly", True):
            clean_yearly_directory(
                preprocessing["yearly_input_dir"],
                preprocessing["yearly_output_dir"],
                pattern=preprocessing.get("yearly_pattern", "discursos_*.csv"),
                keep_first_speaker=preprocessing.get("keep_first_speaker", True),
            )

    if config.get("linguistic_analysis", {}).get("run", False):
        linguistic = config["linguistic_analysis"]
        run_yearly_linguistic_analysis(
            linguistic["input_glob"],
            linguistic["output_dir"],
            spacy_model=linguistic.get("spacy_model", "pt_core_news_lg"),
        )

    if config.get("topic_modeling", {}).get("run", False):
        topic = config["topic_modeling"]
        run_topic_modeling(
            topic["input_glob"],
            topic["output_dir"],
            spacy_model=topic.get("spacy_model", "pt_core_news_sm"),
            sentence_model_name=topic.get("sentence_model_name", "distiluse-base-multilingual-cased-v2"),
            lemmatize_results=topic.get("lemmatize_results", False),
        )

    if config.get("embeddings", {}).get("run", False):
        embeddings = config["embeddings"]
        generate_embeddings(
            embeddings["csv_path"],
            output_base=embeddings.get("output_base"),
            model_name=embeddings.get("model_name", "Linq-AI-Research/Linq-Embed-Mistral"),
            batch_size=embeddings.get("batch_size", 4),
            add_id_column=embeddings.get("add_id_column", True),
            overwrite_csv=embeddings.get("overwrite_csv", False),
            device=embeddings.get("device", "cuda"),
        )

    if config.get("optimization", {}).get("run", False):
        optimization = config["optimization"]
        run_optimization_pipeline(
            optimization["csv_path"],
            optimization["embeddings_path"],
            output_dir=optimization["output_dir"],
            n_trials=optimization.get("n_trials", 1024),
            random_state=optimization.get("random_state", 42),
            timeout=optimization.get("timeout"),
        )

    if config.get("distance_analysis", {}).get("run", False):
        distance = config["distance_analysis"]
        run_distance_analysis(
            distance["csv_path"],
            distance["embeddings_path"],
            tables_dir=distance["tables_dir"],
            figures_dir=distance["figures_dir"],
            parties=distance.get("parties"),
        )


if __name__ == "__main__":
    main()
