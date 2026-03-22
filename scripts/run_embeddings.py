#!/usr/bin/env python3
from __future__ import annotations

"""CLI wrapper for embedding generation."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_json_config, stage_config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate raw and normalized speech embeddings.")
    parser.add_argument(
        "--config",
        default=str(stage_config_path("embeddings.json")),
        help="Path to the embeddings JSON config.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_json_config(args.config)

    from src.embeddings.generation import generate_embeddings

    artifacts = generate_embeddings(
        config["csv_path"],
        output_base=config.get("output_base"),
        model_name=config.get("model_name", "Linq-AI-Research/Linq-Embed-Mistral"),
        text_column=config.get("text_column", "transcricao_limpa"),
        batch_size=config.get("batch_size", 4),
        add_id_column=config.get("add_id_column", True),
        overwrite_csv=config.get("overwrite_csv", False),
        device=config.get("device", "cuda"),
        limit=args.limit,
    )
    print("Embedding artifacts written:")
    print(f"  raw: {artifacts.raw_embeddings_path}")
    print(f"  norm: {artifacts.normalized_embeddings_path}")
    print(f"  ids: {artifacts.ids_path}")
    print(f"  csv: {artifacts.enriched_csv_path}")


if __name__ == "__main__":
    main()
