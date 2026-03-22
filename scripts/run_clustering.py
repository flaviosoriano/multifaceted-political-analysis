#!/usr/bin/env python3
from __future__ import annotations

"""CLI wrapper for party and deputy clustering stages."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_json_config, stage_config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deputy/party clustering and optional optimization.")
    parser.add_argument(
        "--config",
        default=str(stage_config_path("clustering.json")),
        help="Path to the clustering JSON config.",
    )
    parser.add_argument(
        "--stage",
        choices=["deputy", "party", "optuna", "all"],
        default="all",
        help="Which clustering workflow to run.",
    )
    return parser


def run_deputy_stage(config: dict) -> None:
    from src.clustering.deputies import (
        build_cluster_composition,
        build_cluster_distribution,
        build_cluster_profiles,
        cluster_deputies_fixed,
        plot_cluster_centroids,
        plot_deputy_clusters,
    )

    result = cluster_deputies_fixed(config["csv_path"], config["embeddings_path"])
    plot_df = result["plot_df"]
    deputy_cfg = config["deputy"]
    build_cluster_distribution(plot_df).to_csv(deputy_cfg["distribution_path"], index=False)
    build_cluster_composition(plot_df).to_csv(deputy_cfg["composition_path"], index=False)
    build_cluster_profiles(result["raw_df"], plot_df).to_csv(deputy_cfg["profiles_path"], index=False)
    plot_deputy_clusters(plot_df, deputy_cfg["plot_path"])
    plot_cluster_centroids(plot_df, deputy_cfg["centroid_path"])
    print("Deputy clustering complete.")


def run_party_stage(config: dict) -> None:
    from src.clustering.party import (
        cluster_party_years,
        plot_party_clusters,
        plot_party_trajectories,
        project_party_trajectories,
    )

    trajectories = project_party_trajectories(config["csv_path"], config["embeddings_path"])
    plot_party_trajectories(trajectories, config["party_trajectories"]["plot_path"])

    party_plot_df, summary_df = cluster_party_years(config["csv_path"], config["embeddings_path"])
    summary_df.to_csv(config["party_clusters"]["summary_path"], index=False)
    plot_party_clusters(party_plot_df, config["party_clusters"]["plot_path"])
    print("Party trajectory and party clustering complete.")


def run_optuna_stage(config: dict) -> None:
    from src.clustering.optimization import run_optimization_pipeline

    optimization_cfg = config["optimization"]
    run_optimization_pipeline(
        config["csv_path"],
        config["embeddings_path"],
        output_dir=optimization_cfg["output_dir"],
        n_trials=optimization_cfg.get("n_trials", 1024),
        random_state=optimization_cfg.get("random_state", 42),
        timeout=optimization_cfg.get("timeout"),
    )
    print("Optuna optimization complete.")


def main() -> None:
    args = build_parser().parse_args()
    config = load_json_config(args.config)

    if args.stage in {"deputy", "all"}:
        run_deputy_stage(config)
    if args.stage in {"party", "all"}:
        run_party_stage(config)
    if args.stage in {"optuna", "all"} and config.get("optimization", {}).get("enabled", True):
        run_optuna_stage(config)


if __name__ == "__main__":
    main()
