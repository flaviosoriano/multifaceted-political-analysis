from __future__ import annotations

"""Optuna-based deputy clustering optimization."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.clustering.deputies import (
    build_cluster_composition,
    build_cluster_profiles,
    plot_cluster_centroids,
    plot_deputy_clusters,
    prepare_deputy_embeddings,
    run_deputy_clustering,
)


def search_optimal_hyperparameters(
    high_dim_embeddings,
    *,
    n_trials: int = 1024,
    random_state: int = 42,
    timeout: int | None = None,
):
    """Run the Optuna search using the same objective behavior as the legacy script."""

    import hdbscan
    import numpy as np
    import optuna
    import umap.umap_ as umap
    from optuna.pruners import HyperbandPruner
    from optuna.samplers import TPESampler
    from sklearn.metrics import silhouette_score

    def objective(trial: optuna.Trial) -> float:
        umap_n_neighbors = trial.suggest_int("umap_n_neighbors", 5, 100, log=True)
        umap_n_components = trial.suggest_int("umap_n_components", 5, 100, log=True)
        umap_min_dist = trial.suggest_float("umap_min_dist", 0.0, 0.5, step=0.05)
        hdb_min_cluster_size = trial.suggest_int("hdb_min_cluster_size", 5, 30, log=True)
        hdb_min_samples = trial.suggest_int("hdb_min_samples", 1, 10, step=1)
        hdb_cluster_epsilon = trial.suggest_float("hdb_cluster_epsilon", 0.0, 0.5, step=0.05)

        reducer = umap.UMAP(
            n_components=umap_n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric="euclidean",
            random_state=random_state,
            verbose=False,
            n_jobs=1,
        )
        reduced = reducer.fit_transform(high_dim_embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=hdb_min_cluster_size,
            min_samples=hdb_min_samples,
            cluster_selection_epsilon=hdb_cluster_epsilon,
            metric="euclidean",
            cluster_selection_method="eom",
            gen_min_span_tree=False,
        )
        labels = clusterer.fit_predict(reduced)
        mask = labels != -1
        unique_clusters = np.unique(labels[mask])
        if len(unique_clusters) < 6:
            return -1

        score = silhouette_score(reduced[mask], labels[mask])
        trial.set_user_attr("num_clusters", len(unique_clusters))
        return score

    sampler = TPESampler(seed=random_state, multivariate=True)
    pruner = HyperbandPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    return study


def save_optimization_artifacts(study, output_dir: str | Path) -> dict[str, Path]:
    """Persist Optuna search results and figures."""

    import matplotlib.pyplot as plt
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_param_importances,
        plot_slice,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trials_path = output_path / "optuna_trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False, sep=";", decimal=",")

    params_path = output_path / "best_params_optuna.json"
    with params_path.open("w", encoding="utf-8") as handle:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, handle, ensure_ascii=False, indent=2)

    history_ax = plot_optimization_history(study)
    history_path = output_path / "optuna_history.png"
    history_ax.get_figure().savefig(history_path)
    plt.close(history_ax.get_figure())

    importance_ax = plot_param_importances(study)
    importance_path = output_path / "optuna_param_importances.png"
    importance_ax.get_figure().savefig(importance_path)
    plt.close(importance_ax.get_figure())

    slice_ax = plot_slice(study)
    slice_path = output_path / "optuna_slice.png"
    slice_ax.flat[0].get_figure().savefig(slice_path)
    plt.close(slice_ax.flat[0].get_figure())

    return {
        "trials": trials_path,
        "best_params": params_path,
        "history_plot": history_path,
        "importance_plot": importance_path,
        "slice_plot": slice_path,
    }


def run_optimization_pipeline(
    csv_path: str | Path,
    embeddings_path: str | Path,
    *,
    output_dir: str | Path,
    n_trials: int = 1024,
    random_state: int = 42,
    timeout: int | None = None,
) -> dict[str, object]:
    """Run the Optuna search and final clustering export workflow."""

    raw_df, deputy_df = prepare_deputy_embeddings(csv_path, embeddings_path)
    high_dim_embeddings = np.vstack(deputy_df["embedding"].values)
    study = search_optimal_hyperparameters(
        high_dim_embeddings,
        n_trials=n_trials,
        random_state=random_state,
        timeout=timeout,
    )
    artifacts = save_optimization_artifacts(study, output_dir)

    best = study.best_params
    result = run_deputy_clustering(
        raw_df,
        deputy_df,
        analysis_n_components=best["umap_n_components"],
        analysis_n_neighbors=best["umap_n_neighbors"],
        analysis_min_dist=best["umap_min_dist"],
        hdbscan_min_cluster_size=best["hdb_min_cluster_size"],
        hdbscan_min_samples=best["hdb_min_samples"],
        hdbscan_epsilon=best["hdb_cluster_epsilon"],
        viz_n_neighbors=best["umap_n_neighbors"],
        viz_min_dist=0.3,
    )

    output_path = Path(output_dir)
    plot_df = result["plot_df"]
    composition_df = build_cluster_composition(plot_df)
    profiles_df = build_cluster_profiles(raw_df, plot_df)

    plot_path = plot_deputy_clusters(plot_df, output_path / "mapa_politico_clusters_hdbscan_optuna.png")
    centroid_path = plot_cluster_centroids(plot_df, output_path / "clusters_deputados_optuna.png")
    composition_path = output_path / "analise_composicao_clusters_optuna.csv"
    profiles_path = output_path / "perfis_deputados_por_cluster_optuna.csv"

    composition_df.to_csv(composition_path, index=False, sep=";", decimal=",")
    profiles_df.to_csv(profiles_path, index=False, sep=";", decimal=",")

    return {
        "study": study,
        "artifacts": artifacts,
        "plot_path": plot_path,
        "centroid_path": centroid_path,
        "composition_path": composition_path,
        "profiles_path": profiles_path,
        "plot_df": plot_df,
    }
