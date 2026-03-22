from __future__ import annotations

"""Deputy aggregation and HDBSCAN clustering logic."""

from pathlib import Path
from typing import Any

import colorcet as cc
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap

from src.embeddings.aggregation import (
    aggregate_by_deputy,
    embeddings_matrix,
    latest_party_by_deputy,
    load_corpus_with_embeddings,
)
from src.utils.constants import PARTY_COLORS
from src.utils.plotting import save_figure, set_default_style


PROFILE_COLUMNS = [
    "id_deputado",
    "nome",
    "partido",
    "uf",
    "sexo",
    "municipioNascimento",
    "ufNascimento",
    "dataNascimento",
    "escolaridade",
    "situacao",
    "condicaoEleitoral",
    "descricaoStatus",
    "idLegislatura",
]


def prepare_deputy_embeddings(csv_path: str | Path, embeddings_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the speech corpus and aggregate embeddings by deputy."""

    raw_df = load_corpus_with_embeddings(csv_path, embeddings_path, text_column="transcricao_limpa")
    raw_df = raw_df[raw_df["partido"].isin(PARTY_COLORS.keys())].copy()
    deputy_df = aggregate_by_deputy(raw_df, party_filter=False)
    return raw_df, deputy_df


def run_deputy_clustering(
    raw_df: pd.DataFrame,
    deputy_df: pd.DataFrame,
    *,
    analysis_n_components: int,
    analysis_n_neighbors: int,
    analysis_min_dist: float,
    analysis_metric: str = "euclidean",
    analysis_random_state: int = 42,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
    hdbscan_epsilon: float,
    hdbscan_metric: str = "euclidean",
    viz_n_components: int = 2,
    viz_n_neighbors: int,
    viz_min_dist: float,
    viz_random_state: int = 42,
) -> dict[str, Any]:
    """Run the deputy UMAP + HDBSCAN pipeline with configurable parameters."""

    high_dim = embeddings_matrix(deputy_df)

    reducer_analysis = umap.UMAP(
        n_components=analysis_n_components,
        n_neighbors=analysis_n_neighbors,
        min_dist=analysis_min_dist,
        metric=analysis_metric,
        random_state=analysis_random_state,
    )
    reduced_for_clustering = reducer_analysis.fit_transform(high_dim)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        cluster_selection_epsilon=hdbscan_epsilon,
        cluster_selection_method="eom",
        metric=hdbscan_metric,
    )
    cluster_labels = clusterer.fit_predict(reduced_for_clustering)

    reducer_viz = umap.UMAP(
        n_components=viz_n_components,
        n_neighbors=viz_n_neighbors,
        min_dist=viz_min_dist,
        random_state=viz_random_state,
    )
    embedding_2d = reducer_viz.fit_transform(reduced_for_clustering)

    plot_df = pd.DataFrame(embedding_2d, columns=["x", "y"])
    plot_df["nome"] = deputy_df["nome"]
    plot_df["cluster"] = cluster_labels.astype(int)
    plot_df["partido"] = plot_df["nome"].map(latest_party_by_deputy(raw_df))

    return {
        "raw_df": raw_df,
        "deputy_df": deputy_df,
        "high_dim_embeddings": high_dim,
        "analysis_embeddings": reduced_for_clustering,
        "plot_df": plot_df,
        "cluster_labels": cluster_labels,
    }


def cluster_deputies_fixed(csv_path: str | Path, embeddings_path: str | Path) -> dict[str, Any]:
    """Run the fixed-parameter clustering from ``cluster_deputados.ipynb``."""

    raw_df, deputy_df = prepare_deputy_embeddings(csv_path, embeddings_path)
    return run_deputy_clustering(
        raw_df,
        deputy_df,
        analysis_n_components=10,
        analysis_n_neighbors=5,
        analysis_min_dist=0.0,
        hdbscan_min_cluster_size=6,
        hdbscan_min_samples=10,
        hdbscan_epsilon=0.15,
        viz_n_neighbors=5,
        viz_min_dist=0.3,
    )


def build_cluster_distribution(plot_df: pd.DataFrame) -> pd.DataFrame:
    """Return cluster counts and percentages."""

    distribution = plot_df.groupby("cluster").size().reset_index(name="count")
    distribution["percentual"] = distribution["count"] / distribution["count"].sum() * 100
    return distribution


def build_cluster_composition(plot_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-cluster party composition statistics."""

    analysis_rows: list[dict[str, object]] = []
    for cluster_id in sorted(plot_df["cluster"].unique()):
        cluster_df = plot_df[plot_df["cluster"] == cluster_id]
        cluster_size = len(cluster_df)
        composition = cluster_df["partido"].value_counts()
        for party, count in composition.items():
            analysis_rows.append(
                {
                    "cluster_id": cluster_id,
                    "partido": party,
                    "contagem_deputados": count,
                    "percentual_no_cluster (%)": round(count / cluster_size * 100, 2),
                    "tamanho_total_cluster": cluster_size,
                }
            )
    return pd.DataFrame(analysis_rows)


def build_cluster_profiles(raw_df: pd.DataFrame, plot_df: pd.DataFrame) -> pd.DataFrame:
    """Join cluster IDs back to the latest deputy metadata profile."""

    available_columns = [column for column in PROFILE_COLUMNS if column in raw_df.columns]
    profiles = raw_df[available_columns].drop_duplicates("nome", keep="last")
    cluster_df = plot_df[["nome", "cluster"]].drop_duplicates("nome")
    merged = pd.merge(profiles, cluster_df, on="nome", how="inner")
    return merged.sort_values("cluster")


def plot_deputy_clusters(plot_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Render the detailed deputy cluster map from the notebook."""

    set_default_style()
    fig, ax = plt.subplots(figsize=(20, 16))

    cluster_ids = sorted(plot_df["cluster"].unique())
    num_real_clusters = len([cluster_id for cluster_id in cluster_ids if cluster_id != -1])
    real_cluster_colors = cc.glasbey_light[: max(1, num_real_clusters)]

    palette: dict[int, str] = {}
    color_index = 0
    for cluster_id in cluster_ids:
        if cluster_id == -1:
            palette[cluster_id] = "#000000"
        else:
            palette[cluster_id] = real_cluster_colors[color_index]
            color_index += 1

    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="cluster",
        palette=palette,
        s=80,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
        hue_order=cluster_ids,
    )
    ax.set_xlabel("UMAP Dimension 1", fontsize=14)
    ax.set_ylabel("UMAP Dimension 2", fontsize=14)
    ax.legend(title="Cluster ID", bbox_to_anchor=(1.02, 1), loc="upper left")
    target = save_figure(output_path)
    plt.close(fig)
    return target


def plot_cluster_centroids(plot_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Render the centroid-focused cluster view from the notebook."""

    set_default_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    noise_df = plot_df[plot_df["cluster"] == -1]
    real_clusters_df = plot_df[plot_df["cluster"] != -1]
    cluster_info = (
        real_clusters_df.groupby("cluster")
        .agg(x_mean=("x", "mean"), y_mean=("y", "mean"), size=("cluster", "size"))
        .reset_index()
    )

    cluster_ids = sorted(plot_df["cluster"].unique())
    num_real_clusters = len([cluster_id for cluster_id in cluster_ids if cluster_id != -1])
    real_cluster_colors = cc.glasbey_light[: max(1, num_real_clusters)]
    palette: dict[int, str] = {}
    color_index = 0
    for cluster_id in cluster_ids:
        if cluster_id == -1:
            palette[cluster_id] = "#000000"
        else:
            palette[cluster_id] = real_cluster_colors[color_index]
            color_index += 1

    if not noise_df.empty:
        ax.scatter(
            noise_df["x"],
            noise_df["y"],
            color="lightgray",
            s=60,
            alpha=0.6,
            label="Ruído (Não clusterizado)",
        )

    font_base_size = 8
    font_scale_factor = 3.5
    font_max_size = 50
    for _, row in cluster_info.iterrows():
        cluster_id = int(row["cluster"])
        ax.scatter(
            row["x_mean"],
            row["y_mean"],
            s=row["size"] * 80,
            color=palette.get(cluster_id, "#333333"),
            edgecolor="black",
            linewidth=1.5,
            alpha=0.8,
        )
        dynamic_fontsize = min(font_base_size + np.sqrt(row["size"]) * font_scale_factor, font_max_size)
        ax.text(
            row["x_mean"],
            row["y_mean"],
            s=str(cluster_id),
            color="black",
            fontsize=dynamic_fontsize,
            fontweight="bold",
            ha="center",
            va="center",
        )

    ax.set_xlabel("UMAP Dimension 1", fontsize=30, fontweight="bold")
    ax.set_ylabel("UMAP Dimension 2", fontsize=30, fontweight="bold")
    ax.legend().set_visible(False)
    target = save_figure(output_path, bbox_inches="tight")
    plt.close(fig)
    return target
