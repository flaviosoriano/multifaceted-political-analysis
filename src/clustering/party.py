from __future__ import annotations

"""Party trajectory and HDBSCAN analyses derived from the legacy notebooks."""

from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
import umap.umap_ as umap

from src.embeddings.aggregation import (
    aggregate_by_party_year,
    embeddings_matrix,
    load_corpus_with_embeddings,
)
from src.utils.constants import PARTY_COLORS
from src.utils.plotting import save_figure, set_default_style


def project_party_trajectories(
    csv_path: str | Path,
    embeddings_path: str | Path,
    *,
    party_colors: dict[str, str] | None = None,
    n_neighbors: int = 40,
    min_dist: float = 0.1,
    random_state: int = 33,
) -> pd.DataFrame:
    """Project party-year pooled embeddings into 2D for the trajectory plot."""

    palette = party_colors or PARTY_COLORS
    df = load_corpus_with_embeddings(csv_path, embeddings_path)
    aggregated = aggregate_by_party_year(df, party_filter=False)
    aggregated = aggregated[aggregated["partido"].isin(palette.keys())].copy()
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
    )
    embedding_2d = reducer.fit_transform(embeddings_matrix(aggregated))
    plot_df = pd.DataFrame(embedding_2d, columns=["x", "y"])
    plot_df[["ano", "partido"]] = aggregated[["ano", "partido"]].reset_index(drop=True)
    return plot_df


def plot_party_trajectories(
    plot_df: pd.DataFrame,
    output_path: str | Path,
    *,
    party_colors: dict[str, str] | None = None,
    title: str = "Trajetória Semântica (2003-2025) dos partidos",
) -> Path:
    """Render the party trajectory plot from the original notebook."""

    palette = party_colors or PARTY_COLORS
    set_default_style()
    fig, ax = plt.subplots(figsize=(20, 16))

    for party in tqdm(palette.keys(), desc="Plotting party trajectories"):
        party_df = plot_df[plot_df["partido"] == party].sort_values("ano")
        if party_df.empty:
            continue
        color = palette.get(party, "#808080")
        ax.plot(party_df["x"], party_df["y"], marker="o", linestyle="-", color=color, label=party, alpha=0.6)
        for row_index, row in party_df.iterrows():
            year_label = str(row["ano"])[-2:]
            if row_index == party_df.index[0]:
                ax.text(row["x"], row["y"] - 0.1, f"{party} '{year_label}", fontsize=9, color=color, ha="center")
            elif row_index == party_df.index[-1]:
                ax.text(
                    row["x"],
                    row["y"] + 0.1,
                    f"{party} '{year_label}",
                    fontsize=11,
                    color=color,
                    ha="center",
                    weight="bold",
                )
            else:
                ax.text(row["x"], row["y"] + 0.05, f"'{year_label}", fontsize=7, color=color, ha="center", alpha=0.7)

    ax.set_title(title, fontsize=24, pad=20)
    ax.set_xlabel("Dimensão Semântica 1 (UMAP)", fontsize=14)
    ax.set_ylabel("Dimensão Semântica 2 (UMAP)", fontsize=14)
    ax.legend(title="Partido", bbox_to_anchor=(1.02, 1), loc="upper left")
    target = save_figure(output_path)
    plt.close(fig)
    return target


def cluster_party_years(
    csv_path: str | Path,
    embeddings_path: str | Path,
    *,
    party_colors: dict[str, str] | None = None,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_components: int = 2,
    random_state: int = 42,
    hdbscan_min_cluster_size: int = 3,
    hdbscan_min_samples: int = 10,
    hdbscan_epsilon: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the party-year HDBSCAN pipeline from ``partidosHDBSCAN.py``."""

    palette = party_colors or PARTY_COLORS
    df = load_corpus_with_embeddings(csv_path, embeddings_path)
    aggregated = aggregate_by_party_year(df, party_filter=False)
    aggregated = aggregated[aggregated["partido"].isin(palette.keys())].copy()

    matrix = embeddings_matrix(aggregated)
    reducer = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        n_components=umap_components,
        random_state=random_state,
    )
    embedding_2d = reducer.fit_transform(matrix)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        cluster_selection_epsilon=hdbscan_epsilon,
        prediction_data=True,
        metric="euclidean",
    )
    cluster_labels = clusterer.fit_predict(matrix)

    plot_df = pd.DataFrame(embedding_2d, columns=["x", "y"])
    plot_df[["ano", "partido"]] = aggregated[["ano", "partido"]].reset_index(drop=True)
    plot_df["cluster"] = cluster_labels

    summary = (
        plot_df.groupby("cluster")
        .agg(
            count=("partido", "size"),
            parties=("partido", lambda values: ", ".join(values.unique())),
            years=("ano", lambda values: f"{values.min()}-{values.max()}"),
        )
        .reset_index()
    )
    return plot_df, summary


def plot_party_clusters(
    plot_df: pd.DataFrame,
    output_path: str | Path,
    *,
    party_colors: dict[str, str] | None = None,
) -> Path:
    """Render the combined trajectory-plus-cluster visualization."""

    palette = party_colors or PARTY_COLORS
    set_default_style()
    fig, ax = plt.subplots(figsize=(20, 16))

    cluster_ids = sorted(plot_df["cluster"].unique())
    num_clusters = len([cluster_id for cluster_id in cluster_ids if cluster_id != -1])
    cluster_cmap = plt.cm.get_cmap("tab20", max(1, num_clusters))

    for cluster_id in cluster_ids:
        cluster_df = plot_df[plot_df["cluster"] == cluster_id]
        if cluster_id == -1:
            cluster_color = "gray"
            cluster_label = "Ruído"
        else:
            cluster_color = cluster_cmap(cluster_id % cluster_cmap.N)
            cluster_label = f"Cluster {cluster_id}"
        ax.scatter(
            cluster_df["x"],
            cluster_df["y"],
            color=cluster_color,
            label=cluster_label,
            alpha=0.7,
            s=100,
            edgecolor="black",
            zorder=2,
        )

    for party in tqdm(palette.keys(), desc="Drawing party trajectories"):
        party_df = plot_df[plot_df["partido"] == party].sort_values("ano")
        if party_df.empty:
            continue
        color = palette.get(party, "#808080")
        ax.plot(party_df["x"], party_df["y"], linestyle="--", color=color, alpha=0.4, linewidth=1, zorder=1)
        for row_index, row in party_df.iterrows():
            year_label = str(row["ano"])[-2:]
            if row_index == party_df.index[0]:
                ax.text(row["x"], row["y"] - 0.1, f"{party} '{year_label}", fontsize=9, color=color, ha="center")
            elif row_index == party_df.index[-1]:
                ax.text(
                    row["x"],
                    row["y"] + 0.1,
                    f"{party} '{year_label}",
                    fontsize=11,
                    color=color,
                    ha="center",
                    weight="bold",
                )
            else:
                ax.text(row["x"], row["y"] + 0.05, f"'{year_label}", fontsize=7, color=color, ha="center", alpha=0.7)

    party_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette.get(party, "#808080"), markersize=10, label=party)
        for party in palette.keys()
    ]
    party_legend = ax.legend(handles=party_handles, title="Partido", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    ax.add_artist(party_legend)

    cluster_handles = []
    for cluster_id in cluster_ids:
        if cluster_id == -1:
            cluster_color = "gray"
            cluster_label = "Ruído (pontos não agrupados)"
        else:
            cluster_color = cluster_cmap(cluster_id % cluster_cmap.N)
            cluster_label = f"Cluster Semântico {cluster_id}"
        cluster_handles.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cluster_color, markersize=10, label=cluster_label)
        )
    cluster_legend = ax.legend(handles=cluster_handles, title="Grupos Semânticos", bbox_to_anchor=(1.02, 0.7), loc="upper left", fontsize=10)
    ax.add_artist(cluster_legend)

    ax.set_title("Trajetória Semântica e Agrupamentos (2003-2025) dos Partidos", fontsize=24, pad=20)
    ax.set_xlabel("Dimensão Semântica 1 (UMAP)", fontsize=14)
    ax.set_ylabel("Dimensão Semântica 2 (UMAP)", fontsize=14)
    target = save_figure(output_path)
    plt.close(fig)
    return target

