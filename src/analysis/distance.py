from __future__ import annotations

"""Distance-based party experiments derived from the original notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

from src.embeddings.aggregation import load_corpus_with_embeddings
from src.utils.constants import EXTENDED_PARTY_COLORS
from src.utils.plotting import save_figure, set_default_style


def compute_yearly_party_distance_matrices(
    csv_path: str | Path,
    embeddings_path: str | Path,
    *,
    parties: list[str] | None = None,
    year_column: str = "ano",
    party_column: str = "partido",
    output_dir: str | Path | None = None,
) -> dict[int, pd.DataFrame]:
    """Compute Euclidean party-distance matrices for each year."""

    selected_parties = parties or list(EXTENDED_PARTY_COLORS.keys())
    df = load_corpus_with_embeddings(csv_path, embeddings_path)
    df = df[df[party_column].isin(selected_parties)].copy()

    grouped = (
        df.groupby([party_column, year_column])["embedding"]
        .apply(lambda values: np.mean(values.tolist(), axis=0))
        .reset_index(name="embedding")
    )

    matrices: dict[int, pd.DataFrame] = {}
    table_dir = Path(output_dir) if output_dir is not None else None
    if table_dir is not None:
        table_dir.mkdir(parents=True, exist_ok=True)

    for year, year_df in grouped.groupby(year_column):
        vectors = np.vstack(year_df["embedding"].values)
        parties_for_year = year_df[party_column].tolist()
        matrix = pd.DataFrame(
            squareform(pdist(vectors, metric="euclidean")),
            index=parties_for_year,
            columns=parties_for_year,
        )
        matrices[int(year)] = matrix
        if table_dir is not None:
            matrix.to_csv(table_dir / f"distancias_{year}.csv")

    return matrices


def plot_distance_heatmaps(
    matrices: dict[int, pd.DataFrame],
    output_dir: str | Path,
    *,
    cmap: str = "viridis_r",
) -> list[Path]:
    """Render one heatmap per yearly party-distance matrix."""

    set_default_style()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for year, matrix in sorted(matrices.items()):
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            linewidths=0.5,
            cbar_kws={"label": "Distância Euclidiana"},
        )
        plt.title(f"Heatmap de Distâncias Euclidianas entre Partidos - {year}", fontsize=15)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        target = save_figure(output_path / f"heatmap_distancias_{year}.png")
        plt.close()
        saved_paths.append(target)

    return saved_paths


def run_distance_analysis(
    csv_path: str | Path,
    embeddings_path: str | Path,
    *,
    parties: list[str] | None = None,
    tables_dir: str | Path,
    figures_dir: str | Path,
) -> dict[int, pd.DataFrame]:
    """Execute the full distance analysis stage."""

    matrices = compute_yearly_party_distance_matrices(
        csv_path,
        embeddings_path,
        parties=parties,
        output_dir=tables_dir,
    )
    plot_distance_heatmaps(matrices, figures_dir)
    return matrices

