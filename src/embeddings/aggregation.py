from __future__ import annotations

"""Shared helpers for aggregating speech embeddings."""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.constants import PARTY_COLORS, filter_known_parties


def load_corpus_with_embeddings(
    csv_path: str | Path,
    embeddings_path: str | Path,
    *,
    text_column: str | None = None,
) -> pd.DataFrame:
    """Load a corpus CSV and attach a parallel numpy embedding matrix."""

    df = pd.read_csv(csv_path)
    if text_column is not None and text_column in df.columns:
        df = df.dropna(subset=[text_column]).copy()
    embeddings = np.load(embeddings_path)
    if len(df) != len(embeddings):
        raise ValueError("CSV row count and embedding count do not match.")
    df = df.copy()
    df["embedding"] = list(embeddings)
    return df


def aggregate_embeddings(
    frame: pd.DataFrame,
    group_columns: Iterable[str],
    *,
    embedding_column: str = "embedding",
) -> pd.DataFrame:
    """Mean-pool embeddings over the requested grouping columns."""

    grouped = (
        frame.groupby(list(group_columns))[embedding_column]
        .apply(lambda values: np.mean(values.tolist(), axis=0))
        .reset_index()
    )
    return grouped


def aggregate_by_deputy(
    frame: pd.DataFrame,
    *,
    party_filter: bool = True,
    party_column: str = "partido",
    deputy_column: str = "nome",
) -> pd.DataFrame:
    """Aggregate embeddings by deputy name."""

    df = filter_known_parties(frame, party_column, PARTY_COLORS) if party_filter else frame.copy()
    deputy_df = aggregate_embeddings(df, [deputy_column])
    deputy_df.rename(columns={deputy_column: "nome"}, inplace=True)
    return deputy_df


def aggregate_by_party_year(
    frame: pd.DataFrame,
    *,
    party_filter: bool = True,
    party_column: str = "partido",
    year_column: str = "ano",
) -> pd.DataFrame:
    """Aggregate embeddings by party and year."""

    df = filter_known_parties(frame, party_column, PARTY_COLORS) if party_filter else frame.copy()
    aggregated = aggregate_embeddings(df, [year_column, party_column])
    aggregated.rename(columns={year_column: "ano", party_column: "partido"}, inplace=True)
    return aggregated


def embeddings_matrix(frame: pd.DataFrame, embedding_column: str = "embedding") -> np.ndarray:
    """Stack an embedding column into a 2D numpy matrix."""

    return np.vstack(frame[embedding_column].values)


def latest_party_by_deputy(
    frame: pd.DataFrame,
    *,
    deputy_column: str = "nome",
    party_column: str = "partido",
) -> pd.Series:
    """Return the last observed party label for each deputy."""

    return frame.drop_duplicates(deputy_column, keep="last").set_index(deputy_column)[party_column]
