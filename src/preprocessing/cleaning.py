from __future__ import annotations

"""Shared text cleaning logic for the speech corpora."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
from tqdm.auto import tqdm

from src.utils.constants import DEFAULT_KEEP_FIRST_SPEAKER, PARTY_RENAMES


SPEAKER_PATTERN_UNANCHORED = re.compile(
    r"(o sr\.|a sra\.|o senhor|a senhora|o deputado|a deputada) "
    r"[a-z\s]+(\s*\([^)]*\)\s*)?[–-]\s*",
    flags=re.IGNORECASE,
)
SPEAKER_PATTERN_START = re.compile(
    r"^(o sr\.|a sra\.|o senhor|a senhora|o deputado|a deputada) "
    r"[a-z\s]+(\s*\([^)]*\)\s*)?[–-]\s*",
    flags=re.IGNORECASE,
)
SALUTATION_PATTERN = re.compile(
    r"\b("
    r"vossa excelência|vossas excelências|senhor presidente|senhora presidenta|"
    r"Sr Presidente|Sra Presidente|senhores deputados|senhoras deputadas|"
    r"senhor relator|senhora relatora|excelência|excelências|presidente|"
    r"presidenta|parlamentar|deputado|deputada|deputados|deputadas|relator|"
    r"relatora|senhor|senhora|senhores|senhoras|Sras|Srs|Sra|Sr|exa|S|"
    r"vossa|vossas|nobre|nobres|bloco|ordem|pela|sr|sra|v"
    r")\b",
    flags=re.IGNORECASE,
)
PARENTHESIS_PATTERN = re.compile(r"\s*\([^)]*\)\s*")
NON_TEXT_PATTERN = re.compile(r"[^a-zA-ZÀ-ÿ0-9\s]", flags=re.UNICODE)
MULTI_SPACE_PATTERN = re.compile(r"\s+")
ORDINAL_PATTERN = re.compile(r"nº\s*\d+", flags=re.IGNORECASE)


@dataclass(slots=True)
class CleaningSummary:
    """Summary statistics for one cleaned input file."""

    label: str
    original_rows: int
    cleaned_rows: int

    @property
    def removed_rows(self) -> int:
        return self.original_rows - self.cleaned_rows

    @property
    def retained_pct(self) -> float:
        if self.original_rows == 0:
            return 0.0
        return self.cleaned_rows / self.original_rows * 100


def correct_parties(text: str | float | None) -> str | float | None:
    """Normalize party siglas while preserving null values."""

    if pd.isna(text):
        return text
    normalized = NON_TEXT_PATTERN.sub("", str(text))
    return PARTY_RENAMES.get(normalized, normalized)


def preprocess_text(
    text: str | float | None,
    *,
    keep_first_speaker: bool = DEFAULT_KEEP_FIRST_SPEAKER,
) -> str:
    """Clean an individual speech transcript.

    By default, if more than one speaker is present, only the first speaker's
    segment is preserved, matching the behavior from ``flavio/data_cleaning.py``.
    """

    if pd.isna(text):
        return ""

    cleaned = str(text)
    matches = list(SPEAKER_PATTERN_UNANCHORED.finditer(cleaned))
    if len(matches) > 1:
        if keep_first_speaker:
            cleaned = cleaned[: matches[1].start()]
        else:
            return ""

    cleaned = SPEAKER_PATTERN_START.sub("", cleaned)
    cleaned = SALUTATION_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"[–—]", " ", cleaned)
    cleaned = ORDINAL_PATTERN.sub(" ", cleaned)
    cleaned = NON_TEXT_PATTERN.sub(" ", cleaned)
    cleaned = MULTI_SPACE_PATTERN.sub(" ", cleaned).strip()
    cleaned = PARENTHESIS_PATTERN.sub("", cleaned)
    cleaned = cleaned.lower()

    return cleaned


def clean_dataframe(
    frame: pd.DataFrame,
    *,
    text_column: str = "transcricao",
    output_column: str = "transcricao_limpa",
    keep_first_speaker: bool = DEFAULT_KEEP_FIRST_SPEAKER,
    party_columns: Iterable[str] = ("partido", "siglaPartido"),
    normalize_name_column: bool = True,
    drop_empty: bool = True,
) -> pd.DataFrame:
    """Return a cleaned copy of a speeches dataframe."""

    if text_column not in frame.columns:
        raise KeyError(f"Column '{text_column}' not found in dataframe.")

    df = frame.copy()
    df.dropna(subset=[text_column], inplace=True)
    tqdm.pandas(desc="Cleaning speeches")
    df[output_column] = df[text_column].progress_apply(
        lambda value: preprocess_text(value, keep_first_speaker=keep_first_speaker)
    )

    for party_column in party_columns:
        if party_column in df.columns:
            df[party_column] = df[party_column].apply(correct_parties)

    if normalize_name_column and "nome" in df.columns:
        df["nome"] = df["nome"].astype(str).str.title()

    if drop_empty:
        df = df[df[output_column] != ""].copy()

    return df


def clean_combined_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    text_column: str = "transcricao",
    keep_first_speaker: bool = DEFAULT_KEEP_FIRST_SPEAKER,
) -> CleaningSummary:
    """Clean a combined corpus file and write it to disk."""

    source = Path(input_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source, low_memory=False)
    original_rows = len(df)
    cleaned_df = clean_dataframe(
        df,
        text_column=text_column,
        keep_first_speaker=keep_first_speaker,
    )
    cleaned_df.to_csv(target, index=False)

    return CleaningSummary(
        label=source.name,
        original_rows=original_rows,
        cleaned_rows=len(cleaned_df),
    )


def process_year_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    text_column: str = "transcricao",
    keep_first_speaker: bool = DEFAULT_KEEP_FIRST_SPEAKER,
) -> CleaningSummary:
    """Clean a single yearly speech file and save the result."""

    source = Path(input_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source, low_memory=False)
    original_rows = len(df)
    cleaned_df = clean_dataframe(
        df,
        text_column=text_column,
        keep_first_speaker=keep_first_speaker,
    )
    cleaned_df.to_csv(target, index=False)

    return CleaningSummary(
        label=source.stem,
        original_rows=original_rows,
        cleaned_rows=len(cleaned_df),
    )


def clean_yearly_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    pattern: str = "discursos_*.csv",
    text_column: str = "transcricao",
    keep_first_speaker: bool = DEFAULT_KEEP_FIRST_SPEAKER,
) -> tuple[list[CleaningSummary], pd.DataFrame]:
    """Clean all yearly files matching the configured pattern."""

    source_dir = Path(input_dir)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[CleaningSummary] = []
    for source in sorted(source_dir.glob(pattern)):
        target = target_dir / source.name.replace(".csv", "_limpo.csv")
        summary = process_year_file(
            source,
            target,
            text_column=text_column,
            keep_first_speaker=keep_first_speaker,
        )
        summaries.append(summary)

    summary_df = pd.DataFrame(
        [
            {
                "label": item.label,
                "original": item.original_rows,
                "limpo": item.cleaned_rows,
                "removidos": item.removed_rows,
                "percentual_mantido": round(item.retained_pct, 2),
            }
            for item in summaries
        ]
    )
    return summaries, summary_df

