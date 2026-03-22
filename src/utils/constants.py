from __future__ import annotations

"""Shared constants inferred from the original research code."""

from collections.abc import Mapping


PARTY_COLORS: dict[str, str] = {
    "PT": "#E6194B",
    "MDB": "#046600",
    "PSDB": "#3cb44b",
    "PL": "#ffe119",
    "DEM": "#4363d8",
    "UNIÃO": "#4363d8",
    "PP": "#ff7b00",
    "PSB": "#e01f3f",
    "PCdoB": "#3A0000",
    "PDT": "#c42895",
    "PSOL": "#490364",
}

EXTENDED_PARTY_COLORS: dict[str, str] = {
    "PT": "#E6194B",
    "PSDB": "#3cb44b",
    "MDB": "#4363d8",
    "PP": "#911eb4",
    "PSB": "#f18e46",
    "PL": "#ffe119",
    "PCdoB": "#971111",
    "PDT": "#42d4f4",
    "DEM": "#f032e6",
    "PSOL": "#F14545",
    "PTB": "#000075",
    "REPUBLICANOS": "#28a745",
    "PFL": "#808000",
    "PSD": "#469990",
    "PV": "#bfef45",
    "NOVO": "#ff7300",
    "PSL": "#04787c",
    "PSC": "#fabebe",
    "PRB": "#dcbeff",
    "PPS": "#a9a9a9",
    "PMDB": "#4363d8",
    "PR": "#ffe119",
}

PARTY_RENAMES: dict[str, str] = {
    "PMDB": "MDB",
    "PFL": "DEM",
    "PRB": "REPUBLICANOS",
    "PR": "PL",
    "PPS": "CIDADANIA",
}

DEFAULT_KEEP_FIRST_SPEAKER = True


def filter_known_parties(frame, party_column: str = "partido", palette: Mapping[str, str] | None = None):
    """Return only rows whose party is present in the configured palette."""

    palette = dict(palette or PARTY_COLORS)
    return frame[frame[party_column].isin(palette.keys())].copy()

