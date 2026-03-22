from __future__ import annotations

"""Yearly linguistic analysis for Brazilian parliamentary speeches."""

from collections import Counter
from glob import glob
from pathlib import Path
import re

import pandas as pd
from tqdm.auto import tqdm


def load_spacy_model(model_name: str):
    """Load a spaCy model and raise a clear installation error if unavailable."""

    import spacy

    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Install it with: python -m spacy download {model_name}"
        ) from exc


def count_syllables_portuguese(word: str) -> int:
    """Estimate the number of syllables in a Portuguese word."""

    lowered = word.lower().strip()
    lowered = re.sub(r"[^a-záàâãéêíóôõúçü]", "", lowered)
    if not lowered:
        return 0

    vowels = "aeiouáàâãéêíóôõúü"
    count = 0
    previous_was_vowel = False

    for index, char in enumerate(lowered):
        is_vowel = char in vowels
        if is_vowel:
            if not previous_was_vowel:
                count += 1
            elif index > 0:
                previous_char = lowered[index - 1]
                bigram = previous_char + char
                if (
                    (previous_char in "aeo" and char in "aeo")
                    or (previous_char == "i" and char in "aeo")
                    or (previous_char in "aeo" and char == "i")
                ) and bigram not in {"ai", "au", "ei", "eu", "oi", "ou", "ui"}:
                    count += 1
        previous_was_vowel = is_vowel

    return max(1, count)


def calculate_flesch_portuguese(text: str, nlp) -> float:
    """Calculate the Portuguese Flesch readability score for a speech."""

    if not text or not text.strip():
        return 0.0

    doc = nlp(text)
    sentences = list(doc.sents)
    words = [token for token in doc if token.is_alpha and not token.is_space]
    if not sentences or not words:
        return 0.0

    total_words = len(words)
    total_sentences = len(sentences)
    total_syllables = sum(count_syllables_portuguese(token.text) for token in words)
    average_sentence_length = total_words / total_sentences
    average_syllables_per_word = total_syllables / total_words
    return round(206.835 - (1.015 * average_sentence_length) - (84.6 * average_syllables_per_word), 2)


def calculate_type_token_ratio(doc) -> float:
    """Compute lexical diversity from a processed spaCy doc."""

    words = [token.text.lower() for token in doc if token.is_alpha]
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 4)


def extract_pos_percentages(doc) -> dict[str, float]:
    """Return POS-tag percentages for the most relevant categories."""

    valid_tokens = [token for token in doc if token.is_alpha]
    if not valid_tokens:
        return {"noun_pct": 0.0, "verb_pct": 0.0, "adj_pct": 0.0, "adv_pct": 0.0}

    total = len(valid_tokens)
    counts = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0}
    for token in valid_tokens:
        if token.pos_ in counts:
            counts[token.pos_] += 1

    return {
        "noun_pct": round((counts["NOUN"] / total) * 100, 2),
        "verb_pct": round((counts["VERB"] / total) * 100, 2),
        "adj_pct": round((counts["ADJ"] / total) * 100, 2),
        "adv_pct": round((counts["ADV"] / total) * 100, 2),
    }


def extract_named_entities(doc) -> dict[str, list[str]]:
    """Extract named entities grouped by broad type."""

    entities = {"persons": [], "organizations": [], "locations": []}
    for entity in doc.ents:
        value = entity.text.strip()
        if len(value) <= 1:
            continue
        if entity.label_ == "PER":
            entities["persons"].append(value)
        elif entity.label_ == "ORG":
            entities["organizations"].append(value)
        elif entity.label_ in {"LOC", "GPE"}:
            entities["locations"].append(value)
    return entities


def analyze_speech(text: str | float | None, nlp) -> dict[str, object]:
    """Perform the per-speech linguistic analysis used by the yearly pipeline."""

    if pd.isna(text) or not str(text).strip():
        return {
            "word_count": 0,
            "sentence_count": 0,
            "ttr": 0.0,
            "flesch_score": 0.0,
            "noun_pct": 0.0,
            "verb_pct": 0.0,
            "adj_pct": 0.0,
            "adv_pct": 0.0,
            "entities_person": [],
            "entities_org": [],
            "entities_loc": [],
        }

    doc = nlp(str(text))
    sentences = list(doc.sents)
    words = [token for token in doc if token.is_alpha]
    pos_percentages = extract_pos_percentages(doc)
    entities = extract_named_entities(doc)

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "ttr": calculate_type_token_ratio(doc),
        "flesch_score": calculate_flesch_portuguese(str(text), nlp),
        "noun_pct": pos_percentages["noun_pct"],
        "verb_pct": pos_percentages["verb_pct"],
        "adj_pct": pos_percentages["adj_pct"],
        "adv_pct": pos_percentages["adv_pct"],
        "entities_person": entities["persons"],
        "entities_org": entities["organizations"],
        "entities_loc": entities["locations"],
    }


def extract_word_frequencies(text: str | float | None, nlp, pos_filter: str) -> list[str]:
    """Collect lemmatized words for a given POS tag."""

    if pd.isna(text) or not str(text).strip():
        return []
    doc = nlp(str(text))
    return [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop and len(token.text) > 2 and token.pos_ == pos_filter
    ]


def format_top_list(counter: Counter, top_n: int = 20) -> str:
    """Format a counter as a semicolon-separated ranking string."""

    return "; ".join(f"{word}({count})" for word, count in counter.most_common(top_n))


def _extract_year(file_path: str | Path) -> int:
    match = re.search(r"discursos_(\d{4})_limpo\.csv", Path(file_path).name)
    if not match:
        raise ValueError(f"Could not extract year from '{file_path}'.")
    return int(match.group(1))


def analyze_year_file(file_path: str | Path, nlp) -> tuple[pd.DataFrame, dict[str, object]]:
    """Analyze one yearly cleaned CSV and return detailed and aggregate results."""

    year = _extract_year(file_path)
    df = pd.read_csv(file_path, dtype=str)
    text_column = "transcricao_limpa" if "transcricao_limpa" in df.columns else "transcricao"

    analysis_results: list[dict[str, object]] = []
    nouns = Counter()
    verbs = Counter()
    adjectives = Counter()
    persons = Counter()
    organizations = Counter()
    locations = Counter()

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Analyzing {year} speeches",
        unit="speech",
    ):
        text = row[text_column]
        result = analyze_speech(text, nlp)
        analysis_results.append(result)

        if pd.isna(text) or not str(text).strip():
            continue

        nouns.update(extract_word_frequencies(text, nlp, "NOUN"))
        verbs.update(extract_word_frequencies(text, nlp, "VERB"))
        adjectives.update(extract_word_frequencies(text, nlp, "ADJ"))
        persons.update(result["entities_person"])
        organizations.update(result["entities_org"])
        locations.update(result["entities_loc"])

    for column in [
        "word_count",
        "sentence_count",
        "ttr",
        "flesch_score",
        "noun_pct",
        "verb_pct",
        "adj_pct",
        "adv_pct",
    ]:
        df[column] = [result[column] for result in analysis_results]

    df["entities_person"] = ["; ".join(result["entities_person"]) for result in analysis_results]
    df["entities_org"] = ["; ".join(result["entities_org"]) for result in analysis_results]
    df["entities_loc"] = ["; ".join(result["entities_loc"]) for result in analysis_results]

    valid_df = df[df["word_count"] > 0]
    aggregate = {
        "year": year,
        "total_speeches": len(df),
        "valid_speeches": len(valid_df),
        "avg_word_count": round(valid_df["word_count"].mean(), 2) if not valid_df.empty else 0,
        "avg_sentence_count": round(valid_df["sentence_count"].mean(), 2) if not valid_df.empty else 0,
        "avg_ttr": round(valid_df["ttr"].mean(), 4) if not valid_df.empty else 0,
        "avg_flesch_score": round(valid_df["flesch_score"].mean(), 2) if not valid_df.empty else 0,
        "avg_noun_pct": round(valid_df["noun_pct"].mean(), 2) if not valid_df.empty else 0,
        "avg_verb_pct": round(valid_df["verb_pct"].mean(), 2) if not valid_df.empty else 0,
        "avg_adj_pct": round(valid_df["adj_pct"].mean(), 2) if not valid_df.empty else 0,
        "avg_adv_pct": round(valid_df["adv_pct"].mean(), 2) if not valid_df.empty else 0,
        "top_20_nouns": format_top_list(nouns),
        "top_20_verbs": format_top_list(verbs),
        "top_20_adjectives": format_top_list(adjectives),
        "top_20_persons": format_top_list(persons),
        "top_20_organizations": format_top_list(organizations),
        "top_20_locations": format_top_list(locations),
        "unique_nouns": len(nouns),
        "unique_verbs": len(verbs),
        "unique_adjectives": len(adjectives),
        "unique_persons": len(persons),
        "unique_organizations": len(organizations),
        "unique_locations": len(locations),
    }
    return df, aggregate


def write_summary(
    yearly_df: pd.DataFrame,
    output_path: str | Path,
    successful_years: list[str],
    failed_years: list[str],
) -> Path:
    """Write the textual linguistic summary report."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write("=" * 80 + "\n")
        handle.write("RESUMO DA ANÁLISE LINGUÍSTICA DOS DISCURSOS PARLAMENTARES\n")
        handle.write("=" * 80 + "\n\n")
        handle.write(f"Período analisado: {yearly_df['year'].min()} - {yearly_df['year'].max()}\n")
        handle.write(f"Anos processados com sucesso: {len(successful_years)} ({', '.join(successful_years)})\n")
        if failed_years:
            handle.write(f"Anos com falha: {len(failed_years)} ({', '.join(failed_years)})\n")
        handle.write(f"Total de discursos analisados: {yearly_df['total_speeches'].sum():,}\n")
        handle.write(f"Total de discursos válidos: {yearly_df['valid_speeches'].sum():,}\n\n")
        handle.write(f"Contagem média de palavras por discurso: {yearly_df['avg_word_count'].mean():.1f}\n")
        handle.write(f"Contagem média de sentenças por discurso: {yearly_df['avg_sentence_count'].mean():.1f}\n")
        handle.write(f"Riqueza lexical média (TTR): {yearly_df['avg_ttr'].mean():.4f}\n")
        handle.write(f"Legibilidade média (Flesch): {yearly_df['avg_flesch_score'].mean():.1f}\n")
    return target


def run_yearly_linguistic_analysis(
    input_glob: str,
    output_dir: str | Path,
    *,
    spacy_model: str = "pt_core_news_lg",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Run the yearly linguistic analysis for all matching files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    nlp = load_spacy_model(spacy_model)
    files = sorted(glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files found for pattern '{input_glob}'.")

    aggregates: list[dict[str, object]] = []
    successful_years: list[str] = []
    failed_years: list[str] = []

    for file_path in tqdm(files, desc="Processing years", unit="year"):
        try:
            detailed_df, aggregate = analyze_year_file(file_path, nlp)
            year = str(aggregate["year"])
            detailed_output = output_path / f"discursos_{year}_analisados_detalhado.csv"
            detailed_df.to_csv(detailed_output, index=False, encoding="utf-8")
            aggregates.append(aggregate)
            successful_years.append(year)
        except Exception:
            try:
                failed_years.append(str(_extract_year(file_path)))
            except Exception:
                failed_years.append("unknown")

    if not aggregates:
        raise RuntimeError("No yearly linguistic analysis run completed successfully.")

    yearly_df = pd.DataFrame(aggregates).sort_values("year")
    yearly_df.to_csv(output_path / "analise_linguistica_anual.csv", index=False, encoding="utf-8")
    write_summary(yearly_df, output_path / "resumo_analise_linguistica.txt", successful_years, failed_years)
    return yearly_df, successful_years, failed_years

