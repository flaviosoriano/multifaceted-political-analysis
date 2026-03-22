from __future__ import annotations

"""Yearly BERTopic analysis for the cleaned speech corpora."""

from datetime import datetime
from glob import glob
from pathlib import Path
import re

import pandas as pd
from tqdm.auto import tqdm


ADDITIONAL_STOPWORDS = [
    "sr",
    "sra",
    "senhor",
    "senhora",
    "presidente",
    "v",
    "exa",
    "ordem",
    "pela",
    "bloco",
    "deputado",
    "deputada",
    "brasil",
    "brasileiro",
    "brasileira",
    "neste",
    "desta",
    "deste",
    "obrigado",
    "obrigada",
    "palavra",
    "discurso",
    "querida",
    "querido",
    "porque",
    "para",
    "pra",
    "pro",
    "aqui",
    "lá",
    "hoje",
    "ontem",
    "amanhã",
    "sobre",
    "coisa",
    "coisas",
    "toda",
    "todo",
    "todos",
]


def ensure_nltk_resources() -> None:
    """Download NLTK resources on demand."""

    import nltk

    resources = {
        "corpora/stopwords": "stopwords",
        "tokenizers/punkt": "punkt",
    }
    for resource_path, package_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)


def load_models(spacy_model_name: str, sentence_model_name: str):
    """Load the spaCy and sentence-transformer models required by BERTopic."""

    import spacy
    from sentence_transformers import SentenceTransformer

    try:
        nlp = spacy.load(spacy_model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{spacy_model_name}' is not installed. "
            f"Install it with: python -m spacy download {spacy_model_name}"
        ) from exc

    sentence_model = SentenceTransformer(sentence_model_name)
    return nlp, sentence_model


def build_stopwords() -> list[str]:
    """Build the stopword list used by the topic model."""

    import nltk

    ensure_nltk_resources()
    return nltk.corpus.stopwords.words("portuguese") + ADDITIONAL_STOPWORDS


def preprocess_topic_documents(texts: list[str], nlp) -> list[str]:
    """Normalize already-cleaned texts before phrase modeling and BERTopic.

    This intentionally operates on ``transcricao_limpa`` documents, fixing the
    current ``processed_texts`` bug without reintroducing extra structural
    cleaning that the preprocessing stage already handled.
    """

    processed: list[str] = []
    for doc in tqdm(nlp.pipe(texts, batch_size=50), desc="spaCy processing", total=len(texts)):
        tokens = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
        processed.append(" ".join(tokens))
    return processed


def lemmatize_topic_words(words_list: list[str], nlp) -> list[str]:
    """Lemmatize topic representation words if requested."""

    if not words_list:
        return []
    doc = nlp(" ".join(words_list))
    return [token.lemma_.lower() for token in doc if token.is_alpha]


def save_topics(
    topics,
    topic_info: pd.DataFrame,
    year: str,
    output_dir: str | Path,
    *,
    nlp,
    lemmatize_results: bool = False,
) -> Path:
    """Persist BERTopic results in the same text-based format as the legacy script."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "lemmatized" if lemmatize_results else "original"
    target = output_path / f"bertopic_results_{year}_{suffix}_{timestamp}.txt"

    total_docs = len(topics)
    unique_topics = set(topics)
    outlier_count = topics.count(-1) if isinstance(topics, list) else sum(1 for item in topics if item == -1)

    with target.open("w", encoding="utf-8") as handle:
        handle.write(f"# BERTopic Results for Year {year}\n")
        handle.write(f"# Lemmatization of topic keywords: {lemmatize_results}\n")
        handle.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        handle.write(f"Total documents processed: {total_docs}\n")
        handle.write(f"Unique topics found: {len(unique_topics)}\n")
        handle.write(
            f"Topics (excluding outliers): {len(unique_topics) - 1 if -1 in unique_topics else len(unique_topics)}\n"
        )
        handle.write(f"Outlier documents: {outlier_count}\n")
        if total_docs:
            handle.write(f"Outlier percentage: {(outlier_count / total_docs * 100):.2f}%\n\n")
        handle.write("## Topic Overview\n\n")

        for _, row in topic_info.sort_values(by="Count", ascending=False).iterrows():
            topic_id = row["Topic"]
            count = row["Count"]
            words = row["Representation"]
            if topic_id == -1:
                handle.write(f"Topic -1 (Outliers): {count} documents\n")
                continue
            words_to_write = lemmatize_topic_words(words, nlp) if lemmatize_results else words
            handle.write(f"Topic {topic_id}: {', '.join(words_to_write)} (Count: {count})\n")

    return target


def _extract_year(file_path: str | Path) -> str:
    match = re.search(r"discursos_(\d{4})_limpo\.csv", Path(file_path).name)
    if not match:
        raise ValueError(f"Could not extract year from '{file_path}'.")
    return match.group(1)


def analyze_year_data(
    file_path: str | Path,
    *,
    nlp,
    sentence_model,
    output_dir: str | Path,
    stopwords: list[str],
    lemmatize_results: bool = False,
    min_words: int = 10,
    min_topic_size: int = 10,
) -> tuple[str, object, pd.DataFrame]:
    """Run BERTopic for one cleaned yearly file."""

    from bertopic import BERTopic
    from gensim.models.phrases import Phrases, Phraser
    from sklearn.feature_extraction.text import CountVectorizer

    year = _extract_year(file_path)
    df = pd.read_csv(file_path, dtype=str)
    text_column = "transcricao_limpa" if "transcricao_limpa" in df.columns else "transcricao"

    raw_documents = df[text_column].dropna().astype(str).tolist()
    processed_documents = preprocess_topic_documents(raw_documents, nlp)
    processed_documents = [doc for doc in processed_documents if len(doc.split()) >= min_words]
    if len(processed_documents) < 10:
        raise RuntimeError(f"Not enough valid documents for year {year}.")

    tokenized_docs = [doc.split() for doc in processed_documents]
    phrases = Phrases(tokenized_docs, min_count=5, threshold=10)
    phraser = Phraser(phrases)
    final_documents = [" ".join(phraser[doc]) for doc in tokenized_docs]

    vectorizer = CountVectorizer(
        stop_words=stopwords,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        max_features=2000,
    )
    topic_model = BERTopic(
        embedding_model=sentence_model,
        vectorizer_model=vectorizer,
        verbose=True,
        min_topic_size=min_topic_size,
        language="portuguese",
        nr_topics="auto",
    )
    topics, _ = topic_model.fit_transform(final_documents)
    topic_info = topic_model.get_topic_info()
    save_topics(
        topics,
        topic_info,
        year,
        output_dir,
        nlp=nlp,
        lemmatize_results=lemmatize_results,
    )
    return year, topics, topic_info


def run_topic_modeling(
    input_glob: str,
    output_dir: str | Path,
    *,
    spacy_model: str = "pt_core_news_sm",
    sentence_model_name: str = "distiluse-base-multilingual-cased-v2",
    lemmatize_results: bool = False,
    min_words: int = 10,
    min_topic_size: int = 10,
) -> tuple[list[str], list[str]]:
    """Run BERTopic on all matching cleaned yearly files."""

    files = sorted(glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files found for pattern '{input_glob}'.")

    stopwords = build_stopwords()
    nlp, sentence_model = load_models(spacy_model, sentence_model_name)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    successful_years: list[str] = []
    failed_years: list[str] = []
    for file_path in files:
        try:
            year, _, _ = analyze_year_data(
                file_path,
                nlp=nlp,
                sentence_model=sentence_model,
                output_dir=output_path,
                stopwords=stopwords,
                lemmatize_results=lemmatize_results,
                min_words=min_words,
                min_topic_size=min_topic_size,
            )
            successful_years.append(year)
        except Exception:
            try:
                failed_years.append(_extract_year(file_path))
            except Exception:
                failed_years.append("unknown")
    if not successful_years:
        raise RuntimeError("No BERTopic yearly analysis completed successfully.")
    return successful_years, failed_years

