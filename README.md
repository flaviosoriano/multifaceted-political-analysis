# Multifaceted Political Analysis

This repository is a research-grade companion codebase for the analysis of Brazilian parliamentary speeches. It reorganizes the original study code into a reproducible structure while preserving the scientific intent of the notebooks and scripts that were actually used in the research.

The recovered pipeline has six main components: speech preprocessing, yearly linguistic analysis, yearly topic modeling with BERTopic, speech-level embedding generation, party and deputy semantic mapping/clustering, and yearwise distance-based party experiments.

## Repository Structure

```text
.
├── AGENTS.md
├── README.md
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── legacy/
├── docs/
├── notebooks/
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── models/
│   ├── logs/
│   └── legacy/
├── scripts/
└── src/
    ├── analysis/
    ├── clustering/
    ├── embeddings/
    ├── preprocessing/
    └── utils/
```

## Analytical Components

- `src/preprocessing/cleaning.py`
  Speech cleaning, party normalization, and combined/yearly corpus preparation.
- `src/analysis/linguistic.py`
  Yearly readability, lexical diversity, POS, and named-entity analysis.
- `src/analysis/topic_modeling.py`
  Yearly BERTopic modeling with bigram expansion and exportable topic reports.
- `src/embeddings/generation.py`
  Linq embedding generation for cleaned speeches.
- `src/embeddings/aggregation.py`
  Shared mean-pooling utilities used by notebooks and clustering code.
- `src/clustering/deputies.py`
  Deputy aggregation and HDBSCAN clustering.
- `src/clustering/optimization.py`
  Optuna-based search for deputy clustering hyperparameters.
- `src/clustering/party.py`
  Party-year trajectory mapping and party clustering.
- `src/analysis/distance.py`
  Yearwise party-distance matrices and heatmaps.

## Setup

1. Create and activate a Python environment.
2. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install the spaCy models used by the research code:

   ```bash
   python -m spacy download pt_core_news_sm
   python -m spacy download pt_core_news_lg
   ```

4. Ensure the required corpora and cleaned inputs exist in the paths referenced by `configs/*.json`.

## Stage Entry Points

- Preprocessing:

  ```bash
  python scripts/run_preprocessing.py --config configs/preprocessing.json
  ```

- Linguistic analysis:

  ```bash
  python scripts/run_linguistic_analysis.py --config configs/linguistic_analysis.json
  ```

- Topic modeling:

  ```bash
  python scripts/run_topic_modeling.py --config configs/topic_modeling.json
  ```

- Embeddings:

  ```bash
  python scripts/run_embeddings.py --config configs/embeddings.json
  ```

- Clustering:

  ```bash
  python scripts/run_clustering.py --config configs/clustering.json --stage all
  ```

- Distance analysis:

  ```bash
  python scripts/run_distance_analysis.py --config configs/distance_analysis.json
  ```

- Smoke tests:

  ```bash
  python scripts/run_smoke_tests.py
  ```

## Implemented vs Exploratory

- `src/` and `scripts/` contain the productionized research pipeline.
- `notebooks/` preserves the exploratory notebooks and now points back to the relevant Python modules.
- Generated results under `outputs/` are not treated as source code.
- Data files are external research inputs and should be managed separately from the pipeline logic.

## Reproducibility Notes

- The repository preserves two output data shapes because the original research code used both:
  a combined cleaned corpus for embeddings/clustering and yearly cleaned corpora for yearly analyses.
- The default preprocessing behavior keeps only the first speaker segment when a transcript contains multiple speakers.
- The BERTopic refactor fixes the old `processed_texts` bug while keeping the intended workflow: topic modeling runs on already-cleaned documents.
- Downstream configs point to the migrated combined cleaned corpus at `data/processed/discursos/discursos_all_years_combined.csv`, while the single-file combined raw input path remains a configurable placeholder.
- Current configs are intentionally explicit and should be adjusted to match the local placement of the speech corpora.

## Citation

Citation information for the corresponding paper will be added here.
