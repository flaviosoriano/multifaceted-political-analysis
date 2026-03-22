# Methodology Overview

The repository is organized around the workflow that was recoverable from the original notebooks and scripts.

## 1. Preprocessing

- Load raw speech CSV files.
- Drop rows without transcriptions.
- Clean speaker markers, salutations, and extraneous punctuation.
- Normalize party names.
- Keep only the first speaker segment by default when multiple speakers appear in the same transcript.
- Produce both a combined cleaned corpus and yearly cleaned corpora because later stages depend on both layouts.

## 2. Yearly Linguistic Analysis

- Analyze yearly cleaned corpora with spaCy.
- Compute per-speech and yearly aggregates for:
  word count, sentence count, type-token ratio, Portuguese-adapted Flesch readability, POS percentages, and named entities.
- Save detailed yearly outputs and aggregate summaries.

## 3. Topic Modeling

- Run BERTopic separately for each yearly cleaned corpus.
- Use token normalization through spaCy, phrase detection with Gensim, and a multilingual sentence-transformer embedding model.
- Export textual topic summaries for each year.

## 4. Embeddings

- Generate speech-level embeddings from the combined cleaned corpus using the Linq embedding model.
- Save raw embeddings, L2-normalized embeddings, and discourse IDs.

## 5. Party and Deputy Semantic Analyses

- Mean-pool speech embeddings by deputy for deputy-level maps and clustering.
- Mean-pool speech embeddings by `(year, party)` for party trajectories and party clustering.
- Use UMAP for projection and HDBSCAN for clustering.

## 6. Optimization and Distance Experiments

- Use Optuna to search UMAP/HDBSCAN hyperparameters for deputy clustering.
- Compute yearly Euclidean distance matrices between party-level pooled embeddings and render heatmaps.
