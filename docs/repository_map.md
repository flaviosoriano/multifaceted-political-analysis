# Repository Map

This document records how the original research files were migrated into the new structure.

## Code Mapping

- `flavio/data_cleaning.py`
  Primary source for combined-corpus preprocessing behavior.
  Migrated into `src/preprocessing/cleaning.py`.
- `social_final/scripts/models/data_cleaning.py`
  Source of yearly-file processing flow and summary generation.
  Merged into `src/preprocessing/cleaning.py`.
- `social_final/scripts/models/linguistic_analysis.py`
  Migrated into `src/analysis/linguistic.py`.
- `social_final/scripts/models/topic.py`
  Migrated into `src/analysis/topic_modeling.py`.
- `flavio/embedder.py`
  Migrated into `src/embeddings/generation.py`.
- `flavio/agrupamento_deputados.ipynb`
  Moved to `notebooks/deputy_semantic_map.ipynb`.
  Reusable aggregation logic moved into `src/embeddings/aggregation.py` and `src/clustering/deputies.py`.
- `flavio/analise_parti.ipynb`
  Moved to `notebooks/party_semantic_trajectories.ipynb`.
  Reusable logic moved into `src/clustering/party.py`.
- `flavio/cluster_deputados.ipynb`
  Moved to `notebooks/deputy_hdbscan_clustering.ipynb`.
  Reusable logic moved into `src/clustering/deputies.py`.
- `flavio/otimizador.py`
  Migrated into `src/clustering/optimization.py`.
- `flavio/partidosHDBSCAN.py`
  Merged into `src/clustering/party.py`.
- `flavio/euclidean.ipynb`
  Moved to `notebooks/party_distance_heatmaps.ipynb`.
  Reusable logic moved into `src/analysis/distance.py`.
- `flavio/criar_embeddings.py`
  Treated as an incomplete duplicate.
  Any surviving useful aggregation setup was absorbed into `src/embeddings/aggregation.py` and `src/clustering/optimization.py`.

## What Remains Exploratory

- The notebooks remain exploratory artifacts and are not the canonical production entry points.
- Plot styling and ad hoc annotations remain closer to the notebooks than to a fully generalized plotting library.
- Existing result folders are preserved as artifacts, not as validated production outputs.

## Legacy Artifact Relocation

- `social_final/data/discursos/*.csv`
  Relocated to `data/raw/discursos/`.
- `social_final/data/discursos/clean/*.csv`
  Relocated to `data/processed/discursos/`.
- `social_final/data/deputados_ativos/*.csv`
  Relocated to `data/legacy/deputados_ativos/`.
- `social_final/data/bancadas/*.csv`
  Relocated to `data/legacy/bancadas/`.
- `social_final/data/political_brsd/*.csv`
  Relocated to `data/legacy/political_brsd/`.
- `social_final/outputs/*`
  Relocated to `outputs/legacy/social_final_outputs/`.
- `social_final/viz/*`
  Relocated to `outputs/legacy/social_final_viz/`.
- `social_final/scripts/visualizations/*`
  Preserved as legacy visualization artifacts under `outputs/legacy/social_final_viz/`.
- The legacy `flavio/` and `social_final/` trees were removed after migration because their active research logic now lives in `src/`, `scripts/`, `notebooks/`, and the legacy artifact folders above.

## Manual Cleanup Still Expected

- If local data paths differ from the defaults in `configs/*.json`, update those configs before running the pipeline.
- Full runtime validation still requires installing the research dependencies and ensuring the speech corpora are present.
- Existing generated artifacts can be curated further in a later publication-cleanup pass if a lighter public release is needed.
