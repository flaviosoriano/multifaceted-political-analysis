# Agent Guidance

This repository is organized as a reproducible companion codebase for an academic paper.

## Core Rules

- Preserve scientific intent from the original research files in the former `flavio/` workspace.
- Reuse relevant logic from `social_final/scripts/models/` only when it supplements the primary research code without changing the study design.
- Do not use `data/`, `outputs/`, `viz/`, or `visualizations/` as implementation sources.
- Keep notebooks exploratory and keep production logic in `.py` modules under `src/`.
- Prefer minimal behavioral change when refactoring.
- Avoid absolute paths and use `pathlib.Path` throughout the codebase.
- Document assumptions and unresolved ambiguities in docs or TODO notes instead of silently changing behavior.

## Migration-Specific Notes

- The default preprocessing behavior keeps only the first speaker segment when multiple speakers appear.
- Two cleaned data shapes are intentionally supported:
  a combined corpus for embeddings/clustering and yearly cleaned files for linguistic/topic analysis.
- `flavio/criar_embeddings.py` was an incomplete duplicate and is treated as superseded by the refactored aggregation/clustering modules.
- When adding new scripts, parse CLI arguments before importing heavy ML/NLP libraries so `--help` remains a useful smoke test.

## Safe Refactoring Practices

- Keep old-to-new mappings up to date in `docs/repository_map.md`.
- Do not fabricate datasets, figures, or finished experimental results.
- Prefer documenting manual follow-up over guessing when data provenance is unclear.
