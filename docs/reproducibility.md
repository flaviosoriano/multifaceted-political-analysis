# Reproducibility Notes

## What This Repository Guarantees

- The active code path lives in `src/` and `scripts/`.
- Each major stage has a config-backed entry point.
- Original exploratory notebooks are preserved and linked to the refactored modules.
- No stage depends on code hidden in `data/`, `outputs/`, `viz/`, or `visualizations/`.

## External Requirements

- Python dependencies from `requirements.txt`
- spaCy models:
  `pt_core_news_sm`
  `pt_core_news_lg`
- NLTK resources:
  `stopwords`
  `punkt`
- Access to the original speech corpora in the locations configured in `configs/*.json`

## Smoke Testing

Run the static and CLI smoke tests:

```bash
python scripts/run_smoke_tests.py
```

This validates:

- `src/` and `scripts/` compile
- JSON configs are readable
- notebooks remain valid JSON and start with the required context note
- CLI wrappers expose `--help` without importing the heavy research stack too early

## Runtime Validation

Full stage execution still depends on data availability and the research environment. Suggested order for a fresh researcher:

1. Update `configs/preprocessing.json` to match local raw data paths.
2. Run preprocessing.
3. Run embeddings if the combined cleaned corpus is available.
4. Run yearly linguistic analysis and topic modeling from the yearly cleaned files.
5. Run deputy and party clustering.
6. Run the distance analysis.

## Known Ambiguities

- `flavio/criar_embeddings.py` was incomplete, so it was documented as superseded rather than exposed as a standalone entry point.
- The topic-modeling refactor preserves the intended workflow but fixes a latent bug in the original script where `processed_texts` was referenced after its setup had been commented out.
- The repository keeps defaults close to the original research files, but users may need to adapt config paths to their local data placement.
- A combined cleaned corpus is available at `data/processed/discursos/discursos_all_years_combined.csv`; the single-file combined raw input path in preprocessing configs is still a placeholder and should be updated if that raw source is available locally.
