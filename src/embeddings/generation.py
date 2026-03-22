from __future__ import annotations

"""Sentence-embedding generation for cleaned speech corpora."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class EmbeddingArtifacts:
    """Paths produced by an embedding generation run."""

    raw_embeddings_path: Path
    normalized_embeddings_path: Path
    ids_path: Path
    enriched_csv_path: Path


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize a 2D array row-wise."""

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _resolve_output_base(csv_path: Path, output_base: str | Path | None) -> Path:
    if output_base is not None:
        return Path(output_base)
    stem = csv_path.with_suffix("")
    return Path(f"{stem}_Linq")


def generate_embeddings(
    csv_path: str | Path,
    *,
    output_base: str | Path | None = None,
    model_name: str = "Linq-AI-Research/Linq-Embed-Mistral",
    text_column: str = "transcricao_limpa",
    batch_size: int = 4,
    add_id_column: bool = True,
    overwrite_csv: bool = False,
    device: str = "cuda",
    limit: int | None = None,
) -> EmbeddingArtifacts:
    """Generate raw and normalized embeddings for a cleaned CSV corpus."""

    from sentence_transformers import SentenceTransformer

    csv_file = Path(csv_path)
    df = pd.read_csv(csv_file, dtype=str)
    if limit is not None:
        df = df.head(limit).copy()

    if add_id_column and "discurso_id" not in df.columns:
        df.insert(0, "discurso_id", range(len(df)))

    texts = df[text_column].astype(str).tolist()
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    if device.startswith("cuda"):
        model = model.half()

    embeddings_raw = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    embeddings_raw = embeddings_raw.astype(np.float32)
    embeddings_norm = l2_normalize(embeddings_raw.copy())

    base = _resolve_output_base(csv_file, output_base)
    base.parent.mkdir(parents=True, exist_ok=True)

    raw_path = Path(f"{base}_embeddings_raw.npy")
    norm_path = Path(f"{base}_embeddings_norm.npy")
    ids_path = Path(f"{base}_ids.csv")
    enriched_path = csv_file if overwrite_csv else Path(f"{base}_com_id.csv")

    np.save(raw_path, embeddings_raw)
    np.save(norm_path, embeddings_norm)
    df[["discurso_id"]].to_csv(ids_path, index=False)
    df.to_csv(enriched_path, index=False)

    return EmbeddingArtifacts(
        raw_embeddings_path=raw_path,
        normalized_embeddings_path=norm_path,
        ids_path=ids_path,
        enriched_csv_path=enriched_path,
    )

