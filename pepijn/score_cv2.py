from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

DEFAULT_MODEL = "anass1209/resume-job-matcher-all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def _load_model(
    model_name: str = DEFAULT_MODEL, device: Optional[str] = None
) -> SentenceTransformer:
    if device is None:
        return SentenceTransformer(model_name)
    return SentenceTransformer(model_name, device=device)


def score_cvs_multi_jobs_df(
    cv_s_dataframe: pd.DataFrame,
    job_descriptions: Sequence[str],
    job_names: Sequence[str],
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    device: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Scores each CV (one text column) against multiple job descriptions.

    Returns a DF with 2 columns per job:
      "{cvcol}_{jobname}_Raw_Score"   -> raw cosine similarity (~[-1, 1])
      "{cvcol}_{jobname}_Score"       -> mapped to [0, 1]
    """
    if cv_s_dataframe.shape[1] != 1:
        raise ValueError("Expected a DataFrame with exactly 1 CV text column.")
    if len(job_descriptions) != len(job_names):
        raise ValueError("job_descriptions and job_names must have the same length.")
    if len(job_descriptions) == 0:
        raise ValueError("Provide at least one job description/name.")

    cv_col = cv_s_dataframe.columns[-1]
    model = _load_model(model_name=model_name, device=device)

    cvs = cv_s_dataframe[cv_col].fillna("").astype(str).tolist()

    if verbose:
        print(f"Encoding {len(cvs)} CVs (batch_size={batch_size}) with {model_name}...")
    cv_emb = model.encode(
        cvs,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=bool(verbose),
    )

    if verbose:
        print(f"Encoding {len(job_descriptions)} job descriptions...")
    jd_emb = model.encode(
        list(job_descriptions),
        batch_size=min(batch_size, max(1, len(job_descriptions))),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )  # shape: (J, D)

    # cosine similarities (N x J): since embeddings are normalized, dot == cosine
    raw = (cv_emb @ jd_emb.T).astype(np.float32)
    raw = np.clip(raw, -1.0, 1.0).astype(np.float32)
    score01 = ((raw + 1.0) / 2.0).astype(np.float32)

    out = pd.DataFrame(index=cv_s_dataframe.index)
    for j, name in enumerate(job_names):
        out[f"{cv_col}_{name}_Raw_Score"] = raw[:, j]
        out[f"{cv_col}_{name}_Score"] = score01[:, j]

    return out


def score_cvs_multi_jobs_original(
    cv_s_dataframe: pd.DataFrame,
    job_descriptions: Sequence[str],
    job_names: Sequence[str],
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Scores each CV against multiple job descriptions using the original scoring method
    from score_cv.py (QdrantClient with BAAI/bge-base-en model).

    Returns a DataFrame with one column per job:
      "{cvcol}_{jobname}_Score" -> similarity score (0-100 range)
    """

    if cv_s_dataframe.shape[1] != 1:
        raise ValueError("Expected a DataFrame with exactly 1 CV text column.")
    if len(job_descriptions) != len(job_names):
        raise ValueError("job_descriptions and job_names must have the same length.")
    if len(job_descriptions) == 0:
        raise ValueError("Provide at least one job description/name.")

    cv_col = cv_s_dataframe.columns[-1]
    cvs = cv_s_dataframe[cv_col].fillna("").astype(str).tolist()

    # Initialize output DataFrame
    out = pd.DataFrame(index=cv_s_dataframe.index)

    # Initialize the QdrantClient once
    client = QdrantClient(path="./local_model")
    client.set_model("BAAI/bge-base-en")

    # For each job description, score all CVs
    for job_idx, (job_desc, job_name) in enumerate(zip(job_descriptions, job_names)):
        if verbose:
            print(f"Scoring CVs against job {job_idx + 1}/{len(job_names)}: {job_name}")

        scores = []

        for cv_idx, cv_text in enumerate(cvs):
            if verbose and cv_idx % 10 == 0:
                print(f"  Processing CV {cv_idx + 1}/{len(cvs)}...")

            # Delete collection if it exists
            if client.get_collections().collections:
                client.delete_collection(collection_name="demo_collection")

            # Add the CV as a document
            client.add(
                collection_name="demo_collection",
                documents=[cv_text],
            )

            # Query with the job description
            search_result = client.query(
                collection_name="demo_collection", query_text=job_desc
            )

            # Get similarity score and scale to 0-100
            similarity_score = round(search_result[0].score * 100, 3)
            scores.append(similarity_score)

        # Add scores for this job to the output DataFrame
        out[f"{cv_col}_{job_name}_Score"] = scores

    # Clean up the collection
    if client.get_collections().collections:
        client.delete_collection(collection_name="demo_collection")

    return out
