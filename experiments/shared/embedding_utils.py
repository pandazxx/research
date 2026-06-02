"""Reusable embedding helpers used across experiments.

Functions are intentionally simple and dependency-light. Anything that's
genuinely common across more than one experiment lives here; one-off helpers
stay inside the notebook that uses them.
"""

from __future__ import annotations

import functools
from typing import Iterable

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Model and tokenizer loading (cached)
# -----------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def load_hf_model(name: str, device: str | None = None):
    """Load a HuggingFace AutoTokenizer + AutoModel, cached by name.

    Use this when you need access to per-token outputs (last hidden state).
    For sentence-level retrieval, prefer load_sentence_transformer.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


@functools.lru_cache(maxsize=8)
def load_sentence_transformer(name: str):
    """Load a sentence-transformers model, cached by name.

    Use this for standard sentence embeddings. Internally returns numpy arrays.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


# -----------------------------------------------------------------------------
# Similarity primitives
# -----------------------------------------------------------------------------

def cosine(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> float:
    """Cosine similarity between two 1-D vectors. Returns a Python float."""
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix. A: (n, d), B: (m, d) -> (n, m)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T


# -----------------------------------------------------------------------------
# Per-token analysis
# -----------------------------------------------------------------------------

def get_token_vectors(text: str, model_name: str = "BAAI/bge-large-en-v1.5"):
    """Return (token_strings, token_vectors, attention_mask) for a single input.

    token_vectors is a numpy array of shape (num_tokens, hidden_dim).
    """
    tokenizer, model, device = load_hf_model(model_name)

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    token_vectors = outputs.last_hidden_state[0].cpu().numpy()
    token_strings = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    mask = inputs["attention_mask"][0].cpu().numpy()
    return token_strings, token_vectors, mask


def find_token_positions(token_strings: list[str], target: str) -> list[int]:
    """Find all positions where a token string appears (case-insensitive,
    matches substrings to handle wordpiece prefixes like '▁maria')."""
    target = target.lower().strip()
    return [
        i for i, tok in enumerate(token_strings)
        if target in tok.lower().replace("▁", "").replace("##", "")
    ]


def find_first_token_position(token_strings: list[str], target: str) -> int:
    """Return the first position of a token, or -1 if not found."""
    positions = find_token_positions(token_strings, target)
    return positions[0] if positions else -1


# -----------------------------------------------------------------------------
# Pooling
# -----------------------------------------------------------------------------

def mean_pool(token_vectors: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Mean-pool token vectors (ignoring positions with mask=0).

    token_vectors: (num_tokens, hidden_dim)
    mask:          (num_tokens,) optional. 1=real token, 0=padding.
    Returns:       (hidden_dim,)
    """
    if mask is None:
        return token_vectors.mean(axis=0)
    mask = np.asarray(mask, dtype=np.float64)[:, None]
    return (token_vectors * mask).sum(axis=0) / mask.sum().clip(min=1e-9)


# -----------------------------------------------------------------------------
# Diagnostic table builder
# -----------------------------------------------------------------------------

def pairwise_cosine_table(
    labels: Iterable[str],
    vectors: np.ndarray,
) -> "pandas.DataFrame":
    """Build a pretty DataFrame of pairwise cosine similarities."""
    import pandas as pd

    labels = list(labels)
    sims = cosine_matrix(vectors, vectors)
    df = pd.DataFrame(sims, index=labels, columns=labels)
    return df.round(3)
