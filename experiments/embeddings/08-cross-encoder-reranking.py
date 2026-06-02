# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 08 — Cross-encoder reranking
#
# **Question.** How much does adding a cross-encoder reranking step improve
# retrieval over pure dense-embedding similarity?
#
# **Why it matters.** This is the single biggest practical quality lift in
# RAG pipelines and is often missing from research-paper architectures. If you
# care about retrieval quality at all, you should be using a reranker.
#
# **References.** `embedding-applications-survey.md` §2.5 and §3.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))

import numpy as np
import pandas as pd
from experiments.shared.embedding_utils import load_sentence_transformer
from experiments.shared.dataset_loader import load_comparison_dataset

# %% [markdown]
# ## Setup: use the comparison-dataset memories as the corpus

# %%
dataset = load_comparison_dataset()
memories = dataset["memories"]
questions = dataset["questions"]

texts = [m["content"] for m in memories]
print(f"Corpus: {len(texts)} memories")
print(f"Questions: {len(questions)}")

# %% [markdown]
# ## Stage 1: Dense embedding retrieval

# %%
embedder = load_sentence_transformer("BAAI/bge-large-en-v1.5")
doc_embs = embedder.encode(texts)

def dense_top_k(query: str, k: int = 10):
    q_emb = embedder.encode(query)
    sims = doc_embs @ q_emb / (
        np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(q_emb)
    )
    top_idx = np.argsort(sims)[-k:][::-1]
    return [(int(i), float(sims[i]), texts[i]) for i in top_idx]

# %% [markdown]
# ## Stage 2: Cross-encoder reranker

# %%
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")

def rerank(query: str, candidates: list[tuple[int, float, str]]):
    pairs = [(query, text) for _, _, text in candidates]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [(c, float(s)) for c, s in reranked]

# %% [markdown]
# ## Evaluation
#
# For each question with known `requires_facts` (the gold memory ids), measure:
# - Top-1 accuracy: is the right memory at position 1?
# - Top-3 accuracy: is the right memory in the top 3?

# %%
def evaluate(retrieval_fn, name: str):
    top1_hits = 0
    top3_hits = 0
    total = 0
    for q in questions:
        if not q.get("requires_facts"):
            continue
        total += 1
        results = retrieval_fn(q["question"])
        retrieved_ids = [memories[i]["id"] for i, _, _ in results]
        gold = set(q["requires_facts"])
        if retrieved_ids[0] in gold:
            top1_hits += 1
        if any(r in gold for r in retrieved_ids[:3]):
            top3_hits += 1
    print(f"{name:>30}: top1={top1_hits}/{total}, top3={top3_hits}/{total}")

# Dense-only retrieval (top 3 from dense, no rerank)
def dense_only(q):
    return dense_top_k(q, k=3)

# Dense + rerank (top 10 from dense, rerank, take top 3)
def dense_plus_rerank(q):
    candidates = dense_top_k(q, k=10)
    reranked = rerank(q, candidates)
    return [c for c, _ in reranked[:3]]

evaluate(dense_only, "Dense embeddings only")
evaluate(dense_plus_rerank, "Dense + cross-encoder rerank")

# %% [markdown]
# Expected pattern: cross-encoder reranking should give a clear top-1 lift
# (typically 5–20 percentage points) for the cost of one extra model call.

# %% [markdown]
# ## Why it works
#
# Bi-encoders (dense embedders) compute query and document embeddings
# independently, then compare them via cosine. Information is lost when each
# embedding is pooled to a single vector.
#
# Cross-encoders take both the query AND the document as joint input, run them
# through a transformer with full attention, and output a scalar relevance
# score. They preserve information at the cost of being slower per pair.
#
# The standard pattern combines both: bi-encoder for cheap top-50 retrieval,
# cross-encoder for accurate top-5 reranking. Best quality for the lowest
# total cost.

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** Top-1 and top-3 retrieval accuracy with and
#    without a cross-encoder rerank step on the comparison dataset.
# 2. **What did I find?** ___
# 3. **What surprised me?** ___
# 4. **What's next?** If reranking helps, add it as a layer to the HippoRAG
#    and A-Mem pipelines and rerun the comparison dataset.
