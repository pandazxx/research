# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 04 — Semantic chunking vs fixed-size chunking
#
# **Question.** Does semantic chunking (split at topic boundaries) produce
# meaningfully better retrieval than fixed-size or sentence-based chunking?
#
# **Why it matters.** Chunking is upstream of every RAG pipeline. The cost is
# the same; the question is whether the quality difference justifies the
# slightly more complex setup.
#
# **References.** `embeddings-beyond-cosine.md` §2.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))

import numpy as np
from experiments.shared.embedding_utils import (
    cosine,
    load_sentence_transformer,
)

# %% [markdown]
# ## Test document with clear topic shifts

# %%
document = """
Python is a programming language created by Guido van Rossum in 1991.
It is known for its readable syntax and large standard library.
Python is widely used in data science, web development, and automation.

The Eiffel Tower is a wrought-iron lattice tower in Paris, France.
It was constructed from 1887 to 1889 by Gustave Eiffel's company.
The tower stands 330 metres tall and has three observation levels.

Bananas are elongated edible fruits from plants in the genus Musa.
They are grown in over 135 countries, primarily for their fruit.
A medium banana contains about 105 calories and is rich in potassium.
""".strip()

# %% [markdown]
# ## Approach 1 — Fixed-size character chunking (the naive baseline)

# %%
def fixed_size_chunks(text, size=200, overlap=50):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks

chunks_fixed = fixed_size_chunks(document, size=200, overlap=50)
print(f"Fixed-size chunks: {len(chunks_fixed)}")
for c in chunks_fixed[:3]:
    print("---")
    print(c[:150])

# %% [markdown]
# ## Approach 2 — Sentence-based chunking

# %%
import re

def sentence_chunks(text, sentences_per_chunk=2):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return [
        " ".join(sentences[i:i + sentences_per_chunk])
        for i in range(0, len(sentences), sentences_per_chunk)
    ]

chunks_sentence = sentence_chunks(document)
print(f"Sentence chunks: {len(chunks_sentence)}")
for i in range(len(chunks_sentence)):
    print(f"[{i}]: {chunks_sentence[i]}")

# %% [markdown]
# ## Approach 3 — Semantic chunking (embedding-based)

# %%
def semantic_chunks(text, embedder, percentile_threshold=80):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    embs = embedder.encode(sentences)

    distances = [1 - cosine(embs[i], embs[i + 1]) for i in range(len(embs) - 1)]
    threshold = np.percentile(distances, percentile_threshold)

    chunks = []
    current = [sentences[0]]
    for i, d in enumerate(distances):
        if d > threshold:
            chunks.append(" ".join(current))
            current = []
        current.append(sentences[i + 1])
    if current:
        chunks.append(" ".join(current))
    return chunks

embedder = load_sentence_transformer("all-MiniLM-L6-v2")
chunks_semantic = semantic_chunks(document, embedder)
print(f"Semantic chunks: {len(chunks_semantic)}")
for c in chunks_semantic:
    print("---")
    print(c)

# %% [markdown]
# Ideally, each semantic chunk should correspond to one topic
# (Python, Eiffel Tower, Bananas). Verify by reading the chunks.

# %% [markdown]
# ## Retrieval comparison
#
# Embed each chunk set, then query with topic-specific questions. Measure
# which chunking returns the right topic for each query.

# %%
queries = {
    "When was the Eiffel Tower built?": "1887 to 1889",
    "Who created Python?": "Guido van Rossum",
    "How many calories in a banana?": "about 105 calories",
}

def evaluate_chunking(name, chunks, embedder, queries):
    chunk_embs = embedder.encode(chunks)
    print(f"\n=== {name} (chunks: {len(chunks)}) ===")
    for q, expected_topic in queries.items():
        q_emb = embedder.encode(q)
        sims = np.array([cosine(q_emb, c) for c in chunk_embs])
        top = sims.argmax()
        hit = expected_topic.lower() in chunks[top].lower()
        print(f"  Q: {q!r}")
        print(f"    top chunk (sim={sims[top]:.3f}): {chunks[top]}")
        print(f"    expected '{expected_topic}': {'HIT' if hit else 'MISS'}")

evaluate_chunking("Fixed-size", chunks_fixed, embedder, queries)
evaluate_chunking("Sentence", chunks_sentence, embedder, queries)
evaluate_chunking("Semantic", chunks_semantic, embedder, queries)

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** Retrieval accuracy on 3 topic-specific queries
#    against 3 chunking strategies for a 3-topic document.
# 2. **What did I find?** ___
# 3. **What surprised me?** ___
# 4. **What's next?** Test on a longer document where the gap is more visible,
#    or extend with late chunking (`05-late-chunking.py`).
