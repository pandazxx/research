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
# # 06 — Clustering memories for automatic topic discovery
#
# **Question.** Given a corpus of memories, can we automatically discover the
# topic clusters using UMAP + HDBSCAN on their embeddings?
#
# **Why it matters.** Two practical uses for the project: (a) memory
# consolidation — group similar memories for summarisation; (b) discovery —
# understand what's actually in a memory corpus without manual reading.
#
# **References.** `embedding-applications-survey.md` §2.1.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiments.shared.embedding_utils import load_sentence_transformer
from experiments.shared.dataset_loader import load_comparison_dataset

# %% [markdown]
# ## Load the comparison dataset's memories as our corpus

# %%
dataset = load_comparison_dataset()
memories = dataset["memories"]
texts = [m["content"] for m in memories]
print(f"Number of memories: {len(texts)}")
print("First 3:")
for t in texts[:3]:
    print(f"  - {t}")

# %% [markdown]
# ## Embed

# %%
embedder = load_sentence_transformer("BAAI/bge-large-en-v1.5")
embs = embedder.encode(texts, show_progress_bar=True)
print(f"Embedding shape: {embs.shape}")

# %% [markdown]
# ## Dimensionality reduction (UMAP)

# %%
import umap

reducer = umap.UMAP(
    n_neighbors=5,
    min_dist=0.1,
    n_components=2,
    metric="cosine",
    random_state=42,
)
embedding_2d = reducer.fit_transform(embs)

# %% [markdown]
# ## Clustering (HDBSCAN)

# %%
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2)
labels = clusterer.fit_predict(embedding_2d)
print(f"Discovered clusters: {set(labels)}")
print(f"  (-1 = noise / outlier)")

# %% [markdown]
# ## Visualise

# %%
fig, ax = plt.subplots(figsize=(11, 7))
for cluster_id in sorted(set(labels)):
    mask = labels == cluster_id
    color = "lightgray" if cluster_id == -1 else None
    label = "outlier" if cluster_id == -1 else f"cluster {cluster_id}"
    ax.scatter(
        embedding_2d[mask, 0], embedding_2d[mask, 1],
        c=color, label=label, s=80, alpha=0.7,
    )
ax.legend()
ax.set_title("Memory clusters (UMAP + HDBSCAN over BGE embeddings)")
ax.set_xlabel("UMAP dim 1")
ax.set_ylabel("UMAP dim 2")
fig.tight_layout()
fig.show()

# %% [markdown]
# ## Inspect each cluster

# %%
df = pd.DataFrame({"text": texts, "cluster": labels})
for cluster_id in sorted(df["cluster"].unique()):
    if cluster_id == -1:
        continue
    print(f"\n=== Cluster {cluster_id} ===")
    for t in df[df["cluster"] == cluster_id]["text"]:
        print(f"  - {t}")

# %% [markdown]
# ## Use case — memory consolidation
#
# Once memories are clustered, you could:
# - Generate one summary per cluster via LLM
# - Replace the individual memories with the summary (forgetting + consolidation)
# - Or keep individual memories but add a "cluster summary" memory pointing back to them
#
# This is exactly the kind of operation the brain's hippocampus → neocortex
# transfer is doing during sleep (per the brain-memory deep-dive).

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** Automatic clustering of 40 conversational
#    memories using cosine-UMAP + HDBSCAN.
# 2. **What did I find?** ___
# 3. **What surprised me?** ___
# 4. **What's next?** Try generating one LLM summary per cluster and verify
#    summaries are useful (or compare against single-memory retrieval).
