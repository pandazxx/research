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
# # 07 — Novelty scoring as a write-policy signal
#
# **Question.** Can we compute a "novelty score" for a new memory by measuring
# its distance to existing memories? Memories with high novelty should be more
# worth storing; low-novelty memories are near-duplicates and can be dropped or
# merged.
#
# **Why it matters.** This is the embedding-side analogue of the brain's
# dopamine-driven novelty signal for memory consolidation (per the brain memory
# deep-dive). It's a concrete mechanism for a smarter write policy in
# reconsolidation-/forgetting-aware memory systems.
#
# **References.** `embedding-applications-survey.md` §2.4; the brain memory
# deep-dive on novelty/dopamine/LTP.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))

import numpy as np
import pandas as pd
from experiments.shared.embedding_utils import load_sentence_transformer, cosine_matrix
from experiments.shared.dataset_loader import load_comparison_dataset

# %% [markdown]
# ## Setup: the existing memory corpus

# %%
dataset = load_comparison_dataset()
memories = dataset["memories"]
existing_texts = [m["content"] for m in memories]

embedder = load_sentence_transformer("BAAI/bge-large-en-v1.5")
existing_embs = embedder.encode(existing_texts)
print(f"Existing memory corpus: {len(existing_texts)} entries")

# %% [markdown]
# ## Candidate new memories
#
# A mix of: near-duplicates of existing memories, genuinely novel content, and
# moderately-related additions.

# %%
candidates = [
    # Near-duplicates (low novelty expected)
    "Sam works as a software engineer at TechCorp.",
    "Sam's mother Maria lives in Boston.",
    "Sam's brother is in Seattle.",

    # Moderately related (medium novelty)
    "Sam went hiking at Mount Tamalpais last weekend.",
    "Maria visited Sam in Oakland last spring.",

    # Genuinely novel (high novelty expected)
    "Sam published a paper on Rust async runtimes.",
    "The new bakery on 4th Street opens at 6 AM.",
    "Sam adopted a cat named Mochi from the local shelter.",
]
candidate_embs = embedder.encode(candidates)

# %% [markdown]
# ## Compute novelty scores

# %%
def novelty_score(candidate_emb, existing_embs):
    """1 - max similarity to existing. Higher = more novel."""
    sims = cosine_matrix(candidate_emb.reshape(1, -1), existing_embs)[0]
    return 1.0 - float(sims.max())

results = []
for text, emb in zip(candidates, candidate_embs):
    score = novelty_score(emb, existing_embs)
    nearest_idx = int(cosine_matrix(emb.reshape(1, -1), existing_embs)[0].argmax())
    results.append({
        "candidate": text,
        "novelty_score": round(score, 3),
        "nearest_existing": existing_texts[nearest_idx][:60] + "...",
    })

df = pd.DataFrame(results).sort_values("novelty_score", ascending=False)
df

# %% [markdown]
# ## Write-policy thresholds
#
# A simple policy:
#
# - novelty < 0.10 → near-duplicate; merge or update existing memory
# - 0.10 ≤ novelty < 0.30 → related; store with a link to nearest existing
# - novelty ≥ 0.30 → genuinely new; store as standalone with high importance

# %%
def categorise(score):
    if score < 0.10:
        return "near-duplicate"
    elif score < 0.30:
        return "related"
    else:
        return "novel"

df["category"] = df["novelty_score"].apply(categorise)
df

# %% [markdown]
# ## Implications for the reconsolidation project
#
# The novelty signal could feed several mechanisms:
#
# 1. **Selective forgetting** — memories that are low-novelty relative to
#    others are candidates for pruning (their information is already
#    represented).
# 2. **Importance scoring** — novel memories get higher initial importance
#    weights (analogous to dopamine release for novel events).
# 3. **Memory dedup** — near-duplicates trigger update-on-existing rather
#    than new-memory-creation.
# 4. **Trigger for reconsolidation** — a moderately-novel memory whose
#    nearest neighbour is from months ago is a candidate for triggering
#    reconsolidation of the older memory.

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** Novelty (1 - max-cosine) for 8 candidate
#    memories against a 40-memory corpus.
# 2. **What did I find?** ___
# 3. **What surprised me?** ___
# 4. **What's next?** Integrate this scoring into a write-policy that
#    drives A-Mem's memory creation decisions, and measure whether the
#    KG quality improves.
