# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 09 — Anisotropy under context splitting
#
# **Question.** What happens to the within-sequence cosine similarities from
# `01-anisotropy.py` when we *split* the document into one-sentence chunks
# and embed each chunk independently?
#
# **Why it matters.** This is the naïve "early chunking" failure mode that
# motivates **late chunking** (`05-late-chunking.py`). When "She" lives in a
# separate chunk from its antecedent "Joey" / "wife", the embedder has no way
# to bind the pronoun to the entity, *and* both tokens lose the within-sequence
# anisotropy boost they shared. The cross-token similarities — which sat at
# 0.79–0.96 inside one sequence — should collapse toward the cross-sequence
# floor (~0.30–0.55) once the context is cut.
#
# **Reference.** Direct follow-up to `01-anisotropy.py`; preview of
# `05-late-chunking.py`.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parents[1]))

import numpy as np
import pandas as pd
from experiments.shared.embedding_utils import (
    find_first_token_position,
    get_token_vectors,
    pairwise_cosine_table,
)

MODEL_NAME = "BAAI/bge-large-en-v1.5"
LABELS = ["Joey", "she1", "wife", "she2"]

# %% [markdown]
# ## Same data, two views
#
# The original sequence from 01-anisotropy is on the left; on the right it
# has been split into 4 one-sentence chunks. Each chunk will be embedded
# independently — no cross-sentence attention.

# %%
combined_text = (
    "Joey is living in Singapore. She is a software engineer. "
    "My wife is living in Singapore too. She is a house wife."
)

split_sentences = [
    "Joey is living in Singapore.",
    "She is a software engineer.",
    "My wife is living in Singapore too.",
    "She is a house wife.",
]

# Which surface token to extract from each split sentence, in LABELS order.
split_targets = ["joey", "she", "wife", "she"]

print(f"Combined text ({len(combined_text)} chars):\n  {combined_text}\n")
print("Split into 4 sentences:")
for i, s in enumerate(split_sentences):
    print(f"  [{i}] {s}")

# %% [markdown]
# ## View A — Combined (baseline from 01)
#
# Embed the whole document, locate the 4 tokens of interest, take their
# vectors.

# %%
tokens_c, vectors_c, _ = get_token_vectors(combined_text, model_name=MODEL_NAME)

joey_positions = [i for i, t in enumerate(tokens_c) if "joey" in t.lower()]
she_positions  = [i for i, t in enumerate(tokens_c) if t.lower() in ("she", "▁she")]
wife_positions = [i for i, t in enumerate(tokens_c) if "wife" in t.lower()]

idx_combined = {
    "Joey": joey_positions[0],
    "she1": she_positions[0],
    "wife": wife_positions[0],
    "she2": she_positions[1],
}
vecs_combined = np.stack([vectors_c[idx_combined[k]] for k in LABELS])

print(f"Combined sequence length: {len(tokens_c)} tokens")
for k in LABELS:
    print(f"  {k:5s} -> idx {idx_combined[k]:>3d}  token='{tokens_c[idx_combined[k]]}'")

# %%
combined_table = pairwise_cosine_table(LABELS, vecs_combined)
combined_table

# %% [markdown]
# ## View B — Split (one sentence per embedding pass)
#
# For each of the 4 sentences we run a separate forward pass and pull the
# target token's vector from that sentence's own sequence.

# %%
vecs_split = []
for sent, target, label in zip(split_sentences, split_targets, LABELS):
    toks, vecs, _ = get_token_vectors(sent, model_name=MODEL_NAME)
    idx = find_first_token_position(toks, target)
    print(f"  {label:5s} (sentence: '{sent}')  -> idx {idx:>3d}  "
          f"token='{toks[idx]}'  (this sentence has {len(toks)} tokens)")
    vecs_split.append(vecs[idx])
vecs_split = np.stack(vecs_split)

# %%
split_table = pairwise_cosine_table(LABELS, vecs_split)
split_table

# %% [markdown]
# ## Delta — what does splitting cost?

# %%
delta_table = (split_table - combined_table).round(3)
delta_table

# %% [markdown]
# ### Pair-by-pair reading

# %%
pairs = [("Joey", "she1"), ("Joey", "wife"), ("wife", "she2"), ("she1", "she2")]
rows = []
for a, b in pairs:
    rows.append({
        "pair":     f"{a} ↔ {b}",
        "combined": float(combined_table.loc[a, b]),
        "split":    float(split_table.loc[a, b]),
        "delta":    float(split_table.loc[a, b] - combined_table.loc[a, b]),
    })
pd.DataFrame(rows).round(3)

# %% [markdown]
# ## Expected pattern
#
# - **Joey ↔ she1**: largest drop. In the combined sequence, BGE could route
#   some coreference signal between them; in the split, "she1" sits in a
#   one-sentence chunk with no antecedent to bind to.
# - **wife ↔ she2**: smaller drop. Sentence 4 — "She is a house wife." — still
#   contains the word "wife", so "she2" picks up a local coreference cue
#   inside its own chunk. (This is the "lucky" pair.)
# - **she1 ↔ she2**: large drop. Same surface token, but now in completely
#   different sentences, so the within-sequence anisotropy boost is gone.
# - **Joey ↔ wife**: smallest drop in absolute terms — they were already
#   "distant unrelated entities" in the combined view (~0.79), so the floor
#   is closer to where they're heading.
#
# If observed, this is the empirical motivation for **late chunking**: embed
# once over the full document so pronoun-antecedent ties survive in the token
# representations, *then* slice them per chunk afterward. `05-late-chunking.py`
# walks through that recipe with Jina v3.

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** Pairwise cosine between the same four tokens
#    (Joey, she1, wife, she2) computed two ways: in the original 4-sentence
#    document and in 4 independently-embedded one-sentence chunks.
# 2. **What did I find?** ___ (fill in after running)
# 3. **What surprised me?** ___
# 4. **What's next?** Run `05-late-chunking.py` and check whether
#    late chunking recovers most of the similarity lost here.
