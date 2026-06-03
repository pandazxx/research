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
# # 01 — Anisotropy: how similar are tokens within one sentence?
#
# **Question.** When two tokens are in the same sentence, what's the cosine
# similarity between their contextualized vectors? Is it close to 0 (as one
# might expect from "two different tokens"), or much higher?
#
# **Why it matters.** This experiment establishes the *baseline* for any
# subsequent token-level analysis. If within-sequence similarity is already
# 0.85+ for unrelated tokens, then differences smaller than that aren't
# meaningful — they're noise from anisotropy.
#
# **Reference.** `topics/llm-agent-memory/embeddings-101.md` §11.

# %%
import sys
from pathlib import Path

# Make `experiments.shared` importable from this notebook regardless of cwd
sys.path.insert(0, str(Path.cwd().parents[1]))

import numpy as np
import pandas as pd
from experiments.shared.embedding_utils import (
    cosine,
    find_first_token_position,
    get_token_vectors,
    pairwise_cosine_table,
)

# %% [markdown]
# ## The Joey/She/wife setup
#
# A single sequence with two distinct entities and two pronouns referring to them:

# %%
text = (
    "Joey is living in Singapore. She is a software engineer. "
    "My wife is living in Singapore too. She is a house wife."
)
print(text)

# %% [markdown]
# ## Extract per-token vectors

# %%
tokens, vectors, mask = get_token_vectors(text, model_name="BAAI/bge-large-en-v1.5")
print(f"Number of tokens: {len(tokens)}")
print(f"Vector dim:       {vectors.shape[1]}")
print(f"First 20 tokens:  {tokens[:20]}")

# %% [markdown]
# ## Locate the tokens we care about

# %%
joey_positions = [i for i, t in enumerate(tokens) if "joey" in t.lower()]
she_positions = [i for i, t in enumerate(tokens) if t.lower() in ("she", "▁she")]
wife_positions = [i for i, t in enumerate(tokens) if "wife" in t.lower()]

print(f"Joey at positions: {joey_positions}")
print(f"'She' at positions: {she_positions}")
print(f"'wife' at positions: {wife_positions}")

# %% [markdown]
# Use the first occurrence of each:

# %%
joey_idx = joey_positions[0]
she1_idx = she_positions[0]
she2_idx = she_positions[1]
wife_idx = wife_positions[0]

vec_joey = vectors[joey_idx]
vec_she1 = vectors[she1_idx]
vec_she2 = vectors[she2_idx]
vec_wife = vectors[wife_idx]

# %% [markdown]
# ## Pairwise cosine similarities

# %%
labels = [f"Joey@{joey_idx}", f"she1@{she1_idx}", f"she2@{she2_idx}", f"wife@{wife_idx}"]
vecs = np.stack([vec_joey, vec_she1, vec_she2, vec_wife])
pairwise_cosine_table(labels, vecs)

# %% [markdown]
# ## Expected pattern (with BGE-large-en-v1.5)
#
# Based on prior measurement:
#
# | Pair | Cosine | Why |
# |---|---|---|
# | she1 ↔ she2 | ~0.96 | Same token, same sequence — base identity dominates |
# | she2 ↔ wife | ~0.93 | Adjacent tokens, heavy local attention |
# | she1 ↔ Joey | ~0.85 | Distant tokens, same sequence + soft coreference |
# | Joey ↔ wife | ~0.79 | Distant tokens, distinct entities — within-sequence floor |
#
# The whole range is compressed into 0.79–0.96. **Two random tokens from
# different sequences would typically score 0.30–0.55.** That gap (0.79 vs 0.40)
# is the anisotropy.

# %% [markdown]
# ## Bonus — what does a random pair across sequences look like?

# %%
text_unrelated = "The cat sat on the mat. Coffee is a hot beverage."
tokens_unrelated, vectors_unrelated, _ = get_token_vectors(text_unrelated)

# Pick a random content token from the unrelated sequence
cat_idx = find_first_token_position(tokens_unrelated, "cat")
vec_cat = vectors_unrelated[cat_idx]

cross_sequence_baseline = cosine(vec_joey, vec_cat)
print(f"Joey (in our sequence) ↔ cat (different sequence): {cross_sequence_baseline:.3f}")
print(f"Joey ↔ wife (same sequence, distant):              {cosine(vec_joey, vec_wife):.3f}")

# %% [markdown]
# The first number should be substantially lower than the second — confirming
# that "in the same sequence" itself contributes ~0.3 to cosine similarity
# regardless of token identity.

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** Per-token cosine similarity between four tokens
#    within a single 24-token sequence.
# 2. **What did I find?** All within-sequence pairs scored 0.79–0.96, with the
#    ordering matching base-token identity > local attention > coreference >
#    distant unrelated entities.
# 3. **What surprised me?** ___ (fill in after running)
# 4. **What's next?** Run `02-distance-ablation.py` to see how within-sequence
#    similarity decays with token distance specifically.

# %%
