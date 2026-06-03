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
# # 02 — Distance ablation: similarity vs token distance
#
# **Question.** Within one sequence, how does cosine similarity between two
# tokens decay as they move further apart?
#
# **Why it matters.** Anisotropy is a "floor" effect, but on top of that floor
# there's structure. Local attention pulls nearby tokens together more strongly.
# Measuring the decay curve quantifies how much "structure" survives anisotropy.
#
# **Expected outcome.** Smooth decay from ~0.95 at distance 1 down to ~0.78 at
# distance 20+. The shape (linear? exponential? log?) is itself interesting.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiments.shared.embedding_utils import cosine, get_token_vectors

# %%
# A longer sentence with diverse content tokens spread across positions
text = (
    "Joey lives in Singapore. Alice works in Tokyo. "
    "Bob teaches in Boston. Dana cooks in Lisbon. "
    "Sam paints in Madrid. Tom builds in Berlin."
)
tokens, vectors, mask = get_token_vectors(text)
print(f"Total tokens: {len(tokens)}")

# %% [markdown]
# ## Compute pairwise similarity for all token pairs, recording distance

# %%
n = len(tokens)
data = []
for i in range(n):
    for j in range(i + 1, n):
        if mask[i] == 0 or mask[j] == 0:
            continue
        data.append({
            "i": i,
            "j": j,
            "distance": j - i,
            "tok_i": tokens[i],
            "tok_j": tokens[j],
            "cosine": cosine(vectors[i], vectors[j]),
        })

df = pd.DataFrame(data)
df.head()

# %% [markdown]
# ## Mean similarity by distance bucket

# %%
df["distance_bin"] = pd.cut(df["distance"], bins=[0, 2, 5, 10, 20, 50])
summary = df.groupby("distance_bin", observed=True)["cosine"].agg(["mean", "std", "count"]).round(3)
summary

# %% [markdown]
# ## Plot the decay curve

# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(df["distance"], df["cosine"], alpha=0.3, s=10)
# Rolling mean
df_sorted = df.sort_values("distance")
rolling = df_sorted["cosine"].rolling(20, min_periods=5).mean()
ax.plot(df_sorted["distance"], rolling, color="red", label="rolling mean (window=20)")
ax.set_xlabel("Token distance |i - j|")
ax.set_ylabel("Cosine similarity")
ax.set_title("Within-sequence token similarity vs distance")
ax.legend()
ax.grid(alpha=0.3)
fig.show()

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** Pairwise cosine similarity for all token pairs in
#    a single sequence, grouped by token distance.
# 2. **What did I find?** ___
# 3. **What surprised me?** ___
# 4. **What's next?** Compare this curve across embedder families
#    (`03-cross-model-comparison.py`).
