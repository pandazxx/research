# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 03 — Cross-model comparison of anisotropy
#
# **Question.** How does anisotropy differ between BERT-family encoders and
# decoder-LLM-based embedders (E5-Mistral, NV-Embed)?
#
# **Why it matters.** Some recent embedders (E5-Mistral, NV-Embed-v2,
# LLM2Vec) are explicitly trained with isotropy-promoting objectives. This
# experiment quantifies how much that training actually helps.
#
# **References.** `embeddings-101.md` §11; `embedding-applications-survey.md` §1.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))

import numpy as np
import pandas as pd
from experiments.shared.embedding_utils import cosine, get_token_vectors

# %%
MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",   # small classic
    "BAAI/bge-large-en-v1.5",                    # current popular default
    "intfloat/e5-mistral-7b-instruct",           # LLM-based — should be less anisotropic
    "jinaai/jina-embeddings-v3",                 # current state of the art
]

text = (
    "Joey is living in Singapore. She is a software engineer. "
    "My wife is living in Singapore too. She is a house wife."
)

# %% [markdown]
# ## Measure anisotropy per model
#
# Compute: average within-sequence cosine between all distinct-token pairs.
# Higher = more anisotropic.

# %%
results = []
for model_name in MODELS:
    try:
        tokens, vectors, mask = get_token_vectors(text, model_name=model_name)
    except Exception as e:
        print(f"Skipping {model_name}: {e}")
        continue

    n = sum(int(m) for m in mask)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            if mask[i] == 0 or mask[j] == 0:
                continue
            sims.append(cosine(vectors[i], vectors[j]))
    sims = np.array(sims)
    results.append({
        "model": model_name,
        "mean_cosine": sims.mean(),
        "min_cosine": sims.min(),
        "max_cosine": sims.max(),
        "n_pairs": len(sims),
    })

df = pd.DataFrame(results).round(3)
df

# %% [markdown]
# ## Interpretation
#
# - mean_cosine close to 0.85+ → highly anisotropic (BERT-family typical)
# - mean_cosine in the 0.6–0.7 range → moderately isotropic (LLM-based embedders trained with anti-anisotropy)
# - mean_cosine near 0 → ideal isotropy (no current model achieves this)

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** Average within-sequence pairwise cosine across 4
#    embedder families on a 24-token sequence.
# 2. **What did I find?** ___
# 3. **What surprised me?** ___
# 4. **What's next?** If a model has substantially lower anisotropy, test
#    whether retrieval quality (on the comparison dataset) is also higher.
