# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 05 — HippoRAG embedder ablation
#
# **Question.** How much does HippoRAG's retrieval quality depend on the
# choice of embedding model?
#
# **Why it matters.** HippoRAG's synonymy edges and PPR seeding both depend
# heavily on embedding similarity. If a weaker embedder degrades quality
# substantially, the system has a "static embedding bottleneck" that's worth
# flagging as a project gap.
#
# **References.** `hipporag2-study-notes.md` §Why incremental updates are hard.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))

from experiments.shared.system_loaders import HippoRAGSystem
from experiments.shared.dataset_loader import load_comparison_dataset

# %% [markdown]
# ## Setup

# %%
dataset = load_comparison_dataset()

EMBEDDERS = [
    "sentence-transformers/all-MiniLM-L6-v2",   # 384 dim, weakest baseline
    "facebook/contriever",                       # HippoRAG v1 default
    "BAAI/bge-large-en-v1.5",                    # strong free baseline
    "nvidia/nv-embed-v2",                        # HippoRAG v2 default
]

# %% [markdown]
# ## Run HippoRAG with each embedder

# %%
import pandas as pd
results = []
for embedder in EMBEDDERS:
    print(f"\n=== Embedder: {embedder} ===")
    system = HippoRAGSystem(
        llm_model="meta/llama-3.3-70b-instruct",
        embedder=embedder,
        branch="main",  # HippoRAG 2
    )
    system.ingest(dataset["memories"])

    for q in dataset["questions"]:
        resp = system.query(q["question"])
        results.append({
            "embedder": embedder,
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answer": resp["answer"],
            "expected": q["expected_answer"],
        })

df = pd.DataFrame(results)

# %% [markdown]
# ## Score and aggregate

# %%
def is_correct(predicted: str, expected: str, category: str) -> bool:
    pred_norm = predicted.lower()
    if category == "absence_abstention":
        return any(p in pred_norm for p in ("don't know", "not mentioned", "unknown"))
    return any(t.lower() in pred_norm for t in expected.split() if len(t) > 3)

df["correct"] = df.apply(
    lambda r: is_correct(r["answer"], r["expected"], r["category"]), axis=1
)
summary = df.groupby("embedder")["correct"].agg(["sum", "count", "mean"]).round(3)
summary

# %% [markdown]
# ## Per-category breakdown

# %%
df.groupby(["embedder", "category"])["correct"].mean().unstack().round(2)

# %% [markdown]
# ## Interpretation
#
# Expected patterns:
#
# - **Stronger embedders should win** on the implicit-conceptual category
#   (where the embedder's understanding of semantic relationships matters).
# - **All embedders should perform similarly** on single-hop (the answer is
#   direct).
# - **Deep multi-hop** might be embedder-insensitive because PPR does the
#   heavy lifting — but only if synonymy edges (which DO depend on embedder)
#   are correctly identifying the right entities.
#
# If a substantially weaker embedder (`all-MiniLM-L6-v2`) gets within 5% of
# `nv-embed-v2`, then the embedder choice doesn't matter much. If it's 20%
# worse, the static-embedding bottleneck is real and quantified.

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** HippoRAG retrieval quality across 4 embedding
#    models on the comparison dataset.
# 2. **What did I find?** ___
# 3. **What surprised me?** ___
# 4. **What's next?** If the embedder matters a lot, try running with
#    isotropy-promoting embedders (E5-Mistral) and see if they help.
