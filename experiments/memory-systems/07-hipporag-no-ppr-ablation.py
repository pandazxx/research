# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 07 — HippoRAG without PPR
#
# **Question.** What if we keep HippoRAG's graph structure (entity nodes,
# passage nodes, all edges) but skip the Personalized PageRank step, instead
# doing simple top-k embedding retrieval against the passage nodes?
#
# **Why it matters.** PPR is HippoRAG's signature mechanism. If removing it
# doesn't hurt much, then the graph is doing the work, not PPR. If removing
# it hurts a lot — especially on multi-hop questions — that's direct evidence
# that PPR is earning its keep.
#
# **References.** `hipporag-reproduction/docs/paper-notes.md` §PPR.

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

# Standard HippoRAG (with PPR)
hipporag_full = HippoRAGSystem(
    llm_model="meta/llama-3.3-70b-instruct",
    embedder="nvidia/nv-embed-v2",
    branch="main",
)
hipporag_full.ingest(dataset["memories"])

# HippoRAG without PPR — requires modifying the reproduction to expose a
# "skip PPR" flag, or running the retrieval pipeline manually using only the
# passage embeddings.
# TODO: implement the no-PPR variant. Pseudo-code:
#
# class HippoRAGNoPPR(HippoRAGSystem):
#     def query(self, q):
#         # 1. Query-to-triple matching (same as v2)
#         # 2. Triple filtering (same as v2)
#         # 3. SKIP: no PPR — just retrieve top-k passages directly via
#         #    embedding similarity to the query
#         # 4. Pass top-k passages to QA reader
#
# raise NotImplementedError

# %% [markdown]
# ## Per-category comparison

# %%
import pandas as pd
results = []
for q in dataset["questions"]:
    full_resp = hipporag_full.query(q["question"])
    # noppr_resp = hipporag_no_ppr.query(q["question"])
    results.append({
        "id": q["id"],
        "category": q["category"],
        "expected": q["expected_answer"],
        "full": full_resp["answer"],
        # "no_ppr": noppr_resp["answer"],
    })

df = pd.DataFrame(results)

# %% [markdown]
# ## Score and aggregate by category

# %%
def is_correct(predicted, expected, category):
    pred_norm = predicted.lower()
    if category == "absence_abstention":
        return any(p in pred_norm for p in ("don't know", "not mentioned", "unknown"))
    return any(t.lower() in pred_norm for t in expected.split() if len(t) > 3)

df["full_correct"] = df.apply(
    lambda r: is_correct(r["full"], r["expected"], r["category"]), axis=1
)
# df["no_ppr_correct"] = df.apply(
#     lambda r: is_correct(r["no_ppr"], r["expected"], r["category"]), axis=1
# )

# %% [markdown]
# ## Expected pattern
#
# PPR's contribution should be:
#
# - **Large on deep_multi_hop** — PPR propagation through chains is its core
#   value. No-PPR should drop notably here.
# - **Small on single_hop / two_hop** — the answer is one passage away;
#   embedding retrieval alone is sufficient.
# - **Moderate on compositional_aggregation** — PPR aggregates across many
#   linked entities, which embedding retrieval can't replicate.
#
# If the drop is small everywhere, PPR is over-engineering. If it's large only
# on deep_multi_hop, PPR is doing exactly what the paper claims.

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** Quality difference between full HippoRAG and a
#    no-PPR variant on the comparison dataset.
# 2. **What did I find?** ___
# 3. **What surprised me?** ___
# 4. **What's next?** If PPR doesn't pull its weight, the project's
#    foundation might be A-Mem (simpler architecture) rather than HippoRAG.
