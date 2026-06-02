# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 03 — Capacity scaling
#
# **Question.** How do HippoRAG and A-Mem degrade as the memory corpus grows
# from 50 to 5000 entries? Where does each system break first — quality,
# latency, or cost?
#
# **Why it matters.** Reconsolidation and forgetting both become *more*
# important as capacity grows. Knowing where each system breaks tells you
# which problem to attack first.

# %%
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))

import pandas as pd
from experiments.shared.dataset_loader import load_comparison_dataset
from experiments.shared.system_loaders import HippoRAGSystem, AMemSystem

# %% [markdown]
# ## Generate synthetic corpora at three scales
#
# Use the comparison-dataset memories as the seed, then synthetically replicate
# / augment to hit 500 and 5000.

# %%
seed = load_comparison_dataset()["memories"]

def make_corpus(target_size: int, seed_memories: list) -> list:
    """Duplicate and slightly vary seed memories until reaching target_size.

    Real experiment should use diverse synthetic memories, not just duplicates.
    """
    # TODO: replace with realistic memory generation (e.g., LLM-generated
    # variants, or use Memora's synthetic generator).
    out = []
    i = 0
    while len(out) < target_size:
        base = seed_memories[i % len(seed_memories)]
        out.append({**base, "id": f"{base['id']}-rep{len(out)}"})
        i += 1
    return out[:target_size]

corpora = {
    50: make_corpus(50, seed),
    500: make_corpus(500, seed),
    5000: make_corpus(5000, seed),
}

# %% [markdown]
# ## Test queries
#
# Use 5 questions from the comparison dataset across categories.

# %%
test_questions = load_comparison_dataset()["questions"][:5]

# %% [markdown]
# ## Run each system at each scale

# %%
results = []
for size, corpus in corpora.items():
    for system_name, system_cls, system_kwargs in [
        ("HippoRAG", HippoRAGSystem, dict(llm_model="meta/llama-3.3-70b-instruct",
                                          embedder="nvidia/nv-embed-v2")),
        ("A-Mem", AMemSystem, dict(llm_model="meta/llama-3.3-70b-instruct",
                                   embedder="all-MiniLM-L6-v2")),
    ]:
        print(f"\n=== {system_name} @ {size} memories ===")

        # Time indexing
        t0 = time.time()
        system = system_cls(**system_kwargs)
        system.ingest(corpus)
        index_time = time.time() - t0
        print(f"  Index time: {index_time:.1f}s")

        # Time queries
        for q in test_questions:
            t0 = time.time()
            resp = system.query(q["question"])
            query_time = time.time() - t0
            results.append({
                "system": system_name,
                "corpus_size": size,
                "question_id": q["id"],
                "index_time_s": round(index_time, 2),
                "query_time_s": round(query_time, 3),
                "answer": resp["answer"][:80],
            })

df = pd.DataFrame(results)
df

# %% [markdown]
# ## Scaling table

# %%
summary = df.groupby(["system", "corpus_size"]).agg(
    avg_index_time=("index_time_s", "first"),
    avg_query_time=("query_time_s", "mean"),
).round(2)
summary

# %% [markdown]
# ## Plot

# %%
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for system_name, group in summary.reset_index().groupby("system"):
    ax1.plot(group["corpus_size"], group["avg_index_time"], marker="o", label=system_name)
    ax2.plot(group["corpus_size"], group["avg_query_time"], marker="o", label=system_name)
ax1.set_xlabel("Corpus size"); ax1.set_ylabel("Indexing time (s)")
ax1.set_xscale("log"); ax1.set_yscale("log"); ax1.legend(); ax1.set_title("Indexing")
ax2.set_xlabel("Corpus size"); ax2.set_ylabel("Avg query time (s)")
ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.legend(); ax2.set_title("Querying")
fig.tight_layout()
fig.show()

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** Indexing time and per-query time at 3 corpus
#    scales for both systems.
# 2. **What did I find?** ___
# 3. **What surprised me?** ___
# 4. **What's next?** Add a quality metric — does retrieval correctness degrade
#    with scale?
