# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 04 — Long-context baseline
#
# **Question.** What if we just stuff every memory into a single long prompt
# and ask the LLM? How does that compare to HippoRAG and A-Mem?
#
# **Why it matters.** This is the **must-include** baseline that nearly every
# memory paper skips. If a 1M-context model can answer the questions correctly
# by just reading everything, the memory system needs to demonstrate value
# beyond simple context-stuffing.
#
# **References.** `extended-reading-and-experiments.md` Experiment 6.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))

from experiments.shared.dataset_loader import load_comparison_dataset
from experiments.shared.system_loaders import LongContextBaseline

# %% [markdown]
# ## Setup

# %%
dataset = load_comparison_dataset()
memories = dataset["memories"]
questions = dataset["questions"]

# Use a long-context model. Claude Sonnet 4.6+ and GPT-4.1+ both handle 1M
# tokens. The comparison-dataset is small (~3000 tokens), so any modern
# model handles it.
baseline = LongContextBaseline(llm_model="claude-sonnet-4-6")
baseline.ingest(memories)

# %% [markdown]
# ## Run all questions

# %%
import pandas as pd
results = []
for q in questions:
    resp = baseline.query(q["question"])
    results.append({
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "expected": q["expected_answer"],
        "baseline_answer": resp["answer"],
    })

df = pd.DataFrame(results)
df

# %% [markdown]
# ## Score
#
# Use the same scoring as `01-run-comparison-dataset.py`.

# %%
def is_correct(predicted: str, expected: str, category: str) -> bool:
    pred_norm = predicted.lower().strip()
    if category == "absence_abstention":
        return any(p in pred_norm for p in ("don't know", "not mentioned", "unknown"))
    return any(t.lower() in pred_norm for t in expected.split() if len(t) > 3)

df["correct"] = df.apply(
    lambda r: is_correct(r["baseline_answer"], r["expected"], r["category"]), axis=1
)
df.groupby("category")["correct"].agg(["sum", "count"])

# %% [markdown]
# ## Comparison vs memory systems
#
# Pull in the results from `01-run-comparison-dataset.py` (assumes you've
# run it and saved results), and produce a 3-system comparison.

# %%
# TODO: load HippoRAG and A-Mem results from a saved file
# combined = ...
# for category in ['single_hop', 'two_hop', ...]:
#     show side-by-side accuracy for [Baseline, HippoRAG, A-Mem]

# %% [markdown]
# ## Interpretation
#
# Four possible outcomes:
#
# 1. **Long-context wins everywhere.** Your memory systems need substantial
#    work to justify their complexity. Probably means the comparison dataset
#    is too small.
# 2. **Long-context ties on simple, loses on multi-hop and update.** The
#    expected outcome for a well-designed memory benchmark.
# 3. **Long-context loses everywhere.** Possible but suspicious — likely a
#    long-context model behaving badly at long context (the "lost in the
#    middle" problem).
# 4. **Mixed.** Most likely. Memory systems should outperform on update and
#    deep-multi-hop questions specifically.

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** Long-context baseline performance on the
#    comparison-dataset questions.
# 2. **What did I find?** ___
# 3. **What surprised me?** ___
# 4. **What's next?** If the baseline is competitive, scale the dataset up
#    (more memories, longer total context) to find where memory systems
#    start to actually win.
