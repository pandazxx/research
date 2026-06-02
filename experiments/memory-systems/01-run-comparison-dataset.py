# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 01 — Run HippoRAG vs A-Mem on the comparison dataset
#
# **Question.** Do HippoRAG 2 and A-Mem actually behave the way the
# architectural hypotheses predict on the 7-category comparison dataset?
#
# **Why it matters.** This is the headline diagnostic experiment. The results
# directly inform the direction decision (reconsolidation vs forgetting) and
# reveal where each system's strengths and weaknesses actually live.
#
# **References.** `topics/llm-agent-memory/comparison-dataset/` for the dataset
# itself and the predicted outcomes.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))

import pandas as pd
from collections import defaultdict
from experiments.shared.dataset_loader import load_comparison_dataset
from experiments.shared.system_loaders import HippoRAGSystem, AMemSystem

# %% [markdown]
# ## Load dataset

# %%
dataset = load_comparison_dataset()
memories = dataset["memories"]
questions = dataset["questions"]
print(f"Memories:  {len(memories)}")
print(f"Questions: {len(questions)}")
print(f"Categories: {set(q['category'] for q in questions)}")

# %% [markdown]
# ## Initialise both systems
#
# NOTE: This requires the HippoRAG and A-Mem reproductions to be set up.
# See `experiments/shared/system_loaders.py` for path configuration.

# %%
hipporag = HippoRAGSystem(
    llm_model="meta/llama-3.3-70b-instruct",
    embedder="nvidia/nv-embed-v2",
    branch="legacy",  # v1 — switch to "main" for v2
)
amem = AMemSystem(
    llm_model="meta/llama-3.3-70b-instruct",
    embedder="all-MiniLM-L6-v2",  # A-Mem's default
)

# %% [markdown]
# ## Ingest memories

# %%
hipporag.ingest(memories)
amem.ingest(memories)

# %% [markdown]
# ## Run each question through both systems

# %%
results = []
for q in questions:
    hr_resp = hipporag.query(q["question"])
    am_resp = amem.query(q["question"])
    results.append({
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "expected": q["expected_answer"],
        "expected_winner": q["expected_winner"],
        "hipporag_answer": hr_resp["answer"],
        "amem_answer": am_resp["answer"],
        "hipporag_retrieved": hr_resp.get("retrieved_ids", []),
        "amem_retrieved": am_resp.get("retrieved_ids", []),
    })

df = pd.DataFrame(results)
df

# %% [markdown]
# ## Score answers
#
# Use substring match as a cheap proxy for correctness. For absence questions,
# accept any abstention phrase ("don't know", "not mentioned", "unknown").

# %%
def is_abstention(text: str) -> bool:
    text = text.lower()
    return any(phrase in text for phrase in (
        "don't know", "do not know", "not mentioned", "no information",
        "cannot determine", "unknown", "unclear", "not specified",
    ))

def is_correct(predicted: str, expected: str, category: str) -> bool:
    pred_norm = predicted.lower().strip()
    if category == "absence_abstention":
        return is_abstention(pred_norm)
    return any(
        token.lower() in pred_norm
        for token in expected.split() if len(token) > 3
    )

df["hipporag_correct"] = df.apply(
    lambda r: is_correct(r["hipporag_answer"], r["expected"], r["category"]), axis=1
)
df["amem_correct"] = df.apply(
    lambda r: is_correct(r["amem_answer"], r["expected"], r["category"]), axis=1
)

# %% [markdown]
# ## Per-category breakdown

# %%
summary = df.groupby(["category", "expected_winner"]).agg(
    n=("id", "count"),
    hipporag_correct=("hipporag_correct", "sum"),
    amem_correct=("amem_correct", "sum"),
).reset_index()
summary

# %% [markdown]
# ## Compare to predictions
#
# The comparison dataset's `analysis.md` predicts:
#
# - single_hop, two_hop, absence_abstention → tie
# - deep_multi_hop, compositional_aggregation → HippoRAG wins
# - implicit_conceptual, information_update → A-Mem wins
#
# Tick or cross each prediction based on observed results.

# %%
predictions = {
    "single_hop": "tie",
    "two_hop": "tie",
    "deep_multi_hop": "HippoRAG",
    "implicit_conceptual": "A-Mem",
    "information_update": "A-Mem",
    "compositional_aggregation": "HippoRAG",
    "absence_abstention": "tie",
}

for cat, pred in predictions.items():
    cat_rows = df[df["category"] == cat]
    hr_pct = cat_rows["hipporag_correct"].mean() * 100
    am_pct = cat_rows["amem_correct"].mean() * 100
    if hr_pct > am_pct + 10:
        actual = "HippoRAG"
    elif am_pct > hr_pct + 10:
        actual = "A-Mem"
    else:
        actual = "tie"
    confirmed = "✓" if actual == pred else "✗"
    print(f"{cat:>30}: predicted {pred:<10} actual {actual:<10} "
          f"HR={hr_pct:.0f}% AM={am_pct:.0f}% {confirmed}")

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** 22 questions × 2 systems = 44 evaluations,
#    bucketed by category.
# 2. **What did I find?** ___
# 3. **What surprised me?** Look for predictions that DIDN'T hold — those are
#    the most informative cases.
# 4. **What's next?** Investigate the surprising disagreements case-by-case
#    (`02-contradiction-stress-test.py` for the information-update questions).
