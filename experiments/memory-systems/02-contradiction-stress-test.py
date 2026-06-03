# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 02 — Contradiction stress test
#
# **Question.** When a later memory contradicts an earlier one, do HippoRAG
# and A-Mem reliably prefer the latest? Or do they retain both, or pick
# arbitrarily?
#
# **Why it matters.** Contradiction handling is the most direct test of
# reconsolidation. A-Mem's memory evolution mechanism should help; HippoRAG's
# static graph likely fails. The size of the failure quantifies the
# reconsolidation-shaped gap in current systems.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))

import random
from experiments.shared.system_loaders import HippoRAGSystem, AMemSystem

# %% [markdown]
# ## Construct contradiction sequences
#
# Each sequence: 5 base memories, 1 contradicting memory injected at the end,
# and 1 query that should return the new fact.

# %%
contradiction_pairs = [
    {
        "name": "city_change",
        "base": [
            "Alice lives in Berlin.",
            "Alice rides her bicycle to work every day.",
            "Alice's favorite coffee shop is on Friedrichstrasse.",
            "Alice has lived in Berlin for five years.",
        ],
        "update": "Alice moved to Munich last month.",
        "query": "Where does Alice currently live?",
        "expected": "Munich",
    },
    {
        "name": "job_change",
        "base": [
            "Bob works as a data scientist at AcmeCo.",
            "Bob's team uses PyTorch for all their ML work.",
            "Bob's manager at AcmeCo is named Sarah.",
        ],
        "update": "Bob accepted a new role at NovaTech yesterday.",
        "query": "Where does Bob currently work?",
        "expected": "NovaTech",
    },
    {
        "name": "relationship_change",
        "base": [
            "Carol is married to Dave.",
            "Carol and Dave have been together for ten years.",
            "Carol and Dave honeymooned in Iceland.",
        ],
        "update": "Carol and Dave got divorced six months ago.",
        "query": "Is Carol still married to Dave?",
        "expected": "no/divorced",
    },
    # Add more cases as needed
]

# %% [markdown]
# ## Run the stress test

# %%
def run_contradiction_test(system, pair):
    """Ingest base + update memories, then query."""
    memories = [
        {"id": f"{pair['name']}-base-{i}", "content": c, "timestamp": f"2024-01-{i+1:02d}"}
        for i, c in enumerate(pair["base"])
    ]
    memories.append({
        "id": f"{pair['name']}-update",
        "content": pair["update"],
        "timestamp": "2024-02-15",
    })
    system.ingest(memories)
    response = system.query(pair["query"])
    return response

# %% [markdown]
# Initialise systems and run

# %%
hipporag = HippoRAGSystem(
    llm_model="meta/llama-3.3-70b-instruct",
    embedder="nvidia/nv-embed-v2",
)
amem = AMemSystem(
    llm_model="meta/llama-3.3-70b-instruct",
    embedder="all-MiniLM-L6-v2",
)

results = []
for pair in contradiction_pairs:
    print(f"\n=== {pair['name']} ===")
    print(f"Query: {pair['query']}")
    print(f"Expected new answer: {pair['expected']}")

    hr_resp = run_contradiction_test(hipporag, pair)
    am_resp = run_contradiction_test(amem, pair)

    print(f"\nHippoRAG: {hr_resp['answer']}")
    print(f"A-Mem:    {am_resp['answer']}")

    results.append({
        "case": pair["name"],
        "query": pair["query"],
        "expected": pair["expected"],
        "hipporag": hr_resp["answer"],
        "amem": am_resp["answer"],
    })

# %% [markdown]
# ## Classify each answer

# %%
def classify(answer, base_answer, expected):
    """Roughly: did the system prefer new, old, both, or neither?"""
    a = answer.lower()
    if expected.lower() in a and base_answer.lower() not in a:
        return "new_only (correct)"
    if base_answer.lower() in a and expected.lower() not in a:
        return "old_only (wrong)"
    if expected.lower() in a and base_answer.lower() in a:
        return "both (partial)"
    return "neither (confabulation or abstention)"

# Map cases to their base answers manually for now
base_answers = {
    "city_change": "Berlin",
    "job_change": "AcmeCo",
    "relationship_change": "married",
}

import pandas as pd
df = pd.DataFrame(results)
df["hipporag_class"] = df.apply(
    lambda r: classify(r["hipporag"], base_answers[r["case"]], r["expected"]), axis=1
)
df["amem_class"] = df.apply(
    lambda r: classify(r["amem"], base_answers[r["case"]], r["expected"]), axis=1
)
df[["case", "hipporag_class", "amem_class"]]

# %% [markdown]
# ## Interpretation
#
# Predicted pattern:
# - HippoRAG: mostly `both` or `old_only` — graph holds both facts
# - A-Mem: mix of `new_only` (when memory evolution fires) and `both` (when
#   it doesn't). The proportion of `new_only` is a measure of evolution's
#   reliability.
#
# Either result is informative for the project:
# - If A-Mem mostly produces `new_only`: write-time reconsolidation works
#   reasonably well; the project's contribution would be in extending it to
#   read-time.
# - If A-Mem mostly produces `both`: write-time evolution is unreliable; the
#   project's contribution could be in *fixing* the evolution mechanism.

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** Contradiction handling on 3 (or more) base+update
#    sequences for both systems.
# 2. **What did I find?** ___
# 3. **What surprised me?** ___
# 4. **What's next?** Vary the time between base and update memories. Does
#    A-Mem evolution still fire if the contradicting memory comes 100 turns
#    later instead of immediately after?
