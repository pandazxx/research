# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 06 — A-Mem prompt sensitivity
#
# **Question.** How does A-Mem's memory evolution behaviour change with
# different prompt instructions for the evolution step?
#
# **Why it matters.** A-Mem's memory evolution is the closest published thing
# to write-time reconsolidation. Its behaviour is entirely controlled by a
# prompt (P_s3 in the paper). Probing the prompt's sensitivity reveals how
# *reliable* the evolution mechanism actually is.
#
# **References.** A-Mem paper §3.3 Memory Evolution.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))

from experiments.shared.system_loaders import AMemSystem

# %% [markdown]
# ## Prompt variants
#
# Three variants of A-Mem's evolution prompt, each biased differently.

# %%
PROMPTS = {
    "default": """\
Given the new memory and its linked neighbors, decide whether to update
any neighbor's context, keywords, or tags based on the new information.
Return the updated note as JSON.
""",
    "aggressive": """\
Given the new memory, AGGRESSIVELY update any linked neighbors whose
content is now outdated, contradicted, or incomplete. Replace stale
information with the new facts. Return updated notes as JSON.
""",
    "conservative": """\
Given the new memory, CONSERVATIVELY consider whether linked neighbors
need updating. Preserve original memories when possible. Only update if
there is direct, unambiguous contradiction. Return any updated notes as JSON.
""",
    "contradiction-focused": """\
Given the new memory and its linked neighbors, LOOK FOR DIRECT
CONTRADICTIONS between the new memory and each neighbor. When a
contradiction exists, update the older memory to align with the new fact.
For non-contradicting neighbors, do not update. Return updated notes as JSON.
""",
}

# %% [markdown]
# ## Test scenario: a user transition with multiple linked facts

# %%
SCENARIO_MEMORIES = [
    {"id": "m1", "content": "Alice lives in Berlin.", "timestamp": "2024-01-01"},
    {"id": "m2", "content": "Alice's daily commute is via the Berlin U-Bahn.", "timestamp": "2024-01-02"},
    {"id": "m3", "content": "Alice's favorite cafe is on Friedrichstrasse in Berlin.", "timestamp": "2024-01-03"},
    {"id": "m4", "content": "Alice has lived in Berlin for five years.", "timestamp": "2024-01-04"},
    {"id": "m5", "content": "Alice moved to Munich last week.", "timestamp": "2024-02-15"},
]

QUERIES = [
    "Where does Alice currently live?",
    "What city is Alice's favorite cafe in?",
    "How does Alice commute?",
    "How many years has Alice lived in her current city?",
]

# %% [markdown]
# ## Run A-Mem with each prompt variant
#
# Requires modification to A-Mem's reproduction to accept a custom evolution
# prompt at construction time. If your reproduction doesn't expose this,
# patch the prompt directly in the source for each run.

# %%
results = []
for prompt_name, prompt_text in PROMPTS.items():
    print(f"\n=== Prompt: {prompt_name} ===")

    # TODO: pass the prompt to AMemSystem. May require extending the loader.
    system = AMemSystem(
        llm_model="meta/llama-3.3-70b-instruct",
        embedder="all-MiniLM-L6-v2",
        # evolution_prompt=prompt_text,
    )
    system.ingest(SCENARIO_MEMORIES)

    for q in QUERIES:
        resp = system.query(q)
        print(f"  Q: {q}")
        print(f"     {resp['answer']}")
        results.append({
            "prompt": prompt_name,
            "question": q,
            "answer": resp["answer"],
        })

import pandas as pd
df = pd.DataFrame(results)

# %% [markdown]
# ## Classify the answer behaviour
#
# Did the system prefer the new fact (Munich) or stick with the old (Berlin)?

# %%
def prefers_new(answer: str, new: str = "Munich", old: str = "Berlin") -> str:
    a = answer.lower()
    new_present = new.lower() in a
    old_present = old.lower() in a
    if new_present and not old_present:
        return "new_only"
    if old_present and not new_present:
        return "old_only"
    if new_present and old_present:
        return "both"
    return "neither"

df["preference"] = df["answer"].apply(prefers_new)
df.groupby(["prompt", "preference"]).size().unstack(fill_value=0)

# %% [markdown]
# ## Interpretation
#
# - If `default` and `aggressive` both produce mostly `new_only`: A-Mem's
#   evolution is robust to prompt variation. Good news for reproducibility.
# - If `conservative` mostly produces `old_only` or `both`: evolution is
#   highly prompt-sensitive — a finding worth highlighting.
# - If even `aggressive` fails to update: the evolution mechanism has a deeper
#   problem than prompt wording.

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** A-Mem's contradiction-handling behaviour across
#    4 evolution-prompt variants.
# 2. **What did I find?** ___
# 3. **What surprised me?** ___
# 4. **What's next?** Apply automated prompt optimisation (DSPy MIPROv2 or
#    GEPA) to the evolution prompt and see if the system can find a prompt
#    that maximises correct update behaviour.
