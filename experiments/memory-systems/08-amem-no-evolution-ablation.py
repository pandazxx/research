# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 08 — A-Mem without memory evolution
#
# **Question.** What if we disable A-Mem's memory evolution step entirely?
# Does the system still work? Where exactly does evolution earn its keep?
#
# **Why it matters.** Memory evolution is the closest published mechanism to
# write-time reconsolidation. Knowing exactly how much value it provides
# tells you whether read-time reconsolidation (the project's likely
# contribution) would be a meaningful improvement or an incremental tweak.
#
# **References.** A-Mem paper §3.3 (Memory Evolution) and §4.4 (Ablation).

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))

from experiments.shared.system_loaders import AMemSystem
from experiments.shared.dataset_loader import load_comparison_dataset

# %% [markdown]
# ## Setup

# %%
dataset = load_comparison_dataset()

# Standard A-Mem (with evolution)
amem_full = AMemSystem(
    llm_model="meta/llama-3.3-70b-instruct",
    embedder="all-MiniLM-L6-v2",
)
amem_full.ingest(dataset["memories"])

# A-Mem without memory evolution — requires modifying the reproduction
# to either skip the evolution step in note_added(), or set a flag.
# TODO: implement the no-evolution variant. Pseudo-code:
#
# class AMemNoEvolution(AMemSystem):
#     def _add_note(self, note):
#         # 1. Build note (LLM generates K, G, X)
#         # 2. Generate links to existing notes
#         # 3. SKIP: don't trigger memory evolution on linked neighbors
#         pass
#
# raise NotImplementedError

# %% [markdown]
# ## Focused stress: information_update questions

# %%
import pandas as pd
update_questions = [q for q in dataset["questions"] if q["category"] == "information_update"]
print(f"Update-category questions: {len(update_questions)}")
for q in update_questions:
    print(f"  - {q['id']}: {q['question']}")

# %% [markdown]
# ## Run both variants

# %%
results = []
for q in dataset["questions"]:
    full_resp = amem_full.query(q["question"])
    # noevo_resp = amem_no_evolution.query(q["question"])
    results.append({
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "expected": q["expected_answer"],
        "full": full_resp["answer"],
        # "no_evolution": noevo_resp["answer"],
    })

df = pd.DataFrame(results)

# %% [markdown]
# ## Score

# %%
def is_correct(predicted, expected, category):
    pred_norm = predicted.lower()
    if category == "absence_abstention":
        return any(p in pred_norm for p in ("don't know", "not mentioned", "unknown"))
    return any(t.lower() in pred_norm for t in expected.split() if len(t) > 3)

df["full_correct"] = df.apply(
    lambda r: is_correct(r["full"], r["expected"], r["category"]), axis=1
)
# df["no_evolution_correct"] = df.apply(
#     lambda r: is_correct(r["no_evolution"], r["expected"], r["category"]), axis=1
# )

per_cat = df.groupby("category")[
    ["full_correct"]  # add "no_evolution_correct" once implemented
].mean().round(3)
per_cat

# %% [markdown]
# ## Expected pattern
#
# - **information_update**: evolution should help substantially here. This is
#   where its value is concentrated.
# - **single_hop, two_hop**: evolution should be neutral. No reason to update
#   memories that are direct facts.
# - **deep_multi_hop**: ambiguous. Evolution might help by updating linked
#   notes with cross-cutting information, or might hurt by overwriting
#   intermediate steps in chains.
#
# A-Mem's own paper ablation (Table 3) shows:
# - Without link generation OR memory evolution: 9.65 multi-hop F1
# - Without memory evolution (links only): 21.35 multi-hop F1
# - Full A-Mem: 27.02 multi-hop F1
#
# So evolution adds ~6 F1 points on multi-hop in their setup. Whether your
# reproduction shows similar numbers is a basic sanity check on the
# reproduction quality.

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** Per-category quality difference between full
#    A-Mem and a no-evolution variant.
# 2. **What did I find?** ___
# 3. **What surprised me?** ___
# 4. **What's next?** If evolution adds value mostly on update questions
#    *but not all of them*, that's the gap a read-time reconsolidation
#    mechanism could fill.
