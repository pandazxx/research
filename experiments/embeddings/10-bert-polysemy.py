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
# # 10 — BERT-family contextual embeddings: polysemy & cross-context
#
# **Question.** How do **three BERT-family encoders** contextualize the same
# surface word ("apple") across three regimes?
#
# Encoders compared (all BERT architecture, all WordPiece tokenization):
#
# | Model | Params | What's different |
# |---|---|---|
# | `bert-base-uncased`      | 110M | Vanilla BERT, MLM pre-training only |
# | `bert-large-uncased`     | 340M | Same, larger — does scale alone help? |
# | `BAAI/bge-large-en-v1.5` | 335M | BERT-large architecture, **embedding-task fine-tuned**; the kind of model you'd actually use for RAG today |
#
# Three regimes:
# 1. **Polysemy** — same word, different senses, separate sentences.
# 2. **RAG alignment** — long passage vs short query, same sense.
# 3. **Intra-sentence disambiguation** — two senses, *same* sentence.
#
# **Why it matters.** With three encoders side by side we can separate two
# usually-confounded effects: *scale* (base → large) and *fine-tuning*
# (large MLM → large embedding-tuned). RAG systems get both for free when
# they swap a vanilla BERT for BGE; this notebook quantifies what each
# half of that swap actually buys you at the token level.
#
# **First-run note.** Loading all three models downloads ~1 GB from
# HuggingFace. `embedding_utils.load_hf_model` LRU-caches them within the
# kernel session, so each model loads once across all three parts.
#
# **Reference.** `topics/llm-agent-memory/embeddings-101.md` §8 (contextual
# embeddings); companion to `01-anisotropy.py` and `09-anisotropy-context-split.py`.

# %%
import functools
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parents[1]))

import numpy as np
import pandas as pd

from experiments.shared.embedding_utils import (
    cosine,
    find_first_token_position,
    get_token_vectors,
)

# Models to compare, plus short display names for tidy tables.
MODELS: list[str] = [
    "bert-base-uncased",
    "bert-large-uncased",
    "BAAI/bge-large-en-v1.5",
]
SHORT_NAME: dict[str, str] = {
    "bert-base-uncased":      "bert-base",
    "bert-large-uncased":     "bert-large",
    "BAAI/bge-large-en-v1.5": "bge-large",
}

TARGET_WORD = "apple"

# %% [markdown]
# ## Test data (shared across all three parts and all three models)

# %%
sentences_by_sense: dict[str, list[str]] = {
    "fruit": [
        "An apple a day keeps the doctor away.",
        "She picked a ripe apple from the tree behind the house.",
    ],
    "idiom": [
        "His granddaughter is the apple of his eye.",
        "She has always been the apple of her father's eye.",
    ],
    "company": [
        "Apple released a new iPhone last September.",
        "Apple acquired a small AI startup last quarter.",
    ],
}

long_passage = (
    "Apple Inc. was founded on April 1, 1976 by Steve Jobs, Steve Wozniak, "
    "and Ronald Wayne in a garage in Los Altos, California. The company "
    "initially sold the Apple I personal computer kit, which Wozniak had "
    "designed. Within a decade Apple had become one of the most influential "
    "consumer technology companies in the world, releasing the Macintosh in 1984."
)
short_query = "Who founded Apple?"

two_sense_sentence = "Steve Jobs founded Apple in 1976. He often ate an apple at his desk."

# %% [markdown]
# ## Helpers
#
# `collect_apple_vectors` runs the polysemy forward passes once per model
# and LRU-caches the result, so Parts 1 and 3 don't redo the same work.

# %%
@functools.lru_cache(maxsize=8)
def collect_apple_vectors(model_name: str) -> dict[str, tuple[np.ndarray, ...]]:
    """Pull the 'apple' vector from each polysemy sentence for one model."""
    out: dict[str, tuple[np.ndarray, ...]] = {}
    for sense, sentences in sentences_by_sense.items():
        vecs_for_sense = []
        for sent in sentences:
            toks, vecs, _ = get_token_vectors(sent, model_name=model_name)
            idx = find_first_token_position(toks, TARGET_WORD)
            vecs_for_sense.append(vecs[idx])
        out[sense] = tuple(vecs_for_sense)
    return out


def _avg_within(vecs: tuple[np.ndarray, ...]) -> float:
    pairs = [cosine(vecs[i], vecs[j])
             for i, j in itertools.combinations(range(len(vecs)), 2)]
    return float(np.mean(pairs)) if pairs else float("nan")


def _avg_across(va_list, vb_list) -> float:
    return float(np.mean([cosine(va, vb) for va in va_list for vb in vb_list]))

# %% [markdown]
# ## Sanity check — tokenization on the default model
#
# Confirm the right "apple" position is being extracted from each sentence.
# We do this once on the smallest model; the WordPiece tokenizer is shared
# across all three encoders, so positions are identical.

# %%
print(f"Tokenization check using {MODELS[0]}:\n")
for sense, sentences in sentences_by_sense.items():
    for sent in sentences:
        toks, _, _ = get_token_vectors(sent, model_name=MODELS[0])
        idx = find_first_token_position(toks, TARGET_WORD)
        print(f"  {sense:8s} idx={idx:>2d} tok='{toks[idx]}'  | {sent}")

# %% [markdown]
# ## Part 1 — Polysemy across separate sentences
#
# 2 sentences × 3 senses = 6 short sentences. The **separation** column is
# the headline: it's `mean(within-sense pairs) − mean(across-sense pairs)`.
# Higher = the model is producing more distinct "apple" vectors for the
# three senses.
#
# `fruit↔company_Δ` is a sharper version of the same probe, using only the
# most semantically distant pair.

# %%
def summarize_part1(av: dict[str, tuple[np.ndarray, ...]]) -> dict:
    within = {sense: _avg_within(av[sense]) for sense in av}
    across = {
        f"{a}_{b}": _avg_across(av[a], av[b])
        for a, b in itertools.combinations(av, 2)
    }
    within_mean = float(np.mean(list(within.values())))
    across_mean = float(np.mean(list(across.values())))
    return {
        "within_avg":          round(within_mean, 3),
        "across_avg":          round(across_mean, 3),
        "separation":          round(within_mean - across_mean, 3),
        "fruit↔company":       round(across["fruit_company"], 3),
        "fruit_within":        round(within["fruit"], 3),
        "fruit↔company_Δ":     round(within["fruit"] - across["fruit_company"], 3),
    }


rows_p1 = []
for model in MODELS:
    av = collect_apple_vectors(model)
    rows_p1.append({"model": SHORT_NAME[model], **summarize_part1(av)})
part1_df = pd.DataFrame(rows_p1)
part1_df

# %% [markdown]
# **Expected pattern.**
# - `bert-base`: small separation. Vanilla MLM produces anisotropic embeddings
#   where everything looks similar.
# - `bert-large`: modestly larger. More parameters → sharper distinctions,
#   but the loss function is still MLM, not sense separation.
# - `bge-large`: largest separation. Embedding-task fine-tuning explicitly
#   penalises within-batch confusion of unrelated sentences, which
#   incidentally also pushes apart polysemous senses.
#
# If `bge-large` is *not* the winner here, that's the interesting finding:
# embedding-task fine-tuning may collapse fine token-level distinctions in
# favour of sentence-level discrimination.

# %% [markdown]
# ## Part 2 — Long passage ↔ short query (same sense)
#
# Same surface word, same sense (Apple Inc.), but contexts of wildly
# different lengths: a ~70-token paragraph vs a 5-token question. The
# question is whether token-level "apple" alignment survives that
# asymmetry — the exact alignment a RAG retriever depends on.
#
# `mean_vs_query` is a sanity check on the "pool entity mentions" RAG
# trick: averaging the three `Apple` occurrences in the passage and
# comparing *that* to the query.

# %%
def summarize_part2(model_name: str) -> dict:
    toks_p, vecs_p, _ = get_token_vectors(long_passage, model_name=model_name)
    positions_p = [i for i, t in enumerate(toks_p) if TARGET_WORD in t.lower()]
    first_p = vecs_p[positions_p[0]]
    mean_p = vecs_p[positions_p].mean(axis=0)

    toks_q, vecs_q, _ = get_token_vectors(short_query, model_name=model_name)
    pos_q = find_first_token_position(toks_q, TARGET_WORD)
    q = vecs_q[pos_q]

    first_vs_q = cosine(first_p, q)
    mean_vs_q  = cosine(mean_p,  q)
    return {
        "n_apple_in_passage": len(positions_p),
        "first_vs_query":     round(first_vs_q, 3),
        "mean_vs_query":      round(mean_vs_q,  3),
        "mean−first":         round(mean_vs_q - first_vs_q, 3),
    }


rows_p2 = [{"model": SHORT_NAME[m], **summarize_part2(m)} for m in MODELS]
part2_df = pd.DataFrame(rows_p2)
part2_df

# %% [markdown]
# **Expected pattern.** All numbers should be positive and non-trivial
# (same sense). `mean−first > 0` means averaging the multiple passage
# occurrences of `apple` aligns better with the query than the first
# occurrence alone — the standard "pool entity mentions" RAG trick paying
# off. A model with `mean−first ≤ 0` is telling you that trick isn't
# worth implementing for that encoder.

# %% [markdown]
# ## Part 3 — Two senses in the same sentence
#
# Both "apple" tokens live in the same forward pass — they share the
# within-sequence anisotropy floor (~0.79–0.96 from `01-anisotropy.py`),
# *and* they have access to each other's context via attention. Does
# intra-sentence disambiguation survive that pressure?
#
# Two derived metrics:
# - **`anisotropy_boost`** = intra-sentence cos − cross-sentence cos
#   (how much the shared forward pass inflates the score)
# - **`disambig_margin`** = within-sense reference − intra-sentence cos
#   (how much the model still keeps the two senses apart *despite* the boost)

# %%
def summarize_part3(model_name: str, av: dict[str, tuple[np.ndarray, ...]]) -> dict:
    toks, vecs, _ = get_token_vectors(two_sense_sentence, model_name=model_name)
    positions = [i for i, t in enumerate(toks) if TARGET_WORD in t.lower()]
    assert len(positions) == 2, (
        f"Expected 2 'apple' tokens in two-sense sentence, got {len(positions)} for {model_name}"
    )
    v_company, v_fruit = vecs[positions[0]], vecs[positions[1]]

    intra = cosine(v_company, v_fruit)
    cross = _avg_across(av["company"], av["fruit"])
    within_ref = float(np.mean([_avg_within(av["company"]), _avg_within(av["fruit"])]))
    return {
        "intra_sentence":   round(intra, 3),
        "cross_sentence":   round(cross, 3),
        "within_reference": round(within_ref, 3),
        "anisotropy_boost": round(intra - cross, 3),
        "disambig_margin":  round(within_ref - intra, 3),
    }


rows_p3 = []
for model in MODELS:
    av = collect_apple_vectors(model)  # cached from Part 1
    rows_p3.append({"model": SHORT_NAME[model], **summarize_part3(model, av)})
part3_df = pd.DataFrame(rows_p3)
part3_df

# %% [markdown]
# **Expected pattern.**
# - `anisotropy_boost > 0` for every model — within-sequence anisotropy is
#   structural, not model-specific.
# - `disambig_margin > 0` means the model is still separating the two
#   "apple" senses despite the anisotropy boost. A near-zero or negative
#   margin would mean "both apples look identical to me in the same
#   sentence" — a meaningful failure mode worth knowing about.
#
# Watch `bge-large` here especially: fine-tuning may make it focus more on
# sentence-level discrimination at the cost of intra-sentence token
# disambiguation.

# %% [markdown]
# ## All three regimes, side-by-side
#
# One row per model. Higher is "better" for every column.

# %%
combined = pd.DataFrame({
    "model":              [SHORT_NAME[m]              for m in MODELS],
    "P1 separation":      [r["separation"]            for r in rows_p1],
    "P2 mean_vs_query":   [r["mean_vs_query"]         for r in rows_p2],
    "P3 disambig_margin": [r["disambig_margin"]       for r in rows_p3],
})
combined

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** Token-level "apple" cosine across three regimes
#    — polysemy in separate sentences, same-sense in long-vs-short context,
#    and two senses in one sentence — replicated across `bert-base-uncased`,
#    `bert-large-uncased`, and `BAAI/bge-large-en-v1.5`.
# 2. **What did I find?** ___ (fill in after running)
# 3. **What surprised me?** ___ (e.g. did `bge-large` actually underperform
#    base BERT on intra-sentence disambiguation? did scaling base → large
#    help less than embedding-task fine-tuning?)
# 4. **What's next?** Re-run with an asymmetric encoder that has separate
#    query/passage networks (e.g. `intfloat/e5-large-v2`) to see whether
#    the passage↔query alignment in Part 2 improves further.
