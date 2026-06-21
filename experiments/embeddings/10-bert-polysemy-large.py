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
# # 10 — BERT contextual embeddings: polysemy & cross-context
#
# **Question.** How does BERT contextualize the *same* surface word across
# three regimes?
#
# 1. **Polysemy** — same word, different senses, separate sentences.
# 2. **RAG alignment** — same word, same sense, long passage vs short query.
# 3. **Intra-sentence disambiguation** — same word, two senses, *same* sentence.
#
# **Why it matters.** "BERT produces contextual embeddings" is the textbook
# claim. This notebook checks how robust that is in conditions that actually
# matter for memory systems and RAG: cross-document sense disambiguation,
# passage↔query alignment under wildly different context lengths, and
# whether disambiguation survives even when both senses share a single
# forward pass.
#
# We use the polysemous word **"apple"** because it has three crisp senses
# (fruit, idiom, company) that occur in well-known phrasings — easy to write
# clean test sentences for.
#
# **Reference.** `topics/llm-agent-memory/embeddings-101.md` §8 (contextual
# embeddings); companion to `01-anisotropy.py` (within-sequence baseline)
# and `09-anisotropy-context-split.py` (across-sequence collapse).


# %%
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
    pairwise_cosine_table,
)

# bert-base-uncased is the canonical BERT — 110M params, lowercases input.
# Swap to bert-large-uncased (340M) or BAAI/bge-large-en-v1.5 (335M,
# BERT-based, embedding-fine-tuned) to see how the picture changes.
MODEL_NAME = "bert-large-uncased"
TARGET_WORD = "apple"

# %% [markdown]
# ## Part 1 — Polysemy: three senses, separate sentences
#
# Two sentences per sense so we can average within-sense vs across-sense
# cosines and get a cleaner signal than one-shot comparisons.

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

# Pull the "apple" vector from each sentence.
apple_vectors: dict[str, list[np.ndarray]] = {}
for sense, sentences in sentences_by_sense.items():
    apple_vectors[sense] = []
    for sent in sentences:
        toks, vecs, _ = get_token_vectors(sent, model_name=MODEL_NAME)
        idx = find_first_token_position(toks, TARGET_WORD)
        apple_vectors[sense].append(vecs[idx])
        print(f"  {sense:8s} idx={idx:>2d} token='{toks[idx]}'  | {sent}")

# %% [markdown]
# ### 1a. Pairwise cosine table (all 6 sentences)

# %%
labels = [
    f"{sense}_{i + 1}"
    for sense in sentences_by_sense
    for i in range(len(sentences_by_sense[sense]))
]
all_vecs = np.stack(
    [v for sense_vecs in apple_vectors.values() for v in sense_vecs]
)
part1_table = pairwise_cosine_table(labels, all_vecs)
part1_table

# %% [markdown]
# ### 1b. Within-sense vs across-sense averages
#
# A condensed version of the table above.


# %%
def avg_within(sense: str) -> float:
    vs = apple_vectors[sense]
    pairs = [
        cosine(vs[i], vs[j])
        for i, j in itertools.combinations(range(len(vs)), 2)
    ]
    return float(np.mean(pairs))


def avg_across(sense_a: str, sense_b: str) -> float:
    return float(
        np.mean(
            [
                cosine(va, vb)
                for va in apple_vectors[sense_a]
                for vb in apple_vectors[sense_b]
            ]
        )
    )


rows = []
for sense in sentences_by_sense:
    rows.append(
        {
            "sense_a": sense,
            "sense_b": sense,
            "kind": "within",
            "mean_cos": round(avg_within(sense), 3),
        }
    )
for sense_a, sense_b in itertools.combinations(sentences_by_sense, 2):
    rows.append(
        {
            "sense_a": sense_a,
            "sense_b": sense_b,
            "kind": "across",
            "mean_cos": round(avg_across(sense_a, sense_b), 3),
        }
    )
part1_summary = pd.DataFrame(rows)
part1_summary

# %% [markdown]
# **Expected pattern.** Within-sense pairs should be visibly higher than
# across-sense pairs. If they aren't, BERT-base isn't disambiguating these
# senses at the token level — at that point you'd fall back to mean-pooled
# sentence embeddings (or a fine-tuned embedder like BGE) for downstream
# tasks. The strongest separation should be **fruit ↔ company** — those two
# senses share no semantic overlap at all.

# %% [markdown]
# ## Part 2 — Long passage vs short query (same sense)
#
# A multi-sentence passage about Apple Inc.'s founding, and a short retrieval
# query. In a RAG pipeline these two would be matched against each other
# — so the question is whether the **token-level** "apple" vectors in
# passage and query align well despite the wildly different context lengths.

# %%
long_passage = (
    "Apple Inc. was founded on April 1, 1976 by Steve Jobs, Steve Wozniak, and"
    " Ronald Wayne in a garage in Los Altos, California. The company initially"
    " sold the Apple I personal computer kit, which Wozniak had designed."
    " Within a decade Apple had become one of the most influential consumer"
    " technology companies in the world, releasing the Macintosh in 1984."
)
short_query = "Who founded Apple?"

toks_p, vecs_p, _ = get_token_vectors(long_passage, model_name=MODEL_NAME)
apple_positions_p = [
    i for i, t in enumerate(toks_p) if TARGET_WORD in t.lower()
]
apple_first_p = vecs_p[apple_positions_p[0]]
apple_mean_p = vecs_p[apple_positions_p].mean(axis=0)

toks_q, vecs_q, _ = get_token_vectors(short_query, model_name=MODEL_NAME)
apple_pos_q = find_first_token_position(toks_q, TARGET_WORD)
apple_q = vecs_q[apple_pos_q]

print(
    f"Long passage: {len(toks_p)} tokens, '{TARGET_WORD}' at positions"
    f" {apple_positions_p}"
)
print(
    f"Short query:  {len(toks_q)} tokens, '{TARGET_WORD}' at position "
    f" {apple_pos_q}"
)

# %%
part2_rows = [
    {
        "pair": "passage 'apple' (first occurrence) ↔ query 'apple'",
        "cosine": round(cosine(apple_first_p, apple_q), 3),
    },
    {
        "pair": "passage 'apple' (mean of all 3)    ↔ query 'apple'",
        "cosine": round(cosine(apple_mean_p, apple_q), 3),
    },
]
pd.DataFrame(part2_rows)

# %% [markdown]
# **Expected pattern.** Both numbers should be reasonably high (same surface
# word, same sense), but *lower* than the within-sense averages from Part 1
# — Part 1's pairs share short, similar context lengths, whereas here the
# passage is ~70 tokens and the query is 5.
#
# If `mean` of the three passage occurrences scores higher than `first`,
# that's evidence for the usual RAG trick of averaging multiple entity
# mentions to get a more "central" representation.

# %% [markdown]
# ## Part 3 — Two senses, same sentence
#
# Both "apple" tokens live in the *same* forward pass, so they share the
# within-sequence anisotropy that inflated `01-anisotropy.py`'s same-sequence
# pairs to 0.79–0.96. The question is: does BERT's intra-sentence
# disambiguation survive that pressure?

# %%
two_senses = (
    "Steve Jobs founded Apple in 1976. He often ate an apple at his desk."
)

toks_s, vecs_s, _ = get_token_vectors(two_senses, model_name=MODEL_NAME)
apple_positions_s = [
    i for i, t in enumerate(toks_s) if TARGET_WORD in t.lower()
]
assert (
    len(apple_positions_s) == 2
), f"Expected 2 'apple' tokens, got {len(apple_positions_s)}"

vec_company = vecs_s[apple_positions_s[0]]
vec_fruit = vecs_s[apple_positions_s[1]]

print(
    f"Tokens around company-apple (idx {apple_positions_s[0]}): "
    f"{toks_s[max(0, apple_positions_s[0]-3):apple_positions_s[0]+4]}"
)
print(
    f"Tokens around fruit-apple   (idx {apple_positions_s[1]}): "
    f"{toks_s[max(0, apple_positions_s[1]-3):apple_positions_s[1]+4]}"
)

intra_sentence_cos = cosine(vec_company, vec_fruit)
print(
    "\ncosine(company-apple, fruit-apple) within same sentence:"
    f" {intra_sentence_cos:.3f}"
)

# %% [markdown]
# ### Compare against the cross-sentence company ↔ fruit pair from Part 1

# %%
cross_sentence_cos = avg_across("company", "fruit")
within_company = avg_within("company")
within_fruit = avg_within("fruit")

summary = pd.DataFrame(
    [
        {
            "comparison": "intra-sentence (company ↔ fruit, same sentence)",
            "cosine": round(intra_sentence_cos, 3),
        },
        {
            "comparison": "cross-sentence (company ↔ fruit, mean from 1b)",
            "cosine": round(cross_sentence_cos, 3),
        },
        {
            "comparison": "within-sense reference (company ↔ company)",
            "cosine": round(within_company, 3),
        },
        {
            "comparison": "within-sense reference (fruit ↔ fruit)",
            "cosine": round(within_fruit, 3),
        },
    ]
)
summary

# %% [markdown]
# **Expected pattern.** The intra-sentence number is *higher* than the
# cross-sentence one — that's the within-sequence anisotropy boost from
# `01-anisotropy.py` — but it should still be visibly **lower** than the
# within-sense references. If so, BERT really is producing different
# representations for the two "apple"s even though they share a forward
# pass: real intra-sentence sense disambiguation that anisotropy can't
# wash out.

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** Token-level "apple" cosine across three regimes
#    — polysemy in separate sentences, same-sense in long-vs-short context,
#    and two senses in one sentence — using `bert-base-uncased`.
# 2. **What did I find?** ___ (fill in after running)
# 3. **What surprised me?** ___
# 4. **What's next?** Re-run with `bert-large-uncased` (does scale sharpen
#    sense separation?) and `BAAI/bge-large-en-v1.5` (does embedding-task
#    fine-tuning help or hurt token-level disambiguation?).
