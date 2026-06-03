# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 05 — Late chunking vs early chunking
#
# **Question.** Does Jina v3's late chunking (embed whole document, then split
# the token embeddings) actually beat standard early chunking for context-heavy
# documents?
#
# **Why it matters.** Late chunking is "free" in storage and almost free in
# compute, but only available with long-context embedders that expose token
# embeddings. If the quality lift is real, it's a no-brainer for those embedders.
#
# **References.** `embeddings-beyond-cosine.md` §4; `papers/2409.04701-late-chunking.pdf`.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from experiments.shared.embedding_utils import cosine

# %% [markdown]
# ## Test document: heavy with pronoun references
#
# Late chunking should help most on content that uses pronouns / implicit
# subjects spanning multiple chunks.

# %%
document = (
    "Maria is a chemistry teacher who has lived in Boston for fifteen years. "
    "She specialises in organic chemistry and runs a research lab. "
    "Her latest paper covers a novel catalyst for green ammonia synthesis. "
    "She published it in Nature last month and it has received widespread attention."
)

chunk_boundaries_text = [
    "Maria is a chemistry teacher who has lived in Boston for fifteen years.",
    "She specialises in organic chemistry and runs a research lab.",
    "Her latest paper covers a novel catalyst for green ammonia synthesis.",
    "She published it in Nature last month and it has received widespread attention.",
]

# %% [markdown]
# ## Load Jina v3 (long-context embedder with token-level output)

# %%
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
model.eval()

# %% [markdown]
# ## Approach A — Early chunking (standard)
#
# Embed each chunk independently and pool. Each chunk's embedding has NO
# knowledge of the others.

# %%
def embed_chunk_independent(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        out = model(**inputs)
    # mean pool
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return pooled[0].cpu().numpy()

embs_early = np.stack([embed_chunk_independent(c) for c in chunk_boundaries_text])
print(f"Early chunking: {embs_early.shape}")

# %% [markdown]
# ## Approach B — Late chunking
#
# Embed the whole document at once; then pool slices corresponding to each chunk.

# %%
# Tokenize the full document and remember where each chunk's tokens start/end
full_text = " ".join(chunk_boundaries_text)
full_inputs = tokenizer(full_text, return_tensors="pt", truncation=True)
full_tokens = tokenizer.convert_ids_to_tokens(full_inputs["input_ids"][0])

with torch.no_grad():
    full_out = model(**full_inputs)
full_hidden = full_out.last_hidden_state[0]  # (seq_len, hidden_dim)

# Find chunk spans by re-tokenising each chunk and tracking positions
chunk_spans = []
position = 1  # skip [CLS] / [BOS]
for chunk_text in chunk_boundaries_text:
    chunk_token_ids = tokenizer(chunk_text, add_special_tokens=False)["input_ids"]
    span_len = len(chunk_token_ids)
    chunk_spans.append((position, position + span_len))
    position += span_len

# Mean-pool each span from the full hidden state
def pool_span(hidden, start, end):
    return hidden[start:end].mean(dim=0).cpu().numpy()

embs_late = np.stack([pool_span(full_hidden, s, e) for s, e in chunk_spans])
print(f"Late chunking:  {embs_late.shape}")

# %% [markdown]
# ## Retrieval comparison

# %%
queries = [
    "Where does Maria live?",
    "What kind of research does Maria do?",
    "Where was Maria's latest paper published?",
]

def best_match(q_emb, chunk_embs, chunks):
    sims = np.array([cosine(q_emb, e) for e in chunk_embs])
    top = int(sims.argmax())
    return top, sims[top], chunks[top]

for q in queries:
    q_emb = embed_chunk_independent(q)
    early_top, early_sim, early_chunk = best_match(q_emb, embs_early, chunk_boundaries_text)
    late_top,  late_sim,  late_chunk  = best_match(q_emb, embs_late,  chunk_boundaries_text)
    print(f"\nQ: {q}")
    print(f"  EARLY top (sim={early_sim:.3f}): {early_chunk[:80]}")
    print(f"  LATE  top (sim={late_sim:.3f}): {late_chunk[:80]}")

# %% [markdown]
# Expected pattern: queries like "Where was Maria's latest paper published?"
# correspond to a chunk that says "She published it..." — early chunking
# struggles because "She" loses Maria context; late chunking handles it.

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** Top-1 retrieval accuracy with early vs late
#    chunking on 3 pronoun-heavy queries.
# 2. **What did I find?** ___
# 3. **What surprised me?** ___
# 4. **What's next?** Test on a longer, more pronoun-heavy document
#    (a novel passage or conversation transcript).
