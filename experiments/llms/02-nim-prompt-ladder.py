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
# # 02 — NIM prompt ladder (tiny → frontier)
#
# **Question.** Given a single prompt I care about, how do NIM-hosted models
# from ~3B all the way to ~405B differ in output style, latency, and content?
#
# **Why it matters.** Bigger models cost more and add latency. This notebook
# lets you paste any prompt and feel the quality / cost / latency tradeoff
# along the *whole* size spectrum in one pass, so you can pick the smallest
# model that still does the job well for the task you care about.
#
# **Setup.** Needs `NVIDIA_API_KEY` (or `NIM_API_KEY`) in your env. See the
# README's env-var table or `01-nim-model-survey.py` for details. If you use
# direnv, just drop the key in `.env`.

# %% [markdown]
# ## 1. Paste your prompt here
#
# This is the only cell you usually need to edit. Triple-quoted strings keep
# multi-line prompts readable.

# %%
PROMPT_SYSTEM = "You are a careful, concise assistant. Prefer plain prose."

PROMPT_USER = """\
In 3 sentences, explain why late chunking can preserve coreference signal
that early chunking destroys. Assume the reader knows what token embeddings
are but has never heard the term "late chunking".
"""

# %% [markdown]
# ## 2. The model ladder
#
# Tiny → frontier, sorted by parameter count. Multiple providers at most
# tiers (meta, google, microsoft, nvidia, qwen, mistral, deepseek) so you
# can see provider-style differences at a given size.
#
# Edit freely — the NIM catalog at https://build.nvidia.com/ moves. If a
# model 404s the survey continues and the row will show `<error: ...>`.
#
# Notes on the entries below:
# - `a3b` / `a10b` / `a35b` suffixes mean Mixture-of-Experts (only that many
#   parameters are active per token, even though the full model is much bigger).
# - `qwen/qwq-32b` is a *reasoning* model — its output usually contains
#   `<think>…</think>` chain-of-thought before the final answer, so expect
#   longer responses and higher latency.

# %%
MODEL_LADDER: list[tuple[str, str]] = [
    # ─── tiny (~1-4B) ─────────────────────────────────────────────
    ("1B   meta",      "meta/llama-3.2-1b-instruct"),
    ("2B   google",    "google/gemma-2-2b-it"),
    ("3B   meta",      "meta/llama-3.2-3b-instruct"),
    ("4B   ms",        "microsoft/phi-4-mini-instruct"),
    ("4B   nvidia",    "nvidia/nemotron-mini-4b-instruct"),
    # ─── small (~8-9B) ────────────────────────────────────────────
    ("8B   meta",      "meta/llama-3.1-8b-instruct"),
    ("9B   nvidia",    "nvidia/nvidia-nemotron-nano-9b-v2"),
    # ─── medium (~30-50B) ────────────────────────────────────────
    ("32B  qwen",      "qwen/qwq-32b"),                              # reasoning model
    ("47B  mistral",   "mistralai/mixtral-8x7b-instruct"),           # MoE, ~13B active
    ("49B  nvidia",    "nvidia/llama-3.3-nemotron-super-49b-v1.5"),
    # ─── large (~70-141B) ────────────────────────────────────────
    ("70B  meta",      "meta/llama-3.3-70b-instruct"),
    ("80B  qwen",      "qwen/qwen3-next-80b-a3b-instruct"),          # MoE
    ("141B mistral",   "mistralai/mixtral-8x22b-instruct"),          # MoE
    # ─── frontier (>200B) ────────────────────────────────────────
    ("253B nvidia",    "nvidia/llama-3.1-nemotron-ultra-253b-v1"),
    ("480B qwen",      "qwen/qwen3-coder-480b-a35b-instruct"),       # MoE, coder-tuned
    ("v4   deepseek",  "deepseek-ai/deepseek-v4-pro"),
]

# Per-request settings. Lower temperature to make outputs comparable.
TEMPERATURE = 0.2
MAX_TOKENS = 400

# %% [markdown]
# ## 3. Imports + NIM client

# %%
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parents[1]))

import pandas as pd

from experiments.shared.llm_clients import get_nim_client

client = get_nim_client()
print(f"Endpoint: {client.base_url}")
print(f"Ladder:   {len(MODEL_LADDER)} models from '{MODEL_LADDER[0][0]}' to '{MODEL_LADDER[-1][0]}'")

# %% [markdown]
# ## 4. Run the ladder
#
# One chat-completion call per model. Errors are caught per row so one
# unavailable model doesn't abort the survey.

# %%
def query(model: str, system: str, user: str) -> dict:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    elapsed = time.perf_counter() - t0
    return {
        "text": resp.choices[0].message.content,
        "elapsed_s": round(elapsed, 2),
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }


rows: list[dict] = []
for size_label, model in MODEL_LADDER:
    try:
        out = query(model, PROMPT_SYSTEM, PROMPT_USER)
        rows.append({"size": size_label, "model": model, **out})
        print(f"✓ {size_label:<16s} {model:<45s} {out['elapsed_s']:>5.2f}s "
              f"{out['output_tokens']:>4d} out tok")
    except Exception as e:
        rows.append({"size": size_label, "model": model, "text": f"<error: {e}>",
                     "elapsed_s": None, "input_tokens": None, "output_tokens": None})
        print(f"✗ {size_label:<16s} {model:<45s} ERROR: {e}")

results = pd.DataFrame(rows)

# %% [markdown]
# ## 5. Latency / token summary

# %%
results[["size", "model", "elapsed_s", "input_tokens", "output_tokens"]]

# %% [markdown]
# ## 6. Outputs in size order
#
# Read top → bottom. Where does the answer become "good enough" for your
# task? That's usually the model worth picking.

# %%
for _, row in results.iterrows():
    head = (f"=== {row['size']}  —  {row['model']}  "
            f"({row['elapsed_s']}s, {row['output_tokens']} out tok) ===")
    print("\n" + head)
    print(row["text"])

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** Output text, wall-clock latency, and output
#    token count for one prompt across {N} NIM-hosted models spanning ~3B to
#    ~405B parameters.
# 2. **What did I find?** ___
# 3. **What surprised me?** ___ (e.g. did a small model nail it? did the
#    largest one over-explain?)
# 4. **What's next?** Pick the smallest model whose answer was acceptable
#    and use it as the QA reader in `memory-systems/01-run-comparison-dataset.py`.
