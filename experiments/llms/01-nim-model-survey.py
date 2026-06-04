# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 01 — NIM LLM survey
#
# **Question.** Among the LLMs hosted on NVIDIA NIM, which ones look most
# promising as the "reader / QA" component for the memory-systems experiments?
#
# **Why it matters.** Every memory-systems notebook (HippoRAG, A-Mem, the
# long-context baseline) eventually hands a packet of retrieved memories to an
# LLM and asks it to write an answer. The choice of reader changes answer
# quality, latency, and cost more than people expect — and NIM gives us a
# one-API-key way to compare many open-weight models without managing GPUs.
#
# **Setup.** Set `NVIDIA_API_KEY` (or `NIM_API_KEY`) in your env. To point at
# a self-hosted NIM container, also set `NIM_BASE_URL`. The notebook reads
# both from `experiments/shared/llm_clients.py::get_nim_client`.

# %%
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parents[1]))

import pandas as pd

from experiments.shared.llm_clients import NIM_DEFAULT_MODELS, get_nim_client

client = get_nim_client()
print(f"Endpoint: {client.base_url}")
print(f"Models to survey: {NIM_DEFAULT_MODELS}")

# %% [markdown]
# ## Choose which models to survey
#
# `NIM_DEFAULT_MODELS` is just a starting point — edit `models` to add or
# drop anything from the [NIM catalog](https://build.nvidia.com/).

# %%
models = list(NIM_DEFAULT_MODELS)

# %% [markdown]
# ## Prompts
#
# Two memory-style prompts, deliberately small:
#
# 1. **factual_recall** — single-hop lookup from a few notes.
# 2. **synthesis** — multi-hop summary across the same notes.
#
# These mirror the kind of work the QA reader does inside `memory-systems/`
# experiments, so a model's behaviour here is a useful signal for how it
# would fare downstream.

# %%
notes = (
    "- Maria moved to Boston in 2010.\n"
    "- Maria's lab researches iron-cobalt catalysts for green ammonia synthesis.\n"
    "- The lab's process runs at ambient pressure and is ~30% more energy-efficient "
    "than Haber-Bosch.\n"
    "- Maria published the catalyst result in Nature in May 2024."
)

prompts = [
    {
        "name": "factual_recall",
        "system": (
            "Answer using only the provided notes. "
            "If the notes do not contain the answer, say 'I don't know'."
        ),
        "user": f"Notes:\n{notes}\n\nQuestion: How long has Maria lived in Boston as of 2025?",
    },
    {
        "name": "synthesis",
        "system": (
            "Synthesize a one-sentence summary from the provided notes. "
            "Quote at most one phrase verbatim."
        ),
        "user": f"Notes:\n{notes}\n\nQuestion: In one sentence, what is the lab's contribution?",
    },
]

# %% [markdown]
# ## Query helper
#
# Wraps a single chat-completion call and records wall-clock latency plus
# token usage. Errors are caught so one bad model doesn't abort the survey.

# %%
def query(model: str, system: str, user: str, *, max_tokens: int = 300) -> dict:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    elapsed = time.perf_counter() - t0
    return {
        "text": resp.choices[0].message.content,
        "elapsed_s": round(elapsed, 2),
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }

# %% [markdown]
# ## Run the survey

# %%
rows: list[dict] = []
for model in models:
    for p in prompts:
        try:
            out = query(model, p["system"], p["user"])
            rows.append({"model": model, "prompt": p["name"], **out})
            print(f"✓ {model:55s} {p['name']:15s} {out['elapsed_s']:>5.2f}s "
                  f"{out['output_tokens']:>4d} out tok")
        except Exception as e:
            rows.append({"model": model, "prompt": p["name"], "text": f"<error: {e}>",
                         "elapsed_s": None, "input_tokens": None, "output_tokens": None})
            print(f"✗ {model:55s} {p['name']:15s} ERROR: {e}")

results = pd.DataFrame(rows)

# %% [markdown]
# ## Latency / token summary

# %%
results[["model", "prompt", "elapsed_s", "input_tokens", "output_tokens"]]

# %% [markdown]
# ## Read the outputs side-by-side

# %%
for p in prompts:
    print(f"\n=== prompt: {p['name']} ===")
    print(f"    {p['user'].splitlines()[-1]}")
    for _, row in results[results.prompt == p["name"]].iterrows():
        print(f"\n--- {row.model} ---")
        print(row.text)

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** Wall-clock latency, output token count, and
#    answer text for {N} NIM-hosted LLMs on a factual-recall and a synthesis
#    prompt.
# 2. **What did I find?** ___
# 3. **What surprised me?** ___
# 4. **What's next?** Plug the best-looking model into
#    `memory-systems/01-run-comparison-dataset.py` as the QA reader and
#    compare to the OpenAI/Anthropic baselines.
