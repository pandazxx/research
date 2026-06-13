# Experiments

Hands-on experiments for the agent-memory research project. Each notebook is a self-contained learning artifact: load some models, run a focused experiment, draw conclusions.

The notebooks are written as **Jupytext-paired `.py` files** with `# %%` cell markers. This means:

- You edit `.py` files in your editor of choice (vim, VS Code, etc.).
- When you open the `.py` in JupyterLab, Jupytext creates a synced `.ipynb` view for execution.
- Only `.py` files are committed to git. `.ipynb` files are gitignored.
- Git diffs stay clean — no JSON blobs with cell outputs.

This is the standard workflow for ML researchers who don't want to be locked into the Jupyter UI.

---

## Setup

This repo uses **[uv](https://docs.astral.sh/uv/)** for Python environment + dependency management and **[just](https://just.systems/)** for task automation. Both should be installed before starting.

Install uv:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Install just:

```bash
# macOS
brew install just

# Linux (via cargo)
cargo install just

# or download from https://github.com/casey/just/releases
```

Then, from the repo root:

```bash
just setup     # creates .venv, installs deps, registers kernel, installs pre-commit hooks
just lab       # starts JupyterLab in your browser
```

That's it. `just setup` is idempotent — re-run after pulling changes to sync any new dependencies.

### Run a single notebook headless

```bash
just run experiments/embeddings/01-anisotropy.py
```

This converts the `.py` to a `.ipynb`, executes every cell, and writes the executed `.ipynb` next to the `.py`. The executed `.ipynb` is gitignored (only the source `.py` gets committed).

### Just recipes (all available commands)

```bash
just              # list all recipes (default action)
just setup        # install deps + register kernel + install hooks
just install      # install deps only (no kernel setup)
just lab          # start JupyterLab
just run FILE     # run a specific notebook end-to-end
just convert-all  # convert all .py notebooks to .ipynb (no execution)
just strip        # strip outputs from any .ipynb files
just clean        # remove .venv, caches, generated .ipynb files
just lock         # update uv.lock
just sync         # sync env from uv.lock (after pulling)
just python       # show Python version in use
just deps         # show installed packages
```

---

## Directory layout

```
experiments/
├── README.md              ← you are here
├── shared/                ← reusable helpers (regular Python, not notebooks)
│   ├── embedding_utils.py    cosine, token-position lookup, model loading
│   ├── dataset_loader.py     load comparison-dataset/dataset.json
│   ├── llm_clients.py        OpenAI-compatible client for NVIDIA NIM
│   └── system_loaders.py     load HippoRAG / A-Mem (placeholders)
├── embeddings/            ← experiments on embedding behavior
│   ├── 01-anisotropy.py
│   ├── 02-distance-ablation.py
│   ├── 03-cross-model-comparison.py
│   ├── 04-semantic-chunking.py
│   ├── 05-late-chunking.py
│   ├── 06-clustering-discovery.py
│   ├── 07-novelty-scoring.py
│   ├── 08-cross-encoder-reranking.py
│   ├── 09-anisotropy-context-split.py
│   └── 10-bert-polysemy.py
├── llms/                  ← surveys / sandboxes for LLM choices
│   ├── 01-nim-model-survey.py
│   └── 02-nim-prompt-ladder.py
└── memory-systems/        ← experiments comparing / probing HippoRAG and A-Mem
    ├── 01-run-comparison-dataset.py
    ├── 02-contradiction-stress-test.py
    ├── 03-capacity-scaling.py
    ├── 04-long-context-baseline.py
    ├── 05-hipporag-embedder-ablation.py
    ├── 06-amem-prompt-sensitivity.py
    ├── 07-hipporag-no-ppr-ablation.py
    └── 08-amem-no-evolution-ablation.py
```

---

## Experiment index

### Embeddings — `experiments/embeddings/`

| # | Notebook | What it investigates | Status |
|---|---|---|---|
| 01 | `01-anisotropy.py` | Per-token cosine similarity within sequence; reproduces the Joey/She/wife result | Skeleton, partially runnable |
| 02 | `02-distance-ablation.py` | How within-sequence similarity decays with token distance | Skeleton |
| 03 | `03-cross-model-comparison.py` | Compare anisotropy across BGE, MiniLM, E5, Jina | Skeleton |
| 04 | `04-semantic-chunking.py` | Semantic chunker vs fixed-size chunker on retrieval quality | Skeleton |
| 05 | `05-late-chunking.py` | Jina v3 late chunking vs standard early chunking | Skeleton |
| 06 | `06-clustering-discovery.py` | UMAP + HDBSCAN on memory corpus → automatic topic discovery | Skeleton |
| 07 | `07-novelty-scoring.py` | Anomaly / novelty as a write-policy signal (mirrors brain's dopamine novelty) | Skeleton |
| 08 | `08-cross-encoder-reranking.py` | 2-stage retrieval: dense embeddings + cross-encoder rerank | Skeleton |
| 09 | `09-anisotropy-context-split.py` | Re-runs 01's Joey/She/wife pairs after splitting the document into 4 per-sentence embeddings; measures how much within-sequence similarity collapses | Runnable |
| 10 | `10-bert-polysemy.py` | Probes BERT contextual embeddings on polysemous "apple" across 3 regimes: cross-sentence polysemy, long-passage-vs-short-query alignment, and two senses in the same sentence | Runnable |

### LLMs — `experiments/llms/`

| # | Notebook | What it investigates | Status |
|---|---|---|---|
| 01 | `01-nim-model-survey.py` | Side-by-side comparison of several NVIDIA-NIM-hosted LLMs on memory-style QA prompts | Runnable; needs `NVIDIA_API_KEY` |
| 02 | `02-nim-prompt-ladder.py` | Paste a single prompt; runs it through a tiny→frontier ladder of NIM models (~1B → 100s of B) spanning meta / google / microsoft / nvidia / qwen / mistral / deepseek, and shows latency + side-by-side outputs | Runnable; needs `NVIDIA_API_KEY` |

#### NIM environment variables

| Variable | Purpose |
|---|---|
| `NVIDIA_API_KEY` | Required. Get one at https://build.nvidia.com/. `NIM_API_KEY` is accepted as a fallback name. |
| `NIM_BASE_URL` | Optional. Override the endpoint (e.g. point at a self-hosted NIM container). Defaults to `https://integrate.api.nvidia.com/v1`. |

### Memory systems — `experiments/memory-systems/`

| # | Notebook | What it investigates | Status |
|---|---|---|---|
| 01 | `01-run-comparison-dataset.py` | HippoRAG vs A-Mem head-to-head on the comparison-dataset | Needs system loaders |
| 02 | `02-contradiction-stress-test.py` | Inject contradictions; measure each system's update behaviour | Needs system loaders |
| 03 | `03-capacity-scaling.py` | 50 / 500 / 5000 memories — quality and latency degradation | Needs system loaders |
| 04 | `04-long-context-baseline.py` | Stuff everything in one 100K-token prompt vs memory systems | Skeleton |
| 05 | `05-hipporag-embedder-ablation.py` | Swap embedders in HippoRAG; measure quality sensitivity | Needs HippoRAG repro |
| 06 | `06-amem-prompt-sensitivity.py` | Vary A-Mem's evolution prompt; measure behaviour difference | Needs A-Mem repro |
| 07 | `07-hipporag-no-ppr-ablation.py` | Disable PPR; falls back to top-k embedding retrieval | Needs HippoRAG repro |
| 08 | `08-amem-no-evolution-ablation.py` | Disable A-Mem's memory evolution; measure quality drop | Needs A-Mem repro |

---

## Workflow norms

1. **Restart and Run All before committing.** Notebooks must be reproducible top-to-bottom. If a cell only works in a specific order, fix it.
2. **One question per notebook.** If your notebook drifts into investigating a second question, fork it.
3. **Conclusions live in markdown cells at the bottom.** Each notebook should end with a "what did we learn" cell.
4. **Don't commit outputs.** The pre-commit hook (`nbstripout`) handles this if you accidentally save outputs.
5. **Shared code goes in `experiments/shared/`.** If you copy-paste a helper between notebooks more than once, move it to the shared module.

---

## Cost expectations

| Notebook | Approx API cost (single run, default settings) |
|---|---|
| Embeddings 01–08 | $0–2 (free if using local embedders) |
| Memory systems 01–04 | $1–10 (LLM calls for QA reader) |
| Memory systems 05–08 | $5–30 (full reproduction passes) |

If you run with closed-API LLMs (Claude / GPT-4 / etc.), monitor your usage. The memory-systems notebooks make many LLM calls per question; running the full comparison-dataset (22 questions) can cost $5+ per system.

---

## When you finish an experiment

Each notebook's ending markdown cell should answer four questions:

1. **What did I measure?** One sentence.
2. **What did I find?** Specific numbers, not vague claims.
3. **What surprised me?** The interesting part of any honest experiment.
4. **What's next?** One concrete follow-up.

These four sentences are what would eventually become a blog post or a section in the project's eventual paper.
