# Extended Reading & Experiments — Before the Direction Decision

You're ahead of schedule and want more signal before committing to reconsolidation vs active forgetting. Below: six adjacent research themes you haven't fully explored yet, plus eight concrete experiments runnable on your existing demos.

The goal is **decision-quality information**, not breadth. Read selectively — pick the themes whose papers seem most likely to change your mind about direction.

---

## Themes you have NOT yet covered

### Theme 1 — Knowledge editing in LLM weights

**Why this is relevant:** knowledge editing modifies *model weights* to update facts; you're considering modifying *external memory* in similar ways. Same conceptual operation, different target. Papers here have invented techniques (locate-then-edit, lifelong editing, conflict resolution) that map directly onto external memory update.

**Key papers (read in order):**

1. **ROME** (Rank-One Model Editing) — Meng et al., 2022
   https://arxiv.org/abs/2202.05262
   Original locate-then-edit paper. Identifies that factual knowledge lives in specific MLP layers; modifies them via rank-one updates. The "locate-then-edit" idea translates directly to "find the relevant memory node, then edit it" in a graph-based memory.

2. **MEMIT** — Meng et al., 2022
   https://arxiv.org/abs/2210.07229
   Extends ROME to thousands of simultaneous edits. Important because real memory systems need batch updates, not one-at-a-time.

3. **WISE** — Wang et al., 2024
   https://arxiv.org/abs/2405.14768
   Distinguishes "long-term memory editing" from "working memory editing" and shows that direct weight editing creates conflicts with pretraining knowledge. The same lesson applies to external memory: aggressive updates can destabilise the system. **Most relevant of the three for your project.**

4. **MAKE — Memory-Associated Knowledge Editing** — TACL 2025
   https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.26/132652/MAKE-Memory-Associated-Knowledge-Editing
   Recent. Argues that effective knowledge editing requires associating new facts with related existing memories — essentially what A-Mem's memory evolution tries to do, but at the model-weight level.

**Why read these:** these are the most directly transferable analogues to what you want to build. The vocabulary (locality, lifelong editing, conflict, scalability) gives you a way to frame your contribution.

---

### Theme 2 — Episodic memory architectures (pre-LLM foundations + recent)

**Why this is relevant:** the distinction between *episodic* (specific events) and *semantic* (general patterns) memory is load-bearing in cognitive science but rarely cleanly implemented in LLM systems. Recent position papers argue this is the missing piece.

**Key papers:**

1. **Neural Turing Machines** — Graves et al., 2014
   https://arxiv.org/abs/1410.5401
   Foundational. Introduced learnable external memory with attention. Pre-LLM but the architectural concepts (read/write heads, content-based addressing) underlie everything since.

2. **Differentiable Neural Computer (DNC)** — Graves et al., Nature 2016
   The NTM follow-up. Adds temporal links and usage tracking — *very directly* relevant to forgetting (the usage tracker is a forgetting signal).

3. **"Episodic Memory is the Missing Piece for Long-Term LLM Agents"** — position paper, 2025
   https://arxiv.org/abs/2502.06975
   Argues directly for the project's premise: current LLM agents lack proper episodic memory. Worth reading for the framing alone — they articulate the gap you'd be filling.

4. **Continuum Memory Architectures for Long-Horizon LLM Agents** — 2026
   https://arxiv.org/abs/2601.09913
   Recent. Proposes a continuum (not discrete tiers) between episodic and semantic memory. Could inspire how you handle the episodic-to-semantic transition in your project.

**Why read these:** NTM/DNC give you the engineering vocabulary. The position paper gives you the why. The continuum paper shows the field has moved past the simple "episodic store + semantic store" framing.

---

### Theme 3 — Production memory systems in depth

**Why this is relevant:** you've referenced Mem0 and Letta but haven't gone deep into their actual mechanisms. Both have specific update/delete/consolidation strategies worth comparing to what you'd design.

**Key resources:**

1. **Mem0 paper** (arXiv:2504.19413, already in your repo)
   Re-read with the update/delete mechanism in focus. Mem0 has an *explicit* `update_memory` operation in its API — what triggers it, what's the prompt, when does it fire?

2. **Letta (MemGPT) documentation** — https://docs.letta.com
   Letta agents *self-edit* memory through tool calls during the reasoning loop. This is a different paradigm from automatic update: the agent decides what to update, when, via explicit tool use. **Worth understanding deeply** — your reconsolidation mechanism is either implicit (triggered by retrieval) or explicit (triggered by the agent). Letta is the strongest example of explicit.

3. **"Memory Systems for AI Agents: What the Research Says and What You Can Actually Build"** — Steve Kinney
   https://stevekinney.com/writing/agent-memory-systems
   Practitioner perspective. Worth a read for the gap between published research and actual deployed systems.

4. **The Letta GitHub repo** — https://github.com/letta-ai/letta
   The code is public. Their `update_core_memory` and `archival_memory_insert` implementations are short and instructive.

**Why read these:** these systems have made specific design choices that you'd otherwise need to invent. You can either follow them or knowingly deviate.

---

### Theme 4 — Recent continual learning for LLM agents

**Why this is relevant:** you have the general CL-LLM survey. But specific recent work has tackled the **agent** case (not just the model), and that's exactly your niche.

**Key papers:**

1. **MemAct** — Zhang et al., 2026
   Memory-related actions embedded in chain-of-thought prompting. Different from external memory — the agent reasons *about* its memory operations.

2. **The "Rethinking Memory" survey** (arXiv:2505.00675, already noted)
   Re-read with the six core operations as a checklist. For each one (Consolidation, Updating, Indexing, Forgetting, Retrieval, Condensation), ask: which existing system does this well, which does it poorly, where's the open gap?

3. **Memory-R1** (arXiv:2508.19828, already noted)
   You've seen this. Re-read with the specific question: could your project use RL to *learn* when to reconsolidate, instead of using hand-crafted rules?

**Why read these:** these directly compete with what you might build. If you choose reconsolidation, MemAct's reasoning-about-memory approach is an alternative paradigm. Worth knowing what you're rejecting.

---

### Theme 5 — Neuroscience papers one layer deeper

**Why this is relevant:** your brain memory deep-dive established the high-level concepts (LTP, hippocampus, sleep replay, reconsolidation). The next layer down has specific experimental papers that could inspire concrete mechanisms.

**Key papers:**

1. **Nader, Schafe, LeDoux** — 2000 — the foundational reconsolidation paper
   https://www.nature.com/articles/35021052
   "Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval." THE paper that established reconsolidation in modern neuroscience. Short, classic, worth reading even if you don't end up using anything from it directly.

2. **Memory engram cells** — Tonegawa lab — multiple papers
   Optogenetic experiments showing that specific neuronal populations encode specific memories, and activating those neurons triggers recall. The closest biological analogue to "retrieve this specific note" in your system.

3. **Spaced repetition / spacing effect** — many papers, but Cepeda et al., 2008 is a good entry
   The mathematical regularity of how repeated retrievals strengthen memory. Highly relevant if you do reconsolidation: how should repeated retrievals affect a memory's weight or links?

4. **2025 thalamic transcriptional gates paper** (Rajasethupathy lab, Nature)
   Already in your brain memory deep-dive. Re-read with the specific question: could you implement *molecular timers* — a delayed evaluation of whether a memory should consolidate, hours/days after creation?

**Why read these:** you don't need to mimic biology. But specific experimental findings (like the spacing effect) suggest concrete mechanism designs that have been validated in real biological memory.

---

### Theme 6 — Vector DB updates and incremental indexing

**Why this is relevant:** the matrix/incremental-update discussion we had earlier revealed that the *real* bottleneck is synonymy edge recomputation. Vector DB engineering has solved adjacent problems.

**Key resources:**

1. **HNSW (Hierarchical Navigable Small World) papers** — Malkov & Yashunin
   https://arxiv.org/abs/1603.09320
   The algorithm behind most production vector DBs. Supports *dynamic insertion* without full reindex. If you can map HippoRAG's synonymy graph to an HNSW-like structure, incremental updates become feasible.

2. **FAISS dynamic index types** — Facebook AI research blog + docs
   `IndexIVFFlat`, `IndexHNSW`, `IndexIVFPQ` all support `add()` operations. The trade-offs (exactness, memory, latency) are documented.

3. **Pinecone / Weaviate / Milvus update mechanisms** — engineering blogs
   How do production vector DBs handle the case "new vector added, possibly should be in others' neighbour lists"? They mostly use approximate NN structures (HNSW or IVF), accepting some recall loss for incremental update support.

4. **DiskANN** — Microsoft, 2019
   https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/
   Designed for *billion-scale* dynamic vector indexing. The engineering choices here are what you'd need if your memory grew to production scale.

**Why read these:** if your project ends up including incremental updates (which both reconsolidation and active forgetting benefit from), this is the engineering literature you'll need. Two days of reading saves weeks of reinvention.

---

## Eight experiments you can run on your existing demos

These are designed to give you direct signal for the direction decision. Most are runnable in a few hours each.

### Experiment 1 — Run the comparison dataset (most important)

You just designed this. Run it. The results should directly inform which direction has more architectural room for novel contribution.

**Time:** half a day. **Signal level:** high.

### Experiment 2 — Contradiction sequence stress test

Construct 10 small sequences of memories where a later memory contradicts an earlier one ("Sam lives in Oakland" → 30 memories of other content → "Sam moved to LA last month"). For each pair, query *after* the contradiction memory is added.

Measure:
- Does either system mention only the new fact? (Ideal)
- Mention only the old fact? (Failed update)
- Mention both? (Partial — common case)

**Time:** 1 day. **Signal level:** highest for direction decision — directly tests reconsolidation behaviour.

### Experiment 3 — Capacity scaling

Feed each demo:
- 50 memories (baseline)
- 500 memories
- 5,000 memories

Run the same 20 questions at each scale. Measure:
- Retrieval quality degradation
- Latency
- Cost
- Cases where retrieval finds the wrong memory because of capacity

**Time:** 1-2 days (mostly running time). **Signal level:** high — tells you whether the project should prioritise scaling or quality.

### Experiment 4 — Embedding model ablation on HippoRAG

Re-run HippoRAG with three different embedding models:
- The default (whatever you used in the reproduction)
- A weaker one (all-MiniLM-L6-v2 or similar)
- A stronger one (text-embedding-3-large or NV-Embed-v2)

Measure: how much does retrieval quality depend on the embedder?

**Time:** 1 day (mostly re-indexing). **Signal level:** medium — informs whether the static-embedding bottleneck is your highest-priority gap.

### Experiment 5 — A-Mem evolution prompt sensitivity

Take A-Mem's memory evolution prompt (`P_s3` in their paper). Run with:
- The original prompt
- A prompt that says "be aggressive about updating"
- A prompt that says "be conservative; preserve original memories"
- A prompt that explicitly mentions "look for contradictions"

Run the contradiction sequence from Experiment 2 with each. Measure how the prompt affects evolution rate and quality.

**Time:** 1 day. **Signal level:** high — if the system is highly prompt-sensitive, that's both a vulnerability and a research handle.

### Experiment 6 — Long-context baseline

Take your test memories. Concatenate them all into a single 100K-token prompt for Claude/GPT-4. Ask the same questions you'd ask the memory system. Measure how the long-context baseline compares.

This is the **must-include baseline** that almost no memory paper reports honestly. If long-context wins on most of your questions, that's a serious finding — it means your memory system needs to demonstrate value over and above the long-context comparison.

**Time:** half a day. **Signal level:** very high — calibrates everything else.

### Experiment 7 — Ablation: HippoRAG without PPR

Replace HippoRAG's PPR step with simple top-k embedding retrieval (skip the graph traversal entirely). Measure the quality difference.

This isolates PPR's contribution. If PPR adds only 5-10% over plain embedding retrieval, the graph mechanism isn't earning its keep on your dataset, and you should pick A-Mem's note structure over HippoRAG's KG as your project's base.

**Time:** 1 day. **Signal level:** high — directly informs choice of base architecture.

### Experiment 8 — Ablation: A-Mem without memory evolution

A-Mem's paper already includes this ablation. Reproduce it on your reproduction. Measure how much memory evolution contributes vs link generation alone.

If evolution adds little (which the paper suggests), then *write-time* evolution might not be where the value is. That makes the case for *read-time* reconsolidation (your project direction) stronger.

**Time:** half a day. **Signal level:** medium-high.

---

## Suggested execution order

If you have ~2 weeks of buffer time (which is what "ahead of schedule" probably gives you):

**Week 1:** Run experiments 1, 6, 2 (in that order). These three give you the most direction-decision signal.

**Week 2:** Pick one theme to read deeply (recommended: **Theme 1 — Knowledge editing**, because it's the most directly transferable). Read 2-3 papers from that theme. Then run experiments 3 and 8.

**Decision point:** at the end of these 2 weeks, you'll have:
- Concrete comparison data between HippoRAG and A-Mem on your dataset
- Calibration against long-context baseline
- Direct measurement of contradiction handling
- Knowledge of how the knowledge-editing literature has framed similar problems
- Ablation data on what mechanism actually matters

That's enough to make the direction decision with confidence.

---

## Decision criteria (use these after running experiments)

Use this matrix to decide between reconsolidation and active forgetting:

| Signal | Points toward reconsolidation | Points toward active forgetting |
|---|---|---|
| Contradiction stress test | Both systems fail badly | Both handle contradiction OK, but get cluttered |
| Capacity scaling | Quality degrades but contradictions are the issue | Quality degrades because graph/notes get noisy |
| Long-context baseline | Long-context handles contradictions well | Long-context degrades from noise/clutter |
| A-Mem evolution ablation | Evolution helps a lot (validates write-time, motivates read-time extension) | Evolution barely helps (suggests the whole evolution direction is weak) |
| Knowledge editing literature inspiration | Locate-then-edit feels transferable | WISE-style conflict detection feels transferable |
| Personal interest | "Memories that update themselves on recall" | "Memories that compete for limited capacity" |

If the matrix points clearly one direction, go with it. If it's mixed, **default to reconsolidation** — it has more under-explored design space (per the research landscape) and your warmup demos are better positioned for it.

---

## What NOT to do

- Don't read everything. Pick 2-3 themes max. Reading without experimenting won't reduce uncertainty about the right direction.
- Don't run all 8 experiments. Pick 4-5 based on what you most want to learn.
- Don't extend Phase 1A indefinitely. The point of being ahead of schedule is to enter Phase 1B with more confidence, not to delay it. By end of next 2 weeks: decide.
- Don't change the warmup baselines. HippoRAG and A-Mem are committed; doing more reproductions would be scope creep.

The decision criterion is simple: have you reduced your uncertainty about the right direction enough to commit? If yes, commit. If after these 2 weeks you're still ambivalent, that's itself a signal — pick reconsolidation by default and start Phase 1B. Mid-phase pivots are fine if needed.
