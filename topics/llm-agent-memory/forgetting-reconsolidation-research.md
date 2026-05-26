# Forgetting & Reconsolidation in LLM Agent Memory — Recent Research

Papers and systems that directly address *how LLM agents should update, revise, or discard memories* — the specific gap identified in our HippoRAG application study. Most are from late 2025 to mid-2026.

---

## TL;DR

The field is moving fast. As of mid-2026, there are **three systems that explicitly implement reconsolidation-like memory updates** (HiMem, Memory-R1, Human-Inspired Memory Architecture), **three systems with active forgetting mechanisms** (ACT-R-inspired, FadeMem, SYNAPSE), and **one benchmark that tests forgetting explicitly** (Memora). The surveys all identify forgetting/reconsolidation as a top open problem.

**Bottom line for the project:** this is no longer a completely empty gap. There are now concrete systems to compare against and build on. The opportunity has shifted from "nobody has done this" to "several people have started; nobody has done it *well enough* yet, and nobody has applied it to graph-structured memory like HippoRAG."

---

## Category 1 — Systems with reconsolidation / memory update

### HiMem (Jan 2026)

- **Paper:** https://arxiv.org/abs/2601.06377
- **What it does:** hierarchical long-term memory with two tiers — Episode Memory (raw experiences, topic-segmented) and Note Memory (stable extracted knowledge).
- **How reconsolidation works:**
  - Reconsolidation is triggered only when *both* conditions are met: (1) retrieval from Note Memory alone is insufficient, *and* (2) subsequently retrieved Episode Memory provides supporting evidence.
  - When triggered, the system does **conflict-aware** reconsolidation: it checks whether the new episodic evidence contradicts or supplements existing notes, then revises the notes accordingly.
  - This is the closest published mechanism to biological reconsolidation — the memory is re-evaluated and potentially overwritten during retrieval, not just at write time.
- **Relevance to project:** directly implements the mechanism we're considering. The hierarchy (episodic → semantic) maps to the hippocampus → neocortex split from the brain memory study. **This is the most important paper to read.**
- **What it doesn't do:** no graph structure (flat notes, not KG); no PPR or graph-based retrieval; no connection to HippoRAG's architecture.

### Memory-R1 (Aug 2025)

- **Paper:** https://arxiv.org/abs/2508.19828
- **What it does:** uses RL (PPO/GRPO) to train two specialised sub-agents:
  - **Memory Manager** — learns four operations: `ADD`, `UPDATE`, `DELETE`, `NOOP`. Decides what to do with each new piece of information.
  - **Answer Agent** — pre-selects relevant entries and reasons over them.
- **How update/delete works:**
  - The Memory Manager is *trained* via outcome-driven RL to make the right memory management decision. The reward signal is downstream task performance.
  - Unlike prompt-based approaches (Generative Agents, A-Mem), the decision of what to add/update/delete is *learned*, not hand-crafted.
- **Relevance to project:** the RL approach to learning memory operations is orthogonal to our graph-based approach. Could potentially be combined: train an RL agent to decide when to update/delete edges in a HippoRAG KG. But the RL training loop is a different research direction from reconsolidation-on-retrieval.
- **What it doesn't do:** no reconsolidation per se (updates happen at write time based on RL policy, not on retrieval). No graph structure.

### Human-Inspired Memory Architecture (May 2026)

- **Paper:** https://arxiv.org/abs/2605.08538
- **What it does:** implements the full memory lifecycle as deterministic algorithms operating on embeddings and scores:
  - Consolidation: strengthen recently encoded memories.
  - Forgetting: decay-based removal.
  - Maturation: memories evolve from episodic to semantic over time.
  - Reconsolidation: retrieved memories are re-evaluated and potentially revised.
- **How reconsolidation works:**
  - When a memory is retrieved, the system computes a "reconsolidation score" based on how well the memory matches the current query context.
  - If the score is below a threshold, the memory is flagged for revision.
  - Revision involves re-embedding the memory with updated context information.
- **Relevance to project:** algorithmically clean implementation of reconsolidation. The deterministic approach (as opposed to RL or LLM-prompt-based) is appealing for reproducibility.
- **What it doesn't do:** very recent (May 2026); limited empirical evaluation so far. No graph structure.

---

## Category 2 — Systems with active forgetting

### ACT-R-Inspired Memory (HAI 2025)

- **Paper:** https://dl.acm.org/doi/10.1145/3765766.3765803
- **What it does:** integrates the ACT-R cognitive architecture with LLM agents. Memories have an **activation level** that determines retrievability, combining:
  - **Base-level activation**: decays with time, increases with use (power-law decay, matches the spacing effect from psychology).
  - **Spreading activation**: related memories boost each other.
  - **Probabilistic noise**: retrieval has a stochastic element — even well-encoded memories occasionally fail to retrieve.
- **How forgetting works:** memories whose activation drops below a retrieval threshold become effectively forgotten — they still exist but are never surfaced. This is "soft forgetting" (degradation) not "hard forgetting" (deletion).
- **Relevance to project:** the activation function is directly applicable to HippoRAG. Each node/edge in the KG could have an activation level: decays if not accessed, increases when retrieved or reinforced. PPR could be weighted by activation. This is a concrete design path.
- **Key formula:** `activation(i) = ln(Σ t_j^(-d)) + noise`, where `t_j` is time since the j-th access and `d` is the decay parameter (~0.5 in ACT-R).

### FadeMem (Jan 2026)

- **Paper:** https://arxiv.org/abs/2601.18642
- **What it does:** biologically-inspired forgetting for agent memory with two key features:
  - **Dual-layer memory**: short-term (fast decay) and long-term (slow decay), with promotion between layers.
  - **Adaptive exponential decay**: each memory's decay rate is adjusted based on importance and access patterns.
- **Results:** 45% storage reduction while maintaining or improving retrieval quality. Superior multihop reasoning compared to systems without forgetting.
- **Relevance to project:** demonstrates that forgetting *improves* performance — not just saves storage. The 45% reduction with quality preservation is a strong empirical argument for active forgetting. The dual-layer architecture maps naturally to HippoRAG's potential tiered node system.

### SYNAPSE (Jan 2026)

- **Paper:** https://arxiv.org/abs/2601.02744
- **What it does:** episodic-semantic memory integration using spreading activation (from ACT-R) over a graph structure.
  - Propagates activation along specific transitive paths from query anchors.
  - Enforces **sparsity constraints** (only top-k activations survive) and **competition** (memories compete for activation budget).
  - Memories that consistently lose the competition effectively fade.
- **Relevance to project:** the spreading activation mechanism is *very close* to PPR in HippoRAG. SYNAPSE's sparsity/competition constraints are a natural way to implement capacity-bounded memory in a graph. **This is the second-most important paper to read** — it shows how to add forgetting to graph-based memory retrieval.

---

## Category 3 — Surveys that frame forgetting/reconsolidation as open problems

### "Rethinking Memory in LLM based Agents" (Dec 2025)

- **Paper:** https://arxiv.org/abs/2505.00675
- **Key contribution:** defines **six core memory operations** that a complete memory system should support:
  1. Consolidation (stabilise new memories)
  2. **Updating** (revise existing memories with new information)
  3. Indexing (organise for retrieval)
  4. **Forgetting** (remove/decay stale memories)
  5. Retrieval (access stored memories)
  6. Condensation (compress/summarise)
- Operations 2 and 4 are the project's target. This survey is the best existing taxonomy.

### SSGM Framework (Mar 2026)

- **Paper:** https://arxiv.org/abs/2603.11768
- **Angle:** safety and stability of evolving memory. Addresses adversarial injection, memory drift, and the risk of reconsolidation-gone-wrong (a memory being corrupted by a bad update).
- **Relevant concern for the project:** any reconsolidation mechanism we build must also consider: *what happens if the update signal is wrong?* A malicious or confused query shouldn't be able to corrupt the memory graph. SSGM provides a framework for thinking about this.

### "Memory for Autonomous LLM Agents" (Mar 2026)

- **Paper:** https://arxiv.org/abs/2603.07670
- **Key finding:** lists **learned forgetting** and **continual consolidation** as two of the top five emerging frontiers. Identifies the gap between current systems (mostly append-only) and what's needed (dynamic lifecycle management).

### Memora / "From Recall to Forgetting" (Apr 2026)

- **Paper:** https://arxiv.org/abs/2604.20006
- **Key contribution:** the first benchmark that explicitly tests whether an agent **correctly forgets** stale information. Uses the **FAMA metric** (Forgetting-Aware Memory Accuracy) that rewards correct use of valid memory *and* penalises reliance on obsolete memory. **Use this as a benchmark** if the project goes in the forgetting direction.

---

## Category 4 — Related but distinct: machine unlearning / knowledge editing

These target model *weights* rather than external memory, but the concepts are connected:

| Paper | arXiv | What it does | Connection to our work |
|---|---|---|---|
| Survey on LLM Unlearning | 2510.25117 | Comprehensive survey of methods to erase knowledge from model weights | Conceptually related but different target (parametric vs non-parametric memory) |
| Unlearning in LLMs: Methods & Challenges | 2601.13264 | Methods + evaluation for selective knowledge removal | FLAT method (f-divergence maximisation) could inspire edge-weight decay |
| Agentic Unlearning (ALU) | 2602.17692 | Multi-agent approach to unlearning, including from external memory | **Directly relevant** — addresses forgetting from agent memory, not just model weights |
| Editing as Unlearning | 2505.19855 | Knowledge editing as a baseline for unlearning | Shows connection between updating and forgetting at weight level |

**Agentic Unlearning (ALU)** is the most directly relevant: it explicitly asks "what happens when forgotten information is still in external memory?" — which is exactly the HippoRAG KG problem. The others are more about model-weight modification.

---

## Summary map: who does what

| System | Reconsolidation (update on recall) | Active forgetting (decay/delete) | Graph-structured | RL-trained | Benchmark-evaluated |
|---|---|---|---|---|---|
| HiMem | **Yes** (conflict-aware) | No | No (flat notes) | No | Yes |
| Memory-R1 | No (update at write time) | **Yes** (DELETE op) | No | **Yes** | Yes |
| Human-Inspired | **Yes** (re-embedding) | **Yes** (decay-based) | No | No | Limited |
| ACT-R-Inspired | No | **Yes** (activation decay) | No | No | Simulation |
| FadeMem | No | **Yes** (adaptive exponential) | No | No | Yes |
| SYNAPSE | No | **Yes** (competition) | **Yes** (spreading activation) | No | Yes |
| HippoRAG 1/2 | No | No | **Yes** (KG + PPR) | No | Yes |
| *(Your project)* | ? | ? | **Yes** (build on HippoRAG) | ? | ? |

---

## What this means for the project direction

### The opportunity has narrowed but sharpened

Six months ago, "reconsolidation for LLM agent memory" was an empty gap. Now there are concrete systems doing parts of it. But:

1. **Nobody has done reconsolidation on a graph-structured memory system.** HiMem works on flat notes, Memory-R1 on flat records, Human-Inspired on embeddings. None operates on a KG. Combining reconsolidation with graph-based retrieval (HippoRAG's architecture) is still open.

2. **Nobody has combined reconsolidation with graph propagation (PPR).** SYNAPSE's spreading activation is the closest — but it implements forgetting, not reconsolidation. Adding reconsolidation to a PPR-based system would be novel.

3. **The benchmark exists now.** Memora (FAMA metric) lets you measure whether forgetting/reconsolidation actually helps. This was missing 6 months ago.

### Concrete positioning options

| Positioning | What you'd build | What you'd compare against | Novelty claim |
|---|---|---|---|
| "Reconsolidation for graph-based memory" | Add conflict-aware update-on-retrieval to HippoRAG 2's KG | HiMem (reconsolidation but flat), HippoRAG 2 (graph but no reconsolidation) | First system combining graph memory + reconsolidation |
| "Active forgetting via activation decay on KGs" | Add ACT-R-style activation levels to HippoRAG KG nodes/edges | FadeMem, SYNAPSE, HippoRAG 2 | First PPR-based system with decay-driven forgetting |
| "Full memory lifecycle on graph memory" | Both reconsolidation + forgetting on HippoRAG 2 | HiMem + FadeMem + HippoRAG 2 | Most ambitious; highest risk |

### Suggested reading priority

1. **HiMem** (2601.06377) — the most directly relevant reconsolidation mechanism. Read the reconsolidation trigger conditions carefully.
2. **SYNAPSE** (2601.02744) — the closest to combining forgetting with graph-based retrieval. Study how spreading activation relates to PPR.
3. **FadeMem** (2601.18642) — the strongest empirical case that forgetting *helps* performance (45% storage reduction, quality maintained).
4. **Memory-R1** (2508.19828) — the RL angle. Read to understand whether learned memory operations are feasible or overkill for your scope.
5. **"Rethinking Memory"** survey (2505.00675) — for the 6-operation taxonomy. Frame your contribution using their language.
6. **Memora** benchmark (2604.20006) — for the FAMA metric. Know how you'd be evaluated.
7. **ACT-R-Inspired** (HAI 2025) — for the activation formula. Quick read; the math is directly usable.

---

## Sources

- HiMem: https://arxiv.org/abs/2601.06377
- Memory-R1: https://arxiv.org/abs/2508.19828
- Human-Inspired Memory Architecture: https://arxiv.org/abs/2605.08538
- ACT-R-Inspired (HAI 2025): https://dl.acm.org/doi/10.1145/3765766.3765803
- FadeMem: https://arxiv.org/abs/2601.18642
- SYNAPSE: https://arxiv.org/abs/2601.02744
- Rethinking Memory survey: https://arxiv.org/abs/2505.00675
- SSGM Framework: https://arxiv.org/abs/2603.11768
- Memory for Autonomous LLM Agents: https://arxiv.org/abs/2603.07670
- Memora / From Recall to Forgetting: https://arxiv.org/abs/2604.20006
- Agentic Unlearning (ALU): https://arxiv.org/abs/2602.17692
- Survey on LLM Unlearning: https://arxiv.org/abs/2510.25117
