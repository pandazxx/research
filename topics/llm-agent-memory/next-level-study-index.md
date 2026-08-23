# Next-Level Study — RL-Driven Memory Policies (Nov 2025 – June 2026)

*Created 2026-06-14, expanded same day from 4 → 8 papers. Picks up from the Phase-1A extended-reading work and the kill/pivot gate at end of Phase 1C.*

---

## Why this batch, why now

Eight papers, all engaging the same question: **how do you decide what an agent's memory should store, retain, retrieve, and surface — and how do you *train* (or *not train*) that decision-making?** The batch spans Nov 2025 → June 2026 and gives the cleanest snapshot of where memory-policy research sits right now.

The original four-paper framing was "RL-trained memory policies." JitRL breaks that frame (it uses RL *math* without gradient training), and AgeMem / MemSearcher show the *same* broadcast-advantage GRPO trick under different names. The widened theme below is more accurate.

| Cluster | Papers | Common move |
|---|---|---|
| **Memory-as-tools (GRPO, broadcast advantage)** | AgeMem · MemSearcher · MemoPilot | Memory ops are tool calls; trajectory-level reward is broadcast to every turn |
| **Per-turn supervision** | MMPO · Memory-R2 | Sparse outcome reward isn't enough — add per-turn signal (Belief Entropy or local rerollouts) |
| **Decouple retrieval from end-to-end** | HORMA · Memory-T1 | RL only the retrieval/selection policy; reward = Jaccard against gold evidence |
| **No gradient training at all** | JitRL | RL closed-form (KL-constrained policy update) applied at inference time via logit modulation |

Cross-cutting observation: **broadcast-advantage GRPO is the default 2026 recipe** — AgeMem ("step-wise GRPO"), MemSearcher ("multi-context GRPO"), MemoPilot ("turn-wise GRPO with one-step proxy"), and HORMA (GRPO with Jaccard reward) are all instances. Memory-R2 then argues this entire pattern is *unfair* due to memory-induced non-stationarity and proposes LoGo-GRPO as the fix. **That critique applies to four of the eight papers here.** It is probably the most important single technical insight in the batch.

---

## Reading order (one focused week, ~14 hours)

The order is conceptual, not chronological. Read the *frame-setters* first, then specific instances. JitRL goes last because it's a different paradigm entirely and reframes the others retroactively.

| Day | Paper | Why this slot |
|---|---|---|
| 1 (Mon) | **Memory-R2** | Diagnoses the non-stationarity flaw that the GRPO cluster shares. Upstream of everything else. |
| 2 (Tue) | **MMPO** | The other "outcome-reward isn't enough" answer — per-turn shaping via self-supervised Belief Entropy. Composable with Memory-R2 in principle. |
| 3 (Wed) | **AgeMem** | Cleanest of the "memory as tools" trio. Six-tool API + three-stage curriculum. |
| 4 (Thu) | **MemSearcher** | Compact-memory paradigm. Same broadcast-advantage trick as AgeMem, different domain (search agent), much shorter context. |
| 5 (Fri) | **MemoPilot** | Frozen player + trained copilot. Architectural alternative to single-model training. ICML 2026. |
| 6 (Sat) | **HORMA + Memory-T1** | Both decouple retrieval from end-to-end optimisation. Both use Jaccard evidence-grounding rewards. Read together to spot the shared pattern. |
| 7 (Sun) | **JitRL + synthesis** | Forces the question: "do you need training at all?" Then write the synthesis paragraph in `notes.md`. |

**If you can only spend 3 days:** Memory-R2 + JitRL + AgeMem. Memory-R2 for the most important technical insight; JitRL for the most architectural disruption; AgeMem for the cleanest API design.

**If you can only spend 1 day:** Memory-R2 + JitRL. The other six all sit on the spectrum these two papers stake out.

---

## The eight papers

| # | Paper | arXiv | Date | Venue | Notes file |
|---|---|---|---|---|---|
| 1 | **Memory-R2** — Fair Credit Assignment for Long-Horizon Memory-Augmented LLM Agents | [2605.21768](https://arxiv.org/abs/2605.21768) | 2026-05-20 | — | `memory-r2-study-notes.md` |
| 2 | **MMPO** — Meta-Cognitive Memory Policy Optimization for Long-Horizon LLM Agents | [2605.30159](https://arxiv.org/abs/2605.30159) | 2026-05-28 | — | `mmpo-study-notes.md` |
| 3 | **AgeMem** — Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for LLM Agents | [2601.01885](https://arxiv.org/abs/2601.01885) | v2 2026-04-30 | — | `agemem-study-notes.md` |
| 4 | **MemSearcher** — Training LLMs to Reason, Search and Manage Memory via End-to-End RL | [2511.02805](https://arxiv.org/abs/2511.02805) | v2 2026-05-08 | ACL 2026 (per user; verify) | `memsearcher-study-notes.md` |
| 5 | **MemoPilot** — From Player to Master: Enhancing Test-Time Learning of LLM Agents via RL over Memory | [2606.08656](https://arxiv.org/abs/2606.08656) | 2026-06-07 | ICML 2026 | `memopilot-study-notes.md` |
| 6 | **HORMA** — Organize then Retrieve: Hierarchical Memory Navigation for Efficient Agents | [2606.11680](https://arxiv.org/abs/2606.11680) | 2026-06-10 | — | `horma-study-notes.md` |
| 7 | **Memory-T1** — Reinforcement Learning for Temporal Reasoning in Multi-session Agents | [2512.20092](https://arxiv.org/abs/2512.20092) | 2025-12 | ICLR 2026 (per user; verify) | `memory-t1-study-notes.md` |
| 8 | **JitRL** — Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates | [2601.18510](https://arxiv.org/abs/2601.18510) | revised 2026-06-08 | — | `jitrl-study-notes.md` |

All PDFs in `papers/` under their arXiv IDs.

---

## What you already know that supports this batch

| Already-studied | How it connects |
|---|---|
| [[hipporag2-study-notes]] | HORMA's tree-structured memory vs HippoRAG2's KG; Memory-T1's coarse stage echoes graph traversal |
| A-Mem (`2502.12110`, in `notes.md`) | Direct ancestor of AgeMem (notes → tools), MemoPilot (note-style memory as copilot output), Memory-R2 (note manipulation under RL) |
| Memory-R1 (`2508.19828`) | Predecessor to *all* training-based papers in the batch; Memory-R2 critiques it explicitly |
| MemAct (`2510.12635`) | Memory ops in chain-of-thought; AgeMem externalises into tools, MemoPilot externalises into a separate model |
| [[bpo-deep-dive]] | GRPO / PPO / KL-constrained policy optimisation fundamentals — load-bearing for understanding the algorithmic differences across the batch |
| [[brain-memory-deep-dive]] | Belief Entropy in MMPO is a coarse computational analogue of confidence-driven reconsolidation; useful for the reconsolidation-direction argument |
| `extended-reading-and-experiments.md` Theme 4 (CL for agents) | This batch is the 2026 instantiation of that theme |
| `extended-reading-and-experiments.md` Theme 6 (vector DB engineering) | Becomes critical for JitRL (NN search on growing experience buffer) |

---

## Cross-cutting questions to track while reading

Answer these across all eight, not per-paper. The answers are where the project's design decisions live.

### Q1. What is the unit of supervision?

| Paper | Unit |
|---|---|
| Memory-R1 | Trajectory (outcome only) |
| Memory-R2 | Session-local (anchored rerollouts) |
| MMPO | Sub-trajectory (Belief Entropy per turn) |
| MemoPilot | Game-turn (one-step proxy reward) |
| AgeMem | Step (broadcast from trajectory) |
| MemSearcher | Turn (broadcast from trajectory) |
| HORMA | Retrieval-call (Jaccard vs gold evidence) |
| Memory-T1 | Selection (multi-level: accuracy + Jaccard + temporal) |
| JitRL | None (no training; per-decision logit update from retrieved tuples) |

For a reconsolidation mechanism: probably **session-local with a Jaccard-style retrieval reward** — i.e. Memory-R2 × HORMA. None of the eight do exactly this.

### Q2. Architectural decoupling

| Paper | Player | Memory module |
|---|---|---|
| Memory-R1 / R2 / MMPO / AgeMem / MemSearcher / Memory-T1 / HORMA constructor | Same LLM | Same model with role prompts |
| MemoPilot | **Frozen, any LLM** | Separate trained copilot |
| HORMA retriever | Same LLM | Separate lightweight trained model (Qwen 3.5 4B) |
| JitRL | Frozen, any LLM | Non-parametric (no model at all) |

The decoupling spectrum runs from "single fine-tuned policy" (Memory-R1) → "separate trained retrieval module" (HORMA) → "separate trained memory copilot" (MemoPilot) → "no model at all" (JitRL). For a reusable library, the right end of the spectrum is more attractive.

### Q3. How is memory non-stationarity handled?

| Paper | Treatment |
|---|---|
| Memory-R2 | Explicitly diagnosed and fixed (LoGo-GRPO) |
| MemoPilot | Sidestepped via one-step proxy (per-turn reward, not multi-session) |
| MMPO | Sidestepped via dense per-turn rewards (less reliance on long-horizon comparison) |
| AgeMem / MemSearcher / HORMA | **Quietly suffer from it** (broadcast trajectory-level advantage) |
| Memory-T1 / JitRL | Not applicable (no end-to-end multi-session credit assignment) |

This is the single most important technical fault-line in the batch. Memory-R2 names it; four others don't address it.

### Q4. What's the evaluation overlap?

| Benchmark | Papers using it |
|---|---|
| LoCoMo | HORMA, Memory-R2, Memory-T1 |
| LongMemEval | HORMA, Memory-R2 |
| HotpotQA / multi-hop | AgeMem, MemSearcher, MMPO (RULER variant) |
| ALFWorld | HORMA, AgeMem |
| WebArena | JitRL |
| Games (RPS, LHE) | MemoPilot |
| Time-Dialog | Memory-T1 |

LoCoMo, LongMemEval, and HotpotQA are the shared core. **The project's eval harness should support all three** so that any future mechanism produces directly comparable numbers.

### Q5. What rewards / signals could you actually re-use?

- **Jaccard evidence grounding** (HORMA, Memory-T1) — needs gold evidence annotation; trivially reusable on LongMemEval / Time-Dialog.
- **Belief Entropy** (MMPO) — self-supervised, no labels; the most portable signal.
- **One-step proxy reward** (MemoPilot) — only works if next-turn outcome is informative; useful in games, less so in QA.
- **Temporal consistency** (Memory-T1) — needs timestamps; useful narrowly.

---

## Decision criteria — what would make this batch *change* the project's direction?

Read the eight papers; then answer:

1. **If JitRL replicates on LongMemEval at the level it replicates on WebArena, do you abandon training-based RL?** Honest yes/no.
2. **Memory-R2's curriculum delivered a +25 F1 gain by itself.** If you reproduce Memory-R1, is the official-baseline reproduction better done as "Memory-R1 + curriculum" rather than as-published?
3. **Does the "memory as text-tool calls" pattern (AgeMem, MemSearcher) make HippoRAG's KG-based memory look obsolete?** Or does the KG give you something tools can't — e.g. multi-hop traversal that the agent can't easily emulate via tool calls?
4. **None of the eight implements reconsolidation directly.** That's either:
   - Evidence the field has overlooked it (good for the project's positioning), or
   - Evidence it's a dead end nobody's pursuing for unstated reasons (bad — investigate).
5. **Three of the eight use the same broadcast-advantage GRPO that Memory-R2 says is biased.** Do they still beat their baselines? Yes. So how bad is the bias really?

Log answers to these in `project-roadmap.md` §11 by end of the reading week.

---

## Deliverables at end of the week

- One updated paragraph in `notes.md` under "LLM-as-memory-manager paradigm" extending the table with all eight entries.
- A `## Decision update (2026-06-XX)` line in `project-roadmap.md` §11 with the answers to the five decision-criteria questions above.
- One blog-post-shaped draft (or kept in `notes.md`): **"Where is RL-trained memory headed after Memory-R1? And does training matter at all?"** — could become Phase 1A's first blog post if HippoRAG reproduction is delayed. JitRL is the angle.
