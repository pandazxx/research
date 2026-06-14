# Next-Level Study — RL-Trained Memory Policies (May–June 2026)

*Created 2026-06-14. Picks up from the Phase-1A extended-reading work and the kill/pivot gate at end of Phase 1C.*

---

## Why this batch, why now

All four papers below were released within the last 4 weeks (May 20 – June 10, 2026) and all four train a memory policy with reinforcement learning. Together they form the cleanest snapshot we have of where the post–Memory-R1 (`2508.19828`) research frontier is moving.

If the project ends up choosing **reconsolidation** as its mechanism (per the §11 decision log default), the question of *how the memory policy is trained* becomes load-bearing. These four papers stake out four distinct positions on that question:

| Paper | What it adds over Memory-R1 |
|---|---|
| **HORMA** | Decouples the policy: a *retrieval* agent is RL-trained against an evidence reward; *construction* is improved via a non-RL "textual gradient descent" feedback loop. |
| **MemoPilot** | Frozen player + trainable memory copilot. Multi-turn GRPO with a *per-turn* (one-step proxy) reward. |
| **MMPO** | Keeps outcome rewards, adds a *self-supervised* per-turn signal — **Belief Entropy** — as reward shaping. |
| **Memory-R2** | Diagnoses GRPO's hidden flaw (memory-induced non-stationarity makes group comparisons unfair), fixes it with **LoGo-GRPO** + a session curriculum. |

The natural reading order is **Memory-R2 → MMPO → MemoPilot → HORMA**, because:

1. **Memory-R2** explains *why* naive GRPO over memory is broken — and the fix (LoGo) is conceptually upstream of everything else.
2. **MMPO** is the cleanest demonstration of a per-turn reward signal that isn't just outcome-based; it pairs naturally with Memory-R2's fairness fix.
3. **MemoPilot** shows the frozen-player formulation, which is a different *architectural* shape from Memory-R1 / R2 and changes how to think about credit assignment.
4. **HORMA** uses RL only for retrieval, not construction — read it last to see what falls out when you *don't* try to RL-train the whole pipeline end-to-end.

---

## What you already know that supports this batch

| Already-studied | How it connects |
|---|---|
| [[hipporag2-study-notes]] | HORMA's "structured memory" vs HippoRAG2's KG — compare what each represents and how each is retrieved. |
| A-Mem (`2502.12110`, in `notes.md`) | A-Mem is a write-time memory-evolution baseline; Memory-R2 and MMPO are its RL-trained successors. |
| Memory-R1 (`2508.19828`) | Direct predecessor to Memory-R2 (which targets its credit-assignment weakness) and MMPO (which targets its sparse-reward weakness). |
| MemAct (`2510.12635`) | MemAct embeds memory ops in chain-of-thought; MemoPilot externalises them into a separate model. Useful contrast. |
| [[bpo-deep-dive]] | BPO covers RL-for-LLM fundamentals; advantage normalisation and PPO clipping show up identically in MMPO/Memory-R2. |
| `extended-reading-and-experiments.md` Theme 4 (CL for agents) | Same intellectual neighbourhood — MMPO and Memory-R2 are the 2026 instances of that theme. |

---

## The four papers

1. **Memory-R2** — Yan et al. (LMU Munich + Huawei? — verify) — May 20, 2026 — `arXiv:2605.21768`
   → `memory-r2-study-notes.md`
2. **MMPO** — Liu et al. — May 28, 2026 — `arXiv:2605.30159`
   → `mmpo-study-notes.md`
3. **MemoPilot** — Cai et al. (Tsinghua / Peking + Z.ai) — June 7, 2026 — `arXiv:2606.08656` — **ICML 2026**
   → `memopilot-study-notes.md`
4. **HORMA** — Hsu, Kuang, Liu, Yao, He (DeepSpeed team?) — June 10, 2026 — `arXiv:2606.11680`
   → `horma-study-notes.md`

PDFs are in `papers/` under those arXiv IDs.

---

## Cross-cutting questions to keep in mind while reading

These are the questions that, if answered well, change the project's design space. Don't just answer them per-paper — track answers across all four:

1. **What's the unit of supervision?** Outcome only (Memory-R1), per-turn shaped (MMPO), per-session local (Memory-R2), per-game turn-wise (MemoPilot), per-retrieval Jaccard (HORMA)? Which most closely matches what *reconsolidation* would need?
2. **What's the actor architecture?** Single backbone with role-prompts (Memory-R2), frozen player + trained memory (MemoPilot), prompt-only constructor + RL retriever (HORMA), monolithic agent (MMPO)? Which decouples cleanly enough to be a drop-in module?
3. **Where does the evaluation actually live?** All four lean on LongMemEval / LoCoMo / RULER / MSC — so reproductions can share a harness. Plan around that.
4. **How is memory non-stationarity handled?** Only Memory-R2 names it explicitly. Do the others quietly suffer from the same problem? (Plausibly yes.)
5. **What would these methods look like if the underlying *memory representation* were graph-based (HippoRAG) rather than note-based (A-Mem)?** Mostly orthogonal — but worth tracking, because the chosen base architecture from Phase 1A determines which methods are easy to adopt.

---

## Suggested reading schedule (one focused week)

Assuming ~2 hours/day, ~14 hours total — roughly the budget for one paper read at depth.

| Day | Paper | Mode |
|---|---|---|
| 1 (Mon) | Memory-R2 §1–3 + skim experiments | Read + 1-paragraph summary |
| 2 (Tue) | Memory-R2 experiments + ablations | Fill in the study-notes "open questions" |
| 3 (Wed) | MMPO §1–3 + Belief Entropy formula | Verify the 97.1% claim against Table 1 — there's a discrepancy |
| 4 (Thu) | MMPO experiments + MemoPilot §1–3 | Cross-check MMPO's per-turn reward vs MemoPilot's one-step proxy |
| 5 (Fri) | MemoPilot experiments + HORMA §1–3 | StreamBench results are the most generalizable signal |
| 6 (Sat) | HORMA experiments + ablations + write-up | Update notes.md with cross-cutting answers |
| 7 (Sun) | Synthesis | Decide: do any of these change Phase 1B / 1C plans? Log in §11 of roadmap. |

If you can only spend 3 days on this batch: **Memory-R2, MMPO, MemoPilot** — skip HORMA. HORMA's contribution is mostly engineering; Memory-R2 and MMPO are the conceptual cores.

---

## Deliverables at end of the week

- One updated paragraph in `notes.md` under "LLM-as-memory-manager paradigm" extending the table with these four entries.
- A `## Decision update (2026-06-XX)` line in `project-roadmap.md` §11: did this batch change the Phase 1B official-baseline choice or the mechanism direction?
- One blog-post-shaped draft (or kept in `notes.md`): "Where is RL-trained memory headed after Memory-R1?" — could become Phase 1A's first blog post if HippoRAG reproduction is delayed.
