# HORMA — Study Notes

**Title:** Organize then Retrieve: Hierarchical Memory Navigation for Efficient Agents
**Authors:** Hao-Lun Hsu, Nikki Lijing Kuang, Boyi Liu, Zhewei Yao, Yuxiong He
**arXiv:** [2606.11680](https://arxiv.org/abs/2606.11680) · submitted 2026-06-10
**PDF:** `papers/2606.11680-horma.pdf`

(Author list suggests this is from the DeepSpeed / ZeRO group — verify when you open the PDF.)

---

## TL;DR — what it claims

Splits memory into two stages, trains only the second with RL:
1. **Construction (prompt-only, "textual gradient descent")** — organise raw trajectories into a *file-system-like* hierarchy: timestamped directories, structured notes containing compact abstractions + temporal metadata + pointers back to raw trajectories. Construction prompts are refined via a contrastive feedback loop (no RL needed).
2. **Retrieval (RL-trained lightweight agent)** — a small policy navigates the file system with Bash-like ops (`ls`, `grep`, `cd`, `cat`, `select`, `done`) under an evidence-grounded Jaccard reward. Frees up the heavy LLM by selecting *minimal yet sufficient context*.

Headline gain is on **token efficiency**: uses **3.07%–22.17%** of baseline tokens on LoCoMo and **1.24%–16.19%** on LongMemEval, while improving task performance under tight context budgets.

## Why this matters for the project

Most aligned paper with the "engineering quality + reproducibility" arm of the roadmap. The bash-command action space is borrowed from coding-agent work (think SWE-bench style) and turns retrieval into a *generic skill* that's easy to inspect and debug. This is a different bet from the other three — it argues that the highest-leverage place for RL is *retrieval*, not memory write/update.

The "construction = textual gradient descent" pattern is also notable: it's the [DSPy / TextGrad](https://arxiv.org/abs/2406.07496)-style approach extended to memory. Avoids RL where prompts suffice; uses RL where prompts don't.

## Core mechanism

### 1. Memory construction (no RL, "textual gradient descent")

Initial domain-agnostic prompt `P_m^(0)` tells an LLM to organise memory with entity tracking, event abstraction, relation grouping. Each interaction archives raw trajectories in timestamped directories, then synthesises structured notes (abstractions + metadata + references).

The prompt itself is refined by a contrastive analysis:
- `D_exo` = tasks where the *unstructured* history succeeds but the *managed* context fails → indicates information loss in summarisation.
- `D_end` = tasks where the *managed* context succeeds but the *unstructured* history fails → indicates managed memory mitigated lost-in-the-middle / hallucination.

Feedback per task:
```
Feedback_i = LLM(FeedbackInstruction, H_unstructured, H_managed)
```
Aggregated into a skill update:
```
P_m^(k+1) = LLM(SkillAugmentationInstruction, P_m^(k), {Feedback_i})
```

This is "textual gradient descent": the LLM rewrites its own prompt based on observed failures. **No reinforcement learning required for the constructor.**

### 2. Retrieval (RL-trained)

State: the file-system workspace `F_t` + query `q`.
Actions: `{ls, grep, cd, cat, mkdir, nano, mv}` + two terminals `{select, done}`.
Policy: a small model (Qwen 3.5 4B in ablations).
Reward: Jaccard similarity between selected context `C_t` and ground-truth evidence `E`:
```
J(C_t, E) = |C_t ∩ E| / |C_t ∪ E|
```

Trained with GRPO. The reward is **decoupled from downstream reasoning** — only the retrieval is graded. This avoids the credit-assignment ambiguity that hits end-to-end methods (and that Memory-R2 also worries about).

### 3. Why decoupled rewards work here

The construction stage produces structured notes with explicit pointers; retrieval is a graph/tree walk; the right answer is a finite set of nodes; Jaccard against ground-truth evidence is a clean signal. This is the *opposite* design choice from Memory-R2 (which optimises memory ops end-to-end with trajectory reward).

The catch: it requires *evidence annotations* for the training set. That's available on the benchmarks they use but not always elsewhere.

## Headline numbers

**ALFWorld** (134 tasks, success rate %):

| Context budget | Truncation | HORMA |
|---|---|---|
| Small (1,950 tok) | (low) | 56.7% |
| Large (2,200 tok) | (low) | 73.9% |

**LoCoMo** (519 QA, 10K-tok cap, LLM-judge):
- HORMA: 51.6%
- Truncation baseline: 47.8%
- Uses 3.07%–22.17% of baseline tokens.

**LongMemEval** (367 instances, 50K-tok cap, LLM-judge):
- HORMA: 55.9%
- Truncation: 34.1%
- Uses 1.24%–16.19% of baseline tokens.

The huge token reduction is the headline; raw accuracy gains are real but modest.

## Reading questions

1. **How is the file-system updated as new sessions arrive?** Memory is supposed to grow. Either the constructor re-runs (expensive) or the structure is mostly append-only with new directories per session. Verify.
2. **What does "out-of-distribution generalisation on LongMemEval" mean operationally?** Is the retriever trained on LoCoMo and zero-shotted on LongMemEval, or trained on a mix?
3. **What's the wall-clock latency improvement?** Token reduction doesn't always translate to latency — file-system navigation may add round-trips. Check if they report latency separately.
4. **Is the Jaccard reward gameable?** Selecting *all* evidence trivially maximises intersection but tanks the union ratio. The paper claims this incentivises minimal-yet-sufficient selection; verify there's no degenerate optimum.
5. **Can the constructor itself benefit from RL?** The paper argues no (TextGrad suffices) but doesn't ablate against an RL-trained constructor head-to-head. This is the obvious follow-up.

## Open issues

- The contrastive feedback loop requires *both* the unstructured and managed runs of every training task — 2× LLM cost during construction prompt refinement. Manageable but not free.
- "File-system" framing is a UX choice; the underlying structure is a tree with pointers. Whether the bash-command action space generalises beyond toy benchmarks is unproven. The paper notes the retriever "exhibits strong OOD generalisation on LongMemEval" — verify what that means exactly.
- Reward decoupling means the retriever can't learn anything reasoning-specific. If the downstream agent is bad at multi-hop and could benefit from *more* context, the retriever can't know.

## How this could affect Phase 1B / 1C

- **If the warmup baseline winner is HippoRAG:** HORMA's hierarchy concept maps poorly to KG-based memory (HippoRAG is a graph, not a tree). Borrowing is limited to the retrieval-as-navigation framing.
- **If the warmup baseline winner is A-Mem:** A-Mem's notes can be organised into HORMA's hierarchy without rewriting the memory model. A-Mem + HORMA hierarchy is a plausible "extended baseline" for Phase 1B.
- **For evaluation methodology:** the contrastive `D_exo` / `D_end` split is a *very* useful diagnostic. Even if you don't adopt HORMA's mechanism, this split tells you whether your memory system is losing information or surfacing better signal — actionable per-task data, not just average accuracy.
- **Token-budget evaluation:** HORMA reports performance *under context budgets*. The other three papers don't. Worth adopting in your own eval — it makes long-context-stuffing baselines comparable on a fair budget axis.

## Cross-references

- [[hipporag2-study-notes]] — graph-based alternative to HORMA's tree-based memory
- A-Mem (`2502.12110`) — note-based memory that could populate HORMA's hierarchy
- [[memory-r2-study-notes]] — opposite design philosophy (train *everything* end-to-end)
- [[mmpo-study-notes]] — same backbone optimiser (GRPO) used for a totally different subproblem
- MemAct (`2510.12635`) — also uses bash-style memory actions, but inside the reasoning agent rather than as a separate retrieval policy
