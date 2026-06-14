# Memory-R1 — Reading Guide

A guided read of *Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning* (Yan et al., 2025, arXiv v5 Jan 2026).

Paper: `papers/2508.19828-memory-r1.pdf` (in this repo) | arXiv: https://arxiv.org/abs/2508.19828

**Estimated reading time:** 60–75 minutes for the parts that matter to your project. ~2 hours end-to-end including appendices.

Related notes in this repo:
- `forgetting-reconsolidation-research.md` §"Memory-R1 (Aug 2025)" — already has a one-paragraph capsule
- `extended-reading-and-experiments.md` §3 — flags this paper as the place to think about RL-vs-rules for memory ops
- `wise-reading-guide.md` — same template you're reading now, applied to WISE

---

## Why this paper matters for your project

Memory-R1 is the cleanest published statement of "memory operations are a decision problem you can *learn*, not just prompt." Your project is in the same operation space (ADD / UPDATE / DELETE / NOOP, plus reconsolidation as a special UPDATE on retrieval), so this paper is both a **possible baseline** and a **possible direction**.

Two reasons to take it seriously:

1. **It is the RL counterfactual to your design.** You are leaning toward deterministic / biologically motivated update rules (decay, reconsolidation-on-retrieval). Memory-R1 is the opposite stance: *don't write the rules at all, let outcome-driven RL find them.* You need a clean answer to "why not just do what Memory-R1 does?" — and the answer should not be "I didn't read it carefully."
2. **It uses LoCoMo, which your roadmap already lists as a benchmark candidate.** Numbers in this paper are directly comparable to anything you build that reports LoCoMo F1/BLEU-1/J. If you ship a reconsolidation system that beats heuristics but loses to Memory-R1, that's a real result worth reporting; if you beat it without RL, even better.

The single most valuable thing you'll get from this paper: a concrete operating point for the trade-off between *learned* memory management and *hand-crafted* memory management — including data cost (152 QA pairs), training cost (4×H100, PPO/GRPO via VERL), and absolute LoCoMo numbers.

---

## The one big idea: outcome-driven RL replaces hand-crafted memory rules

Memory-R1 splits memory work into two policies and trains both with **answer correctness as the only reward signal** — no labels for what the "right" memory operation is.

| Stage | Policy | Action space | Reward |
|---|---|---|---|
| **1. Memory Construction** | *Memory Manager* π_θ(o, m' \| x, M_old) | {ADD, UPDATE, DELETE, NOOP} + updated content m' | EM of *downstream* Answer Agent on a QA over the new memory bank |
| **2. Memory-Guided Answering** | *Answer Agent* π_θ(y \| q, M_ret) | Free-form answer y after *memory distillation* over 60 retrieved memories | EM(y_pred, y_gold) |

Both are trained with **PPO** and **GRPO** (two variants reported separately). The key claim is that this is enough — no per-operation labels, no teacher trajectories — to beat every static-memory baseline they tried (LoCoMo-RAG, A-Mem, Mem0, MemoryOS) and a GPT-5-distilled supervised variant (Memory-SFT).

**Direct mapping to your project's design choices:**

| Memory-R1's stance | Your project's stance (current draft) |
|---|---|
| Learn the operation policy from QA outcomes | Write the operation policy as deterministic decay + reconsolidation rules |
| Flat memory bank of text records | Graph-structured memory (HippoRAG-style KG) |
| Update happens *at write time*, driven by new info `x` | Update happens *at retrieval time*, driven by query context (reconsolidation) |
| Sparse reward: only the final EM signals the manager | Dense signal: decay / activation evolve continuously per access |

You are not just "doing the same thing with rules"; you are operating on a different memory substrate (graph) and a different update trigger (retrieval, not write). State this distinction explicitly in your paper — Memory-R1's existence makes it sharper, not redundant.

---

## Memory-R1's pipeline in plain English

Stage 1 — **Memory Construction** (Figure 2, blue):
1. Each dialogue turn → Info Extraction (LLM summarises what's worth remembering) → `x`.
2. Search current memory bank → candidate set `M_old` related to `x`.
3. Memory Manager outputs `(operation, m')`. UPDATE keeps the same ID and writes a merged sentence; DELETE returns the same ID with event=DELETE; ADD mints a new ID; NOOP changes nothing.
4. Memory bank is mutated, conversation moves to the next turn.

Stage 2 — **Memory-Guided Answering** (Figure 2, green):
1. Question `q` arrives. Similarity-based RAG retrieves top-60 candidate memories `M_ret` (per participant) from the constructed bank.
2. Answer Agent applies *Memory Distillation*: a learned in-prompt selection of which retrieved memories are actually relevant.
3. Answer Agent generates `y` over `(q, distilled memories)`.

Two facts worth holding onto from the start:

- **Memory Manager training is bootstrapped by a *frozen* Answer Agent.** They alternate: manager trained → manager frozen → answer agent trained. They do not train both jointly (Limitations §). This is the part of the architecture that is most likely to break if you tried to scale or transfer.
- **Memory Distillation is just smart prompting + RL.** There is no separate reranker module. The "distillation" is an emergent behaviour of the Answer Agent learning to ignore distractors when its reward is EM. This is why §4.4 ablation (c) — "w/o Memory Distillation" — degrades performance: it's not removing a component, it's removing the *RL fine-tuning* that taught the agent to filter.

---

## Reading priority by section

If you have 60 minutes, read in this order:

### Must read (40 minutes)

| Section | Pages | Why |
|---|---|---|
| **Abstract + §1 Introduction** | 1–2 | The Andrew/Buddy/Scout case in Fig 1 is the whole motivation in one example. Hold onto it as a mental anchor. |
| **Figure 2 (pipeline)** | 4 | Memorise the two-stage layout. The rest of the paper is "we add RL to each box." |
| **§3.1 RL Fine-tuning for Memory Manager** | 3–4 | Eqs (1)–(4). Task formulation, PPO/GRPO objectives, and the EM-as-reward design. The reward design is the most copy-able idea. |
| **§3.2 RL Fine-Tuning for Answer Agent** | 4–5 | Eqs (5)–(6). Note that the Answer Agent reward is *also* EM, applied to the final answer — there is no intermediate "did you select the right memories" signal. |
| **§4.2 Main Results + Table 1** | 5–6 | LoCoMo numbers per question type (single-hop, multi-hop, open-domain, temporal). Pay attention to the *temporal* column — that's where memory-management actually matters. |
| **§4.4 Ablation Studies + Figure 5** | 7 | Each component (Manager, Answer Agent, Distillation) contributes. Removing the Memory Manager is the largest single drop. |
| **Appendix A.1 case study** | 11 | The Joanna/turtle-allergy example is even better than the Andrew/Buddy one. Read it carefully — it's the failure mode your reconsolidation pitch should be able to handle. |

### Skim (15 minutes)

| Section | Why skim |
|---|---|
| §2 Related Work | Standard list. Useful only to confirm you've also read these (you have — Mem0, MemGPT, A-Mem, MemoryBank, MemoryOS are all in your topic folder). |
| §4.3 Generalisation + Figure 3, 4 | LoCoMo → MSC + LongMemEval transfer. Skim the captions; the headline is "trained only on LoCoMo, still wins." Useful evidence that the policy isn't dataset-overfit. |
| Appendix B (datasets) + Appendix C (prompts) | Read the prompt in Figure 9–10 once — that's literally the Memory Manager system prompt. Useful if you want to reproduce or write a competing prompt-only baseline. |
| Appendix D (implementation) | 4×H100, VERL, lr 1e-6/1e-5, batch 128. Read once so you have a sense of the training cost. |

### Skip entirely (no loss for your purposes)

- Eq derivations of PPO importance ratio (§3.1 PPO box) and GRPO advantage (§3.1 GRPO box) — standard, not Memory-R1-specific.
- Figure 7 (training reward curves) — confirms GRPO converges faster; you don't need the curve.
- Figure 8 latency comparison — interesting but apples-to-oranges; the baseline rerankers aren't strong.

---

## What to actively extract while reading

Keep notes (literal or mental) answering these as you go:

1. **What counts as "one memory" in Memory-R1?** A single text sentence with an integer ID, stored flat. Not a node, not an embedding pair, not a structured fact. **Mapping to your project:** your unit is a KG triple (or a triple plus its attached note). The granularity mismatch is meaningful — UPDATE on a sentence is a string rewrite; UPDATE on a triple is an edge-weight change or a node merge. The RL reward signal would carry through both, but the action space looks very different.

2. **What signal tells the Manager an UPDATE is preferable to DELETE+ADD?** Only the downstream EM. There is no contradiction detector, no entity linker, no semantic similarity threshold. The model learns the distinction implicitly because UPDATEs that preserve old context lead to better answers later. **Implication for your project:** if your reconsolidation rules use *explicit* contradiction detection, the comparison story is "we trade learnability for interpretability." Be ready to say which one wins where.

3. **Why is the Answer Agent trained with EM and not LLM-as-Judge reward?** Table 2 (page 8) shows J-based reward gets the highest J score (63.58) but collapses F1 and BLEU-1 because it encourages verbose, descriptive answers that misalign with string overlap. They picked EM for *balanced* metrics. **Lesson:** if you eventually do RL on any part of your system, the metric you reward becomes the metric you maximise *at the expense of the others*. Pick the one whose externalities you can live with.

4. **How is the manager-vs-answer-agent training decoupled?** Manager is trained against a *frozen* Answer Agent baseline (text says "forwarded to the frozen Answer Agent"). Then they freeze the Manager and train the Answer Agent. They explicitly flag this as a limitation ("end-to-end multi-agent RL ... is promising future work"). **Why it matters to you:** if you ever sketch an end-to-end variant for a workshop paper, this is a *known* open problem with named priors — easy hook for framing.

5. **What does the LoCoMo-Temporal column tell you?** Table 1 — on LLaMA-3.1-8B, Memory-R1-GRPO gets F1 = 49.86 / J = 51.67 on Temporal questions vs Mem0's 28.74 / 22.64. Temporal is the question type that most depends on UPDATE doing the right thing. **Mapping to reconsolidation:** temporal questions are *exactly* the kind your reconsolidation-on-retrieval mechanism is supposed to help with. If you can post a stronger Temporal number with a simpler mechanism, that's a clean, defensible claim.

6. **What's the data cost?** 152 training QA pairs (single split from LoCoMo's 152/81/1307 train/val/test). That is *cheap*. Don't dismiss the method on grounds of "RL needs a lot of data" — this paper's whole point is that it doesn't, given a strong outcome-driven reward.

7. **What's the compute cost?** 4× H100 for 8B/7B; 8× H100 for 14B. VERL framework, joint actor+critic for PPO, actor-only for GRPO. Realistic for a small lab, not realistic for a solo researcher on a single GPU. Note this when you decide whether to *reproduce* Memory-R1 or just *cite* it.

---

## Direct mappings to your project

After reading, you should be able to answer:

### Mapping 1 — Memory-R1's ADD/UPDATE/DELETE/NOOP → your reconsolidation primitives

Their action space is symmetric over (write-time) lifecycle events. Yours, if you go reconsolidation-on-retrieval, is something like:

| Memory-R1 op | Reconsolidation analogue |
|---|---|
| ADD | Insert new triple / note at write time |
| UPDATE | **Two** distinct operations in your world: (a) UPDATE at write time (same as theirs) and (b) UPDATE at retrieval time (reconsolidation — this is the novel part) |
| DELETE | Hard delete vs *soft* delete via activation decay (FadeMem, SYNAPSE) — your project will likely prefer soft |
| NOOP | "Memory passed reconsolidation check" — explicit acknowledgement that retrieval re-evaluated and accepted |

Note the asymmetry: Memory-R1 has *one* UPDATE; your system has *two*. That's a structural distinction worth a bullet in your paper's positioning.

### Mapping 2 — Memory-R1's reward → what *your* outcome metric could be

Memory-R1 rewards downstream QA EM. If your reconsolidation is rule-based, you don't need a reward — but you still need an *evaluation* signal. Candidates:
- **EM on LoCoMo + MSC + LongMemEval** (directly comparable to Memory-R1 Table 1 / Figure 4).
- **FAMA from Memora** (captures "did the right memory survive and the wrong one get suppressed").
- **A locality / interference metric** (analogous to WISE's locality), which Memory-R1 does *not* measure and arguably should — see §"What Memory-R1 doesn't solve" below.

### Mapping 3 — Memory-R1's case studies → your stress tests

The two A.1 case studies (Andrew/Buddy/Scout; Joanna/turtle-allergy) are excellent test cases. Steal them. Hand-construct a dozen similar mini-dialogues where vanilla DELETE+ADD fragments memory and run *your* mechanism on the same inputs. If reconsolidation handles them with no learning, that's a clean qualitative win for the rules-based side.

### Mapping 4 — Memory-R1's Memory Distillation → your retrieval pipeline

Their Answer Agent learns to filter 60 retrieved candidates down to "useful" ones. In a graph-based system, this is closer to PPR's natural top-k cutoff — sparsity is already built in. **You probably don't need a learned distiller.** Be ready to argue why PPR's structural selectivity is competitive with or better than a learned filter.

---

## What Memory-R1 does *not* solve, and you might

The paper is reasonably honest. Concrete gaps:

1. **No retrieval-time update.** Memory-R1 makes all write decisions at ingest time. Once a memory is in the bank, it is only touched again if a *future* turn's information triggers an UPDATE/DELETE. There is no "this query made me realise the stored memory was misleading, let me revise it on the way out." That gap is literally the definition of reconsolidation.

2. **No graph structure.** Flat list of `(id, text)` records. No relations, no PPR, no entity linking. Multi-hop questions still work, but only because the LLM does the hopping in-context over the 60 retrieved memories. Your project, if it sticks with KG, is in a strictly different design space.

3. **No forgetting.** There is DELETE, but it fires only on explicit contradiction. There is no decay, no capacity bound, no scheduled forgetting. The bank grows monotonically unless contradictions arrive. Memora-style longitudinal eval would punish this.

4. **No locality metric.** Memory-R1 reports F1/BLEU-1/J on the *answered* question. It does not measure whether UPDATE-on-X corrupted Y. WISE has the framework for this (the impossible triangle); Memory-R1 ignores the locality leg. **This is a direct gap your project could fill** — instrument LoCoMo or build a small locality benchmark, then show that rules-based reconsolidation preserves locality where RL-learned policies don't (or do — the result would be interesting either way).

5. **Two-agent training is decoupled.** End-to-end multi-agent RL is named as future work in the Limitations section. Not your problem to solve, but worth knowing as a citation hook.

These five gaps are not all yours to fill — but #1, #3, and #4 are. The roadmap-aligned contribution is: *retrieval-triggered reconsolidation on graph-structured memory, evaluated with a locality metric Memory-R1 didn't report.*

---

## Study questions

After reading, you should be able to answer these. If you can't, re-read the relevant section.

**Section A — Comprehension:**

1. What is the action space of the Memory Manager, and what does each operation do to the memory bank? (Bonus: what does NOOP cost the agent, if anything?)
2. Walk through one Memory-R1 turn end-to-end: dialogue turn arrives → ... → memory bank state at t+1. Where does the LLM call happen? How many times?
3. What is the reward used for the Memory Manager, and *whose* output does that reward come from?
4. Why is the Memory Manager trained against a *frozen* Answer Agent rather than jointly?
5. What is "Memory Distillation," architecturally? Is it a separate module?

**Section B — Critique:**

6. The paper trains on 152 QA pairs and evaluates on three benchmarks. Is the 152-pair training set drawn from the same distribution as the eval sets? If yes, what does that imply for the generalisation claim?
7. Memory-R1 uses EM as its training reward. List two failure modes this incentivises that LLM-as-Judge would catch.
8. The DELETE operation fires when a new fact contradicts an existing memory. What happens in Memory-R1 if a memory becomes *stale* (no longer relevant, but not contradicted)? Is there any mechanism for that?
9. Suppose two facts arrive in opposite order: (A) "Andrew got a dog Buddy", (B) "Buddy passed away". Reverse the order: (B') "My dog Buddy is gone", (A') "Andrew adopted a pup Buddy". Will Memory-R1 produce the same final memory state? Why or why not?
10. Locality (WISE's term): updating fact X must not corrupt fact Y. Is there any experiment in Memory-R1 that measures this? If not, sketch one in three sentences.

**Section C — Application to your project:**

11. If you replaced your deterministic reconsolidation rule with a Memory-R1-style RL policy, what is your data cost? Your reward signal? Your action space? Be specific — name the dataset and the action set.
12. Memory-R1's flat memory bank vs your KG: name *one* operation that is trivial in one representation and hard in the other, in each direction.
13. If you reproduce Memory-R1 as a baseline, which model size / variant gives the fairest comparison to your system, and why?
14. If you wrote a paper with the framing "reconsolidation beats Memory-R1 on temporal questions with no RL," what's the single experiment that proves it? Sketch the table.

---

## After reading

Capture three things in your topic notes (you can append to `forgetting-reconsolidation-research.md` or start a fresh `memory-r1-takeaways.md`):

1. **One sentence describing what Memory-R1 learns that your rules don't encode** — and whether that gap matters for the questions on LoCoMo-Temporal.
2. **One design idea you're stealing from Memory-R1.** (Candidates: the EM-as-reward simplicity; the two-stage Manager/Answerer split; the 60→distilled retrieval pattern.)
3. **One thing you'll explicitly do differently from Memory-R1.** (Candidates: retrieval-time reconsolidation; graph substrate; locality metric; decay-based soft delete.)

These three sentences are the seed of your project's positioning relative to the RL-for-memory literature. They're what you'll lead with when someone says "but Memory-R1 already did this."
