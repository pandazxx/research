# GEPA — Reading Guide

A guided read of *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning* (Agrawal et al., ICLR 2026 **Oral**; arXiv v2, Feb 2026).

Paper: `papers/2507.19457-gepa.pdf` (in this repo) | arXiv: https://arxiv.org/abs/2507.19457 | Code: https://github.com/gepa-ai/gepa

**Estimated reading time:** 60–90 minutes for the 14-page main body. The PDF is 96 pages, but pages 15–21 are references and most of the appendix (pp. 31–96) is search-tree figures and full prompt listings you sample rather than read.

Related notes in this repo:
- `topics/llm-agent-memory/memory-r1-study-notes.md` + `memory-r1-reading-guide.md` — the RL-for-memory counterpart; GEPA is the strongest published argument for the *other* side of that debate
- `topics/llm-agent-memory/memsearcher-study-notes.md` — another GRPO-trained agent; same sample-efficiency critique applies
- `topics/llm-agent-memory/bpo-deep-dive.md` — earlier prompt-optimization thread in this workspace

---

## Why this paper matters

GEPA is the current flagship result for a claim you keep running into from the RL side: **when adapting an LLM system to a task, learning in *language space* (reflecting on execution traces in natural language) can beat learning in *weight space* (policy gradients from scalar rewards) — and do it with 35× fewer rollouts.**

Three reasons to read it carefully:

1. **It reframes the RL-vs-rules debate you've been tracking in the memory topic.** Memory-R1, MemSearcher, Memory-T1 all say "don't hand-craft the policy, let GRPO find it." GEPA says: GRPO throws away almost all the learning signal by collapsing rich traces (reasoning chains, tool outputs, compiler errors) into one scalar reward. If the lesson is legible in text, an LLM can extract it from a *handful* of rollouts by reading, not from thousands by gradient estimation. Any future position you take on "learned vs scripted" agent behavior now has a third option: *learned, but in language*.
2. **It's the new state of the art in prompt optimization**, beating MIPROv2 (the DSPy default) on all 6 benchmarks × 2 models, with prompts that are also several times *shorter*. If you ever optimize a compound LLM pipeline, this is the method (it ships in DSPy and as a standalone library).
3. **The Pareto-selection idea is portable.** The single biggest ablation win (+12.44 vs +6.05 aggregate) comes not from reflection but from *how candidates are selected*: keep every candidate that is best on at least one training instance, instead of greedily evolving the global best. That's a general recipe against local optima in any LLM-driven search loop — code optimization, agent-design search, even memory-policy search.

The single most valuable thing you'll get: a concrete, controlled comparison of GRPO (24,000 rollouts, LoRA on Qwen3 8B) vs reflective prompt evolution (~700–6,900 rollouts) on the same tasks, same models, same metrics — with the mechanism-level explanation of *why* the gap exists.

---

## The one big idea: treat rollout traces as text to learn from, not rewards to average

A rollout of a compound AI system serializes naturally into language: module prompts, reasoning chains, tool calls, and the environment's own diagnostics (compiler errors, failed rubric items, per-hop retrieval feedback). GRPO reduces all of that to a scalar. GEPA instead feeds it to a reflection LM that performs **implicit credit assignment in natural language** — "the second-hop query module keeps paraphrasing the question instead of targeting the missing entity" — and rewrites the offending module's prompt accordingly.

The optimizer is a genetic search over *prompts only* (weights stay frozen):

| Component | What it does |
|---|---|
| **Candidate pool P** | Starts with the seed system; every accepted mutation is added with ancestry recorded, so lessons accumulate along the genetic tree |
| **Reflective mutation** | Sample a candidate, roll it out on a minibatch (b=3), collect traces + textual feedback via the feedback function μ_f, pick one module (round-robin), have a reflection LM propose a new instruction for it (meta-prompt in Appendix C) |
| **Pareto-based selection** | Track the best score *per training instance*; keep only candidates that win somewhere; sample the next parent ∝ how many instances it leads. This is the anti-local-optimum device (borrowed from quality-diversity / MAP-Elites thinking) |
| **Acceptance test** | New candidate must beat its parent on the minibatch before it earns a full evaluation on D_pareto (the validation split) |
| **System-aware Merge (GEPA+Merge)** | Crossover: combine two lineages that improved *disjoint modules* of a shared ancestor (Appendix D.1, Algorithms 3–4) |

Two definitions worth internalizing early (§3, "Evaluation traces as diagnostic signals"):
- **Execution trace** — text the *LLM system* produces (reasoning, tool calls).
- **Evaluation trace** — text the *environment/metric* produces while computing the reward (compiler errors, per-constraint pass/fail, per-hop document lists). The feedback function **μ_f** extends the scalar metric μ to return `(score, feedback_text)`. This is GEPA's interface to domain knowledge, and it's where the practitioner effort lives.

Formal setup (§2): a compound AI system Φ = (M, C, X, Y) with modules M_i = (π_i, θ_i, X_i, Y_i); GEPA optimizes the prompts Π_Φ under a rollout budget B (Eq. 2), leaving weights Θ_Φ frozen.

---

## Reading priority by section

### Must read (~50 minutes)

| Section | Pages | Why |
|---|---|---|
| **Abstract + §1** | 1–2 | The whole thesis: language is a richer learning medium than policy gradients. Figure 1's learning curves (GEPA nearly vertical, GRPO a slow crawl over 24k rollouts) are the paper in one image. |
| **Figure 2** | 3 | An actual GEPA-evolved prompt for second-hop query generation. Read it fully — notice it's *declarative task knowledge* ("first-hop docs cover one entity; target the missing linked entity"), not few-shot examples. This is what "lessons in language" means concretely. |
| **§2 Problem Statement** | 2–4 | The Φ = (M, C, X, Y) formalism and the budgeted-optimization framing (Eqs. 1–2). Five minutes; everything downstream uses this vocabulary. |
| **§3 + Algorithm 1 + Figure 3** | 4–6 | The full loop: select → mutate on minibatch → accept if improved → re-evaluate on D_pareto. Note where μ_f enters (line 10) and that module choice is just round-robin. |
| **§3.1 + Algorithm 2 + Figure 6** | 6–7, 10 | Pareto-based candidate selection. Understand the dominance-pruning + frequency-weighted sampling exactly — it's the part most worth stealing. Figure 6a vs 6b (degenerate chain vs balanced tree) is the visual argument. |
| **§4 Tables 1–3 + Observations 1–6** | 8–11 | The evidence. Read the observations as six separate claims (sample efficiency; instructions-beat-few-shot; selection strategy matters; shorter prompts; merge is model-dependent; cross-model transfer) and check each against its table. |
| **Figure 5** | 7 | The annotated evolution trajectory for PUPA (base prompt 82.26 → node 11, 97.6). Shows *what kind* of content accumulates: PII rules, output format, transparency rationale — each mutation adds a compressed lesson. |

### Skim (~20 minutes)

| Section | Why skim |
|---|---|
| §5.1 Inference-time search | Headline: GPT-4o NPU kernels go from 4.25% (10 sequential refinements) to 30.52% mean vector utilization with GEPA, beating RAG (16.33%) and MIPROv2 (19.03%); CUDA fast₁ from ~0% to >20%. The interesting mechanism: μ_f retrieves manual sections keyed on compiler errors — feedback engineering as a first-class design surface. |
| §5.2 Adversarial prompt search | Inverting the reward turns GEPA into a red-teamer: an evolved "trivia distractor" prompt drops GPT-5 Mini from 76% → 10% on AIME-2025 while preserving the task. Fun, and relevant if you care about robustness evals. |
| §6 Related Work | Situates GEPA vs EvoPrompt, AlphaEvolve/OpenEvolve, TextGrad, Trace, MIPROv2, Optimas. Good citation map for the new `prompt-learning` topic. |
| **Appendix C** (p. 22–23) | The *entire* reflection meta-prompt is one page of plain instructions. Read it once — it's demystifying: no chain-of-thought scaffolding, just "read the feedback, identify niche domain knowledge, write a new instruction." |
| **Appendix E** (p. 23–27) | Benchmarks, splits (~150 train / 300 val / 300 test), feedback function per task, GRPO config (LoRA rank 16, 500 steps, 24k rollouts, 1×H100), and costs: **all GPT-4.1-mini experiments under $500 total; GEPA itself $86**. Read E.1 for what μ_f concretely looks like per task. |
| Appendix D.1 + Algorithms 3–4 (p. 23–24) | Merge details. Skim unless you plan to implement crossover; the main-text caveat (helps GPT-4.1 Mini, hurts Qwen3 8B on 3 of 4 tasks) matters more than the pseudocode. |
| Figures 10–17 (App. F–I, p. 28–30) | Budget curves, generalization gap (GEPA's val→test gap is *smaller* than MIPROv2's — instructions now generalize better than exemplars, reversing Wan et al. 2024), and score-vs-prompt-size scatter. One pass over captions suffices. |

### Skip entirely (no loss)

- Pages 15–21: references.
- Figures 19–26 (App. J, p. 31–33): eight benchmark-by-model search-tree collages — Figure 6 already made the point.
- App. K.1 / L / M (p. 34–~95): full-length prompt listings for every intermediate node, benchmark, and kernel task. Sample one PUPA node (you already saw the arc in Figure 5) and one kernel prompt (Figure 27) if curious; do not read linearly.
- App. B: LLM-usage disclosure boilerplate.

---

## Key numbers to hold onto

| Claim | Number | Where |
|---|---|---|
| GEPA vs GRPO, Qwen3 8B aggregate | 54.85 vs 48.91 (baseline 45.23) — **+9.62 vs +3.68**, using ~3,936 avg rollouts vs 24,000 | Table 1 |
| Extreme case | IFBench: GEPA hits 38.61 in **678 rollouts**; GRPO reaches 35.88 after 24,000 | Table 1 |
| Sample efficiency | Matches GRPO's best validation in 243–1,179 rollouts (up to 78×); only 79–737 are *train* rollouts — most budget is validation for candidate selection | Obs 1 |
| GEPA vs prompt optimizers, GPT-4.1 mini aggregate | GEPA +12.19, GEPA+Merge **+13.33** vs MIPROv2 +5.64, TextGrad +6.11, Trace +3.27 | Table 2 |
| Selection ablation (the big one) | Pareto **+12.44** vs SelectBestCandidate +6.05 vs BeamSearch +5.11 aggregate | Table 3 |
| Prompt size | Up to **9.2× shorter** than MIPROv2 (roughly 3× on aggregate, Fig 17–18) — and shorter correlates with better | Obs 4, App. I |
| Cross-model transfer | Prompts optimized on Qwen3 8B → GPT-4.1 Mini: **+9.00** aggregate, beating every optimizer run natively on GPT-4.1 Mini | Table 2, Obs 6 |
| Where GEPA *loses* | AIME-2025 on Qwen3 8B: GRPO 38.00 > GEPA 32.00 — the one benchmark where weight updates win | Table 1 |
| Merge caveat | +Merge adds up to +5% on GPT-4.1 Mini but *degrades* Qwen3 8B on 3/4 tasks (budget-allocation and timing issue, flagged as future work) | Obs 5 |
| Cost | <$500 for all GPT-4.1-mini experiments; GEPA run = $86 | App. E.3 |

---

## What to actively extract while reading

1. **Where does the learning signal actually come from?** Not from the score — from `feedback_text` produced by μ_f (evaluation traces: compiler errors, per-hop doc lists, per-constraint breakdowns). Ask for each benchmark in E.1: how much *engineering* went into μ_f? This is GEPA's analogue of reward shaping, and the honest cost accounting of "35× fewer rollouts" should include it.

2. **Why does Pareto selection beat greedy so badly?** Figure 6a: greedy finds one improvement, then burns the entire budget failing to improve on it (a chain of rejected children). Pareto keeps "specialist" candidates alive because winning on *one instance* is enough to stay in the pool, so complementary strategies coexist and later recombine. Note the pruning step (strictly dominated candidates removed) and the sampling weight (∝ number of instances led). Transferable to any best-of-N-with-iteration loop you build.

3. **What exactly is a "rollout" in the accounting?** Both GEPA's minibatch/validation calls and GRPO's training rollouts count as one system execution + metric evaluation. Obs 1's fine print matters: most of GEPA's budget is *validation* (selection bookkeeping), and only 79–737 rollouts generate learning signal. Also note footnote 1: headline GRPO uses LoRA, but Figure 11 replicates the gap with full finetuning — closes an obvious objection.

4. **Instructions vs few-shot demos.** MIPROv2 optimizes instructions + demos jointly; GEPA writes instructions only, no demos, and wins while being ~3–9× shorter. Obs 2 attributes this to newer models' instruction-following and reflection abilities (and shows the generalization gap has flipped since Wan et al. 2024). Keep this dated: it's a claim about 2025-era models, not a law.

5. **Weights still matter sometimes.** AIME on Qwen3 8B is the counterexample worth remembering: competition math offers little "legible" domain knowledge for a prompt to encode, and the model's ceiling binds. Sketch a rule: **reflective prompt evolution wins when failures are diagnosable in text and fixable by instructions/knowledge; RL wins when the gap is capability, not specification.**

6. **The meta-prompt is trivially simple** (App. C). One generic template, no task-specific tuning. The sophistication lives in the *search structure* (Pareto pool, acceptance test, lineage), not in the reflection prompt. Good design lesson: put intelligence in the loop, not the prompt.

7. **Feedback-as-retrieval (§5.1).** For NPU kernels, μ_f uses compiler errors to retrieve relevant manual sections into the feedback text — the optimizer *pulls* domain documentation on demand, and the final evolved prompt then works with *no RAG at runtime* (26.85% vs 4.25% baseline). This "compile knowledge into the prompt" pattern is the most reusable idea in the extended-applications section.

---

## How GEPA sits against the RL-for-agents papers in this repo

| | Memory-R1 / MemSearcher (GRPO) | GEPA |
|---|---|---|
| Learning medium | Scalar reward → policy gradient into weights | Textual traces → reflection → prompt edits |
| Rollouts to adapt | ~10³–10⁵ | ~10²–10³ (78× reported best case) |
| What is learned | Opaque weight deltas (LoRA/full) | Human-readable declarative rules |
| Hardware | Multi-H100 training runs | API calls only (GEPA run: $86) |
| Works on closed models | No | Yes (GPT-4.1 Mini results, Table 2) |
| Transferable across models | No (weights are model-bound) | Yes (+9% Qwen-optimized → GPT-4.1 Mini) |
| Fails when | Reward sparse, rollouts expensive | Task knowledge not expressible as instructions (AIME/Qwen3), or no rich μ_f available |

Open synthesis question for your notes: the GEPA authors themselves (Soylu et al. 2024, cited in §2) argue finetuning and prompt optimization *compose*. Where does that leave a memory agent — GEPA-optimize the memory-op prompts first, then RL only if a gap remains?

---

## Study questions

**Section A — Comprehension:**

1. Walk one full GEPA iteration: which dataset split does the minibatch come from, what does μ_f return, which module gets updated and how is it chosen, and what two evaluations gate the candidate's entry into the pool?
2. In Algorithm 2, why are strictly dominated candidates pruned *before* frequency-weighted sampling? What failure mode returns if you skip pruning?
3. What is the difference between an execution trace and an evaluation trace? Give one example of each from HoVer and one from NPUEval.
4. GEPA never updates θ (weights). What are the three learnable-parameter regimes Eq. 1 admits, and which do GEPA, GRPO, and MIPROv2 each occupy?
5. What conditions must two candidates satisfy for Merge to combine them (App. D.1)? Why does the paper say merge "occurs sparsely"?

**Section B — Critique:**

6. "35× fewer rollouts" — what costs does this metric *exclude*? (Consider: reflection-LM calls (App. N exists for a reason), μ_f engineering, validation-set evaluations, per-token cost of a stronger reflection model.)
7. Obs 1 admits most of GEPA's budget is validation. If validation cost were the bottleneck, which two mitigations do the authors propose, and what selection risk does a smaller/dynamic D_pareto introduce?
8. The GRPO baseline got 24,000 rollouts and a fixed recipe (LoRA rank 16, one LR schedule, manual exploration of a few hyperparameters). Steelman the RL side: what would a GRPO practitioner change before conceding, and does Figure 11 (full finetuning) address it?
9. All six benchmarks have programmatic, decomposable metrics that yield rich feedback text. Name a task class where μ_f would be nearly as opaque as the scalar reward — what happens to GEPA's advantage there?
10. The prompts GEPA learns encode dataset-level regularities ("first-hop docs cover one entity"). When is that generalization, and when is it just distribution-fitting in words? How would you detect the difference (hint: the paper's own generalization-gap analysis, Fig 16)?

**Section C — Application:**

11. Design μ_f for a memory-agent benchmark (LoCoMo-style QA over a managed memory bank): what would the feedback text contain after a wrong answer, and which module (memory-write prompt vs retrieval prompt vs answer prompt) would reflection blame?
12. Sketch "GEPA-for-memory-ops": seed prompt = Memory-R1's manager prompt (their Appendix C), D_train = 150 LoCoMo QA pairs, μ_f = EM + which memories were retrieved. What do you predict relative to Memory-R1's GRPO numbers, and what would the result mean either way?
13. You have budget for exactly 500 rollouts on a new task. Using the paper's numbers, allocate: minibatch size, expected accepted mutations, D_pareto size. Where does it hurt most?
14. The Pareto trick needs per-instance scores. Your task has one aggregate metric over a whole conversation. How do you manufacture "instances" so the illumination strategy still works?

---

## After reading

Capture three things in `notes.md` (raw) and eventually `summary.md` (clean):

1. **One sentence on when language-space learning beats weight-space learning** — your own version of the rule in extraction point 5, with the AIME counterexample attached.
2. **One design element you're stealing.** Strongest candidates: Pareto per-instance candidate pools; μ_f feedback-as-retrieval; the acceptance test (minibatch-improve before full eval).
3. **One experiment this suggests for the memory topic.** The obvious one is study question 12 — it would connect this new `prompt-learning` topic back to `llm-agent-memory` with a single cheap run (GEPA's whole optimization cost $86 on a commercial API).

Follow-up reading, in priority order:
1. **MIPROv2** (Opsahl-Ong et al., 2024) — the baseline GEPA dethrones; needed to appreciate Obs 2 and 4.
2. **TextGrad** (Yuksekgonul et al., 2025) and **Trace/OptoPrime** (Cheng et al., 2024) — the "textual gradient" alternative lineage; GEPA beats both from a different design point (evolution vs backprop metaphor).
3. **MAP-Elites / illumination** (Mouret & Clune, 2015) — where Pareto-based candidate diversity comes from.
4. **AlphaEvolve** (Novikov et al., 2025) — evolution directly on code; contrast with GEPA's evolution on prompts across tasks.
5. **Soylu et al. 2024** — finetuning + prompt optimization as complements; the natural "GEPA then RL" follow-up.
