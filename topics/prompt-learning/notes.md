# Prompt Learning — Raw Notes

Topic scope: optimizing LLM systems by learning in language space — prompt optimization, reflective evolution, textual gradients — as an alternative/complement to weight-space methods (SFT, RLVR/GRPO).

---

## GEPA (arXiv 2507.19457, ICLR 2026 Oral) — first-pass facts

Source: Agrawal et al., *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*. UC Berkeley / Stanford / BespokeLabs / Notre Dame / Databricks / MIT. v2 Feb 14 2026. PDF: `papers/2507.19457-gepa.pdf`. Reading guide: `gepa-reading-guide.md`.

### Method (§3)
- Genetic-Pareto search over the *prompts* of a compound AI system Φ = (M, C, X, Y); model weights frozen. Budget-constrained: at most B rollouts (Eq. 2).
- Loop (Algorithm 1): SelectCandidate (Pareto) → pick module round-robin → rollout on minibatch b=3 from D_feedback → μ_f returns (score, feedback_text) → reflection LM rewrites the module instruction (meta-prompt = one generic page, App. C) → keep child only if minibatch score improves → then evaluate on D_pareto (validation) and add to pool with ancestry.
- Pareto selection (Algorithm 2): per training instance, record best score across pool; keep candidates that achieve a per-instance best on ≥1 instance; prune strictly dominated ones; sample parent ∝ #instances it leads. Prevents the greedy local-optimum chain of Fig. 6a.
- Feedback function μ_f: extends metric μ with *evaluation traces* — compiler errors, per-hop retrieved-doc lists, per-constraint pass/fail. Distinct from *execution traces* (the LLM's own text). Can embed retrieval: NPU experiments retrieve manual sections keyed on compiler errors (§5.1).
- GEPA+Merge (App. D.1, Alg. 3–4): crossover of two Pareto-optimal lineages that improved *disjoint modules* of a common ancestor; invoked ≤5 times per run; conditions strict so merge is sparse.

### Setup (§4, App. E)
- Benchmarks: HotpotQA (multi-hop QA), IFBench (instruction-following, OOD constraints), HoVer (multi-hop claim verification), PUPA (privacy-aware delegation / PAPILLON 2-module system), AIME-2025, LiveBench-Math. Splits ≈ 150 train / 300 val / 300 test (PUPA 111/111/221; AIME trains on 2022–24, tests 2025×5 runs).
- Models: Qwen3 8B (temp 0.6) and GPT-4.1 Mini (temp 1.0); 16k context.
- GRPO baseline: 24,000 rollouts, 500 steps, LoRA rank 16 on 1×H100 (footnote 1 + Fig. 11: full-finetune comparison on HoVer shows the same gap). MIPROv2 auto=heavy (2,270–6,926 rollouts); GEPA budget capped to match MIPROv2 per benchmark (within ~10%).
- Cost (App. E.3): all GPT-4.1-mini experiments < $500 total — GEPA $86, GEPA+Merge $67, MIPROv2 $76, Trace+TextGrad $172.

### Results
- Table 1 (Qwen3 8B aggregate): Baseline 45.23 | GRPO 48.91 | MIPROv2 47.84 | **GEPA 54.85** | GEPA+Merge 52.40. GEPA avg budget 3,936 rollouts vs GRPO 24,000. IFBench: GEPA 38.61 @ 678 rollouts vs GRPO 35.88 @ 24k.
- Exception: AIME-2025 on Qwen3 8B — GRPO 38.00 > GEPA 32.00. Only benchmark where RL wins; math capability not fixable by instructions.
- Table 2 (GPT-4.1 Mini aggregate): Baseline 53.03 | Trace 56.30 | MIPROv2 58.67 | TextGrad 59.14 | GEPA 65.22 | **GEPA+Merge 66.36**. GEPA works on closed models (prompt-only).
- Cross-model (Obs 6): prompts optimized with Qwen3 8B, evaluated on GPT-4.1 Mini = +9.00 aggregate (up to +27.67 HotpotQA), beating all natively-optimized baselines.
- Obs 1 fine print: GEPA matches GRPO's best validation in 243–1,179 rollouts (≤78× efficiency); only 79–737 are train rollouts — the majority of budget is validation for candidate selection. Proposed mitigations: smaller or dynamically-selected validation subsets (future work).
- Table 3 selection ablation (Qwen3 8B): SelectBestCandidate +6.05 | BeamSearch(4) +5.11 | **Pareto +12.44** aggregate. The selection strategy, not reflection alone, drives much of the gain.
- Obs 4 / App. I: GEPA prompts up to 9.2× shorter than MIPROv2 (≈3× aggregate, Fig. 17–18); higher-performing optimizers trend toward shorter prompts; GEPA spends tokens on declarative instructions, MIPROv2 on few-shot demos.
- Obs 2 / Fig. 16: instruction-only optimization now beats instruction+demos (reverses Wan et al. 2024); generalization gap (val→test) smaller for GEPA than MIPROv2. Attributed to improved instruction-following/reflection in 2025 models.
- Obs 5: Merge helps GPT-4.1 Mini (up to +5%) but degrades Qwen3 8B on 3/4 tasks — budget split and *when* to invoke crossover unresolved; hyperparams were shared across models.

### Extended applications (§5)
- Inference-time search (overfit D_train = task list on purpose): NPUEval (AMD XDNA2) with GPT-4o: Sequential10 4.25% → +RAG 16.33% → +MIPROv2 19.03% → **GEPA 30.52% mean vector utilization** (single kernels up to 70%); one GEPA prompt alone gets 26.85% with no runtime RAG. KernelBench CUDA (35 tasks): fast₁ ~0% → >20% with budget ~3k rollouts. Temperature-0/cached, so gains come from prompt search not sampling luck.
- Adversarial prompt search (§5.2): invert reward → evolve a universal prefix (trivia distractors + strict format demand) that drops GPT-5 Mini AIME-2025 pass@1 from 76% → 10%; failure mode = model outputs the literal `### <final answer>` placeholder. Positioning: automated instruction-level stress tests / regression suites.

### My open questions (to resolve on second pass / experiments)
- How sensitive is GEPA to μ_f quality? All 6 benchmarks have decomposable programmatic metrics. What happens with an opaque scalar (e.g., human preference score)?
- Reflection-LM accounting: App. N counts reflection calls — check whether "35× fewer rollouts" survives if reflection calls (larger model? GPT-4.1 in loop?) are priced in.
- Per-instance Pareto needs instance-level scores — how to adapt for trajectory-level/aggregate metrics (one score per long conversation)?
- Cross-link: GEPA vs Memory-R1/MemSearcher GRPO — could GEPA optimize memory-op prompts on LoCoMo at ~$100 and how close does it get to Memory-R1's trained manager? (Study question 12 in the reading guide.)

## Open Questions (topic-level)
- Where is the boundary between "fixable by instructions" and "needs weight updates"? GEPA's AIME/Qwen3 loss is one data point; collect more.
- Do reflective optimizers compose with RL (Soylu et al. 2024 claim) — order of operations, and does prompt optimization shrink the RL budget needed?
