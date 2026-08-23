# GEPA — Study Notes (Q&A Session)

Consolidated from a study discussion of *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning* (arXiv 2507.19457, ICLR 2026 Oral). Companion to `gepa-reading-guide.md` (reading plan) and `notes.md` (raw first-pass facts). Paper: `papers/2507.19457-gepa.pdf`.

Part 1 records concept clarifications (with paper sections); Part 2 records reproduction planning — cost figures from App. E are paper facts, hardware/wall-clock figures are **our estimates** and should be revised once measured.

---

## Part 1 — Concepts clarified

### "Module" (§2, Algorithms 3–4)

One LLM call site in the pipeline, with its own prompt. Formally `M_i = (π_i, θ_i, X_i, Y_i)`: prompt, weights, input/output schema. Control flow `C` is plain code wiring modules together (maps 1:1 onto a DSPy program). Examples: HoVer = 4 modules (2 query writers + 2 summarizers); PUPA/PAPILLON = 2 (query rewriter + response rewriter); AIME = 1 (ChainOfThought).

Key distinction: a **candidate** in GEPA's pool is a *full system* — the vector `⟨π_1, ..., π_|M|⟩`. Each mutation rewrites exactly one module's prompt (round-robin choice). Merge (Alg. 3–4) exploits this: it recombines two lineages that evolved *disjoint modules* of a common ancestor, taking the evolved prompt per slot. In the experiments all modules share one underlying LLM (App. E.2) — they differ only in prompts.

### How scores are computed (Algorithm 1, lines 13 & 16–18)

`Φ'(x)` outputs text; the number comes from the external metric. Every instance is a pair `(x_i, m_i)` where `m_i` is evaluator metadata (gold answer, rubric, unit tests); `μ : Y × M → [0,1]` compares system output against it. Per benchmark: HotpotQA/AIME/LiveBench = answer match; HoVer = gold-document-set coverage of retrieved docs; IFBench = programmatic constraint checkers; PUPA = composite quality + PII-leak score (LLM-judged, from PAPILLON).

Two-tier evaluation:
1. **Minibatch gate (line 13):** `σ` = parent's avg on the 3-example minibatch (already available from the reflection rollouts), `σ'` = child's avg on the *same* 3 examples (paired comparison; asks "did the rewrite fix what we just reflected on?"). Rejected child costs only 3 rollouts.
2. **Full pass (lines 16–18):** accepted children only — all ~300 `D_pareto` instances with plain `μ`, filling the candidate's row in the scores matrix `S`. This feeds Pareto selection and the final argmax. It's also where most of the rollout budget goes (Obs 1) — the gate's noise (b=3) is corrected here, since a lucky child that leads no instance gets dominated and pruned.

`μ_f` = same evaluator, but returns `(score, feedback_text)` — it keeps the diagnostics (`which constraint failed`, `which gold docs missing`, compiler errors) that `μ` throws away. That text is the learning signal ("evaluation trace").

### "Pareto" / instance-wise Pareto sets (§3.1, Algorithm 2)

The parent-selection strategy — the "Pa" in Genetic-Pareto. Scores are kept *per validation instance*; each instance is treated as its own objective:
- `s*[i] = max_k S[k][i]`; `P*[i]` = all candidates achieving it (ties included) → one winner-set per example ("instance-wise Pareto sets").
- Union the sets; prune strictly *dominated* candidates (≤ everywhere, < somewhere — ties can hide dominated candidates, hence the explicit pruning);
- Sample the next parent ∝ number of instances it leads.

Effect: specialists survive. A candidate mediocre on average but best on some instance type keeps selection probability, so complementary strategies coexist and later recombine — vs greedy best-average selection, which stalls in a local optimum (Fig. 6a: one improvement, then a chain of rejected children). Largest ablation in the paper: Pareto +12.44 vs greedy +6.05 vs beam +5.11 aggregate (Table 3). Lineage: quality-diversity / MAP-Elites "illumination" (Mouret & Clune 2015). This is why the validation split is called `D_pareto`.

### "Few-shot optimizer" (Obs 2, Obs 4)

An optimizer that tunes the *demonstrations* in a prompt (worked input/output examples) rather than/in addition to the instruction. MIPROv2: bootstrap candidate demos from the system's own successful traces, then Bayesian-search (TPE) over instruction × demo-set assignments per module. GEPA optimizes instructions only. Findings: instruction-only now *wins* (reverses Wan et al. 2024; attributed to 2025-era instruction-following/reflection ability) and is far cheaper downstream — MIPROv2 prompts are up to 9.2× longer because demos are token-heavy and paid on every call.

### "Reflection LM" (§3, App. C, App. N)

LM = language model (= LLM here); the label denotes the *role*: the model inside the optimizer loop that reads (current prompt, traces, scores, feedback_text) wrapped in the App. C meta-prompt and writes the new instruction (`UpdatePrompt`, Alg. 1 line 11). Distinct role from the task LM executing the pipeline — but in the main experiments the *same model* fills both ("optimized for (and using) X", Table 2); Qwen3 8B reflects on its own traces. Pluggable in practice (`dspy.GEPA` exposes `reflection_lm`; common to reflect with a stronger model). Self-reflection without self-training: improvement lands in prompt text, never weights. App. N counts reflection calls separately — hundreds per run, cheap relative to rollouts.

### IFBench end-to-end example (App. E.1)

GEPA trains **one system per benchmark per model**; the artifact is the final prompt set. For IFBench: fixed 2-module pipeline (draft answer → rewrite to satisfy constraints); candidate = `⟨π_answer, π_rewrite⟩`. Data: (query, constraint list) pairs; 150 `D_feedback` / 300 `D_pareto` from IF-RLVR train; μ = constraint checkers; μ_f lists pass/fail per constraint in words. Loop as above; budget 3,593 rollouts (Qwen3 8B); final pair scores 38.61 on test vs GRPO 35.88 after 24,000 rollouts.

Why the test is a real generalization test: test = IFBench's 58 *new, out-of-distribution* constraint types, so surviving lessons must be general procedures (enumerate constraints, verify before finalizing, constraint > fluency), not per-constraint tricks. No cross-benchmark sharing; but the prompt artifact is text, so it transfers across models (Obs 6: Qwen-evolved prompts +9.00 aggregate on GPT-4.1 Mini, beating natively-optimized baselines).

---

## Part 2 — Reproduction planning

### Cost anchors (paper facts, App. E.3–E.4, Table 1)

- All GPT-4.1 Mini experiments: **< $500 total**. GEPA $86, GEPA+Merge $67, MIPROv2 $76, Trace+TextGrad $172.
- GEPA rollout budgets (Qwen3 8B): HotpotQA 6,871 | HoVer 7,051 | IFBench 3,593 | PUPA 2,426 | AIME 1,839 | LiveBench-Math 1,839. MIPROv2: 2,270–6,926. GRPO: 24,000 per benchmark.
- GRPO hardware: compound tasks = LoRA rank 16 on 1× H100/A100 80 GB + separate inference GPUs; single-module math tasks = full-parameter FSDP2 on 8 GPUs.

### Per-benchmark reproduction matrix (est. columns are ours)

| Benchmark | GEPA rollouts | Extra infra | Est. GEPA cost @ GPT-4.1 Mini |
|---|---|---|---|
| HotpotQA | 6,871 | Wikipedia retrieval index | ~$25 |
| HoVer | 7,051 | Same index; 4-module program | ~$26 |
| IFBench | 3,593 | Constraint-checker code (released) | ~$13 |
| PUPA | 2,426 | PAPILLON system + LLM judge | ~$9 |
| AIME-2025 | 1,839 | None (string match) | ~$7 |
| LiveBench-Math | 1,839 | Answer checker | ~$7 |

Cheapest meaningful check of the headline: GEPA vs MIPROv2 on IFBench + AIME with GPT-4.1 Mini — no GPUs, est. < $50.

### Hardware options assessed (estimates)

| | V100 32 GB | M4 64 GB | M4 18 GB |
|---|---|---|---|
| Qwen3 8B unquantized | fp16 (no bf16 on Volta) | **native bf16 = paper's checkpoint** | No — must quantize (Q8/Q4) |
| Batched throughput (8B) | ~500–2,000 tok/s (older vLLM pinned; Volta support dropped in modern stacks, no flash-attn) | ~100–300 tok/s (MLX / llama.cpp, actively supported) | ~15–25 tok/s single-stream |
| Wall-clock per benchmark (GEPA) | hours–1 day | ~1 day (math) to ~1 week (HotpotQA/HoVer) | days for AIME only; big benchmarks impractical |
| GRPO baseline | marginal science project | no (no CUDA RL stacks) | no |
| Verdict | speed | fidelity + convenience | orchestrator / small-model shakedown only |

Notes: paper's Qwen3 sampling settings (temp 0.6 / top-p 0.95 / top-k 20) are thinking-mode → verbose rollouts; budget for that. Mac: raise `iogpu.wired_limit_mb` (default Metal cap ~2/3 RAM); 64 GB also fits a Qwen3 32B Q8 (~34 GB) as a stronger local reflection LM. V100: sanity-check fp16-vs-bf16 scores on ~50 examples before trusting bulk runs. Best combo if both available: V100 = headless rollout server, M4 = orchestrator + retriever + reflector. GRPO column: rent an H100 (est. $100–400/benchmark LoRA; $300–800 full-param) or take the paper's numbers.

### Decided reproduction ladder

1. API track: baseline + GEPA + MIPROv2 on AIME + LiveBench-Math with GPT-4.1 Mini (est. < $50, no infra) — verifies the GEPA-vs-MIPROv2 headline and the AIME edge case.
2. Local track: same benchmarks + IFBench with Qwen3 8B (bf16 on M4-64 or fp16 on V100) — verifies the Table 1 GEPA column at $0 API cost.
3. Optional: HotpotQA/HoVer after standing up the Wikipedia index; GRPO column only if renting cloud GPUs.
4. Cross-topic experiment (see reading guide, study question 12): GEPA on memory-op prompts over LoCoMo vs Memory-R1's GRPO-trained manager.

---

## Part 3 — The official repo: gskill and agent skills

Source: https://github.com/gepa-ai/gepa (studied 2026-08-23: `src/gepa/gskill/`, `docs/docs/guides/{gskill,agent-skill,claude-cli-as-proposer}.md`, blog post "Automatically Learning Skills for Coding Agents", 2026-02-18, Tan & Agrawal et al., UC Berkeley). Two distinct "skill" features:

### gskill — GEPA learns repo-specific skills FOR coding agents

The paper's loop with the candidate swapped: evolve the *skill file text* injected into a coding agent (seed = empty string), not a module prompt inside a pipeline.

- **Tasks / where μ comes from:** SWE-smith mines real commits from any GitHub repo, injects bugs, emits hundreds of verifiable instances (problem statement + Docker env + tests). μ = tests pass (1.0) / fail (0.0).
- **Loop:** agent (default mini-SWE-agent + gpt-5-mini) runs task batches in parallel Docker containers with current skills; fitness returns `(score, side_info)` where side_info = patch + agent trace + test output — the paper's μ_f feedback-text idea, verbatim; a stronger reflection model (default gpt-5.2-pro) rewrites the skills. Budget ~600 metric calls; splits ~200 train / 50 val (Pareto) / 100 test. Two proposers: `batch` (all results at once) and `loop` (one failure at a time, then merge — more detailed skills, more LLM calls).
- **Deploy:** output is plain text (`best_skills.txt`) → dropped in as `CLAUDE.md` or proper `.claude/skills/<repo>/SKILL.md`. No retraining; works in any agent that reads skills.

**Results (blog, ~300 rollouts per repo):**

| Setting | jinja (Python) | bleve (Go) |
|---|---|---|
| Mini-SWE-agent (gpt-5-mini) baseline → +skills | 55% → 82% | 24% → 93% |
| Claude Code Haiku 4.5 baseline → +skills (transfer, unmodified) | 93.9% → ~100% | 79.3% → 98.3%, *faster* (173s → 142s) |
| Claude Code Sonnet 4.5 baseline → +skills | 100% → 98.5% (saturated) | 94.8% → 100% (285s → 169s) |

Takeaways: (1) biggest gains where the baseline struggles (bleve ≈ 4×); (2) skills learned with a cheap model on a toy agent transfer to a production agent across model families — Obs 6 cross-model transfer replayed at agent scale; (3) skills cut *duration/cost*, not just errors (less navigation flailing). Learned skills read like compressed operational procedures ("run `go test ./...` early, narrow with `-run TestName -count=1`, treat tests as the bug report, change one behavior at a time"). Stated future work: task sources beyond SWE-smith (their tasks skew simple), evolving skill *scripts*, non-SWE skills (computer use).

**Practical value for Claude Code/Codex:** an automated, data-driven `CLAUDE.md`/`AGENTS.md` generator grounded in what the agent actually got wrong on the target repo. Cross-topic note: gskill is *automated procedural-memory consolidation* (experience → reflection → durable text) — the language-space analogue of the RL-based memory-ops learning in `topics/llm-agent-memory/`.

### The shipped Agent Skill — teaching agents to USE gepa

`.claude/skills/gepa-optimize-anything/` (SKILL.md + references + templates + preflight script): auto-discovered by Claude Code in a repo clone (Cursor/Codex/Copilot/Gemini CLI read the same convention); installable anywhere via `/plugin install gepa-optimize-anything@gepa`. Teaches the agent to drive `optimize_anything` — the generalized API where GEPA is one engine among several — including mode choice (single-task / multi-task / generalization), feedback-rich evaluator design, and budget sizing.

Library has grown past the paper:
- `optimize_anything` engines: `gepa` (default), `best_of_n` (baseline), `autoresearch` and `meta_harness` — the latter two run a *Claude Code subprocess as the optimizer/proposer*. The relationship is bidirectional: GEPA improves agents; agents serve as GEPA's proposer.
- `reflection_lm` accepts any `(str) -> str` callable — documented pattern wraps `claude -p` (Claude subscription as the reflection LM, no API key).
- Candidate = "any string an evaluator can score": prompts, code/CUDA kernels, configs, regex/SQL, agent scaffolds, encoded search solutions.

## Open Questions
- Does fp16 (V100) meaningfully shift Qwen3 8B scores vs the paper's bf16? Measure before bulk runs.
- Reflection-LM cost accounting (App. N): does the 35× rollout advantage survive if reflection calls are priced at a stronger model's rates?
- How does GEPA behave when μ_f is nearly as opaque as the scalar (no decomposable checkers)? None of the six benchmarks test this.
- gskill: do learned skills overfit SWE-smith-style bug-fixing (the blog admits tasks skew simple)? Test on feature-add or refactor tasks.
- Could gskill's recipe learn *memory-management* skills (procedural memory) for a LoCoMo-style agent — connecting to the Memory-R1 comparison experiment?
