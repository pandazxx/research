# MemSearcher — Study Notes

**Title:** MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End Reinforcement Learning
**Authors:** Qianhao Yuan, Jie Lou, Zichao Li, Jiawei Chen, Yaojie Lu, Hongyu Lin, Le Sun, Debing Zhang, Xianpei Han
**Affiliations:** Chinese Information Processing Laboratory (ISCAS), UCAS, Xiaohongshu
**arXiv:** [2511.02805](https://arxiv.org/abs/2511.02805) · v1 Nov 2025, v2 2026-05-08
**Venue:** ACL 2026 (per user; not surfaced in the arxiv page — verify in the published version)
**PDF:** `papers/2511.02805-memsearcher.pdf`

---

## TL;DR — what it claims

A search-agent that maintains **compact, question-relevant memory** between turns instead of concatenating full history. Result: **near-constant context length** across multi-turn interactions while ReAct-style baselines grow linearly.

The training contribution is **multi-context GRPO** — propagates trajectory-level advantages to every turn, even though each turn is conditioned on a different (compacted) context.

Beats Search-R1 across model scales: 3B → 7B → 14B all win on a 7-benchmark QA panel, while staying under 4K tokens per request.

## Why this matters for the project

MemSearcher is the *search-agent* corner of memory work — most other papers in this batch focus on long-horizon dialogue or RAG. It's worth studying because:

1. **It cleanly demonstrates "memory is the context, not a separate store."** The "memory" is just the compressed running context the model maintains as a text blob in `<memory>` tags. No vector DB, no graph, no KG.
2. **The token-cost story is hard to beat.** If the project ever needs to argue "this is better than long-context stuffing," MemSearcher gives the cleanest empirical demonstration: same task, fixed sub-4K context vs ReAct's growing context, better accuracy.
3. **Multi-context GRPO is the same broadcast-advantage trick as AgeMem's step-wise GRPO** — naming convergence is real. Memory-R2 then critiques both.

## Core mechanism

### 1. Single model, three roles via XML tags

One backbone LLM emits per turn:
- `<think>...</think>` — reasoning
- `<tool_call>...</tool_call>` — search engine call (or final answer in `\boxed{}`)
- After observation: `<memory>...</memory>` — updated compact memory

**Input at turn i:** only `(question q, previous memory m_{i-1})`. **Not** the full history. That's the entire compactness mechanism.

Action space per turn: binary — `search(query)` or `answer(\boxed{...})`.

### 2. Multi-context GRPO

The training challenge: each turn is conditioned on *different* contexts (because memory changes), so each turn is effectively in a different MDP. Vanilla GRPO can't compare them as a group.

The fix: compute the trajectory-level reward `R(τ)`, normalise across the rollout group, then **broadcast the trajectory-level advantage to every turn** of every rollout. Each turn's PPO update uses the same advantage.

This is identical in structure to **AgeMem's step-wise GRPO** and **MemoPilot's all-tokens-same-advantage** — three papers in this batch reach the same workaround independently. **Memory-R2's LoGo-GRPO argues this entire approach is biased** (different memory states = different envs = unfair comparisons). MemSearcher does not address that critique.

### 3. Reward design

Two-part terminal reward:
- **Format reward:** correct XML tags + `\boxed{}` presence
- **Answer reward:** F1 score vs ground truth

Composition:
- 0 if format invalid
- 0.1 if format valid but F1 = 0
- F1 otherwise

Trained on NQ + HotpotQA training splits. Group size 5.

## Headline numbers

Averages across 7 QA benchmarks (NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, Musique, Bamboogle):

| Model | Search-R1 | MemSearcher |
|---|---|---|
| 3B | 32.5 | **43.8** |
| 7B-base | 43.1 | **48.9** |
| 14B-base | 47.8 | **51.7** |

**Efficiency:** sub-4K tokens per request across multi-turn vs ReAct's linear growth. Paper reports specific token counts in Table 2 — verify when reading.

## Reading questions

1. **What does the `<memory>` blob actually look like in trained policies?** Is it free-text? Structured JSON? Question-decomposition? This matters for whether the technique generalises — opaque blob compositions are a hallmark of "the trained model figured something out we can't easily port."
2. **Is multi-context GRPO theoretically justified or just empirical?** The trajectory-level advantage is being broadcast to turns conditioned on different states. Memory-R2's argument says this is contaminated. MemSearcher just trains through it; the question is whether the gain is large enough that the bias doesn't matter at the model scales tested.
3. **What's the actual ACL 2026 acceptance status?** The arxiv page does not mention it. The v2 revision in May 2026 is consistent with a camera-ready prep. Confirm by checking ACL 2026 proceedings when they appear.
4. **Why does the 3B model see a larger relative gain (+11.3) than the 14B (+3.9)?** Smaller models benefit more from constrained context — they get less confused by noise. This is itself an interesting finding: memory compaction is more important when the base model is weaker.
5. **Does the agent ever choose to *not* update memory?** The architecture forces a `<memory>` tag every turn. That's wasteful if no new info arrived. Worth checking if the model learns to emit empty memory updates.

## Open issues

- "v2 is a minor revision" per the web extraction — verify by diffing v1 and v2 if you do a deep read; the multi-context GRPO formulation may have changed.
- The benchmark choice (7 QA datasets) is fair but narrow. No long-conversation eval (LoCoMo, LongMemEval). MemSearcher is search-agent territory; its generalisation to long-horizon dialogue is not tested.
- ReAct is the only strong baseline reported. Comparison against Search-R1 is more interesting (same paradigm, no compact memory) — that's where the gain comes from. The 11+ point lift at 3B is substantial; verify whether Search-R1 was trained with comparable compute.

## How this could affect Phase 1B / 1C

- **If the project decides to *not* use a vector DB at all** (compact text memory only), MemSearcher is the proof-of-concept. This is a real architectural fork worth considering — "memory" as just structured context that the model maintains itself, no external store.
- **Multi-context GRPO is cheaper than LoGo-GRPO** (no local rerollouts). If you train memory RL on a budget, this is the default recipe; if budget allows, LoGo-GRPO is the upgrade path.
- **For the comparison-dataset work:** add a MemSearcher-style "compact running memory" baseline alongside HippoRAG and A-Mem. It's a different paradigm entirely (no separate memory store) and would be a fair-fight on token cost.

## Cross-references

- [[agemem-study-notes]] — step-wise GRPO is the same broadcast-advantage trick, different name
- [[memopilot-study-notes]] — also broadcasts a per-turn advantage to all tokens; another instance of the same pattern
- [[memory-r2-study-notes]] — critiques all three of the above as fairness-violating; LoGo-GRPO is the fix
- Search-R1 — the direct baseline; worth a follow-on read
- Memory-R1 (`2508.19828`) — different paradigm (multi-session QA), not directly compared
