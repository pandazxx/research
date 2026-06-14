# Memory-R2 — Study Notes

**Title:** Memory-R2: Fair Credit Assignment for Long-Horizon Memory-Augmented LLM Agents
**Authors:** Sikuan Yan, Ahmed Bahloul, Ercong Nie, Susanna Schwarzmann, Riccardo Trivisonno, Volker Tresp, Yunpu Ma
**arXiv:** [2605.21768](https://arxiv.org/abs/2605.21768) · submitted 2026-05-20
**PDF:** `papers/2605.21768-memory-r2.pdf`

---

## TL;DR — what it claims

GRPO-style RL is the standard recipe for training memory-augmented agents (Memory-R1, MemAgent, MEM1, etc.). Memory-R2 points out that this is **broken**: once different rollouts write different things to memory, they are no longer comparable, so the group-relative advantage is contaminated. Fixed via **LoGo-GRPO** (Global trajectory reward + Local session rerollouts from a shared anchor memory) + a **shared-parameter** fact-extractor / memory-manager + a **8 → 16 → 32 session curriculum**.

Trains on **just 2 LoCoMo conversations**. Outperforms Memory-R1 by **+7.46 F1 / +19.5 LLM-judge** on LoCoMo and generalises to LongMemEval, MSC-Self-Instruct, MemBench.

## Why this matters for the project

This is the most theoretically important paper in the batch. The non-stationarity argument applies to *any* RL-trained memory system, including a reconsolidation system: if updates change the memory and the memory shapes future observations, then standard group-relative comparisons mis-credit. If you train your reconsolidation mechanism with RL, you almost certainly inherit this bug. **Memory-R2's LoGo-GRPO is upstream of everything else in this batch.**

The 2-conversation training set is also a calibration point: large gains from a tiny train set means the reward signal is strong, not that scale is critical.

## Core mechanism

### 1. The non-stationarity problem (the diagnosis)

GRPO assumes rollouts are sampled from the same environment, so group-mean baselines are unbiased. With memory, this fails:

```
rollout 1: writes M_a → environment for session t+1 is (M_a, x_{t+1})
rollout 2: writes M_b → environment for session t+1 is (M_b, x_{t+1})
```

The advantage on session t+1's tokens conflates "good action" with "lucky upstream memory write." Memory-R1 trained through this — the paper argues that's why its gains plateau.

### 2. LoGo-GRPO (the fix)

**Global branch:** one full multi-session rollout, reward `r_G = R(M_T, Q_t, T)` from terminal memory — preserves end-to-end optimisation.

**Local branch:** for each session t (sampled with prob `p_local`), retrieve the cached anchor memory `M_{t-1}^{(i0)}` and run `m` local rerollouts of *only that session*, computing `r_L = R(M_t^{(i0,j)}, Q_t, t)`. All m rerollouts start from the same memory state, so the group-mean advantage is honest.

Unified PPO-style dual-clipped surrogate combines both. (Read §3 of the paper carefully — this is the contribution.)

### 3. Shared-parameter co-learning

Two roles, one Qwen2.5-7B-Instruct backbone, switched via prompt:
- **Fact extractor** `π_ext`: dialogue chunk → atomic, self-contained facts.
- **Memory manager** `π_mgr`: facts + current memory → `{INSERT, UPDATE, DELETE}` op.

Length-normalised step-level importance ratio (see §3.4) prevents the policy from cheating by emitting verbose extractions. Ablation: removing role-sharing costs −5.36 F1 / +26.28 M-Fail — significant.

### 4. Multi-step within a session (K chunks)

Each session is split into K=6 chunks; the extractor/manager pair alternates. Lets memory refine as more evidence arrives within a session instead of treating the whole session as one transition.

### 5. Curriculum (8 → 16 → 32 sessions)

Direct 32-session training: F1 = 0.27, M-Fail = 72.1% (collapse).
Curriculum: F1 = 0.50, M-Fail much lower.
Removing the curriculum is the single biggest hit in the ablation (−25.55 F1 / +39.78 M-Fail). This is the killer finding for anyone training memory RL from scratch.

## Headline numbers (vs Memory-R1, both on Qwen2.5-7B)

| Metric | Memory-R1 | Memory-R2 | Δ |
|---|---|---|---|
| LoCoMo overall F1 | 43.14 | 50.60 | +7.46 |
| LoCoMo BLEU-1 | 36.44 | 44.01 | +7.57 |
| LoCoMo LLM-judge | 61.51 | 80.99 | +19.48 |
| LongMemEval-oracle F1 | 27.88 | 50.60 | +22.72 |
| Temporal F1 | 47.75 | 59.65 | +11.90 |

Generalises across model scale (Qwen2.5-3B: 10.3 → 46.8 F1 — but that small a baseline is suspicious; check the table directly) and across answer-agent swap.

## Reading questions

1. **Is LoGo-GRPO's local branch unbiased, or just less biased?** The anchor memory is itself sampled from one rollout's history, so the local group still inherits that bias. Worth checking the theory section.
2. **The +19.48 LLM-judge gain is much larger than the +7.46 F1 gain.** Either Memory-R1 was much better than its F1 made it look, or the LLM judge is rewarding something other than answer correctness. Which?
3. **What's `p_local` set to?** Determines training cost. If it's 1.0, training cost scales with sessions × m, which gets expensive at 32 sessions.
4. **Can the curriculum be skipped if you start from a good checkpoint?** This determines whether the curriculum is fundamental or just a warm-start crutch.
5. **Why does open-domain F1 drop (23.55 → 20.76)?** Multi-hop, single-hop, temporal all up substantially; open-domain down. Could be ablation table swap noted in the paper; verify.

## Open issues / things to verify before adopting

- M-Fail metric definition: the paper coins it. Verify it's not gameable (e.g., by emitting longer extractions to up recall).
- LoGo-GRPO's reward variance reduction is asserted; check whether it has a stated theoretical justification or is purely empirical.
- The 2-conversation training set may be small enough that LoCoMo test-set leakage via persona overlap is a concern. Check whether eval avoids the same personas.
- All baselines reported use Qwen2.5-7B. What happens if Memory-R1 is given the same curriculum? That ablation isolates *just* LoGo-GRPO's contribution.

## How this could affect Phase 1B / 1C

- **Strongest pull on official baseline choice:** if you choose A-Mem as the warmup winner, Memory-R2's shared-parameter pipeline is essentially "A-Mem + trained extractor + trained manager." A direct extension path exists.
- **Curriculum learning is the cheapest borrow.** Even if you don't adopt LoGo, training on 8 sessions before 16 before 32 might rescue Memory-R1-style baselines you're reproducing.
- **Risk:** if the project goes graph-based (HippoRAG flavour), most of Memory-R2's machinery doesn't transfer cleanly — KG insert/update/delete is structurally different from note insert/update/delete.

## Cross-references

- Memory-R1 (`2508.19828`) — direct predecessor
- [[mmpo-study-notes]] — alternative supervision signal (Belief Entropy), arguably composable with LoGo-GRPO
- [[memopilot-study-notes]] — alternative architecture (frozen player + trained memory)
- [[bpo-deep-dive]] — GRPO / PPO mechanics
