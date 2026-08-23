# MMPO — Study Notes

**Title:** Meta-Cognitive Memory Policy Optimization for Long-Horizon LLM Agents
**Authors:** Ziyan Liu, Zhezheng Hao, Yeqiu Chen, Hong Wang, Jingren Hou, Ruiyi Ding, Yongkang Yang, Wence Ji, Wei Xia, Feng Liu
**arXiv:** [2605.30159](https://arxiv.org/abs/2605.30159) · submitted 2026-05-28
**PDF:** `papers/2605.30159-mmpo.pdf`

---

## TL;DR — what it claims

Outcome-only RL (Memory-R1, MEM1, RL-MemAgent) can't tell *which* memory summary along the trajectory was bad. MMPO adds a per-turn self-supervised signal — **Belief Entropy** — that probes how uncertain the model is about the latent task state given its current memory. Used as reward shaping inside group-relative PPO, it gives fine-grained, memory-specific supervision without needing extra labels.

Headline result: **+3.14% avg on RULER-HotpotQA (224K–3.5M)** over RL-MemAgent; **+9% on WebShop** over MEM1; **+23.4% on 16-objective QA** over MEM1.

## Why this matters for the project

This is the cleanest answer to "how do I get per-step credit without per-step labels?" — a question that comes up immediately if you try to train a reconsolidation mechanism with RL. The Belief Entropy signal is self-supervised and doesn't need annotators, which is the practical concession that makes per-turn rewards feasible on real benchmarks.

Composable with Memory-R2's LoGo-GRPO in principle (different axes of credit assignment), though no paper has yet combined them.

## Core mechanism

### 1. Belief Entropy as a meta-cognitive probe

After each memory update `m_t`, ask the agent (itself!) an anchor question:

> "Based on current memory, what is current task progress and what information is still needed?"

Take the greedy response `y*`. Compute token-level predictive entropy of `y*` under the model conditioned on `(m_t, anchor_q, y*_{<l})`:

```
Ĥ_BE(m_t) = (1/|y*|) · Σ_l Σ_v π(v|...) · −log π(v|...)
```

Intuition: a memory that gives the model a clear belief about the task state produces low entropy; a confused/missing memory produces high entropy. **Note this is unsupervised** — no external label.

Ablation on the anchor question design (Table 5-ish in paper):
- Direct-answer only: 78.17%
- Gap-only: 82.02%
- Progress + gap (their choice): **82.98%** ✓

The "progress" half anchors on what's been done; the "gap" half forces uncertainty over what's missing. Both halves earn their keep.

### 2. Dense sub-trajectory reward

For every sub-trajectory ending at turn k:

```
R_k = α · σ(−Ĥ_BE(m_k)) + r_final
```

where `σ` is the sigmoid (bounds the entropy contribution to (0,1)) and `r_final` is the terminal outcome reward (e.g. F1).

### 3. Turn-level advantage via group normalisation, then averaging

Within G rollouts, normalise per-k:

```
Â_k^(i) = (R_k^(i) − mean(R_k)) / std(R_k)
```

Then aggregate to per-turn:

```
A_t^(i) = (1/(T−t+1)) · Σ_{k=t..T} Â_k^(i)
```

This averages the advantage across all sub-trajectories that pass through turn t. The effect: a memory summary at turn t gets reinforced if *every* later sub-trajectory it enables looks better than the group baseline. Punishes summaries whose downstream consequences are only briefly good.

### 4. Wraps standard clipped PPO

Belief-Entropy supervision enters through the advantages, not as a separate loss. So MMPO is structurally just PPO with shaped rewards — easy to slot into existing RL pipelines.

Information-theoretic motivation (worth verifying): via chain rule, `H(y|m,q) = H(y|m,q,s) + I(y;s|m,q)`. The second term is what we actually want — uncertainty about the latent state s exposed through the anchor response. Lower response entropy ⇒ clearer summary-induced belief about s. Pearson correlation between trajectory-total entropy reduction and accuracy is reported as **r = −0.684**, which is a strong validation of the proxy.

## Headline numbers

**RULER-HotpotQA (vs RL-MemAgent), 7B backbone:**

| Context | RL-MemAgent | MMPO | Δ |
|---|---|---|---|
| 224K | 75.78 | 79.56 | +3.78 |
| 448K | 76.56 | 78.12 | +1.56 |
| 896K | 74.22 | 79.69 | +5.47 |
| 1.75M | 77.34 | 78.91 | +1.57 |

**Multi-objective QA (vs MEM1):**

| Setting | MEM1 | MMPO | Δ |
|---|---|---|---|
| 2-obj EM | 0.709 | 0.725 | +2.3% |
| 8-obj EM | 1.87 | 2.15 | +14.9% |
| 16-obj EM | 1.97 | 2.43 | +23.4% |

**WebShop:** MEM1 70.87 → MMPO 77.25 (+9%).

## ⚠️ Discrepancy to verify

The **abstract** claims "97.1% performance even when scaled to 1.75M-token contexts." Table 1 shows ~79% at 1.75M. The 97.1% number does not appear anywhere obvious. Two possibilities:
- It's *relative* performance: 78.91 / 81.30 (some shorter-context baseline) ≈ 97.1%. Plausible but unusual to frame this way.
- It's a different task variant or metric not shown in the main table.

**Action when reading: verify which it is.** If the framing is the relative interpretation, the abstract is misleading.

## Reading questions

1. **Is the anchor question hardcoded across all tasks?** If yes, how does that prompt generalise outside QA-style tasks?
2. **Is `α` learned, swept, or fixed?** If fixed across tasks, it's a brittle hyperparameter.
3. **Does Belief Entropy correlate with answer correctness, or with answer confidence?** A miscalibrated model produces low entropy even when wrong. The r = −0.684 partly addresses this, but check whether they correct for miscalibration.
4. **Does this composes with LoGo-GRPO from Memory-R2?** In principle yes — LoGo-GRPO fixes the group-comparison fairness, MMPO sharpens the supervision per turn. Worth thinking through how the advantages would combine.
5. **Why does the gap widen so much with objective count (2 → 16-obj)?** Sparse outcome rewards get sparser with more objectives; per-turn supervision helps more. This is the strongest qualitative claim — verify the numbers in detail.

## Open issues

- The anchor question is itself a memory probe — but the probe response is generated by the *same* LLM whose memory is being graded. Risk of pathological optimisation where the model learns to make the anchor answer confidently regardless of memory quality (entropy collapse). Check whether they detect / penalise this.
- The KL term `β · D_KL(π_θ || π_ref)` is standard PPO. Memory-R2 doesn't use it (it uses LoGo's structural fix instead). Either could regularise; check whether MMPO + KL approximates LoGo-GRPO's effect without LoGo.
- "Meta-cognitive" framing is mostly marketing — the paper does not engage substantively with metacognition literature. Don't lean on the name; lean on the formula.

## How this could affect Phase 1B / 1C

- **Reward-shaping handle:** if your reconsolidation mechanism eventually trains via RL, Belief Entropy is the cleanest off-the-shelf per-turn signal. It does not require labels.
- **Cheaper than LoGo-GRPO at training time:** no local rerollouts. If compute is tight, MMPO's recipe wins on cost.
- **Risk:** Belief Entropy is sensitive to model size and calibration. A 4B retriever (like HORMA uses) might not give a useful entropy signal at all. Verify on the model size you'd actually deploy.

## Cross-references

- [[memory-r2-study-notes]] — addresses a different RL pathology (group-comparison fairness); naturally composable
- Memory-R1 (`2508.19828`) — the outcome-only baseline this paper targets
- [[bpo-deep-dive]] — PPO / GRPO mechanics; KL regularisation
