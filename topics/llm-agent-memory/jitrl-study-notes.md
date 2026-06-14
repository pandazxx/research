# JitRL — Study Notes

**Title:** Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates
**Authors:** Yibo Li, Zijie Lin, Ailin Deng, Xuan Zhang, Yufei He, Shuo Ji, Tri Cao, Bryan Hooi
**arXiv:** [2601.18510](https://arxiv.org/abs/2601.18510) · revised 2026-06-08
**PDF:** `papers/2601.18510-jitrl.pdf`

---

## TL;DR — what it claims

**No gradient updates. No fine-tuning. RL mathematics applied at inference time only.**

JitRL maintains a non-parametric replay buffer of `(state, action, return)` tuples from completed episodes. At each new decision point: retrieve top-k similar `(state, action)` tuples; compute Monte-Carlo Q̂ and V̂ from their returns; derive an advantage Â; add `β · Â(s, a)` directly to the LLM's logits. That's it.

The paper proves this additive logit update is the **exact closed-form solution** to the KL-constrained policy-optimisation objective. So it's not a heuristic — it's KL-regularised RL collapsed to a memory-lookup + arithmetic operation.

Headline: **60.0% on WebArena-Lite** vs WebRL's 46.06% — with **34× lower cost** (~$290 API vs ~$9,900 GPU).

## Why this matters for the project

This is the **most surprising paper in the batch and probably the most architecturally important** for a personal/open-source project. The other seven all train an LLM. JitRL doesn't. If you can match or beat trained-memory methods using only retrieval + logit math, the engineering, deployment, and reproducibility stories all become dramatically easier — exactly the differentiation the roadmap explicitly targets ("differentiate on engineering quality + benchmark rigour, not just novelty").

It is also the cleanest realisation of the "LLM-as-memory-manager" paradigm: every memory write is just a tuple append; every memory read is a similarity lookup; every memory-driven action is a closed-form policy update. The memory *is* the policy.

## Core mechanism

### 1. The memory itself

Non-parametric experience store:
```
M = {(s_i, a_i, G_i)}_{i=1..N}
```
where `s_i` is state (tokenised), `a_i` is action, `G_i` is the (Monte-Carlo) discounted return from that trajectory. Grows monotonically; no eviction in the basic version.

### 2. On-the-fly advantage estimation

At decision time, for current state `s`:
- Retrieve neighbours `N(s)` — top-k similar states via Jaccard similarity on token sets.
- Estimate value: `V̂(s) = mean(G_i for i in N(s))`.
- For each known action `a`: estimate Q-value `Q̂(s,a) = mean(G_j for j in N(s,a))`.
- Advantage: `Â(s,a) = Q̂(s,a) − V̂(s)`.
- Unseen actions get an optimism-under-uncertainty bonus with probability λ.

### 3. The closed-form logit update

Optimise:
```
π* = argmax_{π'} E_a[Â(s,a)] − (1/β) D_KL(π' || π_θ)
```
Frozen reference: pre-trained LLM `π_θ`. Lagrangian → softmax-of-logits gives:
```
π*(a|s) ∝ π_θ(a|s) · exp(β · A(s,a))
```
which in logit space is just:
```
z'(s,a) = z(s,a) + β · Â(s,a)
```

That's the entire learning rule. No backprop. No optimiser state. Each "step" is a retrieval + a few arithmetic ops.

## Headline numbers

**WebArena (training-free comparison):**

| Method | Avg / Final SR |
|---|---|
| Static | 35.63 / 36.30 |
| Memory | 41.36 / 43.00 |
| Reflexion | 41.08 / 42.12 |
| AWM | 39.37 / 40.32 |
| EvoTest | 39.24 / 42.49 |
| **JitRL** | **46.98 / 51.35** |

**WebArena-Lite (vs trained methods):**

| Method | Final SR |
|---|---|
| SFT (Llama-3.1-70B) | 23.0 |
| WebRL (Llama-3.1-70B) | 46.06 |
| **JitRL** | **60.00** |

**Jericho (text adventure games), final score:**

| Method | Library | Zork1 | Zork3 |
|---|---|---|---|
| Static | 0 | 10 | 10 |
| EvoTest | 4 | 26 | 54 |
| GRPO | 2 | 11 | 10 |
| **JitRL** | **5** | **30** | **69** |

**Cost:** ~$290 (JitRL inference) vs ~$9,900 (WebRL training). 34×.

## Reading questions

1. **What's the similarity metric's failure mode?** Jaccard on tokenised states is brittle to paraphrase. The fact that this works on WebArena (mostly DOM-string states) suggests states are sufficiently lexical. Would not transfer to natural-language conversation states without changes.
2. **What happens when the buffer is huge?** Nearest-neighbour search is the bottleneck. At 100K+ entries, this needs HNSW or similar — same engineering as your `extended-reading-and-experiments.md` Theme 6.
3. **What β is used, and is it task-tuned?** β controls how strongly retrieved advantages override base-LLM preferences. Too small = retrieval is noise; too large = base LLM is overridden. Almost certainly a key hyperparameter.
4. **Does the closed-form derivation assume on-policy advantages?** Real retrieved tuples come from older versions of the policy or even other agents. Whether this matters in practice (it shouldn't if the LLM is frozen) is worth verifying.
5. **Why does GRPO underperform Static on Jericho?** Trained GRPO at 2/11/10 vs Static at 0/10/10 is a tiny gain. That's a strong negative signal about training-based RL on these benchmarks — consistent with JitRL's pitch.

## Open issues / things to verify

- The "training-free" framing is partly marketing — there's no gradient training, but there is still an LLM call per retrieval to score actions. So compute *per decision* is higher than a vanilla agent.
- The optimism bonus for unseen actions is the only learned-RL-like piece. Worth checking how much of JitRL's gain comes from optimism vs from retrieval-based advantage estimation.
- The Q̂ estimate uses Monte-Carlo returns from completed episodes. If episodes are long, early-episode tuples have noisy `G`. They presumably address this; check whether.
- The KL closed-form solution is well-known in offline RL (advantage-weighted regression, AWAC, etc.). The novelty is *applying it without parameter updates*. Worth understanding what's actually new vs which mathematical machinery is borrowed.

## How this could affect Phase 1B / 1C

- **Strongest disruption candidate in the batch.** If JitRL replicates on a memory benchmark (LongMemEval, LoCoMo), the case for *any* RL-trained memory method weakens dramatically. The Phase 1B official-baseline choice should consider JitRL as a *training-free baseline* — if it ties or wins, the project's entire mechanism design should pivot.
- **Direct experimental call:** apply JitRL's formulation to the contradiction stress test (Experiment 2 in `extended-reading-and-experiments.md`). If retrieved `(s, a, G)` tuples can be modified post-hoc to support reconsolidation (e.g., adjust G when a contradicting later tuple arrives), JitRL becomes a *reconsolidation substrate*, not a competitor.
- **Library shape:** JitRL is ~50 lines of Python plus a retrieval index. That's a very different open-source product shape than a trained model — easier to maintain, easier to adopt, harder to differentiate via engineering complexity.

## Cross-references

- [[memory-r2-study-notes]] — opposite philosophy (train everything end-to-end with curriculum)
- [[memopilot-study-notes]] — also keeps the player frozen, but trains a separate copilot model. JitRL trains nothing.
- Memory-R1 (`2508.19828`) — JitRL implicitly argues you don't need any of this.
- `extended-reading-and-experiments.md` Theme 6 — vector DB engineering becomes load-bearing here.
- [[bpo-deep-dive]] — KL-constrained policy optimisation is the math behind both BPO and JitRL.
