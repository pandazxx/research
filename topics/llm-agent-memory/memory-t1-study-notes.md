# Memory-T1 — Study Notes

**Title:** Memory-T1: Reinforcement Learning for Temporal Reasoning in Multi-session Agents
**Authors:** Yiming Du, Baojun Wang, Yifan Xiang, Zhaowei Wang, Wenyu Huang, Boyang Xue, Bin Liang, Xingshan Zeng, Fei Mi, Haoli Bai, Lifeng Shang, Jeff Z. Pan, Yuxin Jiang, Kam-Fai Wong
**Affiliations:** CUHK, Huawei Noah's Ark Lab, HKUST, University of Edinburgh
**arXiv:** [2512.20092](https://arxiv.org/abs/2512.20092)
**Venue:** ICLR 2026 (per user; not explicitly noted in the arxiv v1 abstract — verify on OpenReview)
**Code:** https://github.com/Elvin-Yiming-Du/Memory-T1/
**PDF:** `papers/2512.20092-memory-t1.pdf`

---

## TL;DR — what it claims

Temporal reasoning specialist. Coarse-to-fine retrieval over long multi-session dialogue:
1. **Coarse:** LLM predicts the query's temporal window → drop sessions outside it. BM25 ranks survivors by lexical relevance → candidate pool `C`.
2. **Fine:** RL-trained agent selects the precise evidence sessions from `C`, citing session IDs explicitly before generating the answer.

Reward: **multi-level** — answer accuracy + Jaccard evidence-grounding + temporal-consistency (chronological proximity + chronological fidelity).

Headline: **67.0% on Time-Dialog (7B)** — beats GPT-4-Full-Prompt (64.8%), Time-R1 (49.4%), MemAgent (49.9%), and Qwen2.5-14B (60.7%). 7B beats 14B by 6.3 points.

## Why this matters for the project

Temporal correctness is one of the project's stated open research directions ("retrieval that is temporally and causally correct, not only semantically similar" — see `summary.md`). Memory-T1 is the first paper in this batch that *specifically* tackles temporal reasoning rather than treating it as a sub-category of long-context QA.

Two design moves are worth borrowing regardless of mechanism:
1. The **temporal-window predictor** as a coarse filter (LLM-predicted, not learned-from-scratch).
2. The **Jaccard evidence-grounding reward** — same as HORMA's retrieval reward — proving this is an emerging standard for "did you select the right sessions" supervision.

## Core mechanism

### 1. Coarse stage — two filters

**Temporal filter:** Same backbone LLM predicts a target temporal window for the query. Sessions whose timestamps fall outside that window are dropped before anything else.

**Relevance filter:** BM25 ranks the temporally-filtered survivors by query overlap. Top-N forms the candidate pool `C`.

This stage is *zero-shot* — no training. The LLM's existing temporal/lexical reasoning is used as-is.

### 2. Fine stage — RL-trained session selection

Same LLM (now in policy mode) performs end-to-end selection + answering. The agent's output structure is:

```
<evidence>session_id_3, session_id_7, ...</evidence>
<answer>...</answer>
```

The structured citation makes evidence selection a first-class action that the reward function can grade independently.

### 3. Multi-level reward

Composite, weights (0.6, 0.2, 0.2):

| Component | What it grades | How |
|---|---|---|
| **R_a** (accuracy) | final answer correctness | task-specific: EM, unit-aware accuracy, ε-EM, Hamming |
| **R_g** (grounding) | evidence selection vs gold | Jaccard between selected and gold session sets |
| **R_t** (temporal) | timestamp + event consistency | sum of R_s (chronological proximity, logistic) + R_f (chronological fidelity, +1/+0.5/−1 per utterance overlap) |

R_a dominates at 0.6 weight; R_g and R_t are shaping signals. **R_t is the only piece in this batch that explicitly rewards temporal correctness.** The other seven papers either don't measure it or fold it into outcome accuracy.

## Headline numbers

**Time-Dialog (the headline benchmark):**

| Method | Overall score |
|---|---|
| Time-R1 | 49.4 |
| MemAgent | 49.9 |
| Qwen2.5-14B | 60.7 |
| GPT-4 Full Prompt | 64.8 |
| **Memory-T1 (7B)** | **67.0** |

**LoCoMo (OOD):** Memory-T1 (3B) at 37.7% overall, 36.7% non-RAG; strong on Temporal and Adversarial subtasks specifically.

**Long-context robustness:** at 64k–128k tokens, Memory-T1 holds while Qwen2.5-7B baseline drops 30+ F1 points → 25-point gap at the long end.

## Reading questions

1. **Where does temporal grounding come from in the LLM?** The coarse-stage temporal-window prediction is a strong claim. LLMs are notoriously bad at temporal reasoning without explicit dating. The paper probably preprocesses session timestamps into prompt-visible form — verify how.
2. **Is the same LLM used in coarse and fine stages, or different roles via prompting?** If same, that's a small footprint; if different, increases system complexity.
3. **The 7B-beats-14B claim is striking.** It's a fair fight only if the 14B baseline gets the same retrieval pipeline. Verify: is the 14B given the full history (likely) or also given the coarse-filtered candidate pool?
4. **R_t's chronological fidelity reward (+1/+0.5/−1) requires per-utterance temporal labels.** Where do those come from? If annotated, this won't scale to other datasets. If LLM-generated, it has its own noise floor.
5. **Why isn't temporal reasoning *the* metric all memory papers report?** Memory-T1 ranks 1st on this and beats GPT-4 by 2.2 points. That's a clean signal that the rest of the field is overfitting to lexical retrieval and missing temporal correctness.

## Open issues

- The user noted "ICLR 2026" — the arxiv abstract doesn't confirm. Could be:
  - Main conference accepted (look at OpenReview)
  - MemAgents workshop submission (the ICLR 2026 MemAgents workshop is real per search results)
  - User remembered wrong about venue
  Worth checking before citing.
- Coarse-stage failure modes: if the temporal-window predictor is wrong, the right session is dropped and never reaches the fine stage. There's no recovery path. Paper should discuss this; check whether they do.
- The reward weights (0.6 / 0.2 / 0.2) are likely tuned. Whether the gains hold under different weightings is the obvious ablation.

## How this could affect Phase 1B / 1C

- **Add a temporal-reasoning slice to the eval harness.** Even if the project's chosen direction is reconsolidation or active forgetting, temporal correctness is downstream of both. Memory-T1's evaluation methodology (Time-Dialog) is a cleaner test than LoCoMo's mixed-bag temporal subset.
- **The Jaccard evidence-grounding reward is the second appearance of this reward shape in the batch** (HORMA was the first). If the project trains any retrieval policy, this is now the safe default reward — used by two independent groups with positive results.
- **Coarse-to-fine framing is a real abstraction.** For reconsolidation work specifically: a *coarse* stage could be "find candidate memories to reconsolidate" (cheap, scalable), and a *fine* stage could be the actual reconsolidation operation (expensive, careful). Worth thinking about this two-stage structure as a design pattern.

## Cross-references

- [[horma-study-notes]] — same Jaccard evidence-grounding reward; HORMA's retrieval policy is a sibling design
- [[memory-r2-study-notes]] — temporal F1 is one of Memory-R2's strongest categories (+11.90 over Memory-R1); Memory-T1 specialises on the same axis
- Time-R1 (referenced as baseline) — worth a follow-on if temporal reasoning becomes a project focus
- `notes.md` benchmark section — LongMemEval's "temporal reasoning" capability is the existing canonical test; Time-Dialog is the new one
