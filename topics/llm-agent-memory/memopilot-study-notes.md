# MemoPilot — Study Notes

**Title:** From Player to Master: Enhancing Test-Time Learning of LLM Agents via Reinforcement Learning over Memory
**Authors:** Yishuo Cai, Xingyu Guo, Xuancheng Huang, Jinhua Du, Can Huang, Wenxuan Huang, Wenhan Ma, Yuyang Hu, Aohan Zeng, Jie Tang, Xu Sun
**arXiv:** [2606.08656](https://arxiv.org/abs/2606.08656) · submitted 2026-06-07
**Venue:** ICML 2026
**PDF:** `papers/2606.08656-memopilot.pdf`

---

## TL;DR — what it claims

Most memory-RL work trains the *whole* agent (memory + reasoning + acting). MemoPilot splits that in two: keep the **player frozen** (an off-the-shelf LLM that plays the game), and train **only a separate small memory copilot** that produces text guidance between games. Trained via multi-turn GRPO with a *one-step proxy reward* (next game's score) so credit isn't entangled with far-future stochasticity.

Beats DeepSeek-V3.2 as a memory model on both Rock-Paper-Scissors (Elo 1590) and Limit Texas Hold'em (Elo 1762). Generalises beyond games to StreamBench (CoSQL +4%, DS-1000 +6.3%).

## Why this matters for the project

Strongest *architectural* signal in this batch. The "frozen player + trained memory copilot" pattern means the memory system can be developed and deployed independently of the underlying LLM. If the project's goal is a reusable, framework-friendly library (per the roadmap), this decoupling is the right shape.

Also: training is on game outcomes, which are cheap, abundant, and unambiguous — a useful contrast to the LoCoMo-style benchmarks the other three papers depend on. Test-time learning in adversarial environments is a domain where you can actually *generate* arbitrarily much training data.

## Core mechanism

### 1. Architecture — what's frozen and what's trained

- **Player `π`**: any LLM (Qwen2.5-14B-Instruct in headline experiments; also tested on Qwen3-235B-A22B). **Completely frozen.** Stateless across games — only conditions on the current memory + game state.
- **Memory copilot `G_θ`**: separate trainable LLM. Reads game trajectory + previous memory → emits updated memory text. Three structured components:
  1. Diagnostic analysis / opponent identification
  2. Maintained belief state
  3. Actionable guidance

Critically: the player is *not* fine-tuned. All cross-game learning flows through what the copilot writes.

### 2. Multi-turn GRPO formulation

**Turn = one game** in a T-game episode. T−1 memory update steps (the first game has no learned guidance).

**One-step proxy return:** `R_{i,t} = r_{i,t+1}` — the reward from the *next* game only.

Why this matters: in poker, far-future returns are dominated by card-draw variance, not memory quality. Truncating to one-step proxy *removes the variance* that would otherwise drown the signal. This is a domain-specific choice that wouldn't transfer directly to a multi-session QA setting.

**Context-independent turn-level advantage:**

```
Â_{i,t,k} = R_{i,t} − mean({R_{i,t}}_{i=1}^G)
```

The advantage is applied **uniformly to every token in the memory generation `m_{i,t}`** — no token-level credit assignment. The paper claims this is finer-grained credit assignment than trajectory-level *and* more stable than token-level. Worth verifying — this is essentially a midpoint between Memory-R1 (trajectory) and full RLHF (token).

### 3. Training environments

- **Rock-Paper-Scissors (RPS):** 6-round matches. Adversaries are diverse strategy bots.
- **Limit Texas Hold'em (LHE):** standard limit poker. Duplicate matches (same cards both sides) reduce variance.

Both are 2-player. The memory copilot's job is to track opponent behaviour and tell the frozen player how to exploit it.

## Headline numbers (Qwen2.5-14B as player)

| Metric | No Memory | DeepSeek-V3.2 memory | MemoPilot | Δ vs DSV3.2 |
|---|---|---|---|---|
| RPS@5 | 0.43 | 1.64 | **3.28** | +1.64 |
| LHE@5 | −1.36 | −0.78 | **2.03** | +2.81 |
| RPS Elo | — | — | **1590** (first) | — |
| LHE Elo | — | — | **1762** (first) | — |

**StreamBench (generalisation outside games):**

| Task | No Memory | MemoPilot | Δ |
|---|---|---|---|
| CoSQL | 69.5% | 73.5% | +4.0% |
| DS-1000 | 50.0% | 56.3% | +6.3% |

**Key negative control:** prompt-based memory baselines using DeepSeek-V3.2 / Qwen2.5-14B *underperformed* no-memory baselines on StreamBench. Heuristic memory updates actively hurt outside of games. **Learned selective memory is necessary.**

## Reading questions

1. **What's the copilot's parameter count?** The paper's argument leans on the copilot being small enough to be a "plug-in." If it's another 14B, the deployment story is much weaker.
2. **Why does the one-step proxy work in poker but not (presumably) in long-horizon QA?** In QA, the relevant question may not arrive for many sessions. Generalising MemoPilot's reward design to non-game settings is non-obvious — check if they discuss this.
3. **"Context-independent turn-level advantage" — what does *context-independent* mean precisely?** That the advantage doesn't depend on the textual context of the memory? Or that it's pre-computed before policy updates? The paper's phrasing suggests the latter but the math should be checked.
4. **How does the player + copilot composition compare to giving the player ChatGPT-tier reasoning + no memory?** With Qwen3-235B-A22B as player + MemoPilot copilot, results are 3.27 (RPS@5), 1.31 (LHE@5). Comparable to 14B+MemoPilot at 3.28/2.03 — bigger player did NOT obviously help. Worth investigating.
5. **Why is StreamBench gain only ~4–6% when in-domain (games) the lift is much larger?** Probably because StreamBench tasks need less behavioural modelling of an adversary. Check whether the copilot was zero-shotted on StreamBench or fine-tuned again.

## Open issues

- "Test-time learning" framing is slightly oversold — the copilot is trained at training time, not test time. What's "test-time" is that the *frozen player* improves across an episode because the copilot's outputs change.
- Game-based training has a known limitation: opponent diversity gates what the copilot can learn. If you train against a small bot pool, the copilot may overfit to those bots. Check the opponent pool.
- The ICML 2026 acceptance is signal that the methodology is solid; expect a polished version of the paper.

## How this could affect Phase 1B / 1C

- **Strongest decoupling argument:** if the project's library should drop into existing agent frameworks (LlamaIndex memory module, DSPy module), MemoPilot's frozen-player + trained-memory shape is the cleanest fit. The memory module *is* the library.
- **Reward design transfer is hard.** The one-step proxy doesn't obviously generalise. For reconsolidation in QA, you might still need MMPO-style per-turn shaping or Memory-R2-style LoGo.
- **Game environments as a sandbox.** Even if the final mechanism isn't deployed to games, RPS/LHE could be a useful internal stress test — cheap, high-throughput, unambiguous reward.

## Cross-references

- MemAct (`2510.12635`, in `papers/`) — opposite shape: memory operations *inside* the player's chain-of-thought. MemoPilot externalises what MemAct internalises.
- [[memory-r2-study-notes]] — same training family (GRPO), different problem decomposition (long-horizon QA vs short-horizon games)
- [[mmpo-study-notes]] — alternative per-turn supervision signal
- Memory-R1 (`2508.19828`) — Memory-R1 trains a single agent end-to-end; MemoPilot's split is the architectural alternative.
