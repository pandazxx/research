# AgeMem — Study Notes

**Title:** Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents
**Authors:** Yi Yu, Liuyi Yao, Yuexiang Xie, Qingquan Tan, Jiaqi Feng, Yaliang Li, Libing Wu
**Affiliations:** Wuhan University, Alibaba Group
**arXiv:** [2601.01885](https://arxiv.org/abs/2601.01885) · v2 revised 2026-04-30
**PDF:** `papers/2601.01885-agemem.pdf`

(Already cross-referenced in `notes.md` under "LLM-as-memory-manager paradigm." This is the full study version.)

---

## TL;DR — what it claims

Exposes **memory operations as tool calls** (Add / Update / Delete / Retrieve / Summary / Filter — six tools, the abstract's "five" is sloppy) and lets the agent's RL-trained policy decide *when* to invoke each. Unifies LTM and STM in a single policy rather than treating them as separately-controlled modules.

Trained with a three-stage progressive curriculum (acquire → distract → query) and step-wise GRPO where the terminal trajectory reward is broadcast back to all preceding memory operations.

**+13.91 pp average across five long-horizon benchmarks (ALFWorld, SciWorld, PDDL, BabyAI, HotpotQA)** vs Qwen2.5-7B no-memory baseline; **+4.82 pp over Mem0**.

## Why this matters for the project

AgeMem is the *cleanest API design* in the batch. Memory operations as named, schema-defined tools is the same shape Letta/MemGPT chose. If the project decides to ship as a library, AgeMem's interface is the most likely to be adopted by downstream agent frameworks.

The three-stage curriculum directly answers a problem the project will face: how do you train memory operations when the task reward only fires at the end? AgeMem's answer (reset context between acquisition and query so the agent *must* use LTM, not residual context) is a clever engineering choice that doubles as an evaluation methodology.

## Core mechanism

### 1. The six memory tools

| Tool | Target | Input → Output | When the policy fires it |
|---|---|---|---|
| **Add** | LTM | content + metadata → memory ID | Stage 1 — salient fact arrives |
| **Update** | LTM | mem ID + new content → updated entry | New info refines/supersedes |
| **Delete** | LTM | mem ID → ack | Entry becomes stale or wrong |
| **Retrieve** | STM | (q, k) → top-k memories | Stage 3 — need knowledge for answer |
| **Summary** | STM | context span → compressed text | Approaching context overflow |
| **Filter** | STM | (criteria, θ_f) → filtered context | Stage 2 — suppress distractors |

The policy chooses one tool per step. All tools are emitted as tool-call JSON in the standard agent loop. No bespoke API surface needed beyond the LLM's existing tool-calling.

### 2. Three-stage progressive curriculum

| Stage | Setup | What the agent must learn |
|---|---|---|
| 1. Acquisition | Casual interaction with information `I_q` provided | Identify and Add salient facts to LTM |
| 2. Distraction | **Context reset.** Distractor messages injected | STM control: Filter/Summary to suppress noise |
| 3. Query | Final question presented | Retrieve from LTM + manage STM + answer |

**Key design choice:** LTM persists across stages while context resets between Stage 1→2. This forces genuine retrieval rather than letting the agent solve everything from residual context — a methodological echo of HORMA's `D_exo`/`D_end` contrastive split.

### 3. Step-wise GRPO

Terminal reward `R(τ)` group-normalised across rollouts:
```
A_t(k,q) = (r_T(k,q) − μ_{G_q}) / (σ_{G_q} + ε)
```
Then **broadcast to all steps** in the trajectory. Every memory operation in stages 1 & 2 gets the same scalar advantage as the final answer step.

This is the same broadcast-style credit assignment as **MemSearcher's multi-context GRPO** and is structurally similar to **MemoPilot's "all tokens get the same per-turn advantage."** It's the easiest way to handle sparse rewards in tool-call-heavy trajectories, and it's a common 2026 pattern. **Memory-R2's LoGo-GRPO is what eventually replaces this when fairness matters.**

Reward composition (Eq. 7 in paper):
```
R = w_task · R_task + w_ctx · R_context + w_mem · R_memory − penalties
```
The dense shaping (`R_context`, `R_memory`) is what saves the broadcast advantage from being too noisy.

## Headline numbers (Qwen2.5-7B backbone)

| Benchmark | No-memory | AgeMem | Δ |
|---|---|---|---|
| ALFWorld | 27.16% | 41.07% | +13.91 |
| SciWorld | 13.80% | 35.55% | +21.75 |
| PDDL | 10.15% | 17.31% | +7.16 |
| BabyAI | 50.80% | 61.42% | +10.62 |
| HotpotQA | 38.36% | 54.44% | +16.08 |
| **Average** | 28.05% | 41.96% | **+13.91** |

vs strongest baseline (Mem0): **+4.82 pp average gain**.
On Qwen3-4B: 54.31% average vs Mem0 44.70% → +23.5% relative.

## Reading questions

1. **Six tools vs the abstract's "five" — which one is the discrepancy?** Possibly Filter is considered a sub-mode of Summary. Worth checking which tool was actually used.
2. **How does the reward shape avoid the agent gaming "tool abuse" penalties?** The paper mentions penalties for overflow/tool abuse but doesn't fully specify. Tool abuse is the obvious failure mode when each tool call gets the same broadcast advantage.
3. **Is the LTM persisted as raw text, embeddings, or structured records?** This is implementation but determines whether AgeMem composes with HippoRAG-style KGs or A-Mem-style notes.
4. **Stage 3's "context reset" is methodologically clever but unrealistic for deployment.** In production, you don't reset context. Does the agent learn behaviours that only work because of the reset?
5. **Why does PDDL gain so little (+7.16) vs SciWorld (+21.75)?** PDDL is symbolic-planning-heavy; SciWorld is exploration-heavy. The memory machinery should help both — unless retrieval is too brittle for symbolic queries.

## Open issues

- The April 30 revision adds Qwen3-4B and Qwen3-14B results; check what other content changed in v2. If the method or training recipe changed, prefer v2's numbers over v1's.
- Step-wise GRPO is essentially what MemSearcher calls "multi-context GRPO" — broadcasting trajectory-level rewards to all turns. **AgeMem and MemSearcher are doing the same trick under different names.** Cross-cite carefully.
- Five-stage training is a lot of moving parts. Ablations should show which stages are load-bearing. Verify whether the paper does this.

## How this could affect Phase 1B / 1C

- **API design template.** If the project ships as a library, AgeMem's six-tool interface is a direct candidate. Letta has a similar surface; combining yields a small de-facto standard.
- **Methodology to borrow:** the Stage 1→2 context reset is a clean evaluation move — it isolates *memory* from *context*. Useful even if you don't adopt AgeMem's mechanism. Should be added to the project's eval harness.
- **Risk against reconsolidation direction:** AgeMem's Update operation is a simple overwrite. Reconsolidation in biology is *partial*, history-aware, and stochastic. If you choose reconsolidation, you're betting that "Update" as a single atomic tool call is insufficient — the agent needs structured access to *how* memory is modified, not just *that* it's modified. Worth being able to articulate this.

## Cross-references

- [[memsearcher-study-notes]] — same broadcast-advantage trick (different name: "multi-context GRPO")
- [[memory-r2-study-notes]] — fixes the fairness flaw in broadcast-advantage methods (LoGo-GRPO)
- MemAct (`2510.12635`) — memory ops inside chain-of-thought; AgeMem externalises them as named tools
- Memory-R1 (`2508.19828`) — closer in spirit; Memory-R1 has 4 ops (Memory-R1's are at the *segment* level), AgeMem has 6 (at the *agent step* level)
- Mem0 — the strongest baseline AgeMem beats
