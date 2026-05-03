# LLM Agent Memory — Working Notes

## Scope
Focus: **recent progress** in memory systems for LLM agents (2023–2026), with emphasis on architectures, retrieval quality, memory management, and evaluation.

## What “memory” means in agent systems
- **Short-term / working memory**: context window, scratchpad, recent turns.
- **Long-term / external memory**: vector DB, key-value stores, episodic logs, structured profiles.
- **Reflective memory**: distilled lessons/rules created from experience.
- **Procedural memory**: reusable tool-use patterns, plans, workflows.

Why it matters: memory is the key mechanism enabling personalization, multi-session continuity, and adaptation without full model fine-tuning.

## Timeline of notable progress

### 2023 foundations
- **Generative Agents (Park et al., 2023)** formalized the memory stream + retrieval + reflection loop in simulated social agents.
- **LongMem (Wang et al., 2023)** proposed augmenting LMs with long-term memory beyond fixed context.
- **MemGPT (Packer et al., 2023)** framed memory management as an OS-like paging problem for LLM agents.

### 2024 consolidation
- More open-source agent stacks began separating:
  - hot context (prompt)
  - warm memory (retrieved summaries/facts)
  - cold archive (full logs/docs)
- Better memory pipelines emerged: write filters, salience scoring, and periodic summarization.
- Benchmarking attention shifted from raw QA toward multi-session assistant behavior.

### 2025 acceleration
- New work emphasized **autonomous memory maintenance** (automatic updates, pruning, and consolidation).
- Increasing focus on **retrieval robustness** and **reasoning over memories**, not just nearest-neighbor recall.
- Benchmarks (e.g., LongMemEval, 2024) started being used to quantify temporal reasoning, updates, and abstention in memory-heavy chat settings.

### Early 2026 direction
- Surveys highlight movement toward:
  - learned forgetting and consolidation,
  - causally grounded retrieval,
  - trustworthy reflection,
  - multimodal/embodied memory.

## Key design patterns in current systems

### 1) Hierarchical memory
- Separate stores by latency and importance.
- Promote/demote entries between tiers.
- Use compressed summaries to save tokens.

### 2) Memory writing policies
- Not every event should be saved.
- Common triggers: novelty, user preference signals, task outcomes, failures.
- Dedupe and contradiction checks are increasingly important.

### 3) Retrieval beyond top-k similarity
- Hybrid retrieval (semantic + symbolic + temporal).
- Query rewriting based on current goal.
- Re-ranking with task-specific relevance and recency.

### 4) Reflection and consolidation loops
- Convert many episodes into compact heuristics or profiles.
- Keep provenance pointers back to source events to reduce hallucinated “memories”.

### 5) Safety and privacy controls
- User-level controls for what is remembered.
- TTL/expiration and deletion semantics.
- Sensitive memory classifiers/redaction.

## Evaluation trends
- Legacy evaluation (single-turn retrieval accuracy) is no longer enough.
- Current evaluations emphasize:
  - multi-session consistency,
  - temporal correctness,
  - update handling (new info overriding old),
  - calibrated abstention when memory is uncertain,
  - downstream task success over long horizons.

## Practical takeaway for your study
If you want to understand the field quickly, study in this order:
1. Memory architecture primitives (stream, store, retrieve, reflect).
2. Memory write/retrieval policies (what gets stored and why).
3. Benchmarks and failure modes (especially temporal and contradiction errors).
4. System trade-offs (quality vs latency vs privacy).

## LLM-as-memory-manager paradigm (focal research direction)

**User observation (2026-05-03):** Generative Agents (2304.03442) is the clearest early example of using LLM prompts to make *all* memory decisions — what is important enough to save, what to retrieve, and when to reflect. This is distinct from hardcoded heuristics. The upside: as the underlying LLM improves, memory quality improves automatically with no architectural changes. The downside identified in the paper itself: the approach was validated almost entirely in a social NPC simulation (Smallville, 25 agents), so generalization to other domains (task agents, coding assistants, multi-step planning) is still open.

**How Generative Agents does LLM-driven memory:**
- **Importance scoring**: an LLM prompt rates each observation on a 1–10 salience scale.
- **Reflection trigger**: when accumulated importance scores cross a threshold, an LLM prompt generates higher-level insights from recent memories.
- **Retrieval ranking**: a weighted combination of recency, importance (LLM-scored), and semantic relevance (embedding similarity) selects what enters the context.

Everything that decides memory is expressed as a prompt, not code.

**Recent work following this paradigm (2025–2026):**

| Paper | Mechanism | Extension over Generative Agents |
|---|---|---|
| A-Mem (2502.12110) | LLM generates notes + links between memories dynamically | Adds explicit inter-memory linking via LLM prompts |
| AgeMem | Exposes memory ops (store, retrieve, update, discard) as tool calls | Agent autonomously decides *when* to invoke each operation |
| MemR3 (2512.20237) | LLM maintains an evidence-gap state over the interaction | Tracks not just what's stored but what's *missing* |
| Agentic Memory (2601.01885) | LLM manages unified short- and long-term memory | Single prompt-driven pipeline without separate retrieval logic |
| Reflective Memory Mgmt (ACL 2025) | Prospect + retrospect reflection loops for personalized dialogue | Generalizes Generative Agents' reflection to open-domain chat |

**Why "scenario-specific" is a real limitation:**
- Importance scoring via LLM prompts is sensitive to domain. A score calibrated for social gossip differs from one for technical task state.
- Reflection quality depends on the LLM having enough background to draw meaningful abstractions from the episode.
- The sandbox evaluation (fixed world, fixed agents) avoids many real problems: adversarial memory injection, privacy, conflicting beliefs from multiple users.

**Open sub-questions in this direction:**
- How to prompt-engineer importance scoring that transfers across domains without retuning?
- Can reflection loops be made safe — i.e., not amplify model biases or hallucinate "lessons"?
- How do tool-call-based memory systems (AgeMem) perform vs. implicit prompt-only systems (Generative Agents)?

## Open Questions
- How should agents decide when to forget versus summarize?
- How can memory provenance be made first-class in generation?
- What is the right benchmark for real user personalization over months?
- Can we align reflective memory without amplifying model biases?
