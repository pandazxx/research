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

## Open Questions
- How should agents decide when to forget versus summarize?
- How can memory provenance be made first-class in generation?
- What is the right benchmark for real user personalization over months?
- Can we align reflective memory without amplifying model biases?
