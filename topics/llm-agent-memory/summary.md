# LLM Agent Memory — Summary

## The big picture
Recent progress has shifted LLM agent memory from “just retrieve similar chunks” to **full memory lifecycle engineering**: write, organize, retrieve, revise, and forget.

## What changed recently
- **From context stuffing to memory systems**: architectures now use multi-tier memory and explicit promotion/compression.
- **From retrieval-only to maintenance loops**: systems increasingly include automated pruning, summarization, contradiction handling, and reflection.
- **From static metrics to agentic evaluation**: newer benchmarks test long-horizon behavior, temporal updates, and abstention.

## 2026-ready mental model
Think of a strong memory-enabled agent as a pipeline:
1. **Ingest**: detect memory-worthy events.
2. **Store**: place them into the right tier with metadata.
3. **Retrieve**: combine semantic relevance with recency/time/task constraints.
4. **Reason**: resolve conflicts and ground outputs in provenance.
5. **Consolidate/Forget**: compress, expire, or remove stale entries.

## Most important current research directions
- Reliable long-term personalization across many sessions.
- Retrieval that is temporally and causally correct (not only semantically similar).
- Learned forgetting and consolidation to control memory bloat.
- Memory safety: privacy boundaries, user controls, and deletion guarantees.

## Suggested study plan (7 days)
- Day 1–2: Read foundational papers (Generative Agents, LongMem, MemGPT).
- Day 3–4: Compare memory architectures and write/retrieval policies.
- Day 5: Review modern benchmarks (e.g., LongMemEval) and map failure modes.
- Day 6: Design your own memory policy template for one agent use case.
- Day 7: Revisit notes and extract a personal checklist for production systems.
