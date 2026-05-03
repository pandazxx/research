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

**A-Mem (2502.12110) — reading notes (2026-05-03)**

A-Mem fits the LLM-as-memory-manager direction well: the LLM generates structured notes per memory, dynamically links related memories as new ones arrive, and uses those links for retrieval. It is a closer match to Zettelkasten than Generative Agents' flat memory stream.

Two open problems identified after reading:

*1. Static embeddings are a long-term bottleneck.*
A-Mem still relies on a fixed embedding model to map notes into vector space for similarity search. The embedding model does not adapt after deployment. Consequences:
- Terminology drift: if the user's vocabulary or the agent's domain evolves, old memories encoded under earlier usage patterns may become harder to retrieve.
- Concept granularity mismatch: general-purpose embeddings flatten domain-specific distinctions that matter for expert tasks.
- The LLM-generated note text can evolve, but the vector representation of old notes is frozen at write time.

Directions being explored:
- *Chain-of-Memory (2601.14287)*: does NOT solve the static embedding bottleneck — see correction note below.
- Re-embedding on retrieval (expensive but correct): re-encode old memories periodically or on-demand.
- Avoid the embedding bottleneck entirely: use LLM-generated structured attributes (keywords, tags, links) as the primary retrieval key instead of dense vectors. A-Mem partially does this via its note structure.

*Chain-of-Memory (2601.14287) — corrected reading (2026-05-03)*

The "dynamic evolution" claim is narrower than the name implies. Precise breakdown:
- *Memory node*: `m = (text, timestamp, speaker_role, embedding)`. Extracted from a single conversation turn. Immutable after creation — content and embedding vector never change.
- *"Dynamic evolution"*: refers to chain assembly at query time only. The algorithm expands from top-retrieved anchor nodes, selecting successors that satisfy both query relevance (fixed cosine similarity) and contextual coherence with the growing chain. The chain structure is different per query; the nodes themselves are static.
- *Retrieval and ranking*: initial retrieval uses a fixed Qwen3-Embedding-8B with cosine similarity. Chain expansion adds a contextual consistency score (evolving chain embedding vs. candidate node). This is adaptive within a single query but nothing is persisted or retrained.

*Conclusion*: Chain-of-Memory improves over flat top-k by reasoning about chain coherence at retrieval time, but it does not address embedding drift at all. Old memories encoded under earlier terminology cannot realign to new vocabulary without re-embedding. The static embedding bottleneck remains unsolved in this paper.

*2. Prompt engineering requires substantial effort and does not generalize.*
A-Mem's memory writing, linking, and retrieval prompts are manually crafted and tuned for the benchmark tasks. Transferring to a new domain means retuning these prompts. The cost compounds because there are multiple interdependent prompts (write, link, retrieve, reflect).

Relevant work on reducing this burden:
- *AutoPDL (2504.04365)*: frames agent prompt selection as an AutoML problem; uses successive halving over a combinatorial space of prompting patterns and demonstrations. Reports 9–69pp accuracy gains across tasks and models.
- *SICA / Evolving Excellence (2512.09108)*: agents that edit their own source prompts using an LLM to propose and evaluate modifications; 17–53% improvement on coding tasks.
- *Self-generated in-context examples*: store successful trajectories as few-shot demonstrations for future runs; lifts performance without manual prompt work.

None of these are memory-specific yet — applying automated prompt optimization to memory prompts (write policy, linking, reflection) is an open gap.

**Open sub-questions in this direction:**
- How to prompt-engineer importance scoring that transfers across domains without retuning?
- Can reflection loops be made safe — i.e., not amplify model biases or hallucinate "lessons"?
- How do tool-call-based memory systems (AgeMem) perform vs. implicit prompt-only systems (Generative Agents)?
- Can automated prompt optimization (AutoPDL, SICA) be applied specifically to memory prompts?
- Is there a retrieval strategy that avoids static embedding drift without requiring full re-indexing?

## Open Questions
- How should agents decide when to forget versus summarize?
- How can memory provenance be made first-class in generation?
- What is the right benchmark for real user personalization over months?
- Can we align reflective memory without amplifying model biases?
