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

## Benchmark deep-dives

### LoCoMo (2402.17753, ACL 2024)

*What it is:* A dataset of very long-term conversations between two agents, grounded on personas and temporal event graphs, with human annotation for long-range consistency. Released set: 10 conversations.

*Scale:* Each conversation ~300 turns, ~9K tokens, up to 35 sessions. ~1,500–2,000 QA pairs total.

*Construction:* Machine-human pipeline. LLM agents generate dialogue grounded in pre-built persona profiles and event graphs (causal + temporal). Human annotators verify consistency and correct drift. Agents can share images — the dataset is multi-modal.

*Tasks:*
- QA: single-hop (one session), multi-hop (across sessions), temporal (ordering/timing), adversarial (misleading)
- Event summarization: extract causal/temporal event graph from conversation history
- Multi-modal dialogue generation: respond to ongoing dialogue using recalled past context including images

*Metrics:* F1 / LLM-as-judge for QA; structured graph match for event summarization.

*Limitations:*
- Only 10 conversations released — small evaluation surface.
- ~9K tokens per conversation is short by current standards (modern LLMs fit this in context with room to spare).
- Social chit-chat domain; limited coverage of task/assistant scenarios.
- Synthetic construction means language patterns may not match real user dialogues.

### LongMemEval (2410.10813)

*What it is:* 500 questions embedded in scalable synthetic user-assistant conversation histories. Designed to stress-test assistant-style agents rather than social agents.

*Scale:* ~115K tokens per question in the standard setting. Conversation histories are "freely scalable" — lengths can be adjusted.

*Construction:* Synthetic user-assistant turns generated to embed specific facts at specific points. Facts are planted; questions require finding and reasoning over them across sessions.

*Five capabilities tested:*
- Information extraction: recall a specific stated fact
- Multi-session reasoning: combine facts from different sessions
- Temporal reasoning: understand when something was true, changed, or expired
- Knowledge updates: new info overrides old (the agent must prefer the latest)
- Abstention: question is unanswerable from memory; agent should say so

*Key finding:* Commercial assistants and long-context LLMs show ~30% accuracy drop compared to single-session baselines.

*Metrics:* Accuracy (exact match or LLM-judge).

*Limitations:*
- Synthetic conversations — topically narrow, may not reflect real distribution of user memory events.
- Fact-planting construction makes it easy to game with extraction heuristics.
- No multi-modal content.
- Abstention category is underweighted relative to real usage.

### Comparison

| Dimension | LoCoMo | LongMemEval |
|---|---|---|
| Format | Two-agent social dialogue | User-assistant |
| Construction | Persona + event graph + human edit | Synthetic fact planting |
| Scale (conversations) | 10 | 500 questions across scalable histories |
| Context length | ~9K tokens / conv | ~115K tokens / question |
| Multi-modal | Yes (images) | No |
| Temporal reasoning | Yes | Yes |
| Knowledge updates | No | Yes |
| Abstention | No | Yes |
| Domain | Social / personal | General assistant |

### Other benchmarks

*MSC — Multi-Session Chat (2107.07567, ACL 2022)*
The original multi-session dialogue benchmark. Human-human crowdworker chats across 5 sessions, each up to 14 utterances, persona-grounded (built on top of PersonaChat). Sessions are separated by hours or days. Scale: relatively small by modern standards. Primary use: training and evaluating models that remember persona facts across sessions. Limitation: very short sessions, narrow persona domain, pre-LLM-agent era design — treated as legacy now but still used as a training corpus.

*MemBench (2506.21605, ACL 2025 Findings)*
Tests memory from two angles: *factual* (store and retrieve explicit facts) vs *reflective* (derive higher-level inferences from experience), and two interaction modes: *participation* (agent was in the conversation) vs *observation* (agent is reading a conversation it wasn't part of). More taxonomically careful than LongMemEval. Covers effectiveness, efficiency (latency/token cost), and capacity (scaling to large memory loads).

*MemoryBench (2510.17281)*
Focus: memory + continual learning together. Includes a user feedback simulation framework — the agent receives corrections and preferences and must update its behavior. Three modules: task provider, user simulator (generates human-like feedback), performance monitor. First benchmark to jointly evaluate memory recall and continual adaptation from feedback. Multi-domain, multi-language. Limitation: feedback simulation quality depends on the LLM used as simulator.

*MemoryAgentBench (2507.05257, ICLR 2026)*
Grounded in cognitive science. Tests four competencies: (1) accurate retrieval, (2) test-time learning (update beliefs mid-task), (3) long-range understanding (synthesize across distant events), (4) selective forgetting (discard stale or contradicted info). Built by reformatting existing long-context datasets into multi-turn incremental format plus newly constructed tasks. Key finding: no current memory agent masters all four — each method wins on some and fails on others.

### Benchmark landscape at a glance

| Benchmark | Year | Format | Sessions | Key strength | Key gap |
|---|---|---|---|---|---|
| MSC | 2022 | Human-human | 5 | First multi-session | Short, narrow, legacy |
| LoCoMo | 2024 | Agent-agent | 35 | Event graphs, multi-modal | Only 10 convs, social domain |
| LongMemEval | 2024 | User-assistant | Scalable | 5 capability types, abstention | Synthetic, pre-1M-ctx |
| MemBench | 2025 | Various | Various | Factual vs reflective, participation vs observation | Newer, less adopted |
| MemoryBench | 2025 | User-feedback | Continuous | Continual learning from feedback | Simulator quality varies |
| MemoryAgentBench | 2026 | Multi-turn incremental | Incremental | Cognitively grounded, 4 competencies | No current method passes all 4 |

*Shared limitation:* All were designed when 32K context was the ceiling. With 1M+ context LLMs available now, a naive baseline of "just stuff everything in context" becomes a serious competitor to any memory system — and none of these benchmarks was designed to account for that. This weakens their usefulness as memory-system discriminators going forward. The 2026-era benchmarks (MemoryAgentBench) are beginning to address this by testing capabilities that long-context stuffing genuinely cannot cover, like selective forgetting and test-time learning.

## Practical takeaway for your study
If you want to understand the field quickly, study in this order:
1. Memory architecture primitives (stream, store, retrieve, reflect).
2. Memory write/retrieval policies (what gets stored and why).
3. Benchmarks and failure modes (especially temporal and contradiction errors).
4. System trade-offs (quality vs latency vs privacy).

## Study pass — synthesis (2026-05-09)

### Core thesis
LLM agent memory is moving from retrieval infrastructure toward **governed state management**. Vector search remains useful, but the frontier systems increasingly focus on the lifecycle around retrieval: deciding what becomes memory, enriching it with structure, detecting missing evidence, revising stale beliefs, and preventing memory corruption.

### What the papers collectively show
- **Generative Agents** remains the clean conceptual baseline: memory stream, LLM-scored importance, retrieval by recency/relevance/importance, and reflection. Its weakness is that the whole policy is domain-prompted and validated mainly in a social simulation.
- **A-Mem** extends the baseline from a flat stream into a Zettelkasten-like note network. The important move is not just graph storage; it is LLM-mediated note writing, link creation, and attribute evolution.
- **HippoRAG / HippoRAG 2** show why graph-structured retrieval matters: dense retrieval struggles when the answer requires associative or multi-hop integration across passages. Personalized PageRank over an LLM-built graph is a practical way to retrieve paths rather than isolated chunks.
- **Mem0** is the production-oriented reference: hybrid vector + graph storage, extraction/consolidation/deduplication, and strong latency/cost claims against full-context baselines. It is useful because it frames memory as an always-on service layer rather than a benchmark-only method.
- **MemR3** shifts attention from storage to retrieval control. Its evidence-gap tracker makes "what do we still need to know?" explicit, which helps avoid both under-retrieval and noisy over-retrieval.
- **Hindsight** sharpens the epistemic design problem: agents should distinguish observed facts, experiences, neutral entity summaries, and evolving opinions/beliefs. This separation is a strong pattern for explainability and preference consistency.
- **SSGM** names the main risk of the 2025-2026 direction: once memory can evolve, errors become persistent. Memory poisoning, semantic drift, procedural drift, and privacy leakage need governance before consolidation, not after.

### Architectural pattern that seems strongest
A robust memory agent should combine:
1. append-only raw event log for provenance,
2. extracted atomic facts/preferences with timestamps,
3. structured links or graph edges for associative recall,
4. synthesized summaries/beliefs kept separate from source facts,
5. closed-loop retrieval that can ask for more evidence,
6. governed consolidation with contradiction, sensitivity, and staleness checks.

This is heavier than naive RAG, but it directly targets the failure modes that long-context stuffing and top-k vector retrieval leave unsolved.

### Key unresolved problems
- **Static embedding drift**: many "evolving memory" systems update text, links, or summaries while old embeddings remain fixed. Chain-of-Memory improves query-time chain construction, but does not persistently adapt old vectors.
- **Prompt policy transfer**: importance scoring, write filters, link generation, and reflection prompts are still hand-designed. AutoPDL/SICA-style prompt optimization is relevant but not yet memory-specific.
- **Trustworthy reflection**: reflections are useful compression, but they can hallucinate causal lessons or amplify early mistakes.
- **Governed forgetting**: selective forgetting is still underdeveloped compared with remembering and retrieval.
- **Benchmark pressure**: LoCoMo and LongMemEval are useful, but modern 1M+ context models make "full context" a stronger baseline. Future benchmarks need to test update correctness, forgetting, privacy boundaries, and long-lived state quality.

### Personal research direction
The most promising angle is not "better memory store" in isolation. It is a **memory policy layer** that treats writes and consolidations like database transactions:
- proposed memory update,
- evidence/provenance check,
- conflict check against existing memory,
- sensitivity and permission check,
- TTL or decay assignment,
- commit with audit trail.

This would make memory systems easier to evaluate, debug, and trust in real assistants.

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

## Code agent benchmarks (added 2026-05-03)

The code agent benchmark space is dominated by SWE-bench and its variants, most of which treat tasks as *independent* — no memory across tasks. Only a few explicitly test memory or context accumulation.

*SWE-bench (original, 2310.06770)*
The baseline: resolve real GitHub issues using repo context. Single-task, no memory component. State of the art is around 50% on Verified subset as of early 2026. Not a memory benchmark — included here as the reference point everything else extends.

*SWE-Bench-CL (2507.00014)*
The most directly memory-relevant code benchmark. Takes SWE-Bench Verified tasks and organizes them chronologically by natural repo evolution, then evaluates how an agent accumulates knowledge across sequential tasks. Uses a FAISS-backed semantic memory module. Metrics: average accuracy, forgetting rate, forward/backward transfer, tool-use efficiency. Tests the stability-plasticity trade-off: learning new things without forgetting old ones. Closest analog to the conversational memory benchmarks but for code.

*SWE-EVO (2512.18470)*
Long-horizon software evolution scenarios: the agent must handle a sequence of changes to a codebase, not just one isolated issue. Tests multi-step planning and state tracking across an evolving repo. Memory here means tracking what has already been changed and why.

*LoCoEval (2603.06358)*
Repository-oriented conversational context management. Simulates a developer having a long multi-turn conversation with a code assistant — requirements, clarifications, noise — until the context balloons to 64K–256K tokens. Tests: topic awareness, information extraction, and code generation (Pass@1). 128 samples, ~50 turns per sample. More about context compression and retrieval than persistent long-term memory.

*LoCoBench (2509.09614)*
Long-context software engineering at scale. 8,000 scenarios with context lengths from 10K to 1M tokens. Measures performance degradation as context grows. Diagnostic rather than memory-system-focused.

*What's missing:*
No current benchmark tests an agent working on the same codebase over weeks or months — accumulating architectural knowledge, learned bug patterns, team conventions, and past decisions. That's the code-agent equivalent of the multi-session personalization problem in conversational memory, and it is an open gap.

The LLM-as-memory-manager paradigm (Generative Agents, A-Mem) has not been applied to code agents in any of these benchmarks. SWE-Bench-CL is the closest beachhead, but its memory is embedding-based (FAISS) with no LLM-driven write/reflect policy.

## Open Questions
- How should agents decide when to forget versus summarize?
- How can memory provenance be made first-class in generation?
- What is the right benchmark for real user personalization over months?
- Can we align reflective memory without amplifying model biases?
