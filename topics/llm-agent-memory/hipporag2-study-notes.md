# HippoRAG 2 — Paper Study Notes

Notes from reading *From RAG to Memory: Non-Parametric Continual Learning for Large Language Models* (Jiménez Gutiérrez et al., ICML 2025).

- Paper: https://arxiv.org/abs/2502.14802
- Code: https://github.com/OSU-NLP-Group/HippoRAG (the `main` branch — `legacy` is HippoRAG 1)
- Companion notes: see `hipporag-reproduction/docs/paper-notes.md` for HippoRAG 1 background. **Read that first if you haven't already.**

---

## TL;DR

HippoRAG 2 is **not a redesign** of HippoRAG. It is the same brain-inspired KG + Personalized PageRank skeleton, with **three refinements** layered on top to fix specific weaknesses of v1:

1. **Dense-sparse integration** — passages become first-class graph nodes alongside entity ("phrase") nodes, eliminating the concept-vs-context tradeoff that hurt v1 on simple factual QA.
2. **Deeper contextualisation** — query→graph linking switches from NER-based to **query-to-triple** matching, capturing more of the query's intent than just its entities.
3. **Recognition memory** — an LLM filters retrieved triples *online* (per query), modelling the recall-vs-recognition distinction from human memory.

Headline empirical result: HippoRAG 2 outperforms standard RAG, dense embedding models (NV-Embed-v2), and competing graph-RAG methods (GraphRAG, LightRAG, RAPTOR, HippoRAG 1) **across all three task types** — simple factual QA, multi-hop QA, and discourse understanding — using an open-weight LLM (Llama-3.3-70B-Instruct).

The reframing also matters: the paper repositions the system as **non-parametric continual learning for LLMs**, not just "smarter RAG." Same architecture; bigger ambition.

---

## What HippoRAG 1 got wrong (the motivation)

The HippoRAG 1 paper claimed wins on multi-hop QA but had a known regression: it underperformed vanilla dense RAG on simple factual questions. The root cause is what the v2 paper calls the **concept-context tradeoff**:

- v1 stores only *phrases* (extracted entities) as graph nodes.
- This is *sparse coding* in neuroscience terms — concise, generalisable, but loses context.
- A simple query like "What is X's birthplace?" needs context, not just the entity X.
- v1's PPR propagates from the entity X through the graph — diffusing probability where you wanted a direct passage lookup.

This same critique applies to GraphRAG, LightRAG, and other graph-augmented RAG methods that emerged in 2024. **All of them sacrifice simple-QA performance for multi-hop gains.** HippoRAG 2's design goal is to fix this without losing the multi-hop strength.

---

## Quick comparison

| | HippoRAG 1 | HippoRAG 2 |
|---|---|---|
| Venue | NeurIPS 2024 | ICML 2025 |
| Graph nodes | Phrase (entity) nodes only | Phrase nodes **+ passage nodes** |
| Edge types | Relation edges + synonymy edges | Relation + synonymy **+ context edges ("contains")** |
| Query→graph linking | NER → top-k node match | **Query→triple matching** (default) |
| Online LLM use | 1 small NER call | NER + **triple-filtering** LLM call |
| LLM in paper | GPT-3.5 | **Llama-3.3-70B-Instruct** (open!) |
| Retriever in paper | Contriever | **nvidia/NV-Embed-v2** |
| Prompt tuning | Hand-crafted | **DSPy MIPROv2** (automated) |
| Best at | Multi-hop QA | Multi-hop AND simple QA AND discourse |
| Worst at | Simple factual QA | (no major regression) |

---

## Brain analogy (extended)

HippoRAG 2 explicitly maps its three components onto a richer neuroscience model:

| Brain region | HippoRAG 2 component |
|---|---|
| Neocortex | The LLM (parametric knowledge) |
| Hippocampus | The KG + PPR (auto-associative memory) |
| Parahippocampal regions | The retrieval encoder (linker between neocortex and hippocampus) |

The paper also leans on **dense-sparse coding theory** (Beyeler et al. 2019):
- **Sparse coding** — small subset of neurons fire, compact, generalisable, but loses context. → maps to *phrase nodes* in v1.
- **Dense coding** — many neurons fire simultaneously, redundant but context-rich. → maps to *passage nodes* added in v2.

Human memory uses both. v2 attempts to integrate both.

---

## Refinement 1 — Dense-Sparse Integration (§3.2)

**Change:** add **passage nodes** to the graph, connected to their constituent phrase nodes by new **context edges** labelled "contains".

The offline indexing pipeline now produces a graph with two node types:

```
[passage1] --contains--> phrase_a
              \--contains--> phrase_b
              \--contains--> phrase_c

phrase_a --relation--> phrase_b      (from OpenIE triple)
phrase_b ↔ phrase_d                  (synonymy edge)
```

Practical effects:

- PPR can now propagate probability *to passages directly*, not just to entities. For a simple factual query, the right passage often becomes a direct seed, bypassing the entity-diffusion problem.
- The graph is bigger — `P + N` nodes (passages + phrases) instead of just `N`.
- Indexing cost increases marginally (passage nodes are cheap to create; the OpenIE step is unchanged).
- Online retrieval gets richer because seed nodes can now be passages.

The same offline indexing process from HippoRAG 1 is retained — this is purely *additive*. Existing phrase-only retrieval still works; passage retrieval is layered on.

---

## Refinement 2 — Deeper Contextualisation (§3.3)

**Change:** switch the query-to-graph linking strategy from NER-based to query-to-triple matching.

The paper considers three approaches, evaluates all three (§6.1), and chooses the third as the default:

| Approach | Mechanism | Problem |
|---|---|---|
| **NER to Node** (v1's approach) | Extract entities from the query with NER, match each to nearest KG nodes via embeddings | Entity-centric; loses contextual signal. "Where does Alice's professor work?" reduces to just "Alice." |
| **Query to Node** | Embed the entire query, match directly to KG nodes | Better — keeps query context — but still matches against single nodes, missing relational structure |
| **Query to Triple** (v2 default) | Embed the entire query, match to **(subject, relation, object) triples** in the KG | Triples encapsulate the relational structure the query is asking about. Much more comprehensive understanding of query intent. |

The system embeds each triple as a sentence-like representation and matches it against the query embedding. Top-k triples are retrieved, and their constituent phrase nodes become the seed candidates for PPR.

**Why this matters:** in v1, a multi-hop question that asked about a relation (not an entity) had no direct way to land on the relevant edges in the graph. v2's query-to-triple matching lets the system identify "this query is about relation R between entities A and B" and seed PPR at the right entities.

---

## Refinement 3 — Recognition Memory (§3.4)

**Change:** an LLM acts as an **online filter** over retrieved triples before they become PPR seed nodes.

The motivation is the recall-vs-recognition distinction in cognitive science:

- **Recall**: actively retrieve a memory without external cues (HippoRAG 1's mostly-passive mode).
- **Recognition**: identify whether retrieved candidates match what you're looking for, with the help of external stimuli.

HippoRAG 2 implements the query-to-triple step in two phases:

1. **Query to Triple** — embedding model retrieves top-k triples `T` from the graph (this is recall).
2. **Triple Filtering** — an LLM reads the query plus the retrieved triples and returns a filtered subset `T' ⊆ T` containing only triples actually relevant to the query (this is recognition).

Only the filtered triples `T'` are used to determine PPR seed nodes. The top-5 most relevant triples (per the filter) determine the reset probabilities.

**Implementation note:** the triple-filtering prompt is *automatically tuned* using **DSPy's MIPROv2** optimiser. This is itself an interesting design choice — instead of hand-crafting the prompt, the authors run black-box prompt optimisation against a held-out set. Connects directly to the BPO deep-dive we did earlier.

---

## Updated online retrieval flow

After all three refinements, the per-query flow is:

1. **Direct passage retrieval (parallel).** Retrieve top-k passages directly via embedding similarity to the query. These are the "raw RAG" candidates — they bypass the graph entirely as a safety net.
2. **Query→triple retrieval.** Embed the query, retrieve top-k triples from the KG.
3. **Triple filtering.** LLM filters retrieved triples to relevant subset `T'`.
4. **Seed-node selection.**
   - From filtered triples: extract constituent phrase nodes (up to k of them, scored by average ranking across `T'`).
   - **Plus all passage nodes** are also seeded (the paper notes "broader activation improves multi-hop reasoning").
5. **Reset probabilities.**
   - Phrase nodes: weighted by their ranking scores from step 4.
   - Passage nodes: weighted by embedding similarity to the query, scaled by a balance factor.
6. **PPR.** Run Personalized PageRank with the combined reset probabilities over the full KG.
7. **Top-K passages** by PageRank score → fed as context to the QA reader LLM.

**Fallback case:** if step 3 returns no triples (filter throws them all out), the system falls back to direct top-k passages from step 1.

---

## Implementation choices in the paper

| Component | v1 | v2 | Why the change matters |
|---|---|---|---|
| LLM (OpenIE + triple filter + reader) | GPT-3.5 (closed, API-only) | **Llama-3.3-70B-Instruct** (open-weight) | Demonstrates the approach works with open models; cheaper to run at scale |
| Retriever / embedding model | Contriever | **nvidia/NV-Embed-v2** (7B) | State-of-the-art embedder; the comparison baseline |
| Prompt tuning method | Hand-crafted | **DSPy MIPROv2** | Reproducible, automated, transferable across models |
| Top-k triples kept | n/a | 5 | Small but enough to seed multi-hop PPR |

The all-open-source stack is notable — this is reproducible without depending on any specific API provider, and the cost story is more controllable.

---

## Results

### QA F1 scores (Table 2 in the paper)

Using Llama-3.3-70B-Instruct as the QA reader. Average across all 7 benchmarks:

| Method | Avg F1 |
|---|---|
| No retrieval | 38.4 |
| Contriever (dense RAG) | 46.9 |
| GTR | 50.4 |
| NV-Embed-v2 (best embedding baseline) | 57.0 |
| RAPTOR | 48.8 |
| GraphRAG | 49.6 |
| LightRAG | 6.6 (broken on these benchmarks) |
| HippoRAG 1 | 53.1 |
| **HippoRAG 2** | **59.8** |

Highlights:
- **Beats NV-Embed-v2 by +2.8 F1 on average** despite using the same retriever as the embedding model.
- **+9.5 F1 over NV-Embed-v2 on 2Wiki**, +3.1 on LV-Eval (multi-hop tasks).
- **+6.7 F1 over HippoRAG 1**, with the biggest jumps on simple-QA where v1 was weakest.

### Passage Recall@5 (Table 3 in the paper)

Where the architectural changes really show:

| Method | NQ (simple) | PopQA (simple) | MuSiQue (multi-hop) | 2Wiki (multi-hop) | HotpotQA (multi-hop) | Avg |
|---|---|---|---|---|---|---|
| NV-Embed-v2 | 75.4 | 51.0 | 69.7 | 76.5 | 94.5 | 73.4 |
| HippoRAG 1 (reproduced) | **44.4** | 53.8 | 53.2 | 90.4 | 77.3 | 63.8 |
| **HippoRAG 2** | **78.0** | 51.7 | 74.7 | 90.4 | 96.3 | **78.2** |

The simple-QA recovery is dramatic: HippoRAG 1 scored 44.4 on NaturalQuestions; HippoRAG 2 scores 78.0. That's the v1 regression fixed.

One mild anomaly: PopQA recall stays roughly flat (53.8 → 51.7). The paper doesn't dwell on this but it's worth noting — passage nodes don't help everywhere.

---

## Cost economics — HippoRAG 1 vs 2

The cost shape shifts. v2 spends more per query, less per index step relative to total.

### Offline indexing

Roughly the same as HippoRAG 1:
- OpenIE on every passage (same as v1, now with Llama-70B locally or via API)
- Entity dedup, encoding, FAISS NN, KG construction
- **New: passage node creation + context edges** — negligible cost, just structural

If you run Llama-3.3-70B locally, indexing is *cheaper* than v1 with GPT-3.5 (no API bill). If you run it via an inference API (e.g., NIM, Together, Replicate), comparable cost.

### Online (per-query) cost

| Step | v1 | v2 |
|---|---|---|
| Query NER / linking | 1 small LLM call | 1 small LLM call |
| **Triple filtering** | — | **1 medium LLM call (reads query + top-k triples)** |
| PPR computation | Local CPU, ~50ms | Local CPU, ~50ms (slightly bigger graph) |
| Total per-query LLM calls | 1 | **2** |
| Wall-clock | ~200-700ms | ~400-1200ms |
| Cost per query | $0.0001-0.005 | **$0.0005-0.01** |

So **v2 is ~2-3× more expensive per query than v1**, mostly from the triple-filtering call. Still much cheaper than fully iterative methods like IRCoT.

The cost is bought in exchange for fixing v1's simple-QA weakness and gaining the recognition-memory filtering quality.

---

## Design choices and quirks

1. **Still no reconsolidation.** v2 has no mechanism to update memories on retrieval — every recall is read-only. The static-graph problem persists.
2. **Still no active forgetting.** No mechanism to prune stale or contradicted memories. Capacity grows monotonically.
3. **Triple filter is the new prompt-sensitive surface.** The system's quality depends on the LLM-as-filter doing its job well. Different prompts produce different filtering behaviour. MIPROv2 tuning helps but the prompt is still a sensitive knob.
4. **Passage nodes inflate the graph.** Index size grows by P (passages). For a 100K-passage corpus, this is a meaningful increase — but still tractable for PPR.
5. **Concept-context balance is a hyperparameter.** v2 uses a weight factor (§3.5) to balance phrase-node vs passage-node influence on reset probabilities. The right value is corpus-dependent.
6. **The recognition memory step adds latency.** Each query waits for the LLM filter to return before PPR runs. If you parallelise the "direct top-k passages" fallback path, you can mask some of this latency.
7. **No incremental updates.** Adding a new passage requires: OpenIE on it, dedup entities into the existing pool, re-running all-pairs FAISS for new synonymy edges, rebuilding the graph. This is the same v1 limitation, not addressed.
8. **Triple matching is still semantic, not symbolic.** "X is the capital of Y" and "Y's capital is X" should be the same triple but might embed differently. The paper doesn't discuss canonicalisation.

---

## Why incremental updates are hard — matrix and data-structure analysis

A common first impression is that HippoRAG's matrix-based computation locks it into fixed dimensions that prevent incremental memory growth. The truth is more nuanced: **the matrices themselves can grow, but the synonymy edges are globally interdependent**, and that is the real bottleneck.

### What data structures HippoRAG uses and whether they can grow

| Data structure | Shape | Can it grow incrementally? |
|---|---|---|
| **Graph** (igraph object) | N nodes, E edges | **Yes** — igraph supports O(1) node/edge addition |
| **Entity embedding matrix** | N × D (D=768 Contriever, 1024 NV-Embed) | **Yes** — append a row per new entity |
| **Passage scoring matrices** (docs_to_facts, facts_to_phrases) | Sparse, P×T and T×N | **Awkward** — scipy CSR needs internal array rebuild; use dict representation instead |
| **FAISS index** | N vectors of dimension D | **Partially** — can add vectors; but see synonymy problem below |
| **PPR transition matrix M** | N × N (implicit) | **Always fresh** — igraph recomputes from current edge list per PPR call |

The embedding dimension D (768 or 1024) is **fixed per model** but does **not change** when memories are added. All entities live in the same D-dimensional space regardless of graph size. D is not the bottleneck.

### The real bottleneck: synonymy edge recomputation

When you add a new entity X:

1. **Compute X's embedding** — one encoder call. Cheap. ✓
2. **Find X's nearest neighbours** — one FAISS query (X against all existing entities). Cheap. ✓
3. **Add synonymy edges from X to its neighbours** — dictionary inserts. Cheap. ✓
4. **Check whether existing entities should now list X as a neighbour** — **this is the problem.** ✗

Step 4 requires re-querying **all N existing entities against X** — O(N) comparisons. The existing NN lists were precomputed without X. If entity Y had neighbours all above 0.8 cosine, and X scores 0.95 with Y, X should be Y's top neighbour — but Y's precomputed list doesn't know X exists.

For one new entity, O(N) is fast. For a batch of 1,000 new entities (a few new documents), it's O(N × 1000), and the synonymy graph can change substantially. This is why HippoRAG does a full rebuild.

### Secondary bottleneck: node specificity cascade

Node specificity = `1 / (passage count for entity i)`. Adding a new passage mentioning existing entities changes their specificity, which cascades into different passage-ranking scores for all queries involving those entities. Cheap to recompute per entity, but the downstream retrieval effect is unpredictable without re-running queries.

### PPR itself is fine

The transition matrix M is **not stored** — igraph computes it from the current edge list each time PPR runs. So PPR always operates on whatever graph exists at query time. There is no "stale matrix" problem for the core retrieval algorithm.

### What LightRAG does differently

LightRAG claims incremental updates by skipping all-pairs synonymy entirely. New entities get only relation edges — no synonymy edges against the full graph. Cheaper, but loses the cross-passage entity linking that makes HippoRAG's retrieval strong.

### Three potential solutions for incremental updates

1. **Lazy synonymy.** Don't recompute synonymy for every new entity. Batch new entities; periodically rebuild synonymy (e.g., nightly). Between rebuilds, new entities participate via relation edges only. *Pragmatic for a research project.*
2. **Approximate NN updates.** Use FAISS `IndexIVFFlat` instead of `IndexFlat`. IVF supports `add()` and the existing NN structure is approximately maintained. Not exact, but fast. *Good if synonymy quality matters but latency matters more.*
3. **Drop synonymy edges entirely.** Rely on v2's query-to-triple matching to find related entities online (via LLM). Shifts cost from offline synonymy to online LLM calls. *The radical design shift that v2 is partially already making.*

### Implications for reconsolidation research

If the research direction involves making HippoRAG's memory evolve over time:
- **Adding new memories:** hits the synonymy bottleneck above. Solution 1 (lazy rebuild) is probably sufficient.
- **Updating existing memories (reconsolidation):** modifying edge weights or replacing nodes doesn't trigger the synonymy problem — it's changes to the existing graph, not additions. PPR automatically uses updated weights on the next query.
- **Deleting memories (forgetting):** removing nodes/edges from igraph is O(1). Node specificity recalculates cheaply. Synonymy edges touching the deleted node just disappear. **Forgetting is actually the easiest operation to support incrementally.**

This is a useful asymmetry: *forgetting is cheap; remembering (adding) is expensive*. A system that aggressively forgets (pruning low-value nodes) while selectively remembering (batched synonymy rebuild) could be both incrementally updatable and high-quality.

---

## Open questions / things v2 still doesn't answer

1. **Where exactly does the recognition-memory step earn its keep?** The paper shows aggregate wins but doesn't ablate the filter prompt or measure how often the filter actually changes the candidate set.
2. **How sensitive is v2 to the LLM choice for triple filtering?** Llama-3.3-70B is strong; would Llama-3.1-8B (much cheaper) still work? Would Claude/GPT-4 do better?
3. **What about cost-quality trade-off via filter k?** Top-5 triples is paper default. Top-3 would halve the filter LLM input; top-10 would double it. Not explored.
4. **Does dense-sparse integration help non-RAG memory tasks?** The paper benchmarks on RAG tasks. Whether passage nodes help long-term multi-session memory (à la MemoryAgentBench) is open.
5. **Does the static-embedding bottleneck change with v2?** With passage nodes, the embedding model now also influences passage-node retrieval, not just entity matching. So the system is *more* embedding-dependent, not less.

---

## Connections to your research project

For the broader agent memory research direction:

- **HippoRAG 2 is now a more credible baseline.** v1's simple-QA weakness was a known limitation; v2 fixes it. Future work in graph-structured memory should benchmark against v2, not v1.
- **The reconsolidation gap is still wide open.** Neither v1 nor v2 implements memories-update-on-recall. This is *exactly* the gap a reconsolidation-focused project can fill.
- **The active-forgetting gap is still wide open.** No pruning, no TTL, no contradiction handling. Another clean direction.
- **Recognition memory ≠ reconsolidation.** v2's triple filter is a *read-time* filter; it doesn't modify the graph. Reconsolidation would *write* during retrieval. These are different mechanisms.
- **Could you build reconsolidation on top of HippoRAG 2?** Yes, plausibly:
  - When a triple is retrieved + filtered as relevant, increase its weight slightly (Hebbian-style strengthening).
  - When a triple is retrieved and filtered out as irrelevant, decrease its weight (or move toward pruning).
  - This would make the graph adapt to query patterns over time.
- **The recognition memory mechanism is itself interesting for reconsolidation.** The filter sees both the query and the candidate memories; it has the information needed to decide *which memories should be updated*. The mechanism is already there — just not connected to the write path.

---

## If you decide to reproduce or build on v2

Suggested entry points:

1. **Start with the `main` branch** of OSU-NLP-Group/HippoRAG. The package is `pip install hipporag` and there's a Colab quickstart.
2. **Use Llama-3.3-70B-Instruct** if you want paper-faithful results. Via NIM (you already have this from the v1 reproduction), Together, or Replicate.
3. **NV-Embed-v2** is the retriever; available on HuggingFace.
4. **Hyperparameters to track** beyond v1:
   - Number of triples retrieved before filtering (`top_k_triples`)
   - Weight factor balancing phrase-node vs passage-node reset probabilities
   - Triple-filter prompt (auto-tuned via MIPROv2 in the paper; you can run MIPROv2 yourself or hand-craft)
5. **Cost expectation:** ~2-3× per-query cost of v1. Budget accordingly if running large evaluations.

---

## Reading order if you study further

1. (You've already read v1 paper notes — `hipporag-reproduction/docs/paper-notes.md`)
2. **HippoRAG 2 paper §3** (Methodology) — the architectural deep-dive
3. **HippoRAG 2 paper §6.1** (Linking strategy ablation) — the NER-to-Node vs Query-to-Node vs Query-to-Triple comparison
4. **HippoRAG 2 paper Appendix A** — the actual filter prompts used
5. **DSPy MIPROv2 docs** — to understand how the filter prompt was tuned (cross-link to BPO deep-dive)
6. **NV-Embed-v2 paper** (arXiv:2405.17428) — the embedding model used

---

## Open questions specifically interesting for the project's research direction

These connect HippoRAG 2 to reconsolidation/forgetting work:

1. If you treat the triple filter's *accept/reject* decision as a write signal back to the graph, what's the cleanest way to update edge weights?
2. Should context edges (passage→phrase "contains") also be modifiable, or only relation edges?
3. Does the dense-sparse split suggest a similar split for reconsolidation — frequent fast updates on phrase nodes vs slow updates on passage nodes?
4. The graph has no temporal information. If you stamp every node and edge with a creation/last-access timestamp, what new behaviours become possible?
5. What would HippoRAG 2 look like with bounded capacity (active forgetting)? Currently the graph grows monotonically — what's the right pruning criterion?
