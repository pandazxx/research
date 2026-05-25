# Structure-Augmented RAG — Study Notes on GraphRAG, LightRAG, and RAPTOR

Background study of the three structure-augmented RAG baselines that HippoRAG 2 compares against in §5. Useful for understanding the competitive landscape around HippoRAG and for situating where the project's research could plausibly contribute.

Related notes:
- `hipporag-reproduction/docs/paper-notes.md` — HippoRAG 1
- `hipporag2-study-notes.md` — HippoRAG 2

---

## Why these three together

All three were published in roughly the same 12-month window (early 2024–late 2024). All three augment classic dense RAG with some additional *structure* on top of the retrieval corpus. All three are referenced by HippoRAG 2 as competing baselines on the same benchmarks.

What they share: an LLM is used **offline** to extract some structured representation from the documents, and that structure is consulted (along with or instead of vectors) at query time.

What separates them: the *kind* of structure they build, and how that structure is used at query time.

| Method | Structure type | Built when | Used when | Best at |
|---|---|---|---|---|
| **GraphRAG** | Knowledge graph + community summaries | Offline | Query time | Global sense-making over a large corpus |
| **LightRAG** | KG + dual-level retrieval (entity + theme) | Offline | Query time | Balanced cost-quality; incremental updates |
| **RAPTOR** | Hierarchical clustering tree of summaries | Offline | Query time | Multi-step reasoning, abstract synthesis |
| (HippoRAG 1) | KG + Personalized PageRank | Offline | Query time | Multi-hop QA |
| (HippoRAG 2) | KG with passage nodes + triple-filtering | Offline | Online (extra LLM call) | Multi-hop + simple QA + sense-making |

---

## 1. GraphRAG (Microsoft, Edge et al., 2024)

- **Paper:** *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* — https://arxiv.org/abs/2404.16130
- **Code:** https://github.com/microsoft/graphrag
- **Tagline:** Build a community-structured knowledge graph offline, summarise communities at multiple granularities, answer "global" questions by combining community summaries.

### Architecture

**Offline indexing (the expensive part):**
1. LLM extracts entities and relationships from documents.
2. Build a knowledge graph: nodes = entities, edges = relationships.
3. Run **community detection** on the graph using the Leiden algorithm (a graph clustering method).
4. Recursively summarise each community: LLM-generated summary of all entities and relationships within that community.
5. Build a **hierarchy of community summaries** — small tightly-clustered communities at the bottom, broader thematic groupings near the top.
6. Persist everything.

**Online retrieval — two modes:**

- **Global mode** (what the paper emphasises): for a query, generate a partial answer from *each* community summary independently, then synthesise all partial answers into a final response. Useful for *query-focused summarisation* — questions like "What are the main themes in this corpus?"
- **Local mode**: classic entity-traversal — find query-relevant entities, expand neighbourhood, return passages. Better for specific factual lookups.

### Strengths

- **Strong on global sense-making.** Questions like "What are the recurring patterns in the documents?" or "Summarise the main themes" — these are where dense RAG fails completely because no single passage contains the answer. GraphRAG's community summaries are designed for exactly this.
- **Scales to large corpora.** Validated on 1M-token corpora in the paper.
- **Hierarchical answers.** Can give answers at different levels of abstraction.

### Weaknesses

- **Indexing is extremely expensive.** Many LLM calls: one for each entity/relation extraction, one per community summary at every hierarchy level. Easily $1000+ for moderate corpora.
- **Static index.** Adding new documents requires re-running community detection and re-summarising affected communities. No incremental update path.
- **Weak on specific factual QA.** The same concept-context tradeoff as HippoRAG 1 — communities capture themes, not facts. Mediocre on simple "Where was X born?" questions.
- **Community-summary quality is a critical knob.** Bad summaries cascade through global retrieval. The paper doesn't deeply discuss summary failure modes.

### Position relative to HippoRAG family

GraphRAG and HippoRAG occupy adjacent niches:
- GraphRAG optimises for *aggregation* — combining information across many passages into a synthesised answer.
- HippoRAG optimises for *retrieval* — finding the specific passages relevant to a query.

HippoRAG 2 outperforms GraphRAG on F1 averaged across simple + multi-hop QA. GraphRAG would likely still win on pure global-summarisation tasks that HippoRAG 2 doesn't directly target.

---

## 2. LightRAG (Guo et al., 2024)

- **Paper:** *LightRAG: Simple and Fast Retrieval-Augmented Generation* — https://arxiv.org/abs/2410.05779
- **Code:** https://github.com/HKUDS/LightRAG
- **Tagline:** Cheaper, simpler version of GraphRAG with a dual-level retrieval mechanism and incremental updates.

### Architecture

**Offline indexing:**
1. LLM extracts entities and relations from chunks (similar to GraphRAG).
2. Build a graph: entity nodes, relation edges.
3. Store **vector embeddings** for both entities and relations.
4. **No community detection step** — this is a key cost-saving move vs GraphRAG.

**Online retrieval — dual-level:**

- **Low-level retrieval**: precise entity/relation matching. The query is parsed for specific entities; their direct neighbourhood in the graph is retrieved. Good for fact-grounded questions.
- **High-level retrieval**: broader theme matching. The query is mapped to relevant *categories* of relations or thematic clusters; passages associated with those themes are retrieved. Good for sense-making.

The two levels run in parallel; results are combined into the context for the final LLM.

**Incremental updates** (LightRAG's main selling point over GraphRAG):
- Adding a new document → extract its entities/relations → add to graph → update relevant entity/relation embeddings. No full reindex.
- This is structurally simpler than GraphRAG's community-summary architecture, which would need to re-cluster.

### Strengths

- **Substantially cheaper than GraphRAG.** No community summarisation step → fewer LLM calls during indexing.
- **Incremental updates.** Production-friendly — corpus can grow without rebuilding from scratch.
- **Dual-level retrieval is conceptually clean.** Handles both specific and broad queries in one architecture.

### Weaknesses

- **Disappointing empirical results.** In HippoRAG 2's reproduction (Table 2), LightRAG scored an average F1 of 6.6 — far worse than every other method including vanilla RAG (46.9). This suggests reproducibility issues or specific weakness on the chosen benchmarks. Treat the public numbers with caution.
- **Weaker on multi-hop.** Without something like Personalized PageRank, multi-hop reasoning is limited to whatever the dual-level retrieval can surface — no graph propagation.
- **Theme/category labels are LLM-defined and possibly noisy.** The "high-level" retrieval depends on the LLM producing coherent thematic categories during indexing.

### Position relative to HippoRAG family

LightRAG is the *budget* version of structure-augmented RAG. It saves on cost vs GraphRAG by skipping community detection, and on cost vs HippoRAG 2 by skipping the per-query LLM filter. The trade-off is empirical performance — HippoRAG 2's evaluations show LightRAG falling well behind.

That said, LightRAG's **incremental update story** is the part HippoRAG family has not addressed. HippoRAG 2 still requires a full reindex when new documents arrive. If you wanted to argue for LightRAG-style updates in HippoRAG, that's a potential contribution.

---

## 3. RAPTOR (Stanford, Sarthi et al., ICLR 2024)

- **Paper:** *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval* — https://arxiv.org/abs/2401.18059
- **Code:** https://github.com/parthsarthi03/raptor
- **Tagline:** Not a graph — a **tree**. Recursively cluster passages and generate hierarchical summaries; retrieve at any tree level.

### Architecture

**Offline indexing:**
1. Embed each passage with a sentence encoder.
2. Cluster passages using a Gaussian Mixture Model (GMM) on the embeddings. (Soft clustering — a passage can belong to multiple clusters.)
3. LLM generates a **summary** for each cluster — that summary becomes a new "node" in the tree at the next level.
4. Re-embed the summaries.
5. Repeat: cluster the summaries, summarise the clusters, etc.
6. Continue until convergence (no more meaningful clustering possible) or a max depth.

Result: a tree where leaves are the original passages and internal nodes are recursive summaries.

**Online retrieval — two strategies:**

- **Tree traversal**: start at the root, descend the tree following the most query-relevant branch at each level. Returns nodes from across multiple levels.
- **Collapsed tree**: flatten all tree nodes (leaves + every internal summary) into one pool; do standard dense retrieval against this pool. Returns the top-k most relevant nodes regardless of level.

The paper finds collapsed-tree retrieval works better in most cases — perhaps surprisingly, the tree structure is more useful for *organising the index* than for *traversing at query time*.

### Strengths

- **Strong on multi-step reasoning.** The paper reports +20% on the QuALITY benchmark vs prior SOTA when combined with GPT-4.
- **Conceptually elegant.** The recursive summarisation naturally captures different abstraction levels in the same index.
- **No knowledge graph required.** Pure embedding-based + LLM summarisation. Simpler dependency stack than KG approaches.
- **Best published numbers among the three baselines** on the HippoRAG 2 evaluation (avg F1 48.8 in the paper's Table 2).

### Weaknesses

- **Indexing cost.** Recursive summarisation = many LLM calls. Roughly: at each level, one LLM call per cluster. For deep trees on large corpora, this adds up — though typically cheaper than GraphRAG's full community summarisation.
- **No relation-level reasoning.** A tree of summaries cannot directly express "X is the parent of Y" the way a KG can. Multi-hop questions that follow specific relations are not as natural a fit.
- **Cluster quality matters a lot.** Bad clustering at any level cascades upward into bad summaries. The paper uses GMM; sensitivity to this choice is not deeply explored.
- **Updates are awkward.** Adding a new passage might invalidate clusters at multiple levels. Static or batch-reindex friendly only.

### Position relative to HippoRAG family

RAPTOR and HippoRAG occupy nearly orthogonal design spaces:
- RAPTOR's structure is **content-based** (clusters by semantic similarity).
- HippoRAG's structure is **relation-based** (graph edges from explicit triples).

They could in principle be combined: build a RAPTOR-style tree of *passages* and a HippoRAG-style graph of *entities*, retrieve from both. HippoRAG 2's "dense-sparse integration" is partly a step in this direction — passage nodes work like RAPTOR's leaf nodes, but without the recursive summarisation layer.

---

## Cross-comparison

### Where each is strongest

| Task | Best of the three | Why |
|---|---|---|
| Specific factual QA ("Where was X born?") | None — all three struggle vs vanilla dense RAG on simple QA. HippoRAG 2 explicitly fixes this. |
| Multi-hop QA ("Connect A to B via intermediate C") | RAPTOR or HippoRAG 1 (close) | Multi-step reasoning needs cross-passage synthesis |
| Global sense-making ("What are the main themes?") | GraphRAG | Designed exactly for this |
| Sense-making over long narratives | RAPTOR | Recursive summaries capture multiple abstraction levels |
| Cost-constrained deployment | LightRAG | Cheapest of the three to index |
| Production with incremental updates | LightRAG | Only one with incremental indexing |
| Pure retrieval quality | HippoRAG 2 (out of competition) | Wins on most benchmarks |

### Cost profile

| Method | Indexing LLM calls | Per-query LLM calls | Per-query latency |
|---|---|---|---|
| Vanilla dense RAG | 0 | 0 | ~50ms |
| LightRAG | P × entity/relation extraction | 0 (dual-level vector retrieval) | ~100ms |
| HippoRAG 1 | P × OpenIE | 1 (NER) | ~300ms |
| RAPTOR | P × passage embedding + summaries at each tree level | 0 (collapsed-tree retrieval) | ~100ms |
| HippoRAG 2 | P × OpenIE + passage node setup | 2 (NER + triple filter) | ~600ms |
| GraphRAG | P × entity extraction + community summaries at every hierarchy level | 0–N (depends on mode, global mode is expensive) | varies wildly |

### LLM-call frequency

A useful way to characterise these methods:

- **Index-heavy, query-cheap**: GraphRAG (heavy), RAPTOR (heavy), LightRAG (medium), HippoRAG 1 (medium)
- **Both index and query use LLM**: HippoRAG 2 (medium for both)
- **All-online (no index-time LLM)**: vanilla dense RAG, IRCoT-style methods

HippoRAG 2's per-query LLM use is what makes it more expensive than these baselines per query — but its retrieval quality wins on average.

---

## What each baseline taught the field

If you read these chronologically:

- **RAPTOR (Jan 2024)** established that **multi-level retrieval** matters. You don't always want to retrieve at the leaf-passage level.
- **GraphRAG (Apr 2024)** showed that **graph community structure** can extract themes from corpora that single passages cannot express.
- **HippoRAG 1 (May 2024)** introduced **graph propagation** (PPR) for multi-hop retrieval — different from GraphRAG's community summaries.
- **LightRAG (Oct 2024)** pushed back on cost — argued that the same graph benefits can be had without GraphRAG's expensive community summarisation.
- **HippoRAG 2 (Feb 2025)** addressed the "concept-context tradeoff" that all the entity-only methods (LightRAG, HippoRAG 1, parts of GraphRAG) suffered from. Added passage nodes, online LLM filtering.

The arc is clear: structure-augmented RAG started with one form of structure (tree/community), evolved through specialised retrieval algorithms (PPR), then converged on richer indices that integrate both dense and sparse signals (HippoRAG 2's dense-sparse hybrid).

---

## What this means for the research project

If the project commits to a memory-mechanism direction (reconsolidation or active forgetting), the baseline competition matters in three ways:

1. **HippoRAG 2 is the reference baseline now.** It's the strongest published structure-augmented method. Any new memory mechanism should beat it (or at least be competitive) on the same benchmarks.
2. **None of these baselines do reconsolidation.** All four (GraphRAG, LightRAG, RAPTOR, HippoRAG 2) treat indices as build-once, read-many. The reconsolidation gap exists across the entire competitive landscape.
3. **LightRAG's incremental updates are the closest existing work to "memory that grows over time."** Worth reading the LightRAG paper carefully for the update mechanism, even though its overall retrieval quality is weak.

If the project wants to position itself in this landscape:
- *"Reconsolidation memory built on top of HippoRAG 2"* — natural framing, clear baseline, fills a known gap.
- *"Active forgetting for graph-based memory"* — applies to GraphRAG, LightRAG, and HippoRAG 2 equally; one mechanism addressing all three.
- *"Dynamic graph-RAG"* — combines incremental updates (LightRAG-style) with reconsolidation. More ambitious; less crisp.

The first framing is probably the cleanest research story.

---

## Reading order if you study these further

1. **GraphRAG** (Edge et al., 2024) — start here; the most thoroughly documented baseline.
2. **RAPTOR** (Sarthi et al., ICLR 2024) — read second; it's the strongest "non-graph" alternative.
3. **LightRAG** (Guo et al., 2024) — read last; mostly for the incremental-update angle, since the retrieval quality is weak.
4. (Then re-read HippoRAG 2's §5 results with the baselines in context — the relative wins become more interpretable.)
