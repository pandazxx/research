# Embeddings Beyond Cosine Similarity — Study Notes

What embeddings can do besides nearest-neighbour retrieval, with particular focus on semantic chunking and multi-vector retrieval. Written specifically to inform improvements to HippoRAG / A-Mem reproductions and the reconsolidation project.

---

## TL;DR

The "embed and cosine-match" pattern is the dominant use of embeddings but not the only one. Three families of techniques worth knowing for your project:

1. **Semantic chunking** — use embeddings to decide *where* to split text into chunks, not just what to retrieve. Topic boundaries become detectable as similarity drops between adjacent sentences.
2. **Late interaction (ColBERT)** — keep per-token embeddings instead of pooling. Match query tokens to document tokens individually. Much more accurate, more storage.
3. **Late chunking (Jina, 2024)** — embed the whole document with full context first, *then* split into chunks. Preserves cross-chunk context that traditional chunking loses.

**What embeddings cannot do:** structured extraction (OpenIE-style triples). Embeddings produce dense vectors; OpenIE produces structured output. These are different output types — you need an LLM (or trained decoder) for extraction. Embeddings can support extraction-adjacent tasks (entity linking, coreference, deduplication) but not extraction itself.

---

## 1. The standard pattern (and its limits)

The default RAG pipeline:

```
Document → split into fixed-size chunks → embed each chunk → store vectors
Query   → embed → cosine match against stored vectors → return top-k
```

This works, but each step has known weaknesses:

- **Fixed-size chunking** cuts mid-thought. A chunk might contain the start of an idea but not its resolution.
- **Pooling to one vector per chunk** loses token-level information. Two semantically different chunks can pool to similar vectors.
- **Embedding chunks in isolation** loses cross-chunk context. The chunk "She was the first female president" embeds without knowing who "she" is.
- **Single-vector retrieval** can't distinguish "the document mentions X" from "the document is about X."

Each of the techniques below addresses one of these.

---

## 2. Semantic chunking

### The idea

Instead of splitting text at arbitrary token boundaries, use embeddings to find *topic shifts*. Sentences with similar embeddings belong together; a sharp drop in adjacent-sentence similarity marks a topic boundary.

### The algorithm (simplest version)

```
1. Tokenise text into sentences.
2. Embed each sentence.
3. Compute cosine similarity between each consecutive pair.
4. Identify positions where similarity drops sharply (below threshold,
   or at local maxima of the difference).
5. Cut chunks at those positions.
```

Result: chunks of *variable* size, each internally coherent.

### Worked example

Suppose we have sentences `s1..s7` with the following adjacent-pair cosine similarities:

```
s1 → s2: 0.92  (closely related)
s2 → s3: 0.89  (closely related)
s3 → s4: 0.34  ← topic shift detected
s4 → s5: 0.91  (closely related)
s5 → s6: 0.88  (closely related)
s6 → s7: 0.41  ← topic shift detected
```

Chunks: `[s1, s2, s3]`, `[s4, s5, s6]`, `[s7]`.

### Implementations to know

- **LangChain `SemanticChunker`** — `from langchain_experimental.text_splitter import SemanticChunker`
- **LlamaIndex `SemanticSplitterNodeParser`** — equivalent in their framework
- **OpenAI's own examples** — they have cookbook notebooks demonstrating semantic chunking

### When it helps

- Long documents with shifting topics (research papers, conversation transcripts, novels)
- Inputs to RAG where chunk coherence matters more than uniform size
- Pre-processing for any pipeline that does per-chunk work downstream (OpenIE, summarisation)

### When it doesn't help

- Short, single-topic documents
- Highly structured documents where structure already implies boundaries (markdown headings, code files)
- Heavily formatted content (tables, lists) where sentence segmentation is ambiguous

### For your project

**HippoRAG improvement:** currently HippoRAG runs OpenIE per passage. If passages have multiple topics, the resulting triples are mixed. Semantic chunking *before* OpenIE → cleaner per-chunk triples → cleaner KG.

**A-Mem improvement:** A-Mem treats each "interaction" as one atomic note. If an interaction covers three topics (e.g., user mentions a job change, a vacation plan, and a coding question in one message), the note's LLM-generated keywords/tags/context are fuzzy. Semantic chunking would split this into three distinct notes, each truly atomic — closer to actual Zettelkasten principles.

**Reconsolidation implication:** the unit of update matters. If "one fact" in your memory store is really three loosely related facts, you can't update them independently. Smaller, semantically-coherent atomic units → cleaner update granularity.

---

## 3. ColBERT and late interaction

**Paper:** Khattab & Zaharia, SIGIR 2020 (arXiv:2004.12832) — already downloaded
**Follow-up:** ColBERTv2 (arXiv:2112.01488) — also downloaded

### The idea

Don't pool to one vector per document. Keep one vector per *token*. At query time, compute similarity per-token and take the maximum.

Mechanically:

```
Document encoding: each token → one embedding. Document = list of embeddings.
Query encoding:    each query token → one embedding.

Similarity: for each query token, find the best-matching document token.
            Sum those max-similarities → document score.
```

Mathematically: `Score(q, d) = Σ_i max_j cos(q_i, d_j)` — the "MaxSim" operator.

### Why it works better

Standard pooled embeddings lose fine-grained information. ColBERT preserves it. A document that contains "Einstein lived in Princeton" and another that contains "Einstein won the Nobel Prize" pool to similar vectors but should be retrieved by different queries. With per-token embeddings, query "Where did Einstein live?" lands on "Princeton" in document 1; query "What prize did Einstein win?" lands on "Nobel" in document 2.

### The cost

- **Storage**: 100× more vectors per document (one per token vs one per chunk).
- **Compute at retrieval**: pairwise similarity scales as `|query_tokens| × |doc_tokens|`.
- **ColBERTv2** addresses both with quantisation and indexing tricks, making it viable at production scale.

### When it helps

- Fine-grained QA where the answer is one specific phrase in a long document
- Domains where vocabulary precision matters (medical, legal, code)
- Heterogeneous documents where pooling averages away the relevant signal

### Status in your project

HippoRAG's legacy code already supports ColBERTv2 as the retriever option (in addition to Contriever). Reading the ColBERT paper helps you understand what that retriever option is actually doing.

For A-Mem, you could in principle store per-attribute embeddings instead of one note-level embedding — that's a kind of multi-vector representation.

---

## 4. Late chunking (Jina, 2024) — deep dive

**Paper:** arXiv:2409.04701 — already downloaded as `papers/2409.04701-late-chunking.pdf`.

### The core insight: when does pooling happen?

All transformer-based embedders work in two phases:

```
Text  →  Tokenize  →  [transformer forward pass]  →  Token-level embeddings
                                                       (one vector per token)
                                                          ↓
                                            Pooling (mean / CLS / weighted)
                                                          ↓
                                              One chunk-level vector
```

The token-level embeddings are **contextual** — that's the whole point of attention. The embedding for token "she" depends on what came before it in the sequence. If "Maria" appeared earlier, "she" has already attended to it and "knows" who she is.

The question is *when* you split into chunks relative to that flow:

| Strategy | Where splitting happens |
|---|---|
| **Early chunking** (the standard) | Split the **text** first, then embed each chunk separately |
| **Late chunking** | Embed the **whole text** first, then split the **token embeddings** |

That's the entire conceptual difference. The downstream architecture is the same.

### Concrete worked example

Take this document:

> "Maria is a chemistry teacher. She lives in Boston."

Suppose your target chunk size is one sentence.

**Early chunking (the broken way for this content):**

```
Step 1: Split text into chunks
  Chunk A: "Maria is a chemistry teacher."
  Chunk B: "She lives in Boston."

Step 2: Embed each chunk independently
  embedder("Maria is a chemistry teacher.")  →  vector_A
  embedder("She lives in Boston.")            →  vector_B
```

The problem: when the embedder processed Chunk B, "She" had no antecedent. The embedder's attention couldn't reach back to "Maria" because Maria isn't in Chunk B's input. The resulting `vector_B` is about "someone living in Boston" — generic.

Query: "Where does Maria live?"

→ `vector_B` doesn't match well because Maria isn't represented in it.
→ Retrieval misses.

**Late chunking (the fix):**

```
Step 1: Embed the WHOLE document at once
  embedder("Maria is a chemistry teacher. She lives in Boston.")
  → token embeddings: [emb("Maria"), emb("is"), emb("a"), emb("chemistry"),
                       emb("teacher"), emb("."), emb("She"), emb("lives"),
                       emb("in"), emb("Boston"), emb(".")]

  Critically: emb("She") attended to emb("Maria") during the forward pass.
  emb("She") now has Maria-context baked into it.

Step 2: Identify chunk boundaries in TOKEN SPACE
  Chunk A spans tokens 0-5 (Maria...teacher.)
  Chunk B spans tokens 6-10 (She...Boston.)

Step 3: Pool the token embeddings within each chunk
  vector_A = mean(token_embeddings[0:6])
  vector_B = mean(token_embeddings[6:11])  ← includes "She" with Maria-context
```

Now `vector_B` encodes "She (= Maria) lives in Boston" because the token embedding for "She" already knew about Maria.

Query: "Where does Maria live?"

→ `vector_B` matches well because the Maria-context is in there.
→ Retrieval hits.

### The algorithm in pseudocode

```python
def late_chunking(text, embedder, chunk_boundaries):
    """
    chunk_boundaries: list of (start_token_idx, end_token_idx) tuples
                      determined by any chunking strategy (semantic,
                      sentence-based, fixed-size — doesn't matter HOW)
    """
    # Step 1: Tokenize and run a single forward pass over the entire text
    tokens = tokenizer(text)
    token_embeddings = embedder.encode_with_hidden_states(tokens)
    # token_embeddings shape: (num_tokens, embedding_dim)

    # Step 2: For each chunk, mean-pool its token embeddings
    chunk_embeddings = []
    for start, end in chunk_boundaries:
        chunk_emb = token_embeddings[start:end].mean(axis=0)
        chunk_embeddings.append(chunk_emb)

    return chunk_embeddings
```

Two important things to note:

1. **You still need to decide chunk boundaries somehow.** Late chunking doesn't replace your boundary-detection logic — it just changes when pooling happens. You can use sentence boundaries, semantic chunking, or fixed sizes to find the boundaries. The boundaries are applied to token positions, not text.

2. **The pooling step is mean-pool (or sometimes attention-weighted mean).** Same operation as standard chunking — just on different tokens.

### What you need to make this work

| Requirement | Reason |
|---|---|
| Long-context embedder | The whole document must fit in the embedder's context. Most sentence embedders cap at 512 tokens; you need ≥4K-32K. |
| Access to token-level outputs | The embedder must expose hidden states per token, not just the pooled vector. |
| Aligned tokenization | Chunk boundaries in text need to map cleanly to token indices. |

**Long-context embedders that support late chunking:**

- **Jina v3** (jinaai/jina-embeddings-v3) — 8K context, has a `late_chunking=True` flag in the API
- **Voyage-3-large** — 32K context
- **bge-m3** (BAAI) — 8K context, supports it manually
- **NV-Embed-v2** — 32K context (used by HippoRAG 2)

Most OpenAI embeddings (text-embedding-3) **don't expose token-level embeddings** through the API, so you can't do late chunking with them directly. That's a real limitation if you're locked into OpenAI.

### Empirical evidence (from the Jina paper)

The Jina paper reports retrieval gains of **5–20%** over early chunking on various IR benchmarks (MS MARCO, NarrativeQA, etc.). The improvement varies by content type:

| Content type | Late chunking gain |
|---|---|
| Heavy pronouns / anaphora (narrative, dialogue) | Largest gain (15–20%) |
| Cross-section references (long papers) | Moderate gain (10–15%) |
| Self-contained chunks (code, structured docs) | Small gain (<5%) |
| Very short documents | Negligible gain |

The intuition: late chunking helps most when chunks would otherwise be missing crucial context from elsewhere in the document. If your chunks are already self-contained, the technique adds little.

### Comparison with similar approaches

| Technique | When pooling happens | Storage | When to use |
|---|---|---|---|
| **Early chunking** (standard) | Per chunk, before storage | 1 vector / chunk | Default; simple content |
| **Late chunking** | After full-document embedding | 1 vector / chunk | Content with cross-chunk references |
| **ColBERT (late interaction)** | Never (no pooling) | 1 vector / token | Maximum retrieval quality, high cost |
| **Sentence-window retrieval** | At retrieval time, surrounding sentences added | 1 vector / chunk + metadata | LangChain pattern; different problem |
| **Document-level embedding** | Once over whole doc | 1 vector / document | Coarse retrieval only |

The mental model: late chunking is *early chunking* + *cross-chunk attention preservation*. Same storage cost as early chunking, better quality for context-heavy content.

### Cost considerations

- **Embedding compute**: roughly equivalent to early chunking total cost (one long pass vs many short passes — usually similar total token-throughput).
- **Memory at indexing time**: higher than early chunking because you hold the full document's token embeddings during pooling. For a 32K-token document with 1024-dim embeddings in float32, that's ~131 MB temporarily.
- **Storage at rest**: identical to early chunking. The final stored vectors are the same shape and count.
- **Retrieval cost**: identical. No change at query time.

So the cost story is "small extra memory at index time, otherwise free." A real win.

### Practical implementation (Jina v3 example)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jinaai/jina-embeddings-v3")

# Late chunking via the Jina API flag
chunk_embeddings = model.encode(
    document_text,
    task="retrieval.passage",
    late_chunking=True,
    chunk_size=500,  # in tokens
)
# Returns a list of chunk-level vectors; chunks have cross-document context
```

Or manually with any HuggingFace transformer that exposes hidden states:

```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3")

inputs = tokenizer(document_text, return_tensors="pt", truncation=True)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# Token-level embeddings from the last layer
token_embs = outputs.last_hidden_state[0]  # (num_tokens, hidden_dim)

# Find chunk boundaries (in token positions) using your chunker of choice
# ... boundaries = [(0, 100), (100, 250), (250, 500), ...]

chunk_embs = [token_embs[s:e].mean(dim=0) for s, e in boundaries]
```

### When late chunking is the wrong tool

- **Documents longer than the embedder's context.** If your document is 1M tokens and your embedder maxes at 32K, you have to pre-split, which defeats the purpose. (You can do "windowed late chunking" — chunk within sub-documents — but the cross-window context is still lost.)
- **You're using OpenAI embeddings exclusively.** OpenAI's API doesn't expose token-level outputs, so late chunking isn't doable.
- **Chunks are already self-contained.** If you're chunking code or highly structured docs, the gain is marginal.
- **Real-time low-latency requirements.** The long forward pass adds latency to indexing.

### When it helps

- Long-form documents with many cross-references (papers, books, conversation transcripts)
- Domains with heavy pronoun use or implicit subjects (narrative, dialogue)
- Any setup where you'd benefit from chunks "remembering" the document they came from

### For your project specifically

**A-Mem improvement:** A-Mem stores conversation turns as separate notes. Late chunking could let you embed *whole conversations* and split into notes after embedding — each note then encodes its place in the conversation, not just its local content. Retrieving a memory that says "she said yes" is useless without context. Late chunking would let the embedding of that memory carry knowledge of *who* said yes, even though only "she said yes" is in the note text.

**HippoRAG 2 improvement:** the passage embeddings used as PPR seeds are currently from independent passage embeddings. Late chunking would let each passage embed with knowledge of the broader document it came from. Could moderately improve seed quality on long documents.

**The catch:** NV-Embed-v2 (used by HippoRAG 2) supports 32K context but doesn't have a "late chunking mode" exposed in its API — you'd need to implement it manually using the HuggingFace transformers interface.

### Experiment path if you want to validate the technique

If you want to run this as a contained experiment:

1. Pick Jina v3 (easiest API — `late_chunking=True` flag).
2. Take 5–10 documents that have cross-chunk references (narratives or papers with references to earlier sections).
3. Embed them once with early chunking, once with late chunking.
4. Run a retrieval task against both with queries that require the cross-chunk context.
5. Measure top-k recall difference.

Should take half a day and give you direct intuition for when it helps.

---

## 5. Other applications worth knowing (briefly)

### Clustering for memory consolidation

UMAP + HDBSCAN on memory embeddings → automatic discovery of topic clusters. Could feed reconsolidation: memories in the same cluster are candidates for consolidation into a higher-level summary memory.

### Embedding-based deduplication

For incoming memories, compute embedding and check against the nearest existing memory. If similarity > threshold, treat as duplicate (update the existing rather than create new). Prevents A-Mem from accumulating near-identical notes when the user repeats themselves.

### Anomaly / novelty detection

A new memory whose embedding is far from any existing cluster is *novel*. This is a write-policy signal: novel memories are probably more worth storing than mundane ones. Mirrors the brain's novelty-driven dopamine signal for memory consolidation (covered in your brain memory deep-dive).

### Multi-aspect / multi-vector embeddings

Each memory gets multiple embeddings — one per facet. For A-Mem: embed the keywords separately from the context description. At retrieval, score against the relevant facet for the query type ("what topics did Sam mention?" → keywords; "what was Sam thinking about climbing?" → context).

### Embedding arithmetic (research-y)

Linear directions in embedding space sometimes encode attributes. E.g., a "recency" direction could be derived from old vs new memories. Subtracting/adding this direction to a query could let you bias retrieval toward old or new memories. Less mature; mostly research curios at this point.

---

## 6. What embeddings cannot do

**OpenIE is not in this list.** Embeddings produce dense vectors; OpenIE produces (subject, predicate, object) triples. These are different output types. To get structured output from an embedding model, you need a decoder — and at that point, you have a generative model, not an embedding model.

Embeddings *support* OpenIE-adjacent tasks:

| Task | How embeddings help |
|---|---|
| Entity linking | Embed entity mentions, find nearest KG node by similarity |
| Coreference resolution | Embed mentions, cluster similar ones |
| Relation classification | Given a (head, tail) pair, embed and classify with a linear probe |
| Triple deduplication | Embed extracted triples, merge near-duplicates |
| Disambiguation | Embed in context, separate by cluster |

But the *extraction* itself — finding the triples in raw text — still needs an LLM or a specialised supervised model.

---

## 7. Concrete experiments for your project

### Experiment A — Semantic chunking on HippoRAG

1. Take the HippoRAG reproduction.
2. Replace the default passage tokeniser with LangChain's `SemanticChunker`.
3. Re-index your corpus.
4. Re-run the comparison dataset.

**Measure:** does multi-hop QA improve? Does single-hop change? Are the extracted triples cleaner (manually inspect 20 of them)?

**Expected outcome:** moderate improvement on multi-hop (cleaner chunks → cleaner triples → better KG). Possible slight regression on single-hop if some chunks become too small.

### Experiment B — Semantic chunking on A-Mem

Trickier because A-Mem already treats each interaction as atomic. But you can:

1. For multi-topic incoming "interactions" (or simulated long messages), semantically chunk *before* creating notes.
2. Create one note per chunk instead of one note per interaction.
3. Measure: does memory evolution behave more cleanly? Do contradiction queries find the right note?

### Experiment C — Late chunking on either system

Requires a long-context embedding model (jinaai/jina-embeddings-v3 supports this; so does Voyage-3-large for 32K).

1. Replace the embedder with a long-context one.
2. For inputs longer than ~2K tokens, embed the whole document first, then split.
3. Compare retrieval quality vs standard chunking.

**Expected outcome:** noticeable improvement on questions whose answer requires context from elsewhere in the document.

### Experiment D — Multi-vector A-Mem

Instead of one embedding per note (computed from concat of K, G, X), embed K, G, X separately. At retrieval, score against the most relevant of the three.

**Why interesting:** addresses a known A-Mem weakness — the single embedding blurs together different aspects of the note.

---

## 8. Papers to read (in priority order)

| # | Paper | Why | Time |
|---|---|---|---|
| 1 | **Late Chunking** (Jina, 2024, papers/2409.04701) | Most directly actionable for your project | 30 min |
| 2 | **ColBERT** (papers/2004.12832) | Foundational; understand what HippoRAG's ColBERTv2 option does | 60 min |
| 3 | ColBERTv2 (papers/2112.01488) | Production-grade evolution; skim for the indexing tricks | 30 min |

**ColBERT is dense.** If you're short on time, read the abstract + Sections 3 (Architecture) and 4 (Late Interaction). Skip the IR theory section. The MaxSim operator is the one concept you must internalise.

**Late Chunking is short.** Read end-to-end. The core idea is in Section 3 and Figure 1.

---

## 9. Connection to the broader project

Three places this thread feeds into your work:

1. **Reproduction quality** — running A-Mem and HippoRAG with semantic chunking *before* your dataset evaluation may shift the comparison results. Worth a side experiment.
2. **Reconsolidation design** — finer-grained atomic memory units (via semantic chunking) makes targeted updates possible. Coarser units force all-or-nothing updates.
3. **Forgetting design** — clustering (a non-cosine use of embeddings) is a natural way to identify *what to forget*. Tiny isolated clusters could be candidates for pruning; dense well-connected clusters are likely important.

If you choose reconsolidation, expect semantic chunking to make its way into your design — fine-grained units are what reconsolidation needs to work properly.

---

## Sources

- ColBERT (Khattab & Zaharia, SIGIR 2020) — papers/2004.12832-colbert.pdf
- ColBERTv2 (Santhanam et al., 2021) — papers/2112.01488-colbertv2.pdf
- Late Chunking (Jina AI, 2024) — papers/2409.04701-late-chunking.pdf
- LangChain SemanticChunker docs: https://api.python.langchain.com/en/latest/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html
- LlamaIndex SemanticSplitterNodeParser: https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/semantic_splitter/
