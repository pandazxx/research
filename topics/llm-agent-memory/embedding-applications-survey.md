# Embedding Applications — A Practitioner's Survey

What embeddings can do beyond "embed text, search by cosine." This is a broader follow-up to `embeddings-beyond-cosine.md`, with focus on:

1. The recent generation of embedding models (Jina v3, E5-Mistral, BGE-M3, LLM2Vec)
2. Less-obvious applications of embeddings beyond retrieval
3. Multimodal and code embeddings
4. Concrete experiments / techniques to know

If `embeddings-beyond-cosine.md` is "how do I chunk better," this is "what *else* can I do with embeddings."

---

## Section 1 — The current generation of embedding models

Embedding models have evolved a lot since 2022. Here's the landscape of recent influential models, all with PDFs in the repo:

### 1.1 Jina Embeddings v3 (2024, papers/2409.10173)

**Key innovations:**
- **Task-LoRA**: one shared backbone (570M params, XLM-RoBERTa) with separate LoRA adapters for different tasks (retrieval-query, retrieval-document, classification, clustering, text-matching). Same model, different head depending on what you want.
- **Long context**: 8K tokens — much more than typical sentence embedders.
- **Multilingual**: trained on 89 languages.
- **Matryoshka Representation Learning (MRL)**: default 1024 dims, but you can truncate to any size down to 32 without retraining. Storage / quality knob.
- **Late chunking support**: the API has a `late_chunking=True` flag (the technique from yesterday's discussion).

**Why it matters for your project:** it's the most production-friendly embedder for the experiments you'd want to run — long context for late chunking, free truncation for storage trade-offs, free task-conditioning so you can use the same model for retrieval and clustering.

### 1.2 Matryoshka Representation Learning (2022, papers/2205.13147)

**Key idea:** train the embedding such that the first `k` dimensions are themselves a usable embedding for any `k ≤ d`. Store once at full dimension, truncate at query time to trade quality for storage / latency.

**Why it matters:** decouples storage cost from retrieval quality. You can index at 3072 dim, then query at 256 dim for a fast first pass and re-rank with 3072. Now standard in OpenAI text-embedding-3 and Jina v3.

### 1.3 E5-Mistral (2024, papers/2401.00368)

**Key idea:** take a decoder-only LLM (Mistral 7B), turn it into an embedding model via supervised contrastive fine-tuning. Specifically, mean-pool the last hidden state and train on (query, positive, negative) triplets.

**Why it matters:** broke the assumption that you need a dedicated encoder architecture (BERT-family) for embeddings. Decoder LLMs make excellent embedders. Performance on MTEB jumped significantly.

### 1.4 LLM2Vec (2024, papers/2404.05961)

**Key idea:** convert *any* decoder-only LLM into an embedder unsupervised, via three steps:
1. Enable bidirectional attention (modify the attention mask)
2. Masked next-token prediction (MNTP) — light additional training
3. Unsupervised contrastive learning via SimCSE-style dropout-as-positive

**Why it matters:** any open LLM (Llama, Mistral, Qwen, ...) can become an embedder for ~hours of fine-tuning. You're no longer locked into proprietary or specialized embedding models.

### 1.5 BGE-M3 (2024, papers/2402.03216)

**Key idea:** one model that produces three types of embeddings simultaneously:
1. **Dense**: standard pooled embedding (what we've been discussing)
2. **Sparse**: SPLADE-style token-weight vector (high-dim sparse vector indexed by vocabulary)
3. **Multi-vector**: ColBERT-style per-token embedding for late interaction

**Why it matters:** hybrid retrieval (combining dense + sparse + late-interaction scores) often beats any single method. BGE-M3 makes that hybrid retrieval feasible with one model.

### 1.6 SPLADE (2021, papers/2107.05720)

**Key idea:** learned sparse embeddings. Instead of a dense vector of 1024 floats, produce a *sparse* vector over the entire vocabulary (~30K dims, but mostly zeros). Each non-zero entry is a learned token weight that captures both lexical match (which words appear) and semantic match (which related words *should* match).

**Why it matters:** combines the speed of inverted index lookup (like BM25) with the semantic quality of dense embeddings. Cheaper to store than dense embeddings at scale.

### 1.7 Instructor (2023, papers/2212.09741)

**Key idea:** condition embeddings on a natural-language *instruction* describing the task. The same text embedded with "Represent the document for retrieval" vs "Represent the document for clustering" produces *different* embeddings.

**Why it matters:** one model, many tasks. Removes the need for task-specific embedding models. Jina v3's task-LoRA is the next-generation refinement of this idea.

### 1.8 CLIP (2021, papers/2103.00020)

**Key idea:** train a text encoder and an image encoder jointly such that matching (image, caption) pairs are close in a shared embedding space. Suddenly you can search images by text and vice versa.

**Why it matters:** the entire multimodal embedding field flows from CLIP. Modern descendants (BLIP-2, SigLIP, etc.) extend the idea to video, audio, and structured modalities.

---

## Section 2 — Applications of embeddings beyond retrieval

The "embed and cosine-match" pattern is so dominant that everything else feels exotic. Here's a tour of less-obvious applications.

### 2.1 Semantic clustering and topic discovery

Take your corpus, embed everything, run UMAP for dimensionality reduction, then HDBSCAN for clustering. Each cluster = a topic. No labels needed.

```python
import umap, hdbscan
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jina-embeddings-v3", task="clustering")
embs = model.encode(documents)
reduced = umap.UMAP(n_components=5).fit_transform(embs)
labels = hdbscan.HDBSCAN(min_cluster_size=10).fit_predict(reduced)
```

**Use cases:**
- Memory consolidation (group similar memories for summarization)
- Topic discovery in unlabeled data
- Auto-routing (which model / pipeline to use based on cluster)
- Visualization for QA over a corpus

**Tools:** BERTopic is the most popular library for this. Worth bookmarking.

### 2.2 Zero-shot and few-shot classification

You don't need a trained classifier if you have embeddings. For each class, write a sentence describing it (e.g., "This text is about cooking"). Embed those class descriptions. To classify a new document, embed it and pick the closest class.

```python
classes = ["cooking", "sports", "politics", "science"]
class_embs = model.encode([f"This text is about {c}" for c in classes])
doc_emb = model.encode(new_document)
predicted_class = classes[np.argmax(cosine_sim(doc_emb, class_embs))]
```

**Use cases:**
- Quick prototyping without labeled data
- Routing user queries to specialised handlers
- Memory tagging (auto-tag memories with categories)

### 2.3 Semantic deduplication

For each new memory, find the nearest existing memory by cosine. If similarity > threshold (e.g., 0.92), treat as duplicate and merge / update instead of creating a new one.

```python
def is_duplicate(new_emb, existing_embs, threshold=0.92):
    sims = cosine_sim(new_emb, existing_embs)
    return sims.max() > threshold, sims.argmax()
```

**Use cases:**
- Prevent A-Mem from accumulating near-identical notes when user repeats themselves
- Deduplicate scraped corpora before indexing
- Online dedup as new content arrives

### 2.4 Novelty / anomaly detection

A new item whose embedding is *far* from any cluster center is novel.

```python
def novelty_score(new_emb, existing_embs):
    sims = cosine_sim(new_emb, existing_embs)
    # High novelty = low similarity to nearest neighbour
    return 1 - sims.max()
```

**Use cases:**
- Memory write policy: novel memories are more worth storing (mirrors the brain's dopamine-driven novelty signal)
- Quality control: flag inputs that don't match expected distribution
- Active learning: select novel examples for labeling

### 2.5 Reranking with cross-encoders

Standard 2-stage pattern:
1. **Stage 1:** dense retrieval gets top-k candidates fast (say, top-50).
2. **Stage 2:** a cross-encoder scores each (query, candidate) pair jointly — much more accurate but slower.

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base")

top_50 = embedding_search(query, k=50)
scores = reranker.predict([(query, doc) for doc in top_50])
top_5 = [top_50[i] for i in np.argsort(scores)[-5:][::-1]]
```

**Why it works:** the bi-encoder (separate query/doc embeddings) loses information when matching. The cross-encoder sees both at once with full attention.

**Use cases:** the single biggest practical quality lift in RAG. Most production RAG pipelines do this. Often missing from research papers.

### 2.6 Concept arithmetic / steering vectors

In word2vec, "king - man + woman ≈ queen." That phenomenon survives (less cleanly) in modern embeddings.

```python
# Find the "recency" direction
new_memory_avg = mean(embs_of_new_memories)
old_memory_avg = mean(embs_of_old_memories)
recency_vector = new_memory_avg - old_memory_avg

# Steer a query toward recent memories
query_emb = model.encode(query)
biased_query = query_emb + alpha * recency_vector
```

**Use cases:**
- Bias retrieval toward recent / specific / formal content
- Concept-level filtering ("subtract the 'cooking' direction" to avoid cooking results)
- Interpretability (linear probes for what's encoded)

**Mature version:** the field of "representation engineering" / "control vectors" applies this to LLM activations, not just static embeddings, but the principle is the same.

### 2.7 Embedding-based in-context-example selection

For LLM prompting, the choice of few-shot examples matters a lot. Embed the test query, retrieve the most similar training examples to use as ICL examples.

```python
def select_examples(query, training_set, k=3):
    query_emb = model.encode(query)
    train_embs = model.encode([ex["input"] for ex in training_set])
    top_k = np.argsort(cosine_sim(query_emb, train_embs))[-k:]
    return [training_set[i] for i in top_k]
```

**Use cases:**
- Dynamic few-shot prompting
- Personalisation (retrieve examples most similar to the user)
- Curriculum learning ordering

### 2.8 Hallucination detection

If a model's output should be grounded in retrieved context, embed both and check similarity. A model output that's far from any retrieved chunk is likely hallucinated.

```python
def hallucination_score(model_output, retrieved_chunks):
    output_emb = embedder.encode(model_output)
    chunk_embs = embedder.encode(retrieved_chunks)
    return 1 - cosine_sim(output_emb, chunk_embs).max()
```

**Use cases:** safety / quality monitoring in production RAG. Cheap, model-agnostic.

### 2.9 Code search and similarity

Code embeddings (CodeBERT, UniXcoder, jina-code, voyage-code-3) understand the semantic meaning of code, not just keyword matching. Search "function that downloads a file" → returns code that does that, regardless of variable names.

**Use cases:**
- Memory for code agents (HippoRAG-CL territory)
- Cross-codebase search
- Vulnerability matching (find code patterns similar to known vulnerabilities)

### 2.10 Cross-modal retrieval and bridging

With CLIP and successors, text and images live in the same embedding space.

```python
text_emb = clip_model.encode_text("a black cat on a sofa")
image_embs = clip_model.encode_images(image_corpus)
matching_images = top_k(cosine_sim(text_emb, image_embs), k=10)
```

**Use cases:**
- Text-to-image search (and reverse)
- Memory systems that handle screenshots / diagrams
- Visual instruction following
- Multimodal QA

### 2.11 Embedding as features for downstream tasks

Feed embeddings into a tiny classifier / regressor for a domain-specific task. Far more sample-efficient than training a neural net from scratch.

```python
from sklearn.linear_model import LogisticRegression
embs = model.encode(train_texts)
clf = LogisticRegression().fit(embs, labels)
```

**Use cases:**
- Sentiment / toxicity / quality scoring on tiny labeled datasets
- Memory importance scoring (linear regression on embedding → "this memory is worth keeping" 0/1)
- Pre-built signals for any downstream pipeline

---

## Section 3 — Hybrid retrieval (the production pattern)

The dominant production pattern is *not* "dense embeddings alone" or "BM25 alone." It's a hybrid:

```
Query → [BM25 sparse search] → top-50 by keyword match
      → [Dense embedding search] → top-50 by semantic match
      → [Union and rerank with cross-encoder] → final top-5
```

**Why hybrid:**
- BM25 catches rare terms and exact keyword matches that dense embeddings miss.
- Dense embeddings catch paraphrases and semantic relationships that BM25 misses.
- Cross-encoder reranks the union with full attention.

Modern variants:
- **Dense + SPLADE** (BGE-M3 supports this natively)
- **Dense + ColBERT late interaction** (heavier but higher quality)
- **Dense + sparse + cross-encoder** (most production systems)

If your memory system isn't doing hybrid retrieval, it's probably leaving 10-20% recall on the table.

---

## Section 4 — Embedding model evaluation

How do you know if an embedder is good? Three sources of truth:

### 4.1 MTEB (Massive Text Embedding Benchmark)

https://huggingface.co/spaces/mteb/leaderboard

The de facto leaderboard. 56+ tasks across 8 task types (retrieval, classification, clustering, etc.). Cumulative score is the standard headline metric.

**Caveat:** MTEB scores have inflated over time due to test-set contamination. Treat 1-2 point differences between models as noise.

### 4.2 BEIR (Benchmarking IR)

https://github.com/beir-cellar/beir

Focused on retrieval specifically. 18 datasets. Goes deeper on retrieval than MTEB.

### 4.3 Domain-specific evals

Generic benchmarks may not match your use case. For memory specifically:
- LongMemEval, Memora, MemoryAgentBench (as covered in your existing notes)
- Build a small set of representative queries for your actual application

**Practical advice:** don't just trust MTEB. The embedder that wins for *retrieval over Wikipedia* may not be the best for *retrieval over conversation memories*. The MTEB top-3 are usually within 1 point of each other; the domain difference can be 5-10 points.

---

## Section 5 — Concrete experiments worth running

The matrix of "embedding technique × your project area":

| Technique | Worth trying in HippoRAG | Worth trying in A-Mem | Worth trying in reconsolidation project |
|---|---|---|---|
| Late chunking | Yes — long passages | Yes — long conversations | Probably |
| Matryoshka truncation | Yes — for storage | Yes | If memory grows large |
| Task-conditioned embeddings (Jina v3) | Yes — different retrieval task = different embedding | Yes | Yes |
| Cross-encoder reranking | High priority | High priority | Almost mandatory |
| Hybrid sparse+dense | Moderate priority | Moderate priority | Worth exploring |
| Semantic dedup | Low priority (HippoRAG already merges) | High priority (A-Mem can duplicate) | High priority |
| Novelty score for write policy | Low priority | Moderate priority | High priority — directly informs forgetting |
| Concept arithmetic / steering | Research-y | Research-y | Research-y |
| Multi-vector / ColBERT | Already in HippoRAG | Moderate priority | Worth exploring |

The two highest-leverage additions for your reconsolidation project:

1. **Cross-encoder reranking** — almost any retrieval-based system improves with this layer added.
2. **Novelty score for write policy** — directly maps to the dopamine-driven novelty signal in the brain memory deep-dive. A natural input to a forgetting / consolidation policy.

---

## Section 6 — Reading priority

The papers, ranked by usefulness for your project:

| # | Paper | Why | Time |
|---|---|---|---|
| 1 | **Late Chunking** (2409.04701) — already covered | Most actionable | 30 min |
| 2 | **Jina v3** (2409.10173) | Most production-relevant embedder | 45 min |
| 3 | **Matryoshka** (2205.13147) | Foundation for all variable-dim embedders | 30 min |
| 4 | **BGE-M3** (2402.03216) | Hybrid retrieval done in one model | 45 min |
| 5 | E5-Mistral (2401.00368) | LLM-as-embedder, current state of the art | 30 min |
| 6 | LLM2Vec (2404.05961) | How to make your own embedder from any LLM | 30 min |
| 7 | SPLADE (2107.05720) | Sparse learned embeddings | 30 min |
| 8 | Instructor (2212.09741) | Task-conditioned embeddings (background for Jina v3's task-LoRA) | 20 min |
| 9 | CLIP (2103.00020) | Multimodal embeddings (skip if you don't care about images) | 20 min |

Read in this order. If you only have time for two, pick **Jina v3** and **BGE-M3** — together they cover the current production frontier.

---

## Section 7 — Things I deliberately didn't cover

For completeness, here's what else exists that you might encounter:

- **Lexical embeddings (word2vec, GloVe)** — superseded by contextual embeddings; mostly historical interest now
- **Graph embeddings (node2vec, GraphSAGE)** — relevant if you go deep on HippoRAG's KG, but a separate field
- **Time-series embeddings** — niche; mostly for forecasting work
- **Audio embeddings (Whisper, CLAP)** — multimodal but different field
- **Embedding-based ICL example selection (DSP, kNN-LM)** — niche but worth knowing exists
- **Knowledge graph embeddings (TransE, RotatE, etc.)** — pre-LLM era; mostly superseded

If any of these become relevant to your project, the field is large enough that a focused literature search will turn up plenty.

---

## Sources (all downloaded as PDFs)

- **Jina v3:** papers/2409.10173-jina-v3.pdf
- **Matryoshka:** papers/2205.13147-matryoshka.pdf
- **E5-Mistral:** papers/2401.00368-e5-mistral.pdf
- **LLM2Vec:** papers/2404.05961-llm2vec.pdf
- **BGE-M3:** papers/2402.03216-bge-m3.pdf
- **SPLADE:** papers/2107.05720-splade.pdf
- **Instructor:** papers/2212.09741-instructor.pdf
- **CLIP:** papers/2103.00020-clip.pdf
- **Late Chunking:** papers/2409.04701-late-chunking.pdf (already in repo)
- **ColBERT:** papers/2004.12832-colbert.pdf (already in repo)
- **ColBERTv2:** papers/2112.01488-colbertv2.pdf (already in repo)

**MTEB leaderboard:** https://huggingface.co/spaces/mteb/leaderboard
**BEIR:** https://github.com/beir-cellar/beir
