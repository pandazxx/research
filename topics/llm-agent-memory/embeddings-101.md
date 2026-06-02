# Embeddings — Entry-Level Learning Guide

A foundational introduction to what embeddings are, why they exist, and how to use them. Written as a starting point — no prerequisites beyond comfort with Python and a vague familiarity with the idea that AI models exist.

If you already understand cosine similarity and have used `sentence-transformers`, this is too basic. Read `embeddings-beyond-cosine.md` and `embedding-applications-survey.md` instead.

---

## 1. The one-paragraph answer

An **embedding** is a list of numbers (a vector) that represents the meaning of something — most commonly a piece of text. The key property: two pieces of text with similar meaning produce *similar* vectors. So "dog" and "puppy" end up close in the vector space; "dog" and "quantum mechanics" end up far apart. Once you have this, you can do math on meaning. "Find documents like this query?" Embed both, find the closest vectors. "Group these sentences by topic?" Cluster the vectors. "Detect duplicates?" Check if vectors are nearly identical.

That's it. Everything else is engineering details.

---

## 2. The intuition: geometric meaning

Imagine a 2D map of every word in English. Hand-drawn. Where do you put each word?

A reasonable rule: **put related words close to each other.** So you might end up with:

```
                    ↑
                    │
       cat • • dog              banana • • apple
            • puppy             • fruit
                    │
       car ← ← ← ← ←┼→ → → →   chair
        • truck             •  table
                    │       •  desk
                    ↓
```

Now if I ask "find me the word closest to 'kitten'," you can answer geometrically: look at the position of kitten, find the nearest neighbour, return that.

An embedding is exactly this idea, but:

- Instead of 2 dimensions, typical embeddings use **384, 768, 1024, or 3072 dimensions**.
- Instead of hand-drawn, the positions are **learned by a neural network** from massive text data.
- It works for sentences and paragraphs, not just single words.

That's the whole concept. Everything else is "how do we actually build this map," "how do we measure distance in many dimensions," and "what can we do once we have it."

---

## 3. Why embeddings exist (the problem they solve)

Before embeddings, the standard way to "find similar text" was **keyword matching**:

> Query: "How do I install Python?"
> Documents that contain the words "install" AND "Python" → relevant.

This is what classical search engines (and `grep`) do. It's fast and exact. But it has obvious failures:

- **Synonyms:** a document that says "setting up Python" doesn't match the query.
- **Paraphrasing:** "Python installation guide for beginners" matches less well than a document that just contains both words.
- **Cross-language:** "Cómo instalar Python" doesn't match at all.
- **Conceptual relationship:** a document about "pip install" doesn't even contain the word "Python" but is highly relevant.

Embeddings solve this. The query "How do I install Python?" and the document "Python installation guide for beginners" become *vectors* whose mathematical similarity is high — because the embedding has learned that "install" and "installation" mean similar things, that "Python" appears in both, and that the topic is the same.

The trade-off: embeddings are slower and less exact than keyword matching. But for semantic similarity, they're transformative.

---

## 4. What an embedding looks like

A real embedding from a real model:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

vector = model.encode("The cat sat on the mat.")
print(vector.shape)   # (384,)
print(vector[:10])    # [0.012, -0.045, 0.103, 0.024, -0.018, 0.067, ...]
```

So an embedding is just an array of 384 floating-point numbers (or 768, or 1024 depending on the model). You can't read it. You can't meaningfully interpret any single number. The "meaning" is encoded in the *pattern* of all 384 numbers together, and especially in the *relative position* of this vector compared to other vectors.

This is important: **you never inspect individual embedding values.** You only compare them to other embeddings.

---

## 5. Measuring similarity: cosine similarity

Once you have vectors, you need a way to ask "how similar are these two vectors?"

The standard answer is **cosine similarity**:

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

vec_dog = model.encode("dog")
vec_puppy = model.encode("puppy")
vec_quantum = model.encode("quantum mechanics")

print(cosine_similarity(vec_dog, vec_puppy))    # 0.78  (very similar)
print(cosine_similarity(vec_dog, vec_quantum))  # 0.12  (unrelated)
```

Cosine similarity returns a number between -1 and 1:

| Value | Meaning |
|---|---|
| 1.0 | Identical |
| 0.7–0.9 | Highly similar (synonyms, paraphrases) |
| 0.4–0.7 | Related but distinct |
| 0.0–0.3 | Largely unrelated |
| Below 0 | Rare with modern embeddings; usually near 0 |

**Why cosine specifically?** It measures the *angle* between two vectors, ignoring their magnitudes. Two vectors pointing in the same direction have cosine similarity 1, regardless of how long either one is. This turns out to be the right way to compare text embeddings — sentence length shouldn't matter for similarity.

Two other distance measures exist (Euclidean, dot product), but cosine is the default for almost all text embeddings. You can mostly forget the other two until you need them.

---

## 6. The famous word2vec example

The classic demonstration that embeddings encode *meaning* in their geometry:

> Embed "king", "man", "woman", "queen" as vectors.
> Compute: `vec("king") - vec("man") + vec("woman")`
> The result is closest to: `vec("queen")`.

In words: the *direction* from "man" to "king" represents something like "royalty," and applying that same direction starting from "woman" lands you at "queen." The embedding space has learned a "royalty" direction.

This works (approximately) because the model saw enough text where (king, man) and (queen, woman) appeared in analogous contexts. The structure of the embedding space ends up encoding these relationships geometrically.

Modern sentence embeddings show this less cleanly than word2vec did, but the principle is the same: **the geometry of the space encodes meaning**.

---

## 7. Where embeddings come from (training in one paragraph)

Embeddings are produced by a neural network — specifically a *transformer*, the same family of models behind ChatGPT. The model is trained by feeding it pairs of texts and telling it which pairs should be "similar" and which "different." For example:

- (question, correct answer) → make their embeddings close
- (question, random unrelated answer) → make their embeddings far

After training on millions of such pairs, the model learns to produce embeddings such that *meaning-similar* texts end up close. This training process is called **contrastive learning**.

You don't need to understand the training math to use embeddings — but it explains why an embedding for cooking text from a model trained mostly on news won't be as good as one trained on cooking. The model encodes whatever's in its training data.

---

## 8. Your first embedding code

```python
# Install: pip install sentence-transformers
from sentence_transformers import SentenceTransformer
import numpy as np

# Load a small, fast model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed some sentences
sentences = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",
    "Python is a programming language.",
    "Snakes are reptiles.",
]
embeddings = model.encode(sentences)  # shape: (4, 384)

# Compute similarity between sentence 0 and each other sentence
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

for i in range(1, 4):
    sim = cosine(embeddings[0], embeddings[i])
    print(f"Similarity between '{sentences[0]}' and '{sentences[i]}': {sim:.3f}")
```

Expected output:

```
Similarity between 'The cat sat on the mat.' and 'A feline rested on the rug.': 0.781
Similarity between 'The cat sat on the mat.' and 'Python is a programming language.': 0.084
Similarity between 'The cat sat on the mat.' and 'Snakes are reptiles.': 0.221
```

Notice:
- The cat sentence and the feline sentence are *semantically* very similar (0.78), even though they share no exact words.
- The Python sentence is unrelated (0.08).
- The snake sentence is mildly related (0.22) because both are about animals, even though cats and snakes are very different.

That single experiment teaches more than reading any paper. **Run it.**

---

## 9. The main thing embeddings are used for: semantic search

The pattern:

```python
# Step 1: Embed your "corpus" (the set of documents to search)
corpus = ["doc 1", "doc 2", ..., "doc 1000"]
corpus_embeddings = model.encode(corpus)  # shape: (1000, 384)

# Step 2: When a query comes in, embed it
query = "user question"
query_embedding = model.encode(query)

# Step 3: Find the document with highest similarity
similarities = corpus_embeddings @ query_embedding  # (1000,)
top_3_indices = np.argsort(similarities)[-3:][::-1]
top_3_docs = [corpus[i] for i in top_3_indices]
```

This is **semantic search** — finding documents by *meaning* rather than by exact word match.

In production this scales to millions or billions of vectors using **vector databases** (FAISS, Pinecone, Weaviate, Milvus, pgvector, Chroma). The vector DB handles efficient nearest-neighbour search over huge corpora.

---

## 10. What embeddings encode (and what they don't)

An embedding captures *whatever was useful for the contrastive training objective*. In practice, this includes:

| What | How well |
|---|---|
| Topic / domain | Very well |
| Sentiment / tone | Moderately well |
| Synonyms and paraphrasing | Very well |
| Concept-level similarity | Well |
| Cross-language equivalence (multilingual models) | Well |
| Logical relationships (cause, sequence) | Poorly |
| Numerical precision (numbers, dates) | Poorly |
| Negation ("not happy" vs "happy") | Surprisingly poorly — often considered similar! |

The negation issue is worth remembering: a model that hasn't been specifically trained to handle negation will often embed "I love this movie" and "I do not love this movie" as quite similar. Modern instruction-tuned embedders handle this better but it's still a known weakness.

---

## 11. Different embedding models, different perspectives

Important: **embeddings from different models live in different spaces and cannot be compared.**

```python
model_a = SentenceTransformer("all-MiniLM-L6-v2")           # 384 dim
model_b = SentenceTransformer("BAAI/bge-large-en-v1.5")     # 1024 dim

emb_a = model_a.encode("hello")
emb_b = model_b.encode("hello")

# Different dimensions, fundamentally incomparable.
# Even if you somehow projected to the same dim, the spaces are different.
```

Practical consequence: **once you pick a model and embed your data, you're committed to that model.** If you switch, you have to re-embed everything. This is the "embedding drift" / "vendor lock-in" problem we covered in earlier notes.

For an entry-level project, this means: pick one model and stick with it. Recommended defaults for English text:

- **`all-MiniLM-L6-v2`** — small, fast, free (384 dim). The "hello world" of embeddings.
- **`BAAI/bge-large-en-v1.5`** — much better quality, free, local (1024 dim).
- **`text-embedding-3-small`** (OpenAI API) — cheap and good (1536 dim, truncatable).
- **`voyage-3`** (Voyage AI API) — long-context, very good (1024 dim).

If in doubt, start with `all-MiniLM-L6-v2`. It's the model everyone uses for prototyping.

---

## 12. The mental model to internalise

Three principles that hold across almost all uses of embeddings:

### Principle 1 — Embeddings encode meaning as geometry.

If two pieces of text have similar meaning, they have similar vectors. Distance in the vector space corresponds (roughly) to semantic distance. This is the fundamental property and everything else follows from it.

### Principle 2 — Embeddings are model-specific.

Different models produce incompatible vectors. You can't mix and match. Once you commit to a model, you've committed to a perspective.

### Principle 3 — Embeddings compress meaning, lossily.

A 384-dim vector cannot capture every nuance of a paragraph. Similar embedding ≠ identical meaning. Use them as a *first-pass filter* for finding candidates, not as the final word on semantic equivalence.

---

## 13. Common beginner mistakes (and how to avoid them)

| Mistake | Why it's wrong | Fix |
|---|---|---|
| Comparing embeddings from different models | Different vector spaces are not comparable | Always use one model end-to-end |
| Treating cosine similarity as absolute | A score of 0.6 doesn't mean "60% similar" in any meaningful sense | Use it for *ranking*, not absolute interpretation |
| Forgetting to normalise text | Different casing, whitespace, punctuation can shift embeddings | Many models handle this internally; if in doubt, lowercase + strip |
| Embedding very long text | Most models silently truncate at 512 tokens | Use a long-context model (Jina v3, Voyage-3) for long inputs |
| Storing in float64 | Embeddings are fine in float16 or even int8 with quantisation | Use float32 by default; quantise if storage matters |
| Embedding raw HTML / markdown | Tags and formatting waste embedding capacity | Strip formatting first; embed clean text |

---

## 14. What to learn next

Now that you understand the basics:

1. **Vector databases** — how to scale beyond a million vectors efficiently. Read about FAISS, then look at a vector DB like Chroma or Pinecone.
2. **Cosine vs other metrics** — when to use Euclidean distance, dot product, or learned similarity.
3. **Reranking** — how a 2-stage pipeline (embeddings for candidates + cross-encoder for final scoring) almost always beats embeddings alone.
4. **Hybrid retrieval** — combining keyword search (BM25) with semantic search.
5. **Late chunking and other "what else can embeddings do" tricks** — covered in `embeddings-beyond-cosine.md`.
6. **Embedding model evaluation** — the MTEB benchmark and how to read it.
7. **Multimodal embeddings** — CLIP and friends, for image+text.

Each of these is worth 1–2 hours of dedicated learning. Don't try to absorb all at once.

---

## 15. A 30-minute hands-on exercise

If you do nothing else after reading this, do this exercise:

```python
# pip install sentence-transformers numpy

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

corpus = [
    "Python is a programming language.",
    "Cats are popular pets.",
    "Machine learning is a field of artificial intelligence.",
    "Dogs are loyal animals.",
    "JavaScript is used for web development.",
    "Neural networks are a type of machine learning model.",
    "Snakes are cold-blooded reptiles.",
    "Apple Inc. makes the iPhone.",
    "TypeScript adds types to JavaScript.",
    "Coffee is a popular morning beverage.",
]

corpus_embs = model.encode(corpus)

# Try these queries:
queries = [
    "What language do I use for the web?",
    "Tell me about AI",
    "I love my pet dog",
    "Hot drinks",
    "How do mobile phones work?",
]

for query in queries:
    query_emb = model.encode(query)
    sims = corpus_embs @ query_emb / (
        np.linalg.norm(corpus_embs, axis=1) * np.linalg.norm(query_emb)
    )
    top_3 = np.argsort(sims)[-3:][::-1]
    print(f"\nQuery: {query}")
    for i in top_3:
        print(f"  {sims[i]:.3f}  {corpus[i]}")
```

Run this. Observe how:
- "What language do I use for the web?" surfaces JavaScript and TypeScript even though "web" doesn't appear in either.
- "Tell me about AI" surfaces machine learning and neural networks.
- "I love my pet dog" finds the dog sentence first, then cats, then snakes.
- "Hot drinks" finds coffee despite no shared words.

After 30 minutes of playing with this, you'll have stronger intuition for embeddings than reading 10 papers.

---

## Glossary

- **Embedding**: A vector (array of numbers) that represents the meaning of text (or image, or other input).
- **Vector**: An ordered list of numbers. A 384-dimensional vector has 384 numbers.
- **Cosine similarity**: A measure of how similar two vectors are, between -1 and 1.
- **Encoder model**: A neural network that takes text and produces an embedding.
- **Contrastive learning**: The training procedure that teaches a model to produce embeddings where similar things are close.
- **Vector database**: A specialised database for efficiently searching over many embeddings.
- **Semantic search**: Finding documents by *meaning* rather than by exact word match. The main use of embeddings.
- **Embedding dimension**: The number of numbers in a single vector. Common values: 384, 768, 1024, 1536, 3072.
- **MTEB**: Massive Text Embedding Benchmark — the standard leaderboard for comparing embedding models.

---

## Where to go from here

Once this is comfortable, read in this order:

1. `embeddings-beyond-cosine.md` — what else embeddings can do (semantic chunking, late chunking, ColBERT)
2. `embedding-applications-survey.md` — broader applications and the current generation of models
3. `commercial-embeddings-deep-dive.md` — the commercial provider landscape

These three together cover the practitioner-relevant landscape.
