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

## 9. Going one level deeper — what happens inside an embedder

So far we've treated the embedder as a black box: text goes in, a vector comes out. Now let's open the box. This section is optional for using embeddings, but understanding it prevents a lot of confusion later — especially around advanced topics like late chunking.

### The pipeline from text to vector

What you call `model.encode("some text")` actually involves four distinct stages:

```
"Maria is a chemistry teacher."   ← Raw text (Python string)
            ↓
       Tokenization                 ← Split into subword pieces
            ↓
"[CLS]", "Maria", "is", "a", "chemistry", "teacher", ".", "[SEP]"
            ↓
       ID lookup                    ← Each piece → an integer
            ↓
[101, 3854, 2003, 1037, 6248, 3836, 1012, 102]   ← Token IDs
            ↓
       Tensor packaging             ← Wrap as tensor with batch dim
            ↓
tensor([[101, 3854, 2003, 1037, 6248, 3836, 1012, 102]])   ← shape (1, 8)
            ↓
       Forward pass through transformer layers
            ↓
Per-token vectors (last hidden state)   ← shape (1, 8, 768)
            ↓
       Pooling (mean / CLS / etc.)
            ↓
Sentence embedding                  ← shape (768,)
```

So the input the model actually consumes is **not** text — it's a 2D tensor of integers with shape `(batch_size, sequence_length)`.

### Tokenization: text becomes integer IDs

A tokenizer splits raw text into **subword pieces** using a fixed vocabulary learned during model training:

```python
tokenizer.tokenize("Antidisestablishmentarianism is a long word.")
# → ['Anti', 'dis', 'establish', 'ment', 'arianism', ' is', ' a', ' long', ' word', '.']
```

Notice the long word was split into five sub-pieces. The tokenizer breaks unfamiliar words into known sub-pieces from its vocabulary. This is why we say "tokens" rather than "words" — they often don't align with word boundaries.

**Rough rule of thumb:** for English, you get about **1.3 tokens per word**. A 100-word sentence becomes ~130 tokens.

After tokenization, each piece is looked up in the vocabulary table and replaced by an integer ID:

```python
tokenizer.vocab["maria"]    # → 3854
tokenizer.vocab["[CLS]"]    # → 101
tokenizer.vocab_size        # → 30522 (for BERT-base)
```

### Special tokens

The tokenizer automatically adds special markers to your sequence:

| Token | Purpose |
|---|---|
| `[CLS]` | Classification token, prepended to every input. Its final hidden state is often used as the pooled sentence embedding. |
| `[SEP]` | Separator, appended at the end (and between segments if you input two sentences). |
| `[PAD]` | Padding token, used to fill out shorter sequences to a fixed length. |
| `[MASK]` | Used during pretraining for masked language modeling. Not relevant for inference. |
| `[UNK]` | Unknown token, used when the input has characters the tokenizer can't handle. |

You don't manage these directly — the tokenizer adds them — but they do take up positions in the sequence.

### The input shape, formally

```
Input tensor shape: (batch_size, sequence_length)
Input tensor dtype: int64 (integer token IDs)
```

Where:

- `batch_size` = number of input sequences processed together
- `sequence_length` = number of tokens per sequence (padded to a uniform length within a batch)

### Batching and the attention mask

You almost always embed many sentences at once for efficiency. The model processes a batch in roughly the same time as one sentence:

```python
sentences = [
    "Maria is a chemistry teacher.",  # 8 tokens
    "The cat sat on the mat.",        # 9 tokens
    "Python is short.",                # 6 tokens
]
inputs = tokenizer(sentences, padding=True, return_tensors="pt")
print(inputs["input_ids"].shape)  # torch.Size([3, 9])
```

Shape `(3, 9)` means 3 sentences, 9 tokens each. The shorter sentences are **padded** with the `[PAD]` token (ID 0) until they all match the longest in the batch.

The tokenizer also returns an **attention mask** — a parallel tensor of 1s and 0s marking which positions are real tokens vs padding:

```python
inputs["attention_mask"]
# tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1, 1, 0, 0, 0]])
```

The model uses this mask to **ignore** padding positions during attention. Padding tokens take up space in the tensor (they have to, for shape uniformity), but they don't affect the meaningful tokens' representations.

### Inside the model: layers and hidden states

Once the integer IDs enter the model, they go through a stack of identical **transformer layers** (12, 24, or more depending on the model). Each layer transforms the representations:

```
Token IDs                  shape: (1, 8)         dtype: int64
        ↓
[Embedding lookup layer]   ← maps each ID to an initial 768-dim vector
        ↓
Initial embeddings         shape: (1, 8, 768)    dtype: float32
        ↓
[Transformer layer 1]      ← self-attention + feed-forward
        ↓
Hidden state (layer 1)     shape: (1, 8, 768)    dtype: float32
        ↓
[Transformer layer 2]
        ↓
Hidden state (layer 2)     shape: (1, 8, 768)    dtype: float32
        ↓
   ...
        ↓
[Transformer layer N]
        ↓
Last hidden state          shape: (1, 8, 768)    dtype: float32
        ↓
[Pooling]                  ← mean or CLS-token
        ↓
Pooled embedding           shape: (1, 768)       dtype: float32
```

A **hidden state** is just the output of one transformer layer. It's a 3D tensor `(batch_size, sequence_length, hidden_dim)` — one vector per token at that particular layer's level of refinement.

Each layer does two operations:
1. **Self-attention**: each token "looks at" every other token and updates itself based on what it sees.
2. **Feed-forward**: each token's representation is further refined.

The deeper you go in the stack, the more contextually-rich each token's vector becomes. By the final layer:
- "Maria" has integrated context from the whole sentence
- "She" has integrated context that includes "Maria"
- Every token "knows" about every other token

The **last hidden state** — the output of the final transformer layer, before any pooling — is what you want for late chunking. It contains the fully-contextualized per-token vectors.

The word "hidden" is a relic of older neural network terminology. It just means "internal" — not the input, not the final pooled output, but the intermediate per-token state inside the network. Nothing mysterious.

### The two "embeddings" (don't confuse them)

The word "embedding" is used in two different ways inside a transformer:

1. **Initial token embeddings** (from the embedding lookup layer): one 768-dim vector per token ID, looked up from a learned table. These are the *starting* vectors before the transformer layers do their work. They're not contextualized.
2. **Sentence embedding** (the final pooled output): one 768-dim vector summarising the whole sentence. This is what you use for retrieval.

The transformer's job is to convert (1) into (2), via the intermediate hidden states. When people say "embedding" without qualification, they almost always mean (2).

### Maximum sequence length

Every embedder has a maximum input length:

| Model | Max sequence length |
|---|---|
| `all-MiniLM-L6-v2` | 256 tokens |
| BERT-base | 512 tokens |
| Most OpenAI embeddings | 8191 tokens |
| Jina v3 | 8192 tokens |
| Voyage-3-large | 32768 tokens |

If your input exceeds the limit, the tokenizer silently **truncates** (by default) — meaning the trailing portion of your text is discarded before reaching the model. This is a common source of silent quality issues: you embed a long document, the embedder only sees the first 512 tokens, and you don't realise the rest was lost.

**Always check the model's max length.** If you might exceed it, either chunk your input or switch to a long-context embedder.

### Putting it together: complete example with shapes

```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3")

texts = [
    "Maria is a chemistry teacher.",
    "She lives in Boston.",
]

# Tokenize and batch
inputs = tokenizer(texts, padding=True, return_tensors="pt")
print(inputs["input_ids"].shape)       # torch.Size([2, 9])
print(inputs["attention_mask"].shape)  # torch.Size([2, 9])

# Run the model
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

print(outputs.last_hidden_state.shape)  # torch.Size([2, 9, 1024])

# Each text's per-token vectors:
text1_vectors = outputs.last_hidden_state[0]  # shape (9, 1024)
text2_vectors = outputs.last_hidden_state[1]  # shape (9, 1024)

# To get the standard sentence embedding, apply mean pooling
# (respecting the attention mask so padding is ignored)
def mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

sentence_embs = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
print(sentence_embs.shape)  # torch.Size([2, 1024])
```

### Summary of shapes

| Stage | Shape | dtype |
|---|---|---|
| Raw text | a Python string | str |
| Tokenized text | list of token strings | list[str] |
| Token IDs | (batch_size, sequence_length) | int64 |
| Attention mask | (batch_size, sequence_length) | int64 |
| Initial token embeddings | (batch_size, sequence_length, hidden_dim) | float32 |
| Hidden state at any layer | (batch_size, sequence_length, hidden_dim) | float32 |
| Last hidden state | (batch_size, sequence_length, hidden_dim) | float32 |
| Pooled sentence embedding | (batch_size, hidden_dim) | float32 |

The two shapes you'll see most often:
- **Input**: `(batch_size, sequence_length)` of integers
- **Output (pooled)**: `(batch_size, hidden_dim)` of floats

For late chunking, you skip the pooling step and work directly with the `(batch_size, sequence_length, hidden_dim)` last hidden state.

### Why this all matters

Once you internalise this picture, several things become clear:

- **Why different models have incompatible vector spaces** — they have different vocabularies, different layer counts, different training. The "768" in BERT and the "768" in another model don't refer to the same coordinate system.
- **Why some embedders support late chunking and others don't** — late chunking requires access to the last hidden state, not just the pooled output. OpenAI's API doesn't expose it; most open-source models do.
- **Why max sequence length matters** — your text becomes integer tokens, and the model's architecture has a fixed window. Exceed it and the rest is dropped.
- **Why batching matters** — efficiency comes from processing many sequences in one forward pass. Padding + attention masks enable batches of mixed lengths.

This is enough understanding to read advanced papers without losing the thread.

---

## 10. The BERT family of encoders

When we say "embedder" or "encoder," we almost always mean a model from the **BERT family**. This is a lineage of encoder-only transformer models that started with Google's BERT in 2018 and has dominated text embedding ever since.

### The original BERT (2018)

BERT — **B**idirectional **E**ncoder **R**epresentations from **T**ransformers — was introduced by Devlin et al. at Google in 2018. Two architectural choices made it landmark:

1. **Encoder-only.** BERT keeps only the encoder half of the original transformer architecture (Vaswani et al., 2017 — "Attention Is All You Need"). No decoder, no generation. Just a stack of transformer encoder layers that take tokens in and produce contextualized token vectors out.
2. **Bidirectional attention.** Earlier language models (like GPT-1) used *causal* attention — each token could only attend to tokens before it. BERT removed this restriction. Every token attends to every other token. This makes BERT terrible at left-to-right generation but excellent at *understanding* existing text.

BERT was pretrained on the **Masked Language Modeling (MLM)** task: randomly mask 15% of input tokens, ask the model to predict them from the surrounding context. This forces the model to build rich contextual representations of every token.

Two sizes were released:
- **BERT-base**: 12 layers, 768 hidden dim, ~110M parameters.
- **BERT-large**: 24 layers, 1024 hidden dim, ~340M parameters.

### The BERT family tree

After BERT, dozens of variants and successors emerged. The ones you'll encounter in this domain:

| Model | Year | What's new |
|---|---|---|
| **RoBERTa** | 2019 | Same architecture, better training: more data, longer training, removed next-sentence-prediction. Facebook AI. |
| **DistilBERT** | 2019 | Distilled to 6 layers — 60% smaller, ~95% of quality. |
| **ALBERT** | 2019 | Parameter sharing across layers; much smaller memory footprint. |
| **DeBERTa** | 2021 | Disentangled position–content attention. State-of-the-art for many classification tasks. |
| **MPNet** | 2020 | Combined MLM + permuted language modeling objectives. |
| **Sentence-BERT** | 2019 | First sentence-level fine-tuning of BERT for embeddings. Source of the sentence-transformers library. |
| **MiniLM** | 2020 | Distilled into a small/fast sentence embedder — basis of `all-MiniLM-L6-v2`. |
| **E5, BGE, GTE families** | 2022–2024 | Modern instruction-tuned sentence embedders, mostly BERT-family backbones (XLM-RoBERTa is the common base). |
| **Jina v3** | 2024 | XLM-RoBERTa backbone + task-specific LoRA adapters. |

All of these share BERT's core architecture: an encoder-only transformer stack with bidirectional attention. They differ in size, training data, training objectives, and downstream fine-tuning recipes.

### Why this matters for embeddings

Every embedder I've recommended in these notes (`all-MiniLM-L6-v2`, `BAAI/bge-large-en-v1.5`, `jinaai/jina-embeddings-v3`, etc.) is a BERT-family model. They all:
- Use bidirectional attention.
- Were initially pretrained with some form of MLM.
- Were then fine-tuned (usually with contrastive learning) for sentence-level similarity.
- Output per-token vectors that get pooled into one sentence embedding.

The recent shift to decoder-only LLMs as embedders (E5-Mistral, LLM2Vec, NV-Embed-v2) is the first serious departure from this lineage — but for now, "embedder" still mostly means "a BERT-family model."

### Encoder vs decoder distinction

Worth being clear about this since it explains why GPT-style models aren't great default embedders:

| Architecture | Attention | Best at | Examples |
|---|---|---|---|
| Encoder-only | Bidirectional | Understanding, classification, embedding | BERT, RoBERTa, MiniLM, BGE |
| Decoder-only | Causal (one-way) | Generation, completion | GPT, Llama, Mistral, Claude |
| Encoder-decoder | Both (different layers) | Translation, summarization | T5, BART |

Encoder-only models can "see the whole context at once" via bidirectional attention. This is exactly what you want for producing a representation of a piece of text. Decoder-only models build up understanding strictly left-to-right, which is less natural for producing position-symmetric outputs like sentence embeddings.

(LLM2Vec and similar techniques essentially convert decoder-only LLMs to act bidirectionally for embedding purposes, getting around this limitation.)

---

## 11. Anisotropy — why token vectors cluster

Once you start running experiments with token-level embeddings, you'll notice something surprising: **the cosine similarity between any two tokens in the same sentence is unexpectedly high** — typically 0.80–0.95, even for tokens that should be conceptually different.

This isn't noise or a reproduction bug. It's a well-documented property of BERT-family encoders called **anisotropy**.

### The phenomenon

In an ideal embedding space, token vectors would be spread roughly uniformly across a high-dimensional sphere. Two random tokens would have cosine similarity near 0; semantically related tokens would have meaningfully higher similarity than that.

In practice, **BERT-family encoders produce token vectors that occupy a narrow cone of the embedding space.** Two random tokens have cosine similarity ~0.4 as a baseline floor. Two tokens in the same sequence have ~0.8 or higher.

Empirically, the typical similarity ranges for a vanilla BERT-family model:

| Comparison | Typical cosine similarity |
|---|---|
| Same token, same sequence | 0.95+ |
| Adjacent tokens, same sequence | 0.90–0.95 |
| Distant tokens, same sequence | 0.75–0.85 |
| Same token, different sequences | 0.65–0.80 |
| Random tokens, different sequences | 0.30–0.55 |

The whole "interesting" range of cosine similarity is compressed into a small band, with a hard floor (~0.3+) imposed by anisotropy.

### Why it happens

Several factors compound:

1. **Self-attention is a weighted average.** Stacks of attention layers cumulatively pull token vectors toward each other. By the last layer of a 12- or 24-layer encoder, every token has been blended with every other token in its sequence many times.
2. **Residual connections only partially preserve token identity.** They help but the cumulative mixing dominates.
3. **Encoders aren't trained to keep tokens distinct.** They're trained so that *pooled sentence vectors* discriminate between similar and dissimilar sentences. Per-token discrimination isn't an objective.
4. **Position embeddings drift.** Position information gets gradually diluted across layers.

Net effect: by the time you reach the last hidden state, the per-token vectors look very similar within a sequence — they all live in the same anisotropic cone.

### Why does anything work, then?

If per-token vectors are nearly identical, how do embedders retrieve correctly?

**Pooling washes out the within-sequence noise.** When you mean-pool many token vectors that are all ~0.9 similar to each other, you get a vector that's essentially the centroid of that cone. But two *different sentences* produce centroids in *different* parts of the embedding space — and those centroids are well-separated, even though the individual tokens within each sentence are not.

The retrieval signal lives in the **between-sentence** dimension, which the pooling step preserves. The **within-sentence** dimension is noisy and largely uninformative.

### A concrete example

Take this sentence:

> "Joey is living in Singapore. She is a software engineer. My wife is living in Singapore too. She is a house wife."

Running with BGE-large-en-v1.5, comparing per-token vectors from the last hidden state:

```
she1 ↔ she2:   0.959   (same token, same sequence)
she2 ↔ wife:   0.934   (adjacent tokens, same sequence)
she1 ↔ Joey:   0.849   (distant tokens, same sequence)
joey ↔ wife:   0.794   (very distant tokens, same sequence)
```

All four pairs are above 0.79, even though the underlying tokens are very different — proper noun, common noun, pronoun. The *ordering* is meaningful (0.96 > 0.93 > 0.85 > 0.79 tracks the intuitive similarity), but the entire scale is compressed into the 0.79–0.96 band rather than spanning the theoretical 0.0–1.0.

For comparison, two random tokens from *different* sequences typically score 0.30–0.55. The within-sequence floor of ~0.79 is a direct measure of how strong the anisotropy is.

### Implications

1. **Don't trust token-level cosine similarity at face value.** A score of 0.85 doesn't mean "85% similar in meaning"; it might just mean "in the same sentence."
2. **Pooled sentence embeddings still work** because pooling extracts the centroid of the anisotropic cone, and centroids of different cones (different sentences) are well-separated.
3. **Coreference resolution at the embedding level is weak.** The "she refers to Joey" signal exists — you can detect it as a small bump in cosine similarity — but it's swamped by the anisotropic baseline.
4. **Some embedders are designed to be less anisotropic.** E5-Mistral, NV-Embed-v2, and other recent decoder-LLM-based embedders are explicitly trained with isotropy-promoting objectives. Older BERT-family models (vanilla BERT, RoBERTa) are highly anisotropic.
5. **Whitening transformations can post-process embeddings** to be more isotropic. Standard trick if you need cleaner per-token signals.

### Further reading

The phenomenon was rigorously characterised in 2019–2021 by three papers worth knowing:

- **Ethayarajh (EMNLP 2019)**, *"How Contextual are Contextualized Word Representations?"* — first thorough documentation that BERT-family token vectors are highly anisotropic. Showed empirically that contextualized vectors are *less* isotropic than the static word vectors (word2vec, GloVe) they were supposed to improve over. arXiv:1909.00512 — saved as `papers/1909.00512-ethayarajh-contextual.pdf`.
- **Li et al. (EMNLP 2020)**, *"On the Sentence Embeddings from Pre-trained Language Models"* — introduced **BERT-flow**, a learned invertible mapping that transforms BERT's anisotropic distribution into an isotropic Gaussian. Demonstrated significant retrieval improvements after this transformation. arXiv:2011.05864 — saved as `papers/2011.05864-bert-flow.pdf`.
- **Su et al. (2021)**, *"Whitening Sentence Representations for Better Semantics and Faster Retrieval"* — showed that simple statistical whitening (centring and decorrelating) gives most of BERT-flow's benefits without learning a transformation. The current standard preprocessing trick. arXiv:2103.15316 — saved as `papers/2103.15316-whitening.pdf`.

If you find yourself doing token-level analysis on BERT-family embeddings, these papers are worth reading in order. For typical sentence-level retrieval, modern instruction-tuned embedders (BGE, E5, Jina v3) handle most of the anisotropy issue internally.

---

## 12. The main thing embeddings are used for: semantic search

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

## 13. What embeddings encode (and what they don't)

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

## 14. Different embedding models, different perspectives

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

## 15. The mental model to internalise

Three principles that hold across almost all uses of embeddings:

### Principle 1 — Embeddings encode meaning as geometry.

If two pieces of text have similar meaning, they have similar vectors. Distance in the vector space corresponds (roughly) to semantic distance. This is the fundamental property and everything else follows from it.

### Principle 2 — Embeddings are model-specific.

Different models produce incompatible vectors. You can't mix and match. Once you commit to a model, you've committed to a perspective.

### Principle 3 — Embeddings compress meaning, lossily.

A 384-dim vector cannot capture every nuance of a paragraph. Similar embedding ≠ identical meaning. Use them as a *first-pass filter* for finding candidates, not as the final word on semantic equivalence.

---

## 16. Common beginner mistakes (and how to avoid them)

| Mistake | Why it's wrong | Fix |
|---|---|---|
| Comparing embeddings from different models | Different vector spaces are not comparable | Always use one model end-to-end |
| Treating cosine similarity as absolute | A score of 0.6 doesn't mean "60% similar" in any meaningful sense | Use it for *ranking*, not absolute interpretation |
| Forgetting to normalise text | Different casing, whitespace, punctuation can shift embeddings | Many models handle this internally; if in doubt, lowercase + strip |
| Embedding very long text | Most models silently truncate at 512 tokens | Use a long-context model (Jina v3, Voyage-3) for long inputs |
| Storing in float64 | Embeddings are fine in float16 or even int8 with quantisation | Use float32 by default; quantise if storage matters |
| Embedding raw HTML / markdown | Tags and formatting waste embedding capacity | Strip formatting first; embed clean text |

---

## 17. What to learn next

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

## 18. A 30-minute hands-on exercise

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
