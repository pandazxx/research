# WISE — Reading Guide

A guided read of *WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models* (Wang et al., NeurIPS 2024).

Paper: `papers/2405.14768-wise.pdf` (in this repo) | arXiv: https://arxiv.org/abs/2405.14768
Code: https://github.com/zjunlp/EasyEdit

**Estimated reading time:** 60–90 minutes for the parts you need. 2-3 hours if you read end-to-end.

---

## Why this paper matters for your project

WISE solves a problem in *model-weight editing* that maps almost directly onto what you'd face building reconsolidation for external memory. **Don't worry about the weight-editing details** — what matters is the *framing* and the *architectural ideas*, both of which are highly transferable.

The single most valuable thing you'll get from this paper: a precise vocabulary for the trade-off you'll be designing around.

---

## The one big idea: the impossible triangle

WISE argues that any lifelong editing system must balance three properties:

| Property | What it means | External memory analogue |
|---|---|---|
| **Reliability** | After editing, the new fact is correctly applied | Retrieving the updated memory when asked |
| **Generalization** | The edit works for paraphrased / related queries, not just verbatim | Updated memory applies to "what city is Sam in?" *and* "where does Sam live now?" |
| **Locality** | Unrelated facts are undisturbed | Updating "Sam moved to Munich" doesn't corrupt "Sam's mother is Maria" |

**The claim:** existing methods can do 2 of 3 but not all 3 simultaneously when many edits happen sequentially.

- **Long-term memory editors** (ROME, MEMIT, fine-tuning): edit base model weights. Good reliability and generalization initially, but locality collapses as edits accumulate. *Edits interfere with prior knowledge.*
- **Working memory editors** (GRACE, DEFER): retrieve cached edits at inference. Good reliability and locality, but poor generalization. *Retrieval can't generalize to paraphrased queries.*

**Direct mapping to your project:**

| WISE's problem | Your problem |
|---|---|
| Edit model weights → conflicts pile up | Update A-Mem notes → memory evolution can corrupt unrelated content |
| Retrieve cached edits → can't generalize | HippoRAG-style override layer → edits don't generalize to paraphrases |
| **Impossible triangle for lifelong editing** | **Impossible triangle for reconsolidation in external memory** |

This is the framing your eventual paper should use, whichever direction you pick.

---

## WISE's solution in plain English

WISE proposes a **third memory location** — a "side memory" that sits between the model's weights (long-term) and inference-time retrieval (working). The side memory is itself parametric (real weights, like the main model) but **separate from the main weights**.

Three mechanisms:

1. **Side memory** — a copy of one FFN value matrix from a mid-to-late layer (e.g., layer 26 of LLaMA-2-7B). Initialised as a clone of the main memory; all edits go here.
2. **Routing** — at inference, a per-query decision: send this query to the side memory (if it's about an edited fact) or the main memory (if not). Routing is trained with a margin-based loss.
3. **Knowledge sharding + merging** — when many edits arrive, partition them into `k` subspaces of the side memory (random gradient masks). Each subspace gets its own edits, no overlap → no conflicts. Then merge the `k` subspaces into one using *Ties-Merge* (a model-merging technique).

**The architectural insight that transfers to your project:**

> "Don't try to edit the original memory in place. Maintain a separate update layer with a routing mechanism that decides which layer answers each query, and a merging mechanism that prevents update-on-update interference."

That single sentence is the take-away. The specific weight-editing implementation details are not.

---

## Reading priority by section

If you have 60 minutes, read these in order:

### Must read (40 minutes)

| Section | Pages | Why |
|---|---|---|
| **Abstract** | 1 | Get the impossible triangle framing in one paragraph |
| **§1 Introduction** | 1–2 | Specifically the "impossible triangle" box and the contributions list |
| **Figure 1 (radar chart)** | 2 | Visual proof that existing methods fail at one of three metrics |
| **§2.2 Rethinking the Memory Design** | 3 | Table 1 — the comparison matrix — is the single most useful summary |
| **§2.3 WISE overview + Figure 2** | 3–4 | The architecture diagram. Spend time here. |
| **§3.2 Main Results** | 6–7 | Tables 2, 4. See how badly the baselines degrade vs WISE at T=1000 edits |

### Skim only (15 minutes)

| Section | Why skim |
|---|---|
| §2.3.1 Side Memory Design | The math is specific to FFN weight editing. Read the prose; skip equations 2-6. |
| §2.3.2 Knowledge Sharding and Merging | The random mask + Ties-Merge mechanism. Read the *idea*; skip equation 7 and the Subspace Overlap theorem. |
| §3.3 Further Analysis | Mostly hyperparameter ablations. Read the captions of Figures 3, 4, 5; skip the rest. |

### Skip entirely (no loss)

- §3.1 Experimental Settings (unless you plan to reproduce WISE — you don't)
- Appendices (specific to model-editing benchmarks)
- All the equations in §2.3 (you don't need the gradient masks to understand the *idea*)

---

## What to actively extract while reading

Keep a notebook (literal or mental) with answers to these questions as you read:

1. **What's an edit, in WISE's terminology?** A factual update like (subject, relation, object) — e.g., "Steve Jobs founded Apple" → "Steve Jobs founded NeXT." Map this onto your project: an edit in your system would be a single piece of new information that should override or modify an existing memory.

2. **Where is the side memory located?** §2.3.1 says mid-to-late FFN layers, specifically value matrix `W_v` of layer 26 in LLaMA-2-7B. Why? Because mid-late layers encode high-level linguistic features without disturbing fundamental grammar (early layers) or the final output structure (last layer). **Analogue for your project:** which "layer" of your memory system holds the most update-friendly representation? A-Mem's K/G/X note attributes? HippoRAG's edge weights? The note text itself?

3. **How does the router decide which memory to use?** Activation indicator `Δ_act(x) = ||A(x) · (W_v' - W_v)||_2`. Effectively: "how different does the side memory respond to this query vs the main memory?" If different enough → it's about an edit → use side memory. Otherwise use main memory. **Analogue:** for your project, what's the signal that a query is about an updated memory? Is it embedding similarity to recently-edited notes? Time-since-last-edit? Explicit metadata?

4. **What's the failure mode of "just edit the original memory"?** §3.2 Table 2: at T=1000 edits, ROME's Locality drops to 0.16; MEMIT's to 0.04. The model has been so heavily modified that unrelated queries get wrong answers. **Lesson for your project:** if you let A-Mem's memory evolution rewrite notes aggressively, the system will progressively degrade on unrelated queries. The dataset's Q20-Q22 (absence questions) test this directly.

5. **What does Ties-Merge actually do?** Conceptually: when you have multiple parameter updates that might conflict, (a) trim small-magnitude changes, (b) align signs, (c) average the rest. Map this to your project: if multiple updates affect the same memory node, you need a similar reconciliation mechanism — not just last-write-wins.

---

## Direct mappings to your project

After reading, you should be able to answer:

### Mapping 1 — WISE's side memory → your project

**If you choose reconsolidation:** introduce a *delta layer* in your memory store. Original notes/edges live in the main store. When a memory needs to be updated (triggered by retrieval + new context), the update goes to the delta layer, not the original. A routing mechanism decides whether a query reads the delta or the original.

**If you choose active forgetting:** the delta layer becomes a *deprecation layer* — entries get moved here when they're scheduled for forgetting, rather than being deleted outright. Queries can still find them but with reduced weight. Eventually they age out entirely.

### Mapping 2 — WISE's knowledge sharding → handling concurrent updates

In WISE, multiple edits are partitioned into different parameter subspaces. In your project, this maps to: when multiple updates affect the same memory region, partition them. For example, partition by user (each user's updates are independent) or by time window (this month's updates vs last month's). Then merge at retrieval time.

### Mapping 3 — WISE's routing → reconsolidation trigger

WISE's router decides per-query: is this about an edited fact? In your project, the analogous decision is: should this retrieval *trigger* reconsolidation? Maybe most retrievals are read-only, but some retrievals (those that surface contradictions, or those with novel context) trigger updates.

### Mapping 4 — WISE's impossible triangle → your evaluation criteria

When you eventually measure your reconsolidation system, you should report all three: reliability (does the updated memory get used?), generalization (does it apply to paraphrased queries?), locality (are unrelated memories undisturbed?). The Memora benchmark's FAMA metric captures something like the locality dimension. The contradiction stress test (Experiment 2 in the extended-reading plan) tests reliability.

---

## What WISE does *not* solve, and you might

The paper is honest about its limits:

1. **Read-time updates are not addressed.** WISE handles writing edits to side memory but doesn't have a mechanism for "every retrieval re-evaluates the memory." That's still the biological-reconsolidation gap.
2. **No forgetting.** WISE accumulates side-memory capacity indefinitely. After enough edits even the sharding can't help.
3. **Weight-level, not external-memory-level.** All of WISE's mechanisms operate on LLM parameters. Adapting them to external memory (which is what your project would do) is itself a contribution — the *adaptation* is novel work.

These three gaps are exactly the contributions your project could make. WISE provides the framework; your project provides the external-memory instantiation + the reconsolidation extension + (optionally) the forgetting layer.

---

## Study questions

After reading, you should be able to answer these. If you can't, re-read the relevant section.

**Section A — Comprehension:**

1. What is the "impossible triangle" and why is it called impossible? Give a one-sentence summary of each of the three properties.
2. Walk through what happens at inference time in WISE: query arrives → router fires → side or main memory used → output. Where does each step happen architecturally?
3. Why is `W_v` (the value matrix of a specific FFN layer) chosen as the side memory, rather than the entire layer or all layers?
4. How does knowledge sharding prevent edit-on-edit interference?

**Section B — Critique:**

5. WISE's experiments are on QA, hallucination, and OOD generalisation datasets. Are these representative of your project's use case (multi-month agent memory)? What does this paper not tell you about your target domain?
6. The routing mechanism is trained on a margin-based loss using known edit examples vs irrelevant examples. In a real deployment where edits arrive online, you don't have a labelled "irrelevant" set. How would WISE's routing degrade in that setting?
7. WISE shards edits into `k` subspaces. With `k=2` (the paper's recommendation), what's the maximum number of edits it can handle before all subspaces are "full"?

**Section C — Application to your project:**

8. If you were to apply WISE's "side memory + routing" idea to A-Mem, what would the side memory look like? What would it store?
9. If you applied it to HippoRAG, what would the side memory look like? Would it be parallel KGs? Edge-weight overrides?
10. Where does WISE's framework break down for *agent memory* specifically? What's a scenario the paper doesn't consider that your project would face?
11. If you wrote a paper with the framing "WISE for external memory," what's the single most important contribution that distinguishes it from just porting WISE one-to-one?

---

## After reading

When you're done, capture three things in your project notes (`research` repo, perhaps as `wise-takeaways.md`):

1. **One sentence describing the impossible triangle in your project's terms.**
2. **One design idea you're stealing from WISE.**
3. **One thing you'll explicitly do differently from WISE.**

These three sentences are the seed of your project's positioning relative to the editing literature.
