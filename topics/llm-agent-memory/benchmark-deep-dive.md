# LLM Agent Memory Benchmarks — A Practitioner's Guide

*A survey of benchmarks for evaluating long-term memory in LLM agents, 2022–2026. Covers conversational, code-agent, and personalisation benchmarks; the cognitive framework behind them; metrics; and what the field still does not measure.*

---

## TL;DR (one screen)

**The state of the field.** As of mid-2026, LLM agent memory has its own first-class benchmark suite. The space has matured from "can the model recall what was said three turns ago" (MSC, 2022) to "can the agent selectively forget stale information, learn new facts mid-task, and stay consistent across weeks-long interactions" (Memora, MemoryAgentBench, 2026).

**The big shift in 2025–2026.** Three things changed at once:
1. **Context windows grew to 1M+ tokens**, making "just stuff everything in context" a serious baseline. Benchmarks designed for 32K contexts (MSC, LoCoMo, LongMemEval) are now partially solved by long-context models alone, weakening them as discriminators.
2. **The cognitive-science framing won.** MemoryAgentBench (ICLR 2026) crystallised four memory competencies — retrieval, test-time learning, long-range understanding, selective forgetting — that any serious benchmark now structures itself around.
3. **Forgetting became a first-class metric.** FAMA (Forgetting-Aware Memory Accuracy) rewards systems that correctly *abandon* outdated information, not just recall valid information.

**The open gap.** No benchmark yet measures the practically-important case: an agent serving the *same user* over *months*, learning their preferences, vocabulary, and history, with the user's life evolving (new job, new project, new relationships). Memora (2026) gets closest but is still synthetic.

---

## 1. What a memory benchmark should test

A well-designed memory benchmark needs to separate several capabilities that are easy to conflate:

| Capability | What it measures | Example test |
|---|---|---|
| **Retrieval** | Find a specific fact buried in history | "What was my dog's name from session 3?" |
| **Multi-hop reasoning** | Combine facts from different sessions | "How many years between when I started yoga and when I moved cities?" |
| **Temporal reasoning** | Understand when something was true or expired | "Am I still vegetarian?" given conflicting statements over time |
| **Knowledge updates** | Prefer the latest fact over older contradicting ones | User said "I work at A" in session 1, "I work at B" in session 8 → answer B |
| **Test-time learning** | Update beliefs *during* the interaction | New instruction or correction mid-conversation is followed |
| **Long-range understanding** | Synthesise across distant events | "What pattern do you see in my mood entries from the past month?" |
| **Selective forgetting** | Discard stale, contradicted, or sensitive info | After user says "forget my old phone number" the system stops using it |
| **Abstention** | Recognise when memory is insufficient | "I don't have that information" rather than confabulating |
| **Consistency** | Same question gets same answer across reruns | Robustness to retrieval order, paraphrase |

Older benchmarks tested 2–3 of these; modern ones (MemoryAgentBench, Memora) aim for all of them.

---

## 2. The four-competency framework (MemoryAgentBench, ICLR 2026)

The most influential framing of the past year. MemoryAgentBench distils memory evaluation down to **four core competencies** drawn from cognitive science:

1. **Accurate retrieval** — get the right fact when asked.
2. **Test-time learning** — update beliefs and behaviour mid-task from new information.
3. **Long-range understanding** — synthesise across temporally distant events.
4. **Selective forgetting** — actively discard information that is stale, contradicted, or marked for deletion.

The paper's key finding is striking: **no current memory agent masters all four competencies.** Each method wins on some and loses on others. Methods optimised for retrieval (vector DB + top-k) fail at forgetting. Methods with explicit pruning lose on long-range understanding. The four competencies appear to be in tension and a unified architecture has not yet emerged.

This is the closest thing the field has to a unifying taxonomy. Read this paper first.

---

## 3. The benchmark landscape

### 3.1 Conversational / personalisation benchmarks

#### MSC — Multi-Session Chat (2022)
- **Paper:** https://arxiv.org/abs/2107.07567 (ACL 2022)
- **Format:** Human-human crowdworker chats, 5 sessions, ~14 utterances each, persona-grounded (PersonaChat).
- **Strength:** First multi-session benchmark. Real human dialogue.
- **Weakness:** Tiny sessions, narrow persona domain, pre-LLM-agent era design. Solved by modern models.
- **Status:** Legacy. Still used as a training corpus, not as a discriminator.

#### LoCoMo — Long Conversational Memory (2024)
- **Paper:** https://arxiv.org/abs/2402.17753 (ACL 2024)
- **Format:** Two-agent social dialogue, ~300 turns/~9K tokens per conversation, up to 35 sessions. Multi-modal (images allowed). Grounded in pre-built persona profiles + temporal event graphs. Human-annotated.
- **Tasks:** Single-hop QA, multi-hop QA, temporal QA, adversarial QA; event summarisation; multi-modal dialogue generation.
- **Released set:** Only 10 conversations.
- **Strength:** First benchmark with explicit event graphs and human-validated long-range consistency. Multi-modal.
- **Weakness:** Tiny released set; ~9K tokens per conversation fits in modern context windows; social-chat domain only.
- **Status:** Still widely cited; partially solved by long-context models.

#### LongMemEval (2024)
- **Paper:** https://arxiv.org/abs/2410.10813
- **Format:** 500 questions embedded in synthetic user-assistant conversation histories at ~115K tokens per question; lengths "freely scalable".
- **Five capabilities tested:** information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention.
- **Key finding:** Commercial assistants and long-context LLMs show ~30% accuracy drop versus single-session baselines.
- **Strength:** First to explicitly test abstention and knowledge updates as named capabilities. Scalable to longer contexts.
- **Weakness:** Synthetic conversations; fact-planting construction is gameable by extraction heuristics; no multi-modal.
- **Status:** Currently the most-cited memory benchmark; the workhorse of head-to-head comparisons.

#### MemBench (ACL 2025 Findings)
- **Paper:** https://arxiv.org/abs/2506.21605
- **Format:** Two-axis taxonomy:
  - *Factual* (store and retrieve explicit facts) vs *Reflective* (derive higher-level inferences from experience).
  - *Participation* (agent was in the conversation) vs *Observation* (agent reads a conversation it wasn't part of).
- **Strength:** Most taxonomically careful design. Also measures efficiency (latency, token cost) and capacity (scaling).
- **Weakness:** Newer, less adopted; smaller cited footprint.
- **Status:** Underused but methodologically the cleanest.

#### MemoryBench (2025)
- **Paper:** https://arxiv.org/abs/2510.17281
- **Format:** Continual-learning + memory combined. Three modules: task provider, user simulator (generates human-like feedback), performance monitor. Multi-domain, multi-language.
- **Strength:** First benchmark to test memory + continual adaptation from feedback in one suite. Includes user-feedback simulation.
- **Weakness:** Feedback-simulation quality depends on the LLM used as simulator (circular dependency).
- **Status:** Useful when your system involves online feedback (RLHF-style updates), not just passive memory.

#### MemoryAgentBench (ICLR 2026)
- **Paper:** https://arxiv.org/abs/2507.05257
- **Code:** https://github.com/HUST-AI-HYZ/MemoryAgentBench
- **Format:** Reformats existing long-context datasets into multi-turn incremental format + adds newly constructed tasks. Built around the four core competencies from §2.
- **Key finding:** **No current method masters all four competencies.** This is the unifying empirical finding of the year.
- **Strength:** Cognitively grounded; tests competencies that long-context stuffing cannot cover (especially forgetting and test-time learning).
- **Weakness:** Newer, ecosystem still forming.
- **Status:** The current state-of-the-art evaluation. Use this for any serious 2026 work.

#### Memora (2026)
- **Paper:** https://arxiv.org/abs/2604.20006 (*From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents*)
- **Format:** Synthetic user-agent conversations spanning **weeks to months**, much longer horizon than prior benchmarks. Three task families: remembering, reasoning, recommending.
- **Strength:** Longest temporal horizon to date. Explicitly tests personalised recommendation, not just QA.
- **Weakness:** Synthetic; user behaviour models are LLM-generated.
- **Status:** The closest current proxy for the practical "agent serving a user for a year" scenario.

### 3.2 Code-agent memory benchmarks

#### SWE-bench (2023, baseline reference)
- **Paper:** https://arxiv.org/abs/2310.06770
- **Format:** Resolve real GitHub issues using repo context. Single-task, no memory.
- **Why included here:** Reference point all memory-aware code benchmarks extend.

#### SWE-Bench-CL (2025)
- **Paper:** https://arxiv.org/abs/2507.00014
- **Format:** Takes SWE-Bench Verified tasks and organises them chronologically by repo evolution; evaluates accumulated knowledge across sequential tasks.
- **Memory module:** FAISS-backed semantic memory.
- **Metrics:** Average accuracy, forgetting rate, forward/backward transfer, tool-use efficiency.
- **Strength:** Tests stability-plasticity trade-off (learning new without forgetting old) in code.
- **Status:** The closest code-agent analogue to the conversational memory benchmarks.

#### SWE-EVO (2025)
- **Paper:** https://arxiv.org/abs/2512.18470
- **Format:** Long-horizon software evolution — agent handles a sequence of changes to a codebase, not isolated issues.
- **Strength:** Tests state tracking across an evolving repo.

#### LoCoEval (2026)
- **Paper:** https://arxiv.org/abs/2603.06358
- **Format:** Repository-oriented conversational context management. 128 samples, ~50 turns each, context 64K–256K tokens. Tests topic awareness, information extraction, code generation (Pass@1).
- **Status:** More about context compression than persistent long-term memory.

#### LoCoBench (2025)
- **Paper:** https://arxiv.org/abs/2509.09614
- **Format:** 8,000 scenarios from 10K to 1M tokens.
- **Strength:** Diagnostic tool for context-length degradation curves.
- **Status:** Use to measure how performance scales with context, not to evaluate a memory system per se.

---

## 4. Cross-benchmark comparison

### 4.1 Coverage matrix

| Benchmark | Retrieval | Multi-hop | Temporal | Updates | Test-time learn | Long-range | Forgetting | Abstention | Multi-modal |
|---|---|---|---|---|---|---|---|---|---|
| MSC | ✓ | – | – | – | – | – | – | – | – |
| LoCoMo | ✓ | ✓ | ✓ | – | – | ✓ | – | – | ✓ |
| LongMemEval | ✓ | ✓ | ✓ | ✓ | – | – | – | ✓ | – |
| MemBench | ✓ | – | – | – | – | ✓ | – | – | – |
| MemoryBench | ✓ | – | – | ✓ | ✓ | – | – | – | – |
| MemoryAgentBench | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – |
| Memora | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | – | – |

Reading: MemoryAgentBench and Memora are the only two that cover both **forgetting** and **multi-month-style long-range understanding**. They are complementary — MemoryAgentBench is broader on capabilities, Memora is longer in horizon.

### 4.2 Scale comparison

| Benchmark | Sessions / horizon | Context per question | Test set size |
|---|---|---|---|
| MSC | 5 sessions | ~few K tokens | Small |
| LoCoMo | 35 sessions | ~9K tokens | 10 conversations |
| LongMemEval | Scalable | ~115K tokens | 500 questions |
| MemBench | Mixed | Mixed | Mid-sized |
| MemoryBench | Continuous | Mixed | Multi-domain |
| MemoryAgentBench | Multi-turn incremental | Scalable | Multi-task |
| Memora | Weeks–months | Very long | Multi-task |

### 4.3 What each benchmark is *best* at testing

| If you want to test... | Use |
|---|---|
| Basic multi-session recall | LongMemEval |
| Multi-modal long conversation | LoCoMo |
| Continual learning from user feedback | MemoryBench |
| The full four competencies (current SOTA) | MemoryAgentBench |
| Multi-month personalisation horizon | Memora |
| Factual vs reflective memory separation | MemBench |
| Code-agent memory across tasks | SWE-Bench-CL |
| Long-context degradation in code | LoCoBench |

---

## 5. Metrics

### 5.1 Traditional metrics
- **Accuracy / Exact Match** — most QA-style benchmarks.
- **F1** — softer string overlap.
- **LLM-as-judge** — increasingly common; cheap but circular when the judge shares biases with the system under test.
- **Pass@1** — code-generation benchmarks.

### 5.2 Memory-specific metrics introduced in 2024–2026

- **Forgetting rate** (SWE-Bench-CL): performance drop on earlier tasks after training on new ones.
- **Forward transfer / Backward transfer** (SWE-Bench-CL, classic CL terms): benefit of earlier learning on later tasks, and vice versa.
- **Tool-use efficiency** (SWE-Bench-CL): cost-aware metric.
- **FAMA — Forgetting-Aware Memory Accuracy** (introduced in Memora area, 2026): explicitly rewards correct use of *valid* memory and penalises reliance on *obsolete or deleted* memory. The first metric that treats forgetting as a positive achievement, not just measured noise.
- **Capacity-aware scoring** (MemBench): performance as a function of memory store size — measures graceful degradation as memory bloats.

FAMA is the most important new metric. Almost every prior benchmark implicitly penalised systems for *forgetting*, even when forgetting was the correct behaviour. FAMA inverts that.

---

## 6. Limitations shared across the current benchmark suite

### 6.1 The 1M-context-window problem
Every benchmark designed before 2025 was sized for 32K context. Modern models (Gemini 1.5 Pro, Claude 3.5+, GPT-4.1+) handle 1M+ tokens natively. On LoCoMo (~9K tokens/conv), a 1M-context model can stuff every conversation entirely in-context and skip memory architecture altogether. This means **strong scores on these benchmarks no longer demonstrate good memory design** — they may just demonstrate good context handling.

MemoryAgentBench is the first benchmark to explicitly test capabilities that *cannot* be solved by long-context stuffing (selective forgetting, test-time learning under instruction).

### 6.2 Synthetic-data brittleness
Every benchmark except MSC uses synthetic or semi-synthetic conversations. LLM-generated dialogues have characteristic patterns (over-explicit reference, low ellipsis, uniform tone) that real users do not produce. Methods overfit to these patterns and degrade on real conversation logs.

### 6.3 Single-user, single-purpose
Real agents handle many users, many concurrent tasks, with cross-user privacy constraints. No public benchmark tests this. A good agent should *not* mix memories across users; current benchmarks cannot measure that property.

### 6.4 Static evaluation
Benchmarks are run once on a frozen test set. Real memory systems decay over time (vector drift, embedding model updates, summarisation errors compounding). No benchmark measures *temporal stability* — does the same agent on the same user produce consistent answers next month?

### 6.5 No production-realistic latency or cost budget
A method that takes 30 seconds to retrieve from a 100M-item memory may score perfectly on accuracy but be unusable in production. MemBench is the only one that explicitly measures efficiency, and even that is limited.

### 6.6 Evaluator circularity
Many benchmarks use LLM-as-judge for scoring. When the judge LLM and the system-under-test share training data and biases, scores are optimistically inflated.

### 6.7 Missing the practical question
Every enterprise user of memory systems wants to answer: "Will this agent, talking to my real users, get measurably more helpful over weeks and months?" No benchmark answers that. The benchmarks measure *what could go wrong* in memory, not *what value is added by it*.

---

## 7. Open gaps — what is not measured yet

1. **Multi-user privacy and cross-user contamination.** Critical for any production memory system; absent from all benchmarks.
2. **Real-time temporal stability.** Run the same evaluation a month later on the deployed system; does it still score the same?
3. **Personalisation lift.** The fundamental product metric: does the agent become more helpful *to a specific user* over time? Memora gets closest, but with synthetic users.
4. **Memory cost-effectiveness.** Token cost per useful recall, dollars per personalisation improvement.
5. **Robustness to adversarial memory injection.** Can a user manipulate the agent's memory to make it misbehave on others?
6. **Cross-domain transfer of memory architectures.** A vector-DB memory might excel in social chat and fail in code. None of the conversational benchmarks test code-style content; none of the code benchmarks test social.
7. **Real human-in-the-loop evaluation at scale.** The closest is MemoryBench's user simulator, but it is itself an LLM.
8. **Memory-driven *behavioural* improvement, not just QA.** Today's benchmarks ask "do you remember?"; production cares about "do you *act* differently based on what you remember?".

---

## 8. Practical recommendations

### For 2026 research papers
- **Always include MemoryAgentBench.** It is the current standard for capability breadth.
- **Add LongMemEval** for backwards compatibility with prior work.
- **Add Memora** if your claim involves long-horizon personalisation.
- **Report FAMA explicitly** if your system has any selective-forgetting mechanism.
- **Include a long-context baseline.** Stuff everything in a 1M context window and report that score. If your memory architecture cannot beat it, that is an important finding.

### For production teams choosing a benchmark
- If you build a personal-assistant style product: **Memora + MemoryAgentBench**.
- If you build a coding assistant: **SWE-Bench-CL**.
- If your concern is mostly long context degradation: **LoCoBench**.
- If you're optimising for cost: **MemBench** (the only one with explicit efficiency metrics).

### For applying memory-management prompt optimisation (cross-link to BPO deep-dive)
The clean experimental setup is:
1. Pick MemoryAgentBench (covers all four competencies).
2. Wrap Generative Agents' importance-scoring prompt in a DSPy module.
3. Optimise with MIPROv2 → GEPA.
4. Critically: report scores per competency, not just overall. The competencies are in tension (§2), so a single overall score can hide regressions.

---

## 9. Predictions for the next benchmark wave (2026–2027)

These are gaps that someone will fill within 12–18 months:

1. **A real long-horizon benchmark.** Recruit users for 3–6 months of natural interaction with an instrumented agent. Score on retained personalisation. Expensive but tractable.
2. **A multi-user privacy benchmark.** N users sharing one model; measure both personalisation and cross-user information leakage as a joint objective.
3. **A "deployed-system aging" benchmark.** Same system, same evaluation, run at t=0 and t=3 months. Score on stability and decay characteristics.
4. **A unified meta-benchmark.** Aggregate MemoryAgentBench, Memora, SWE-Bench-CL, MemBench under one harness with cost-normalised scoring. The community needs the equivalent of MMLU for memory systems.
5. **Adversarial memory benchmarks.** Memory injection attacks, prompt-based memory poisoning, persistent jailbreak via planted "facts".

---

## 10. Bibliography

- **MSC** — https://arxiv.org/abs/2107.07567
- **LoCoMo** — https://arxiv.org/abs/2402.17753
- **LongMemEval** — https://arxiv.org/abs/2410.10813
- **MemBench** — https://arxiv.org/abs/2506.21605
- **MemoryBench** — https://arxiv.org/abs/2510.17281
- **MemoryAgentBench** — https://arxiv.org/abs/2507.05257 | Code: https://github.com/HUST-AI-HYZ/MemoryAgentBench
- **Memora / From Recall to Forgetting** — https://arxiv.org/abs/2604.20006
- **SWE-Bench-CL** — https://arxiv.org/abs/2507.00014
- **SWE-EVO** — https://arxiv.org/abs/2512.18470
- **LoCoEval** — https://arxiv.org/abs/2603.06358
- **LoCoBench** — https://arxiv.org/abs/2509.09614
- **Memory survey** — Meng et al. 2026, https://arxiv.org/abs/2603.07670
- **State of AI Agent Memory 2026** — https://mem0.ai/blog/state-of-ai-agent-memory-2026
- **Mem0 (production system, comparison reference)** — https://arxiv.org/abs/2504.19413
