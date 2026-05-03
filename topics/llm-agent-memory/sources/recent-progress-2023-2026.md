# Sources — LLM Agent Memory Recent Progress (2023–2026)

## Foundational / core papers
1. Park et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior.*
   - URL: https://arxiv.org/abs/2304.03442
   - Why it matters: establishes memory stream, retrieval, reflection pattern.

2. Wang et al. (2023). *Augmenting Language Models with Long-Term Memory (LongMem).*
   - URL: https://arxiv.org/abs/2306.07174
   - Why it matters: early long-context memory augmentation strategy.

3. Packer et al. (2023). *MemGPT: Towards LLMs as Operating Systems.*
   - URL: https://arxiv.org/abs/2310.08560
   - Why it matters: introduces OS-inspired memory hierarchy and virtual context management.

## Evaluation and recent developments
4. LongMemEval (2024). *Benchmarking Chat Assistants on Long-Term Interactive Memory.*
   - URL: https://arxiv.org/abs/2410.10813
   - Why it matters: measures multi-session extraction, reasoning, temporal updates, and abstention. User-assistant format; 500 questions; ~115K tokens per question. Limitation: synthetic, designed pre-1M-context-window era.

4b. LoCoMo (2024, ACL). *Evaluating Very Long-Term Conversational Memory of LLM Agents.*
    - URL: https://arxiv.org/abs/2402.17753
    - Why it matters: social two-agent dialogue, multi-modal (images), grounded in persona + temporal event graphs, human-annotated. 10 conversations, ~300 turns, up to 35 sessions. Limitation: tiny released set, short by modern standards, no abstention task.

5. Mao et al. (2025). *How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior.*
   - URL: https://arxiv.org/abs/2505.16067
   - Why it matters: empirical analysis of how memory-management choices shape long-term behavior.

6. Zhao et al. (2025, EMNLP). *MemInsight: Autonomous Memory Augmentation for LLM Agents.*
   - URL: https://aclanthology.org/2025.emnlp-main.1683/
   - Why it matters: explores autonomous memory update/augmentation mechanisms.

7. Meng et al. (2026). *Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers.*
   - URL: https://arxiv.org/abs/2603.07670
   - Why it matters: up-to-date survey synthesizing architectures, benchmarks, and open problems.

## LLM-as-memory-manager direction (added 2026-05-03)
8. A-Mem (2025). *Agentic Memory for LLM Agents.*
   - URL: https://arxiv.org/abs/2502.12110
   - Why it matters: LLM generates structured notes and dynamic inter-memory links; closest direct successor to Generative Agents' prompt-driven memory.

9. Agentic Memory (2026). *Learning Unified Long-Term and Short-Term Memory Management for LLM Agents.*
   - URL: https://arxiv.org/abs/2601.01885
   - Why it matters: prompt-driven unified memory pipeline, no separate retrieval module.

10. MemR3 (2025). *Memory Retrieval via Reflective Reasoning for LLM Agents.*
    - URL: https://arxiv.org/abs/2512.20237
    - Why it matters: tracks evidence gaps — what the agent still needs to know — as a first-class state alongside stored memories.

11. Hindsight (2025). *Building Agent Memory that Retains, Recalls, and Reflects.*
    - URL: https://arxiv.org/abs/2512.12818
    - Why it matters: separates objective facts from subjective opinions; adds confidence-scored opinion evolution, reducing hallucinated "memories."

12. Reflective Memory Management for Long-term Personalized Dialogue (ACL 2025).
    - URL: https://aclanthology.org/2025.acl-long.413
    - Why it matters: applies prospect + retrospect reflection loops to open-domain chat agents; directly generalizes Generative Agents' reflection.

13. SSGM Framework (2026). *Governing Evolving Memory in LLM Agents: Risks, Mechanisms, and the Stability and Safety Governed Memory Framework.*
    - URL: https://arxiv.org/abs/2603.11768
    - Why it matters: safety and stability angle on LLM-driven memory; addresses adversarial injection and memory drift.

14. MemAgents Workshop (ICLR 2026). *Memory for LLM-Based Agentic Systems.*
    - URL: https://openreview.net/pdf?id=U51WxL382H
    - Why it matters: community snapshot of open problems and emerging directions as of early 2026.

## Additional benchmarks (added 2026-05-03)
18. MSC — Multi-Session Chat (2022, ACL). *Beyond Goldfish Memory: Long-Term Open-Domain Conversation.*
    - URL: https://arxiv.org/abs/2107.07567
    - Why it matters: original multi-session benchmark; human-human chats, 5 sessions, persona-grounded. Now legacy but still used as training corpus.

19. MemBench (2025, ACL Findings). *Towards More Comprehensive Evaluation on the Memory of LLM-based Agents.*
    - URL: https://arxiv.org/abs/2506.21605
    - Why it matters: factual vs reflective memory × participation vs observation modes; also tests efficiency and capacity.

20. MemoryBench (2025). *A Benchmark for Memory and Continual Learning in LLM Systems.*
    - URL: https://arxiv.org/abs/2510.17281
    - Why it matters: jointly evaluates memory recall and continual adaptation from user feedback; multi-domain, multi-language.

21. MemoryAgentBench (2026, ICLR). *Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions.*
    - URL: https://arxiv.org/abs/2507.05257
    - Why it matters: cognitively grounded; tests 4 competencies (retrieval, test-time learning, long-range understanding, selective forgetting). No current method passes all four.

## Addressing A-Mem open problems (added 2026-05-03)
15. Chain-of-Memory (2026). *Lightweight Memory Construction with Dynamic Evolution for LLM Agents.*
    - URL: https://arxiv.org/abs/2601.14287
    - Why it matters: directly targets the static embedding bottleneck; triggers re-indexing as memory context shifts.

16. AutoPDL (2025). *Automatic Prompt Optimization for LLM Agents.*
    - URL: https://arxiv.org/abs/2504.04365
    - Why it matters: AutoML-style search over prompting patterns and demonstrations; reduces manual prompt engineering burden. Not memory-specific, but directly applicable.

17. SICA / Evolving Excellence (2025). *Automated Optimization of LLM-based Agents.*
    - URL: https://arxiv.org/abs/2512.09108
    - Why it matters: agents that rewrite their own prompts and heuristics via LLM-proposed edits; 17–53% gains on coding tasks.

## Code agent benchmarks (added 2026-05-03)
22. SWE-bench (2023). *Can Language Models Resolve Real-world GitHub Issues?*
    - URL: https://arxiv.org/abs/2310.06770
    - Why it matters: the reference code agent benchmark; single-task, no memory. Baseline for all extensions.

23. SWE-Bench-CL (2025). *Continual Learning for Coding Agents.*
    - URL: https://arxiv.org/abs/2507.00014
    - Why it matters: only code benchmark with explicit memory across tasks; chronological repo evolution; FAISS memory; forgetting/transfer metrics.

24. SWE-EVO (2025). *Benchmarking Coding Agents in Long-Horizon Software Evolution Scenarios.*
    - URL: https://arxiv.org/abs/2512.18470
    - Why it matters: long-horizon sequential changes to a codebase; tests state tracking across evolving repo.

25. LoCoEval (2026). *A Scalable Benchmark for Repository-Oriented Long-Horizon Conversational Context Management.*
    - URL: https://arxiv.org/abs/2603.06358
    - Why it matters: 50-turn code assistant conversations; tests context compression and extraction under 64K–256K token pressure.

26. LoCoBench (2025). *A Benchmark for Long-Context LLMs in Complex Software Engineering.*
    - URL: https://arxiv.org/abs/2509.09614
    - Why it matters: 8,000 scenarios from 10K to 1M tokens; diagnostic tool for context length degradation.

## Reliability / relevance notes
- Priority given to primary sources (arXiv/ACL anthology) over blog summaries.
- For very recent papers (2025–2026), validate claims against the PDF before citing specific numbers.
