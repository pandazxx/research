# Agent Memory Project — Roadmap & Milestones

*Living planning document. Edit as decisions are made. Last updated: 2026-05-21.*

---

## 0. Project identity

**Working title:** *[TBD — pick by end of Week 2]*

**One-line description:** An open-source implementation and benchmark study of *[reconsolidation | active competitive forgetting | TBD]* as a memory mechanism for LLM agents. Direction to be committed by end of Week 2.

**North-star outcome (month 12):**
- A maintained open-source library implementing the mechanism, with clean API and docs.
- Reproducible benchmark results against ≥2 existing memory baselines on MemoryAgentBench + one other.
- One workshop paper draft (acceptance is a bonus, not the goal).
- 5+ substantive blog posts documenting the research.
- A real research contribution: at least one finding that is novel, defensible, and would not have been obvious to someone who hadn't done the work.

**Why this project survives one year:**
The mechanism being implemented (reconsolidation / active forgetting) is biologically motivated, underexplored in production LLM systems, and does not get solved by larger context windows. If 1M context becomes 10M context, the mechanism remains relevant because it targets capacity-driven memory management, not context-stuffing limits.

**Non-goal:** This roadmap intentionally excludes career/job-hunt planning. That gets its own document if and when needed. Focus here is the research and the artifact.

---

## 1. Phase overview

| Phase | Weeks | Granularity in this doc | Focus | Primary output |
|---|---|---|---|---|
| **Foundation 1A** | 1–5 | Weekly | Public setup + two warmup reproductions + direction decision | Blog launched, HippoRAG + A-Mem both running, direction committed |
| **Foundation 1B** | 6–9 | Weekly | Official baseline reproduction + first design + first variant | Baseline matched, design doc, first variant numbers |
| **Foundation 1C** | 10–14 | Weekly | Iterate variants, kill/pivot gate | 3+ variants, results blog post, kill/pivot decision |
| **Build** | 15–26 | Bi-weekly | Implement properly, run comparative experiments, polish engineering | v0.5 release, comparative benchmark numbers |
| **Amplify** | 27–50 | Monthly | Documentation, workshop paper, community engagement, sustain the project | v1.0, paper submitted, sustained community presence |

Total timeline: ~12.5 months. The extra 2 weeks vs the original 12-month plan came from extending each warmup to 2 weeks (see §10).

---

## 2. Phase 1 — Foundation (Months 1–3, weekly)

### Goal
Stop researching, start operating publicly. Lock in the direction. Build the lowest-stakes possible version of the mechanism to confirm the approach.

### Phase 1A — Setup + warmup reproductions (Weeks 1–5)

Deliberate structure: **public setup → two warmup reproductions → direction decision**. The warmup is both practice (learning the reproduction workflow and tooling) and informational (you learn things from running real baselines that change how you'd choose between candidate directions). Either way the work is not wasted: one of the warmup baselines is likely to also be the official baseline for the chosen direction.

#### Week 1 — Public setup (direction-agnostic)
- [ ] Personal blog set up (Substack, Hashnode, or self-hosted).
- [ ] GitHub repo created. Private for this week, public by end of week 2.
- [ ] Initial repo scaffold: README placeholder, LICENSE (MIT), .gitignore, pyproject.toml.
- [ ] Tentative project name (can change once direction is decided).
- [ ] Read 2–3 papers from *each* candidate direction (reconsolidation + active forgetting) to inform the warmup and direction decision.
- [ ] Confirm warmup baselines: **HippoRAG** (https://github.com/OSU-NLP-Group/HippoRAG) and **A-Mem** (https://github.com/agiresearch/A-mem). Document choice in `docs/warmup.md`.

#### Weeks 2–3 — Warmup reproduction #1: HippoRAG
- [ ] **Week 2 — go public + start HippoRAG:**
  - First blog post published: "A Survey of LLM Agent Memory Research (2023–2026)". Use existing notes as basis. Target ~3000 words.
  - Repo made public with placeholder README explaining the goal.
  - File one comment/issue on each of: Mem0, Letta, A-Mem, HippoRAG, DSPy.
  - Read HippoRAG paper carefully. Clone the repo, set up environment, install dependencies (OpenIE / spaCy / API keys).
  - Get a "hello world" run going — any input through the pipeline.
- [ ] **Week 3 — HippoRAG end-to-end:**
  - Run HippoRAG on a small slice (10–20 examples) of MemoryAgentBench or LongMemEval.
  - Goal is *running*, not *matching published numbers* — that comes later.
  - Log learnings in `docs/warmup.md`: what was easy, what was hard, codebase quality, what would need to change for a deeper reproduction.
- [ ] **Hard cap: 10 working days.** If genuinely stuck after that, abandon and document why (per risk register).

#### Weeks 4–5 — Warmup reproduction #2: A-Mem + direction decision
- [ ] **Week 4 — start A-Mem:**
  - Read A-Mem paper carefully. Clone repo, set up environment.
  - Note the architectural contrast with HippoRAG: LLM-generated dynamic notes vs explicit knowledge graph.
  - Get a "hello world" run going.
- [ ] **Week 5 — A-Mem end-to-end + decide:**
  - Run A-Mem on the same small slice used for HippoRAG.
  - Update `docs/warmup.md` with the second baseline's learnings.
  - **Direction decision (end of Friday Week 5): reconsolidation or active competitive forgetting.** The warmup should now have given you concrete signal:
    - which codebase / approach is easier to build on,
    - which mechanism would be more naturally expressed in the existing tooling,
    - whether either candidate direction faces an obvious blocker.
  - Project name finalised based on direction.
  - Design doc started in `docs/design.md` — one paragraph describing the chosen mechanism, that's enough for now.
- [ ] **Hard cap: 10 working days.**

**Phase 1A checkpoint (end of Week 5):**
- ✓ Public blog with at least 1 post
- ✓ Public repo with running code
- ✓ Two warmup reproductions running (even if rough)
- ✓ Direction chosen and committed
- ✓ `docs/warmup.md` documents both reproductions

If any of these are missing, the discipline-collapse risk is already materialising. Don't move to Phase 1B until they're done.

### Phase 1B — Official baseline reproduction (Weeks 6–9)

#### Week 6 — Commit to official baseline
- [ ] Decide which of the two warmup baselines becomes the *official* baseline for the project. Most often it will be the one aligned with the chosen direction, but engineering-quality observations from the warmup can override that.
- [ ] Begin full reproduction of the chosen official baseline on a meaningful slice of the benchmark.
- [ ] Numbers logged in a public experiment log (a simple CSV in the repo is fine).
- [ ] Repo README updated with what's there and what's coming.

#### Week 7 — Full reproduction
- [ ] Run the baseline on the full MemoryAgentBench (or chosen benchmark). Numbers should be within ~5% of published, or you understand why they aren't.
- [ ] Document any deviations from the published methodology.

#### Week 8 — Reproduction write-up
- [ ] **Second blog post published:** "Reproducing [Baseline]: What I Learned." Honest write-up including the warmup experiences and what differed from the paper.
- [ ] Reproduction reports do disproportionately well — they're high-value, low-novelty, and researchers especially appreciate them.

#### Week 9 — Design v0.1 + first implementation
- [ ] Full design doc for the mechanism in `docs/design.md`. 5 pages max.
- [ ] Specifically: what gets stored, what gets updated/forgotten, when, based on what signal, with what cost.
- [ ] Identify the smallest testable variant. Implement it on top of the reproduced baseline.
- [ ] Run it against the benchmark. Get *some* number for the variant.

**Phase 1B checkpoint (end of Week 9):**
- ✓ Full baseline reproduction
- ✓ Reproduction blog post live
- ✓ Design doc written
- ✓ First variant produces a number on the benchmark

### Phase 1C — First results, kill/pivot gate (Weeks 10–14)

#### Week 10 — Iterate
- [ ] Try at least 2 variants of the core mechanism. Different hyperparameters, different design choices.
- [ ] Run each on the benchmark. Log numbers publicly.

#### Week 11 — Iterate
- [ ] Try at least 1 more substantially different variant.
- [ ] By end of this week: 3+ variants benchmarked against the baseline.

#### Week 12 — Analysis
- [ ] Honest analysis: do any variants show signal? Where? On which competencies?
- [ ] Sketch what would need to be true for the strongest variant to be a real research contribution.

#### Weeks 13–14 — Decision and write-up
- [ ] **Third blog post published:** "First results: does [reconsolidation/forgetting] actually help LLM agent memory?" Honest, technical, includes numbers.
- [ ] **Kill/pivot/proceed decision logged in §11.** Three possible outcomes:
  - **Strong signal:** continue as planned.
  - **Mixed signal:** the more common case. Proceed to Phase 2.
  - **No signal at all:** off-ramp. Pivot now, not in month 6.

**Phase 1C checkpoint (end of Week 14):**
- ✓ 3+ variants benchmarked
- ✓ Honest assessment of signal
- ✓ Third blog post live
- ✓ Decision recorded in §11 (kill / pivot / proceed)

---

## 3. Phase 2 — Build (Months 4–6, bi-weekly)

### Goal
Take the prototype to a real implementation with credible comparative benchmark results. Every two weeks, ship something visible.

Note: Phase 2 starts at Week 15 (two weeks later than the original plan because the warmup phase was extended).

### Month 4

#### Weeks 15–16 — Generalisation
- [ ] Test the mechanism with at least 2 different base LLMs (e.g. Claude + GPT-4 or Claude + open-weight).
- [ ] Test with at least 2 different embedding providers.
- [ ] Document which combinations work well and which fail.

#### Weeks 17–18 — Additional benchmark
- [ ] Add one benchmark beyond MemoryAgentBench. Recommended: Memora (long-horizon personalisation) or LongMemEval (broader capability coverage).
- [ ] Run the mechanism and the baseline on the additional benchmark.
- [ ] **Fourth blog post:** technical deep-dive on the mechanism design — the "how it works" post.

### Month 5

#### Weeks 19–20 — Engineering quality pass
- [ ] Rewrite the implementation with production-quality code: clean API, type hints, tests, docs.
- [ ] CI set up on GitHub Actions (lint, type-check, tests).
- [ ] Documentation site (mkdocs or similar) deployed.

#### Weeks 21–22 — v0.5 release
- [ ] PyPI package published as v0.5.
- [ ] Tag the release, write release notes.
- [ ] Make sure the README is good enough that a stranger can install and run a quickstart in under 10 minutes.

### Month 6

#### Weeks 23–24 — Comparative landscape
- [ ] Reproduce 1–2 additional memory methods so you have honest comparison numbers beyond just the original baseline. Candidates: simple top-k retrieval, long-context-stuff baseline, second memory method (Mem0 if you reproduced A-Mem first, or vice versa).
- [ ] Make sure all comparisons use the same evaluation harness — fair-fight rules.

#### Weeks 25–26 — Hyperparameter sweep
- [ ] Run a proper hyperparameter sweep on your mechanism: at least 20 configurations across the key knobs.
- [ ] Track results in a public experiments dashboard (Weights & Biases public projects work well).
- [ ] **Fifth blog post:** "How [your mechanism] compares to N alternatives on M benchmarks." This is the *evidence post* — must be rigorous.

**Phase 2 checkpoint (end of month 6):**
- ✓ v0.5 on PyPI
- ✓ Engineering quality clearly above typical research code
- ✓ Comparative numbers on 2+ benchmarks against 2+ baselines
- ✓ Hyperparameter sweep complete
- ✓ 5 blog posts published

---

## 4. Phase 3 — Amplify (Months 7–12, monthly)

### Goal
Stop building. Start polishing, documenting, and getting the work seen and used. Most engineers skip this phase; that's where the project earns its keep.

### Month 7 — Workshop paper drafting

- [ ] Identify 1–2 workshop submission targets and their deadlines.
  - NeurIPS workshops on agents/memory/foundation models (typical deadline: mid–late September)
  - ICLR workshops (typical deadline: January–February)
  - ACL / EMNLP workshops
  - Specialised workshops on memory, continual learning, RAG
- [ ] Draft the workshop paper. 4–8 pages depending on venue.
- [ ] Get at least one external reader on the draft (could be someone met through community engagement).
- [ ] Begin to think about the paper's "story" — the single claim and the single experiment that supports it.

### Month 8 — Workshop submission + polish

- [ ] Submit workshop paper.
- [ ] Whether accepted or not, the paper becomes a permanent artifact you can link to.
- [ ] Polish v1.0 of the library: README, quickstart, examples folder with real working code, FAQ.
- [ ] Create a 5-minute demo video. Loom is fine.

### Month 9 — Documentation as artifact

- [ ] Long-form retrospective blog post: "What I learned building [project] over 9 months." This is the most-read post hiring managers, peers, and other researchers will reach for.
- [ ] Re-read the design doc from month 2 and write a *what changed and why* commentary. Public design evolution is itself a valuable artifact.
- [ ] Update README and documentation to reflect the final state of the project.

### Month 10 — Visibility push

- [ ] One coordinated launch moment: HN post, Twitter/X thread, possibly newsletter outreach (Latent Space, The Sequence, Practical AI, etc.).
- [ ] Submit a talk proposal to a meetup or smaller conference.
- [ ] Goal here is sustained, not spiky, attention. One good HN post is great; one good HN post plus follow-up engagement is better.

### Month 11 — Sustain and respond

- [ ] By now there should be issues, PRs, and discussion happening on the repo. Triage them properly — open-source maintenance is part of the project, not a distraction.
- [ ] If the workshop paper was accepted: prepare poster/presentation.
- [ ] If rejected: revise based on reviewer comments, target a different venue.
- [ ] Identify the next research question. What's worth doing if you wanted to continue this for another 12 months?

### Month 12 — Closing reflection

- [ ] Final retrospective blog post.
- [ ] Tag a v1.0 release.
- [ ] Decision: continue actively maintaining the project, or hand it off, or freeze it as a portfolio artifact.
- [ ] Re-read this entire roadmap. What was right? What was wrong? What would you tell yourself if you were starting over?

**Phase 3 checkpoint (end of month 12):**
- ✓ v1.0 released
- ✓ Workshop paper submitted (acceptance is bonus)
- ✓ 7+ blog posts published over the year
- ✓ Some form of external recognition: stars, citations, mentions, podcast, talk
- ✓ A defensible answer to "what was the actual research contribution?"

---

## 5. Success metrics by phase

| Phase | Metric | Threshold |
|---|---|---|
| Foundation | First blog post published | Yes/no |
| Foundation | Warmup reproductions | 2 baselines running on a small slice by end of week 3 |
| Foundation | Direction commitment | Yes/no, locked by end of week 3 (informed by warmup) |
| Foundation | Official baseline reproduced | Within ~5% of published by end of month 2 |
| Foundation | Variants benchmarked | 3+ by end of month 3 |
| Foundation | Kill/pivot gate honoured | Yes/no — pivots happen if signal absent |
| Build | Number of blog posts | 5+ by end of month 6 |
| Build | Benchmark numbers vs baselines | At least one signal on one competency |
| Build | v0.5 on PyPI | Yes/no |
| Build | Comparative landscape | 2+ baselines, 2+ benchmarks |
| Amplify | Workshop paper | Submitted |
| Amplify | v1.0 released | Yes/no |
| Amplify | External recognition | At least one external citation, mention, talk, or podcast |
| Amplify | Defensible contribution | Can you state it in one sentence? |

---

## 6. Risk register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Discipline collapse around month 4–6 | High | High | Weekly cadence with explicit Friday self-review; blog cadence forces visible progress; consider one accountability peer for biweekly check-ins. |
| Scope creep | High | Medium | Hard cap: 1 mechanism, 2 benchmarks, 1 paper, 5–7 blog posts. Anything beyond is bonus. |
| Isolation — building without community | High | High | Start community engagement in week 2, not month 8. File one PR/issue per week on existing memory projects. |
| "Perfect plan" trap — research forever, never build | Medium | High | Hard cutoff: direction chosen by end of week 1, public repo by end of week 2. |
| Direction turns out to be a dead end | Medium | High | Month 3 kill/pivot gate. Take it seriously — 3 months lost beats 9 months lost. |
| Big company ships a similar feature | Medium | Medium | Differentiate on engineering quality + benchmark rigour, not just novelty. Even if commoditised, a well-engineered reference implementation has value. |
| Burnout | Medium | High | Take 1 full week off every 2 months. The blog cadence helps — visible incremental progress is psychologically protective. |
| Benchmark contamination | Low | High | If using an LLM that was trained on the benchmark, note this explicitly. Use held-out splits where available. |
| Warmup reproductions take much longer than 1 week each | Medium | Medium | Scope the warmup explicitly to "small slice, just running" — not "matching published numbers." If a warmup is genuinely stuck after 5 days, abandon that baseline and pick another; document why in `docs/warmup.md`. |

---

## 7. Operating rhythm

**Weekly (Friday afternoon, ~30 min)**
- Self-review: what shipped this week? What's next week's top 3?
- At least one public commit. Not necessarily code — could be a blog draft, a doc, a benchmark log.

**Bi-weekly (months 4–6, on the same day of the week as week start)**
- Same as weekly but extended to ~60 min.
- Specific check: are bi-weekly deliverables in this roadmap on track?

**Monthly (last Sunday of month, ~2 hours)**
- Deep review: update this roadmap. Check phase progress. Refresh risk register.
- Publish *something* publicly: blog post, release, or community engagement.

**Quarterly (end of months 3, 6, 9, 12, ~half day)**
- Step-back: is the strategy still right? Re-read this roadmap. Edit it.
- Check the field: has someone published your idea? Has the landscape shifted? Has a bigger context window made the mechanism less relevant?

---

## 8. Open decisions

**By end of Week 1:**
- [ ] **Blog platform:** Substack vs Hashnode vs self-hosted.
- [ ] **Tentative project name** (can revise after direction is chosen).
- [ ] **Two warmup baseline papers**, one closer to each candidate direction.
- [ ] **Primary benchmark for the warmup small slice**: MemoryAgentBench (recommended) or LongMemEval.
- [ ] **Primary base LLM**: Claude, GPT-4, or open-weight (Llama 3 via Together/Replicate). Used for both warmup reproductions.
- [ ] **Primary embedding model**: OpenAI text-embedding-3-large or Voyage-3-large (see embeddings deep-dive).

**By end of Week 3:**
- [ ] **Project focus:** reconsolidation OR active competitive forgetting. Informed by what the warmup reproductions revealed.
- [ ] **Final project name.**

**By end of Week 4:**
- [ ] **Official baseline for the full reproduction.** Usually one of the two warmup baselines; may be a third paper if the warmup revealed both candidates have blockers.

---

## 9. Time-commitment assumption

This roadmap assumes ~20 hours/week effective project time. If less:
- 10–12 hours/week (evenings + half a weekend day): stretch timeline to ~16–18 months proportionally.
- 5–8 hours/week (evenings only): the 12-month version is not feasible; consider scoping down to a smaller artifact (e.g. a reproduction + analysis post, no original mechanism).

If more (30+ hours/week, e.g. transitioning between jobs):
- 12-month timeline can compress to 9 months with the same artifacts.
- Use the extra budget for additional benchmarks, more variants explored, or an additional standalone analysis post — not for more features.

Revisit this assumption monthly.

---

## 10. Hardware and budget

For both the warmup reproductions and the full project, hardware needs depend on the chosen compute path. None of the directions we're considering require local fine-tuning, so for this project a laptop is sufficient.

### Compute paths

| Path | Hardware | Monthly cost (build phase) | Pros | Cons |
|---|---|---|---|---|
| **A — API-only (recommended)** | Laptop, 16GB RAM | $100–500 | Simple setup, no equipment investment | Ongoing API costs |
| **B — Local embeddings + API LLMs** | GPU 8–16GB VRAM (e.g. RTX 4060 ~$450) | $50–300 | Cheaper at high embedding volume | GPU setup, driver maintenance |
| **C — Fully local** | GPU 24GB+ VRAM (RTX 4090 ~$1700) | ~$0 (electricity) | Predictable cost, no API limits, full reproducibility | High upfront cost, slower than top API models |

### Recommendation

**Path A unless you specifically want to fine-tune local models.** Both warmup baselines (HippoRAG, A-Mem) are designed to work with API LLMs. The mechanisms under consideration (reconsolidation, active forgetting) are also API-friendly.

Move to Path B if your embedding volume grows enough that API costs dominate (rough crossover: >50M tokens/month of embeddings). Move to Path C only if you want full independence from API providers or if you decide to fine-tune.

### Warmup-specific requirements

| Baseline | LLM needs | Embedding needs | Estimated cost (2 weeks) |
|---|---|---|---|
| HippoRAG | GPT-3.5 (or equivalent) for OpenIE extraction | ContrieverQA in original; API embeddings work fine | $50–150 |
| A-Mem | Heavy LLM use for note generation and linking; cost scales with memory size | Standard sentence embedder | $100–300 |

A laptop is enough for both. No GPU is required if using API endpoints.

### Storage

Trivial. Modern SSD is enough. Benchmark datasets are small (under a few GB).

### Budget guidance for the year

| Phase | Estimated total cost |
|---|---|
| Foundation 1A (warmups) | $150–450 |
| Foundation 1B–1C (full reproduction + first variants) | $200–500 |
| Build (months 4–6) | $500–1500 (more experiments, more benchmarks) |
| Amplify (months 7–12) | $300–800 (lighter usage, more writing) |
| **Total for the year** | **$1150–3250** |

This is the API-only estimate. If you switch to Path B mid-year, ongoing costs drop. Budget tighter if necessary by:
- Using cheaper LLMs for variant exploration (GPT-3.5, Claude Haiku) and saving GPT-4 / Claude Sonnet for final runs.
- Running smaller benchmark slices during iteration; full benchmark only for milestone numbers.
- Using Google's $0.006/M embedding model (see embeddings deep-dive) for the embedding-heavy paths.

---

## 11. Decision log

*Use this to record significant decisions and their rationale. Future-you needs this when re-reading the roadmap in month 7.*

| Date | Decision | Rationale | Revisit? |
|---|---|---|---|
| 2026-05-21 | Roadmap drafted (job-hunt removed) | Goal = "interesting work + portfolio + learning"; career planning to be handled separately if needed | End of month 1 |
| 2026-05-22 | Warmup pairing: HippoRAG + A-Mem | HippoRAG aligned with brain-inspired graph search interest; A-Mem provides paradigm contrast (LLM-managed notes vs explicit KG) | After warmups complete (end of W5) |
| 2026-05-22 | Warmup duration extended from 1 week to 2 weeks each | Honest estimate from PR review: 1 week is too tight for first-time reproductions of unfamiliar codebases. Adds 2 weeks total to Phase 1 (12 → 14 weeks); total timeline ~12.5 months. | End of warmup phase |
| 2026-05-22 | Compute path: API-only (Path A) | No fine-tuning planned; both warmup baselines and candidate mechanisms are API-friendly. Laptop sufficient. | If embedding volume crosses ~50M tokens/month |
| 2026-06-14 | Next-level study batch added (8 papers): Memory-R2, MMPO, AgeMem, MemSearcher, MemoPilot, HORMA, Memory-T1, JitRL | All cluster around RL-driven memory policies (Nov 2025 – June 2026). Original 4-paper "RL-trained" framing widened to "RL-driven" to include JitRL (no gradient updates). Reading guide in `next-level-study-index.md`. | End of one-week reading block — re-evaluate whether any change the Phase 1B official-baseline choice or the mechanism direction. Critical question: if JitRL replicates on memory benchmarks, does the project abandon training-based RL entirely? |
| | | | |

---

## 12. Notes / scratch

*Ongoing thoughts, idea fragments, questions for later. Trim quarterly.*

- Project naming: prefer a memorable name over a descriptive one — easier to talk and write about.
- Library vs research code positioning: lean library (people install and use it) — harder but more impactful.
- Twitter/X presence: 1–2 substantive posts per week + light replies is the right cadence. Not constant posting.
- Open question: should the implementation be designed to be drop-in with existing frameworks (LlamaIndex memory module, DSPy module) or standalone? Drop-in increases adoption potential significantly.
