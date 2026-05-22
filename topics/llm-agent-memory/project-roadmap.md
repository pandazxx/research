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

| Phase | Months | Granularity in this doc | Focus | Primary output |
|---|---|---|---|---|
| **Foundation** | 1–3 | Weekly | Pick direction, reproduce one baseline, ship first prototype | Public blog launched, baseline reproduced, kill/pivot gate at end of month 3 |
| **Build** | 4–6 | Bi-weekly | Implement properly, run comparative experiments, polish engineering | v0.5 release, comparative benchmark numbers |
| **Amplify** | 7–12 | Monthly | Documentation, workshop paper, community engagement, sustain the project | v1.0, paper submitted, sustained community presence |

---

## 2. Phase 1 — Foundation (Months 1–3, weekly)

### Goal
Stop researching, start operating publicly. Lock in the direction. Build the lowest-stakes possible version of the mechanism to confirm the approach.

### Month 1 — Direction & public setup

#### Week 1 — Public setup (direction-agnostic)
- [ ] Personal blog set up (Substack, Hashnode, or self-hosted).
- [ ] GitHub repo created. Private for this week, public by end of week 2.
- [ ] Initial repo scaffold: README placeholder, LICENSE (MIT), .gitignore, pyproject.toml.
- [ ] Tentative project name (can change once direction is decided).
- [ ] Read 2–3 papers from *each* candidate direction (reconsolidation + active forgetting) to inform the Week 2 decision.

#### Week 2 — Direction decision + go public
- [ ] **Binary direction decision: reconsolidation or active competitive forgetting.** Friday deadline. No backtracking after this week.
- [ ] Project name finalised based on direction.
- [ ] **First blog post published:** "A Survey of LLM Agent Memory Research (2023–2026)". Use existing notes as the basis. Target ~3000 words.
- [ ] Repo made public with placeholder README explaining the goal.
- [ ] File one comment/issue on each of: Mem0, Letta, A-Mem, HippoRAG, DSPy. Just establish that you exist in those communities.
- [ ] Design doc started in `docs/design.md` — one paragraph describing the chosen mechanism, that's enough for now.

#### Week 3 — Reproduction target
- [ ] Pick the *one baseline paper to reproduce*. Recommended candidates: Mem0, A-Mem, HippoRAG.
- [ ] Set up local dev environment: which LLM API, embedding API, benchmark scoring harness.
- [ ] Get the baseline running end-to-end on a small slice (10–20 examples) of MemoryAgentBench or LongMemEval. Numbers don't have to match published yet.

#### Week 4 — Baseline numbers
- [ ] Full reproduction of the chosen baseline on a meaningful slice of the benchmark.
- [ ] Numbers logged in a public experiment log (a simple CSV in the repo is fine).
- [ ] Repo README updated with what's there and what's coming.

**Month 1 checkpoint (end of week 4):**
- ✓ Public blog with at least 1 post
- ✓ Public repo with running code
- ✓ Direction chosen and committed
- ✓ First benchmark numbers (even if just baseline)

If any of these are missing, the discipline-collapse risk is already materialising. Don't move to month 2 until they're done.

### Month 2 — Reproduce, then prototype

#### Week 5 — Full reproduction
- [ ] Run the baseline on the full MemoryAgentBench (not a slice). Numbers should be within ~5% of published, or you understand why they aren't.
- [ ] Document any deviations from the published methodology.

#### Week 6 — Reproduction write-up
- [ ] **Second blog post published:** "Reproducing [Baseline]: What I Learned." Honest write-up including what was hard, what differed from the paper.
- [ ] Reproduction reports do disproportionately well — they're high-value, low-novelty, and researchers especially appreciate them.

#### Week 7 — Design v0.1
- [ ] Full design doc for the mechanism in `docs/design.md`. 5 pages max.
- [ ] Specifically: what gets stored, what gets updated/forgotten, when, based on what signal, with what cost.
- [ ] Identify the smallest testable variant.

#### Week 8 — First implementation
- [ ] Implement the smallest variant on top of the reproduced baseline.
- [ ] Run it against MemoryAgentBench. Get *some* number — any number — for the variant.

**Month 2 checkpoint (end of week 8):**
- ✓ Full baseline reproduction
- ✓ Reproduction blog post live
- ✓ Design doc written
- ✓ First variant produces a number on the benchmark

The number being good is not the test yet. The test is whether you have a variant to iterate on.

### Month 3 — First results, kill/pivot gate

#### Week 9 — Iterate
- [ ] Try at least 2 variants of the core mechanism. Different hyperparameters, different design choices.
- [ ] Run each on MemoryAgentBench. Log numbers publicly.

#### Week 10 — Iterate
- [ ] Try at least 1 more substantially different variant (different mechanism for the same biological principle).
- [ ] By end of this week you should have 3+ variants benchmarked against the baseline.

#### Week 11 — Analysis
- [ ] Honest analysis: do any variants show signal? Where? On which competencies?
- [ ] Sketch what would need to be true for the strongest variant to be a real research contribution.

#### Week 12 — Decision and write-up
- [ ] **Third blog post published:** "First results: does [reconsolidation/forgetting] actually help LLM agent memory?" Honest, technical, includes numbers.
- [ ] **Kill/pivot/proceed decision logged in §10.** Three possible outcomes:
  - **Strong signal** (clear improvement on at least one competency): continue as planned.
  - **Mixed signal** (improvement on some, regression on others): expected, the more common case. Proceed to Phase 2.
  - **No signal at all**: this is the off-ramp. Pivot now, not in month 6.

**Month 3 checkpoint (end of week 12):**
- ✓ 3+ variants benchmarked
- ✓ Honest assessment of signal
- ✓ Third blog post live
- ✓ Decision recorded in §10 (kill / pivot / proceed)

This is the most important checkpoint of the year. Loss of 3 months from an honest pivot is much better than loss of 9 months from a denied pivot.

---

## 3. Phase 2 — Build (Months 4–6, bi-weekly)

### Goal
Take the prototype to a real implementation with credible comparative benchmark results. Every two weeks, ship something visible.

### Month 4

#### Weeks 13–14 — Generalisation
- [ ] Test the mechanism with at least 2 different base LLMs (e.g. Claude + GPT-4 or Claude + open-weight).
- [ ] Test with at least 2 different embedding providers.
- [ ] Document which combinations work well and which fail.

#### Weeks 15–16 — Additional benchmark
- [ ] Add one benchmark beyond MemoryAgentBench. Recommended: Memora (long-horizon personalisation) or LongMemEval (broader capability coverage).
- [ ] Run the mechanism and the baseline on the additional benchmark.
- [ ] **Fourth blog post:** technical deep-dive on the mechanism design — the "how it works" post.

### Month 5

#### Weeks 17–18 — Engineering quality pass
- [ ] Rewrite the implementation with production-quality code: clean API, type hints, tests, docs.
- [ ] CI set up on GitHub Actions (lint, type-check, tests).
- [ ] Documentation site (mkdocs or similar) deployed.

#### Weeks 19–20 — v0.5 release
- [ ] PyPI package published as v0.5.
- [ ] Tag the release, write release notes.
- [ ] Make sure the README is good enough that a stranger can install and run a quickstart in under 10 minutes.

### Month 6

#### Weeks 21–22 — Comparative landscape
- [ ] Reproduce 1–2 additional memory methods so you have honest comparison numbers beyond just the original baseline. Candidates: simple top-k retrieval, long-context-stuff baseline, second memory method (Mem0 if you reproduced A-Mem first, or vice versa).
- [ ] Make sure all comparisons use the same evaluation harness — fair-fight rules.

#### Weeks 23–24 — Hyperparameter sweep
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
| Foundation | Baseline reproduced | Within ~5% of published |
| Foundation | Direction commitment | Yes/no, locked by end of week 1 |
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

**By end of Week 2:**
- [ ] **Project focus:** reconsolidation OR active competitive forgetting.
- [ ] **Final project name.**

**By end of Week 3:**
- [ ] **Primary benchmark:** MemoryAgentBench (recommended) or alternative.
- [ ] **Reproduction target:** Mem0, A-Mem, or HippoRAG.
- [ ] **Primary base LLM:** Claude, GPT-4, or open-weight (Llama 3 via Together/Replicate).
- [ ] **Primary embedding model:** OpenAI text-embedding-3-large or Voyage-3-large (see embeddings deep-dive).

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

## 10. Decision log

*Use this to record significant decisions and their rationale. Future-you needs this when re-reading the roadmap in month 7.*

| Date | Decision | Rationale | Revisit? |
|---|---|---|---|
| 2026-05-21 | Roadmap drafted (job-hunt removed) | Goal = "interesting work + portfolio + learning"; career planning to be handled separately if needed | End of month 1 |
| | | | |

---

## 11. Notes / scratch

*Ongoing thoughts, idea fragments, questions for later. Trim quarterly.*

- Project naming: prefer a memorable name over a descriptive one — easier to talk and write about.
- Library vs research code positioning: lean library (people install and use it) — harder but more impactful.
- Twitter/X presence: 1–2 substantive posts per week + light replies is the right cadence. Not constant posting.
- Open question: should the implementation be designed to be drop-in with existing frameworks (LlamaIndex memory module, DSPy module) or standalone? Drop-in increases adoption potential significantly.
