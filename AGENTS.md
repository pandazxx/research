# Repository Guidelines

## Purpose & Scope
This repository is a personal research workspace. It is used to study multiple subjects, collect source material, keep working notes, and consolidate takeaways into durable summaries. Treat clarity, traceability, and retrieval as the primary goals.

## Research Structure
Organize content by subject. Each topic should live in its own folder, for example:

- `topics/llm-agents/notes.md`
- `topics/llm-agents/summary.md`
- `topics/llm-agents/sources/`

Use `notes.md` for raw findings, `summary.md` for cleaned conclusions, and `sources/` for PDFs, links, excerpts, or metadata. Keep filenames descriptive and lowercase with hyphens, for example `retrieval-augmented-generation.md`.

## Agent Workflow
When working in this repository, the agent should:

- add new findings to the relevant topic instead of scattering notes at the root
- preserve source attribution for claims, quotes, and statistics
- consolidate repeated notes into short summaries when a topic grows
- prefer updating existing topic folders before creating new ones

If a subject spans several subtopics, create subfolders rather than mixing unrelated notes in one file.

## Note-Taking Conventions
Write in Markdown with short sections and scannable headings. Separate raw notes from synthesized conclusions. Mark open questions clearly, for example `## Open Questions`, and keep action items explicit.

When summarizing a source, capture:

- what it says
- why it matters
- how reliable or relevant it appears

## Reference Management
Store reference documents close to the topic they support. Prefer a small companion note alongside each source that records title, author, date, URL, and key takeaways. Avoid saving duplicate copies of the same document in multiple places.

## Maintenance Guidelines
Use `rg --files` to inspect the repository, `git status` to confirm changes, and `git diff --staged` before committing. Commit messages should be short and descriptive, such as `Add notes on agent memory systems` or `Consolidate transformer scaling references`.

The agent should commit and push whenever a meaningful unit of work is completed, such as adding a new topic, consolidating a set of notes, organizing source documents, or producing a useful summary. Avoid batching unrelated research changes into one commit. Prefer small, reviewable commits that leave the repository in a coherent state.
