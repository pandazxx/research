"""Loaders for HippoRAG and A-Mem reproductions.

These are PLACEHOLDERS. The actual reproduction code lives in separate repos:
- HippoRAG: https://github.com/pandazxx/hipporag-reproduction
- A-Mem:    https://github.com/pandazxx/a-mem-reproduction

To use them from this experiments directory, either:
  (a) git-clone the reproductions side-by-side with this repo, and update
      HIPPORAG_PATH / AMEM_PATH below to point at them, or
  (b) pip-install the reproductions as editable packages and import their
      public APIs directly.

The wrapper classes below define the minimum interface that the
memory-systems experiments expect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# -----------------------------------------------------------------------------
# Configure these paths
# -----------------------------------------------------------------------------

HIPPORAG_PATH = Path.home() / "code" / "hipporag-reproduction"
AMEM_PATH = Path.home() / "code" / "a-mem-reproduction"


# -----------------------------------------------------------------------------
# Common interface that experiments use
# -----------------------------------------------------------------------------

class MemorySystem:
    """Minimal interface that both HippoRAGSystem and AMemSystem implement."""

    def ingest(self, memories: list[dict[str, Any]]) -> None:
        """Feed memories (with id, content, timestamp) to the system."""
        raise NotImplementedError

    def query(self, question: str) -> dict[str, Any]:
        """Answer a question. Return dict with:
          - answer: str — the final natural-language answer
          - retrieved_ids: list[str] — memory ids surfaced by retrieval
        """
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Adapters — fill in the actual implementations
# -----------------------------------------------------------------------------

class HippoRAGSystem(MemorySystem):
    def __init__(self, llm_model: str, embedder: str, branch: str = "legacy"):
        """Initialise HippoRAG. Should pick branch='legacy' for v1 or
        'main' for v2 in the OSU-NLP-Group/HippoRAG repo."""
        self.llm_model = llm_model
        self.embedder = embedder
        self.branch = branch
        # TODO: import and instantiate HippoRAG from HIPPORAG_PATH
        raise NotImplementedError(
            f"Wire up your HippoRAG repro from {HIPPORAG_PATH}"
        )

    def ingest(self, memories: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    def query(self, question: str) -> dict[str, Any]:
        raise NotImplementedError


class AMemSystem(MemorySystem):
    def __init__(self, llm_model: str, embedder: str):
        self.llm_model = llm_model
        self.embedder = embedder
        # TODO: import and instantiate A-Mem from AMEM_PATH
        raise NotImplementedError(
            f"Wire up your A-Mem repro from {AMEM_PATH}"
        )

    def ingest(self, memories: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    def query(self, question: str) -> dict[str, Any]:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Long-context baseline (no memory system at all)
# -----------------------------------------------------------------------------

class LongContextBaseline(MemorySystem):
    """Stuffs all memories into a single prompt and asks the LLM directly.

    The control baseline that every memory system must beat to justify its
    existence.
    """

    def __init__(self, llm_model: str = "claude-sonnet-4-6"):
        self.llm_model = llm_model
        self.memories: list[dict[str, Any]] = []

    def ingest(self, memories: list[dict[str, Any]]) -> None:
        self.memories = memories

    def query(self, question: str) -> dict[str, Any]:
        context = "\n".join(
            f"[{m['timestamp']}] {m['content']}" for m in self.memories
        )
        prompt = (
            "Below are facts told to you in chronological order. "
            "Answer the question based ONLY on these facts. "
            "If the answer is not in the facts, say so.\n\n"
            f"FACTS:\n{context}\n\n"
            f"QUESTION: {question}\n\nANSWER:"
        )
        # TODO: actually call the LLM API. Stubbed for now.
        raise NotImplementedError("Plug in your preferred LLM API")
