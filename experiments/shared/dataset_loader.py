"""Load the HippoRAG vs A-Mem comparison dataset."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_DATASET_PATH = (
    REPO_ROOT / "topics" / "llm-agent-memory" / "comparison-dataset" / "dataset.json"
)


def load_comparison_dataset() -> dict:
    """Load the comparison dataset (40 memories, 22 questions, 7 categories).

    Returns the parsed JSON dict with keys: metadata, memories, questions.
    """
    with COMPARISON_DATASET_PATH.open() as f:
        return json.load(f)


def get_questions_by_category(dataset: dict, category: str) -> list[dict]:
    """Filter questions by category (e.g., 'single_hop', 'deep_multi_hop')."""
    return [q for q in dataset["questions"] if q["category"] == category]


def get_memory_by_id(dataset: dict, memory_id: str) -> dict | None:
    """Look up a memory by its id field."""
    for m in dataset["memories"]:
        if m["id"] == memory_id:
            return m
    return None
