# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 09 - Simulate Memory-R1 with off-the-shelf agents
#
# **Question.** Can a prompted Memory Manager plus a prompted Answer Agent
# reproduce the core Memory-R1 loop before training either network?
#
# **Why it matters.** Memory-R1's trained manager is not publicly available
# yet. This notebook creates a runnable proxy: it extracts facts from short
# dialogue events, retrieves related memories, asks an LLM manager to choose
# `ADD` / `UPDATE` / `DELETE` / `NOOP`, stores the result, and then asks an
# LLM answer agent to answer questions from retrieved memory.
#
# **Recommended local vector DB.** Use LanceDB for the durable version of this
# experiment: it is local-first, file-backed, easy to inspect, and supports
# vector search without running a separate service. This notebook defaults to a
# dependency-free in-memory store for quick iteration, and includes an optional
# `LanceDBVectorStore` adapter for runs where `lancedb` is installed.
#
# **Requires.**
# - `NVIDIA_API_KEY` or `NIM_API_KEY` for NIM chat completions.
# - `OPENAI_API_KEY` only when `EMBEDDING_PROVIDER = "openai"`.
# - `lancedb` only when `STORE_BACKEND = "lancedb"`.

# %%
from __future__ import annotations

import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

sys.path.insert(0, str(Path.cwd().parents[1]))

from experiments.shared.embedding_utils import cosine_matrix, load_sentence_transformer
from experiments.shared.llm_clients import get_nim_client

# %% [markdown]
# ## Configuration
#
# Toggle the manager and answer-agent model independently. The defaults use
# Llama 3.1 8B Instruct through NVIDIA NIM; switch to Qwen 2.5 7B Instruct by
# changing the two role model names below.

# %%
NIM_MODELS = {
    "llama31-8b": "meta/llama-3.1-8b-instruct",
    "qwen25-7b": "qwen/qwen2.5-7b-instruct",
}

MANAGER_MODEL = NIM_MODELS["llama31-8b"]
ANSWER_MODEL = NIM_MODELS["llama31-8b"]

EmbeddingProvider = Literal["bge", "e5", "openai"]
EMBEDDING_PROVIDER: EmbeddingProvider = "bge"

STORE_BACKEND = "memory"  # "memory" or "lancedb"
LANCEDB_PATH = Path("experiments/.local/memory-r1-lancedb")
LANCEDB_TABLE = "memories"

TOP_K_MANAGER_RETRIEVAL = 4
TOP_K_ANSWER_RETRIEVAL = 5

DRY_RUN = os.environ.get("NIM_DRY_RUN", "0") == "1"

EMBEDDING_MODELS = {
    "bge": "BAAI/bge-small-en-v1.5",
    "e5": "intfloat/e5-base-v2",
    "openai": "text-embedding-3-small",
}

# %% [markdown]
# ## Toy dialogue and QA set
#
# Keep the data tiny at first. The point is to observe write-time behavior:
# does the manager update stale facts instead of blindly appending?

# %%
EVENTS = [
    {
        "id": "turn-001",
        "text": "Alice lives in Boston and usually drinks green tea.",
        "timestamp": "2024-01-01",
    },
    {
        "id": "turn-002",
        "text": "Alice moved to Seattle last month.",
        "timestamp": "2024-03-15",
    },
    {
        "id": "turn-003",
        "text": "Alice stopped drinking green tea and now mostly drinks coffee.",
        "timestamp": "2024-04-02",
    },
    {
        "id": "turn-004",
        "text": "Bob works at AcmeCo as a data scientist.",
        "timestamp": "2024-04-03",
    },
    {
        "id": "turn-005",
        "text": "Bob accepted a machine learning engineer role at NovaTech.",
        "timestamp": "2024-05-01",
    },
]

QUESTIONS = [
    {
        "id": "q1",
        "question": "Where does Alice currently live?",
        "expected": "Seattle",
    },
    {
        "id": "q2",
        "question": "What does Alice mostly drink now?",
        "expected": "coffee",
    },
    {
        "id": "q3",
        "question": "Where does Bob currently work?",
        "expected": "NovaTech",
    },
]

# %% [markdown]
# ## Data structures

# %%
Operation = Literal["ADD", "UPDATE", "DELETE", "NOOP"]


@dataclass
class MemoryRecord:
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source_ids: list[str] = field(default_factory=list)


@dataclass
class ManagerDecision:
    operation: Operation
    content: str
    target_id: str | None
    rationale: str


class VectorStore(Protocol):
    def add(self, record: MemoryRecord, vector: np.ndarray) -> None:
        ...

    def update(self, record: MemoryRecord, vector: np.ndarray) -> None:
        ...

    def delete(self, record_id: str) -> None:
        ...

    def search(self, query_vector: np.ndarray, *, top_k: int) -> list[tuple[MemoryRecord, float]]:
        ...

    def all_records(self) -> list[MemoryRecord]:
        ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.records: dict[str, MemoryRecord] = {}
        self.vectors: dict[str, np.ndarray] = {}

    def add(self, record: MemoryRecord, vector: np.ndarray) -> None:
        self.records[record.id] = record
        self.vectors[record.id] = vector

    def update(self, record: MemoryRecord, vector: np.ndarray) -> None:
        self.add(record, vector)

    def delete(self, record_id: str) -> None:
        self.records.pop(record_id, None)
        self.vectors.pop(record_id, None)

    def search(self, query_vector: np.ndarray, *, top_k: int) -> list[tuple[MemoryRecord, float]]:
        if not self.records:
            return []
        ids = list(self.records)
        matrix = np.vstack([self.vectors[memory_id] for memory_id in ids])
        scores = cosine_matrix(np.asarray([query_vector]), matrix)[0]
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.records[ids[i]], float(scores[i])) for i in order]

    def all_records(self) -> list[MemoryRecord]:
        return list(self.records.values())


class LanceDBVectorStore:
    """Optional durable local vector store.

    Install with `uv add lancedb` or `uv pip install lancedb` before using this
    backend. The in-memory store above is enough for quick notebook iteration.
    """

    def __init__(self, path: Path, table_name: str) -> None:
        import lancedb
        import pyarrow as pa

        self.db = lancedb.connect(path)
        self.table_name = table_name
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("content", pa.string()),
                pa.field("timestamp", pa.string()),
                pa.field("source_ids", pa.list_(pa.string())),
                pa.field("vector", pa.list_(pa.float32())),
            ]
        )
        self.table = self.db.create_table(table_name, schema=schema, exist_ok=True)

    def _row(self, record: MemoryRecord, vector: np.ndarray) -> dict[str, Any]:
        return {
            "id": record.id,
            "content": record.content,
            "timestamp": record.timestamp,
            "source_ids": record.source_ids,
            "vector": np.asarray(vector, dtype=np.float32).tolist(),
        }

    def add(self, record: MemoryRecord, vector: np.ndarray) -> None:
        self.table.add([self._row(record, vector)])

    def update(self, record: MemoryRecord, vector: np.ndarray) -> None:
        self.delete(record.id)
        self.add(record, vector)

    def delete(self, record_id: str) -> None:
        self.table.delete(f"id = '{record_id}'")

    def search(self, query_vector: np.ndarray, *, top_k: int) -> list[tuple[MemoryRecord, float]]:
        if self.table.count_rows() == 0:
            return []
        rows = (
            self.table.search(np.asarray(query_vector, dtype=np.float32).tolist())
            .limit(top_k)
            .to_list()
        )
        results = []
        for row in rows:
            score = 1.0 - float(row.get("_distance", 0.0))
            results.append((self._record_from_row(row), score))
        return results

    def all_records(self) -> list[MemoryRecord]:
        return [self._record_from_row(row) for row in self.table.to_list()]

    def _record_from_row(self, row: dict[str, Any]) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            timestamp=row["timestamp"],
            source_ids=list(row.get("source_ids") or []),
        )

# %% [markdown]
# ## Embeddings

# %%
class Embedder:
    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider
        self.model_name = EMBEDDING_MODELS[provider]
        self._local_model = None
        self._openai_client = None

    @property
    def dim(self) -> int:
        return len(self.embed_document("dimension probe"))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_many([text], input_type="query")[0]

    def embed_document(self, text: str) -> np.ndarray:
        return self.embed_many([text], input_type="passage")[0]

    def embed_many(
        self,
        texts: list[str],
        *,
        input_type: Literal["query", "passage"] = "passage",
    ) -> np.ndarray:
        if self.provider == "openai":
            return self._embed_openai(texts)
        return self._embed_local(texts, input_type=input_type)

    def _embed_local(
        self,
        texts: list[str],
        *,
        input_type: Literal["query", "passage"],
    ) -> np.ndarray:
        if self._local_model is None:
            self._local_model = load_sentence_transformer(self.model_name)
        if self.provider == "e5":
            texts = [f"{input_type}: {text}" for text in texts]
        return np.asarray(
            self._local_model.encode(texts, normalize_embeddings=True),
            dtype=np.float32,
        )

    def _embed_openai(self, texts: list[str]) -> np.ndarray:
        from openai import OpenAI

        if self._openai_client is None:
            self._openai_client = OpenAI()
        response = self._openai_client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return np.asarray([item.embedding for item in response.data], dtype=np.float32)


def make_store() -> VectorStore:
    if STORE_BACKEND == "memory":
        return InMemoryVectorStore()
    if STORE_BACKEND == "lancedb":
        return LanceDBVectorStore(LANCEDB_PATH, LANCEDB_TABLE)
    raise ValueError(f"Unknown STORE_BACKEND={STORE_BACKEND!r}")

# %% [markdown]
# ## NIM chat helpers

# %%
def chat_json(
    *,
    client: Any,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
) -> dict[str, Any]:
    if DRY_RUN:
        raise RuntimeError("chat_json should not be called in DRY_RUN mode.")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    content = response.choices[0].message.content or "{}"
    return parse_json_object(content)


def chat_text(
    *,
    client: Any,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
) -> str:
    if DRY_RUN:
        raise RuntimeError("chat_text should not be called in DRY_RUN mode.")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if not match:
        match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(1))
    raise ValueError(f"Expected JSON object, got: {text[:200]}")

# %% [markdown]
# ## Fact extraction, memory management, and answering

# %%
FACT_EXTRACTION_SYSTEM = """Extract durable user facts from the text.
Return JSON with this schema:
{"facts": ["atomic fact 1", "atomic fact 2"]}
Only include stable facts useful for future personalization or QA.
"""

MANAGER_SYSTEM = """You are a Memory-R1-style memory manager.
Given one extracted fact and related existing memories, choose exactly one
operation:
- ADD: store the new fact as a new memory.
- UPDATE: revise one existing memory to reflect the newest correct state.
- DELETE: remove an obsolete or invalid memory.
- NOOP: skip facts that are not useful as durable memory.

Return JSON with this schema:
{
  "operation": "ADD|UPDATE|DELETE|NOOP",
  "target_id": "existing memory id, or null",
  "content": "memory text to add or updated replacement text, or empty string",
  "rationale": "brief reason"
}
Prefer UPDATE over ADD when a new fact supersedes an old one.
"""

ANSWER_SYSTEM = """Answer using only the provided memory records.
If the memories do not contain the answer, say "I don't know".
Return a concise answer, not a chain of thought.
"""


def extract_facts(client: Any, text: str) -> list[str]:
    if DRY_RUN:
        return heuristic_extract_facts(text)
    payload = chat_json(
        client=client,
        model=MANAGER_MODEL,
        system=FACT_EXTRACTION_SYSTEM,
        user=f"Text:\n{text}",
    )
    return [str(item).strip() for item in payload.get("facts", []) if str(item).strip()]


def manage_memory(
    *,
    client: Any,
    fact: str,
    retrieved: list[tuple[MemoryRecord, float]],
) -> ManagerDecision:
    if DRY_RUN:
        return heuristic_manager(fact, retrieved)

    related = [
        {
            "id": record.id,
            "content": record.content,
            "timestamp": record.timestamp,
            "similarity": round(score, 3),
        }
        for record, score in retrieved
    ]
    payload = chat_json(
        client=client,
        model=MANAGER_MODEL,
        system=MANAGER_SYSTEM,
        user=json.dumps({"new_fact": fact, "related_memories": related}, indent=2),
    )
    operation = str(payload.get("operation", "NOOP")).upper()
    if operation not in {"ADD", "UPDATE", "DELETE", "NOOP"}:
        operation = "NOOP"
    return ManagerDecision(
        operation=operation,  # type: ignore[arg-type]
        target_id=payload.get("target_id"),
        content=str(payload.get("content", "")).strip(),
        rationale=str(payload.get("rationale", "")).strip(),
    )


def answer_question(
    *,
    client: Any,
    question: str,
    retrieved: list[tuple[MemoryRecord, float]],
) -> str:
    if DRY_RUN:
        return heuristic_answer(question, retrieved)

    memories = [
        {
            "id": record.id,
            "content": record.content,
            "similarity": round(score, 3),
        }
        for record, score in retrieved
    ]
    return chat_text(
        client=client,
        model=ANSWER_MODEL,
        system=ANSWER_SYSTEM,
        user=json.dumps({"question": question, "memories": memories}, indent=2),
    )


def apply_decision(
    *,
    store: VectorStore,
    embedder: Embedder,
    decision: ManagerDecision,
    source_id: str,
) -> None:
    if decision.operation == "NOOP":
        return
    if decision.operation == "DELETE":
        if decision.target_id:
            store.delete(decision.target_id)
        return
    if not decision.content:
        return

    if decision.operation == "UPDATE" and decision.target_id:
        record_id = decision.target_id
    else:
        record_id = str(uuid.uuid4())
    record = MemoryRecord(
        id=record_id,
        content=decision.content,
        source_ids=[source_id],
    )
    store.update(record, embedder.embed_document(record.content))

# %% [markdown]
# ## Dry-run heuristics
#
# Set `NIM_DRY_RUN=1` to exercise the notebook without API calls. These
# heuristics are intentionally simple; they are only a smoke-test path.

# %%
def heuristic_extract_facts(text: str) -> list[str]:
    return [part.strip().rstrip(".") + "." for part in re.split(r"\band\b|;", text) if part.strip()]


def heuristic_manager(
    fact: str,
    retrieved: list[tuple[MemoryRecord, float]],
) -> ManagerDecision:
    fact_lower = fact.lower()
    for record, score in retrieved:
        content_lower = record.content.lower()
        same_subject = content_lower.split(" ", 1)[0] == fact_lower.split(" ", 1)[0]
        if same_subject and score > 0.35:
            return ManagerDecision(
                operation="UPDATE",
                target_id=record.id,
                content=fact,
                rationale="Heuristic update for same subject.",
            )
    return ManagerDecision(
        operation="ADD",
        target_id=None,
        content=fact,
        rationale="Heuristic add.",
    )


def heuristic_answer(
    question: str,
    retrieved: list[tuple[MemoryRecord, float]],
) -> str:
    question_lower = question.lower()
    joined = " ".join(record.content for record, _ in retrieved)
    if "alice" in question_lower and "live" in question_lower and "Seattle" in joined:
        return "Seattle"
    if "alice" in question_lower and "drink" in question_lower and "coffee" in joined:
        return "coffee"
    if "bob" in question_lower and "work" in question_lower and "NovaTech" in joined:
        return "NovaTech"
    return "I don't know"

# %% [markdown]
# ## Run ingestion

# %%
client = None if DRY_RUN else get_nim_client()
embedder = Embedder(EMBEDDING_PROVIDER)
store = make_store()

manager_trace = []

for event in EVENTS:
    facts = extract_facts(client, event["text"])
    print(f"\n{event['id']} facts:")
    for fact in facts:
        print(f"- {fact}")

        retrieved = store.search(embedder.embed_query(fact), top_k=TOP_K_MANAGER_RETRIEVAL)
        decision = manage_memory(client=client, fact=fact, retrieved=retrieved)
        apply_decision(
            store=store,
            embedder=embedder,
            decision=decision,
            source_id=event["id"],
        )
        manager_trace.append(
            {
                "event_id": event["id"],
                "fact": fact,
                "operation": decision.operation,
                "target_id": decision.target_id,
                "content": decision.content,
                "rationale": decision.rationale,
                "retrieved": [(record.content, round(score, 3)) for record, score in retrieved],
            }
        )
        print(f"  -> {decision.operation}: {decision.content or decision.target_id}")

# %% [markdown]
# ## Inspect final memory bank

# %%
for record in store.all_records():
    print(f"{record.id[:8]} | {record.content}")

# %% [markdown]
# ## Answer QA from memory

# %%
qa_results = []
for qa in QUESTIONS:
    retrieved = store.search(
        embedder.embed_query(qa["question"]),
        top_k=TOP_K_ANSWER_RETRIEVAL,
    )
    answer = answer_question(
        client=client,
        question=qa["question"],
        retrieved=retrieved,
    )
    qa_results.append(
        {
            "id": qa["id"],
            "question": qa["question"],
            "expected": qa["expected"],
            "answer": answer,
            "retrieved": [(record.content, round(score, 3)) for record, score in retrieved],
        }
    )
    print(f"\nQ: {qa['question']}")
    print(f"A: {answer}")
    print(f"Expected: {qa['expected']}")

# %% [markdown]
# ## Score the toy run

# %%
def contains_expected(answer: str, expected: str) -> bool:
    return expected.lower() in answer.lower()


correct = sum(contains_expected(row["answer"], row["expected"]) for row in qa_results)
print(f"Toy QA accuracy: {correct}/{len(qa_results)}")

# %% [markdown]
# ## Conclusions
#
# 1. **What did I measure?** Whether a prompted Memory-R1-style loop can keep
#    contradictory personal facts current and answer from the resulting memory.
# 2. **What did I find?** Run the notebook with NIM enabled and record the
#    manager operation trace plus toy QA accuracy here.
# 3. **What surprised me?** Inspect cases where the manager appends instead of
#    updating, or where the answer agent selects a stale retrieved memory.
# 4. **What's next?** Replace the toy events with LoCoMo-style turn/QA tuples,
#    then convert answer correctness into PPO/GRPO rewards for manager training.
