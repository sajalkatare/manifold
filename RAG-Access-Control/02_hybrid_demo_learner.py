"""Vector vs BM25 vs hybrid (RRF) retrieval with access control.

Run:
    docker compose up -d
    export OPENAI_API_KEY=sk-...
    python build_corpus_learner.py
    python 02_hybrid_demo_learner.py
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, TypedDict
from dotenv import load_dotenv
load_dotenv()
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langgraph.graph import END, START, StateGraph

try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever

EMBED_MODEL = "text-embedding-3-small"
COLLECTION = "session3_secure_kb"
DEFAULT_DSN = "postgresql+psycopg://rag:rag@localhost:5433/ragdb"


def die(msg: str):
    print(f"\n[setup] {msg}", file=sys.stderr)
    sys.exit(1)


def pg_connection() -> str:
    dsn = os.getenv("DATABASE_URL", DEFAULT_DSN)
    if dsn.startswith("postgresql://"):
        dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    return dsn


def get_embedder() -> OpenAIEmbeddings:
    if not os.getenv("OPENAI_API_KEY"):
        die("OPENAI_API_KEY is not set.")
    return OpenAIEmbeddings(model=EMBED_MODEL)


def load_json(path: str):
    if not os.path.exists(path):
        die(f"{path} not found. Run: python build_corpus_learner.py")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_documents(chunks: list[dict]) -> list[Document]:
    return [
        Document(
            page_content=c["content"],
            metadata={
                "source": c.get("source"),
                "section": c.get("section"),
                "chunk_id": c.get("chunk_id"),
                "department": c.get("department", "all"),
                "access_level": c.get("access_level", "public"),
                "tenant_id": c.get("tenant_id", "acme"),
            },
        )
        for c in chunks
    ]


@dataclass
class UserContext:
    name: str
    role: str
    department: str
    access_levels: list = field(default_factory=lambda: ["public"])
    tenant_id: str = "acme"


def can_see(user: UserContext, d: Document) -> bool:
    m = d.metadata
    if m.get("tenant_id") != user.tenant_id:
        return False
    if m.get("access_level") not in user.access_levels:
        return False
    return m.get("department", "all") in ("all", user.department) or user.role == "admin"


def access_filter(user: UserContext) -> dict:
    clauses = [
        {"tenant_id": {"$eq": user.tenant_id}},
        {"access_level": {"$in": user.access_levels}},
    ]
    if user.role != "admin":
        clauses.append({"department": {"$in": ["all", user.department]}})
    return {"$and": clauses}


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", text.lower()) if t]


class BM25Retriever(BaseRetriever):
    """Keyword retriever using rank_bm25."""

    k: int = 4
    docs: list[Document] = Field(default_factory=list)
    bm25: Any = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_documents(cls, documents: list[Document]) -> "BM25Retriever":
        corpus = [_tokenize(d.page_content) for d in documents]
        return cls(docs=documents, bm25=BM25Okapi(corpus))

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.docs[i] for i in ranked[: self.k]]


def rule(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)


def make_retrievers(store, docs, user, k=5):
    vector = store.as_retriever(search_kwargs={"k": k, "filter": access_filter(user)})
    allowed = [d for d in docs if can_see(user, d)]
    bm25 = BM25Retriever.from_documents(allowed)
    bm25.k = k
    hybrid = EnsembleRetriever(retrievers=[bm25, vector], weights=[0.5, 0.5])
    return vector, bm25, hybrid


def sections(results):
    return [d.metadata["section"] for d in results]


def main():
    chunks = load_json("data/chunks.json")
    golden = load_json("data/golden_queries.json")
    docs = to_documents(chunks)
    emb = get_embedder()

    rule(f"BUILD  (PGVector '{COLLECTION}')")
    try:
        store = PGVector(
            embeddings=emb,
            collection_name=COLLECTION,
            connection=pg_connection(),
            use_jsonb=True,
            pre_delete_collection=True,
        )
    except Exception as e:
        die(f"cannot connect to Postgres ({e}). Start: docker compose up -d")
    store.add_documents(docs)
    print(f"   stored {len(docs)} chunks")

    admin = UserContext("Asha (admin)", "admin", "all", ["public", "internal", "restricted"])
    vector, bm25, hybrid = make_retrievers(store, docs, admin, k=5)

    def show(label, results):
        print(f"\n  {label}:")
        for r, d in enumerate(results[:3], 1):
            print(f"     {r}. {d.metadata['section']:24s} ({d.metadata['source']})")
        if not results:
            print("     (no results)")

    q1 = "What does error ERR-5021 mean?"
    rule(f"1) EXACT-TERM QUERY: {q1!r}")
    show("vector-only", vector.invoke(q1))
    show("BM25-only", bm25.invoke(q1))
    show("hybrid(RRF)", hybrid.invoke(q1))

    q2 = "How much time off do new parents get?"
    rule(f"2) PARAPHRASE QUERY: {q2!r}")
    show("vector-only", vector.invoke(q2))
    show("BM25-only", bm25.invoke(q2))
    show("hybrid(RRF)", hybrid.invoke(q2))

    rule("3) HIT-RATE@3 (golden set, admin view)")
    print(f"   {'query':46s} vector bm25 hybrid")
    tot = {"vector": 0, "bm25": 0, "hybrid": 0}
    for g in golden:
        v, b, h = make_retrievers(store, docs, admin, k=3)
        marks = {}
        for name, ret in (("vector", v), ("bm25", b), ("hybrid", h)):
            ids = [d.metadata["chunk_id"] for d in ret.invoke(g["query"])]
            marks[name] = int(any(cid in ids for cid in g["relevant"]))
            tot[name] += marks[name]
        print(
            f"   {g['query'][:46]:46s}   {marks['vector']}     "
            f"{marks['bm25']}     {marks['hybrid']}"
        )
    n = len(golden)
    print("   " + "-" * 64)
    print(
        f"   {'HIT-RATE@3':46s}  {tot['vector']/n:.2f}  "
        f"{tot['bm25']/n:.2f}   {tot['hybrid']/n:.2f}"
    )

    rule("4) Secure hybrid retrieval as LangGraph node")

    class S(TypedDict):
        query: str
        user: UserContext
        results: list

    def secure_retrieve(state: S) -> dict:
        _, _, hyb = make_retrievers(store, docs, state["user"], k=3)
        return {"results": sections(hyb.invoke(state["query"]))}

    graph = StateGraph(S)
    graph.add_node("secure_retrieve", secure_retrieve)
    graph.add_edge(START, "secure_retrieve")
    graph.add_edge("secure_retrieve", END)
    app = graph.compile()
    public = UserContext("Pat (public)", "public", "none", ["public"])
    out = app.invoke({"query": "leadership compensation", "user": public})
    print(f"   public user -> {out['results']}")


if __name__ == "__main__":
    main()
