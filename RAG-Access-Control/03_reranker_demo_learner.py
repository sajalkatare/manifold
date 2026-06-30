import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, TypedDict
from dotenv import load_dotenv
load_dotenv()
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, model_validator
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
RECALL_K = 10
RERANK_TOP_N = 3


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
    """Keyword retriever using rank_bm25 (no langchain_community)."""

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


DEFAULT_FLASHRANK_MODEL = "ms-marco-MultiBERT-L-12"


class FlashrankRerank(BaseDocumentCompressor):
    """Cross-encoder reranker via flashrank (no langchain_community)."""

    client: Any = None
    top_n: int = 3
    score_threshold: float = 0.0
    model: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _init_client(cls, values: dict) -> dict:
        if values.get("client") is not None:
            return values
        try:
            from flashrank import Ranker
        except ImportError as e:
            raise ImportError(
                "flashrank is not installed. Run: pip install flashrank"
            ) from e
        model = values.get("model", DEFAULT_FLASHRANK_MODEL)
        values["model"] = model
        values["client"] = Ranker(model_name=model)
        return values

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        from flashrank import RerankRequest

        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(documents)
        ]
        response = self.client.rerank(RerankRequest(query=query, passages=passages))
        results = []
        for item in response[: self.top_n]:
            if item["score"] >= self.score_threshold:
                results.append(
                    Document(
                        page_content=item["text"],
                        metadata={
                            "relevance_score": item["score"],
                            **item["meta"],
                        },
                    )
                )
        return results


def get_reranker(top_n: int = RERANK_TOP_N) -> FlashrankRerank:
    try:
        return FlashrankRerank(top_n=top_n)
    except ImportError:
        die("flashrank is not installed. Run: pip install flashrank")


def rule(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)


def make_hybrid(store, docs, user, k=RECALL_K):
    vector = store.as_retriever(search_kwargs={"k": k, "filter": access_filter(user)})
    allowed = [d for d in docs if can_see(user, d)]
    bm25 = BM25Retriever.from_documents(allowed)
    bm25.k = k
    return EnsembleRetriever(retrievers=[bm25, vector], weights=[0.5, 0.5])


def show_results(label, results):
    print(f"\n  {label}:")
    if not results:
        print("     (no results)")
        return
    for rank, d in enumerate(results[:RERANK_TOP_N], 1):
        score = d.metadata.get("relevance_score")
        score_txt = f"  score={score:.4f}" if score is not None else ""
        print(
            f"     {rank}. {d.metadata['section']:24s} "
            f"({d.metadata['source']}){score_txt}"
        )


def rerank(reranker: FlashrankRerank, query: str, candidates: list[Document]):
    return list(reranker.compress_documents(candidates, query))


def main():
    chunks = load_json("data/chunks.json")
    golden = load_json("data/golden_queries.json")
    docs = to_documents(chunks)
    emb = get_embedder()
    reranker = get_reranker(RERANK_TOP_N)

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
    hybrid = make_hybrid(store, docs, admin, k=RECALL_K)

    q = "How much time off do new parents get?"
    rule(f"1) hybrid recall@{RECALL_K} vs rerank top {RERANK_TOP_N}")
    candidates = hybrid.invoke(q)
    show_results("hybrid recall", candidates)
    show_results("after rerank", rerank(reranker, q, candidates))

    q2 = "What does error ERR-5021 mean?"
    rule("2) exact-term query")
    candidates2 = hybrid.invoke(q2)
    show_results("hybrid recall", candidates2)
    show_results("after rerank", rerank(reranker, q2, candidates2))

    rule("3) HIT-RATE@3")
    print(f"   {'query':46s} hybrid  +rerank")
    tot = {"hybrid": 0, "rerank": 0}
    for g in golden:
        hyb = make_hybrid(store, docs, admin, k=RECALL_K)
        cands = hyb.invoke(g["query"])
        hybrid_ids = [d.metadata["chunk_id"] for d in cands[:3]]
        reranked_ids = [
            d.metadata["chunk_id"] for d in rerank(reranker, g["query"], cands)[:3]
        ]
        relevant = set(g["relevant"])
        marks = {
            "hybrid": int(any(cid in relevant for cid in hybrid_ids)),
            "rerank": int(any(cid in relevant for cid in reranked_ids)),
        }
        for key in tot:
            tot[key] += marks[key]
        print(
            f"   {g['query'][:46]:46s}   {marks['hybrid']}        {marks['rerank']}"
        )
    n = len(golden)
    print("   " + "-" * 64)
    print(
        f"   {'HIT-RATE@3':46s}  {tot['hybrid']/n:.2f}     {tot['rerank']/n:.2f}"
    )

    rule("4) LangGraph node: secure retrieve + rerank")

    class S(TypedDict):
        query: str
        user: UserContext
        results: list

    def secure_retrieve_and_rerank(state: S) -> dict:
        hyb = make_hybrid(store, docs, state["user"], k=RECALL_K)
        cands = hyb.invoke(state["query"])
        top = rerank(reranker, state["query"], cands)
        return {"results": [d.metadata["section"] for d in top]}

    graph = StateGraph(S)
    graph.add_node("secure_retrieve_and_rerank", secure_retrieve_and_rerank)
    graph.add_edge(START, "secure_retrieve_and_rerank")
    graph.add_edge("secure_retrieve_and_rerank", END)
    app = graph.compile()
    out = app.invoke({"query": q, "user": admin})
    print(f"   {out['results']}")


if __name__ == "__main__":
    main()
