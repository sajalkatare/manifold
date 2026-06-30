import os
import sys
import json
import argparse
from typing import TypedDict
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()
EMBED_MODEL = "text-embedding-3-small"
COLLECTION = "session2_kb"
DEFAULT_DSN = "postgresql+psycopg://rag:rag@localhost:5433/ragdb"



def pg_connection() -> str:
    """PGVector needs the psycopg3 SQLAlchemy URL (postgresql+psycopg://...)."""
    dsn = os.getenv("DATABASE_URL", DEFAULT_DSN)
    if dsn.startswith("postgresql://"):
        dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    return dsn


def get_embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBED_MODEL)


def load_chunks(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"{path} not found. Run: python build_corpus.py")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_documents(chunks: list[dict]) -> list[Document]:
    """Each chunk -> a LangChain Document: page_content + metadata travel together."""
    return [Document(page_content=c["content"],
                     metadata={"source": c.get("source"), "page": c.get("page"),
                               "section": c.get("section"), "chunk_id": c.get("chunk_id"),
                               "document_type": c.get("document_type")})
            for c in chunks]


def build_store(docs: list[Document], emb: OpenAIEmbeddings) -> PGVector:
    try:
        store = PGVector(embeddings=emb, collection_name=COLLECTION,
                         connection=pg_connection(), use_jsonb=True,
                         pre_delete_collection=True)   # fresh build each run
    except Exception as e:
        print(f"cannot connect to Postgres ({e}). Start it with: docker compose up -d")
    # NOTE: no custom ids. langchain_pg_embedding's id is globally unique across
    # collections, so reusing chunk_ids as ids would make a second collection's
    # inserts hit ON CONFLICT DO NOTHING. chunk_id lives in metadata instead.
    store.add_documents(docs)
    return store


def rule(s): print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)


def main(query):
    chunks = load_chunks("RAG\\data\\chunks.json")
    docs = to_documents(chunks)

    emb = get_embedder()
    rule(f"1) EMBED + STORE  (LangChain OpenAIEmbeddings -> PGVector '{COLLECTION}')")
    store = build_store(docs, emb)
    print(f"   embedded + stored {len(docs)} chunks (openai:{EMBED_MODEL} -> pgvector)")
    print("   content + embedding + metadata persisted in one pgvector row each")

    rule("2) FIRST VECTOR SEARCH  (LangChain similarity_search_with_score, top-5)")
    print(f"   query: {query!r}\n")
    for rank, (doc, distance) in enumerate(store.similarity_search_with_score(query, k=5), 1):
        m = doc.metadata
        print(f"   {rank}. dist={distance:.3f}  [{m['source']} · {m['section']}]")
        print(f"      {doc.page_content[:70]}...")
    print("\n   Every result has a source you can cite. That's production retrieval.")

    # ---- the SAME search, as a minimal LangGraph retrieval graph -------------
    rule("3) THE SAME SEARCH AS A LANGGRAPH NODE  (seed of the Session 4 pipeline)")

    class RetrievalState(TypedDict):
        query: str
        results: list

    def retrieve(state: RetrievalState) -> dict:
        hits = store.similarity_search_with_score(state["query"], k=5)
        return {"results": [(d.metadata["section"], round(s, 3)) for d, s in hits]}

    graph = StateGraph(RetrievalState)
    graph.add_node("retrieve", retrieve)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", END)
    app = graph.compile()

    out = app.invoke({"query": query})
    print(f"   graph: START -> retrieve -> END")
    print(f"   query: {query!r}")
    for rank, (section, dist) in enumerate(out["results"], 1):
        print(f"   {rank}. dist={dist:.3f}  [{section}]")
    print("\n   Identical results — now retrieval is a graph node you can compose.")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="How many weeks of parental leave do I get?")
    main(ap.parse_args().query)
